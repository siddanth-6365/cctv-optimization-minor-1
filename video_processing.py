import cv2
import numpy as np
import time as tm
import logging
from ultralytics import YOLO
from models import Box, MovingObject
from utils import (get_centres, distance, get_nearest, sec2HMS, frame2HMS,
                   cut, overlay, write_output_video)
import tempfile

# Define constants
YOLO_MODEL_PATH = 'yolov8n.pt'       
YOLO_NAMES_PATH = 'coco.names'       

CONTINUITY_THRESHOLD = 10            # Threshold to consider an object as continuous
MIN_SECONDS = 2                      # Minimum duration to consider an object
INTERVAL_BW_DIVISIONS = 10           # Interval between divisions
GAP_BW_DIVISIONS = 1.5               # Gap between divisions

def process_video(VID_PATH):
    # Initialize variables
    cap = cv2.VideoCapture(VID_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read the video file.")
        return [], None

    all_conts = []
    avg2 = np.float32(frame)  

    # Load YOLO model
    model = YOLO(YOLO_MODEL_PATH) 

    with open(YOLO_NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    relevant_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

    frame_count = 0

    # Initialize counters for Retention Ratio
    total_detections = 0

    # While loop to read frames and detect objects
    while ret:
        cv2.accumulateWeighted(frame, avg2, 0.01)

        results = model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf)
                label = classes[class_id]

                if confidence >= 0.5 and label in relevant_class_names:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)

                    # Append bounding box
                    detections.append([x, y, w, h])

        # Convert detections to numpy array
        contours = np.array(detections)
        all_conts.append(contours)

        # Update total detections
        total_detections += len(detections)

        frame_count += 1
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
    background = cv2.convertScaleAbs(avg2)

    # Proceed with associating boxes into moving objects and creating summary video
    moving_objs, detections_in_summary = associate_moving_objects(all_conts, fps)
    final_video_length, final_video, all_texts, logs = overlay_moving_objects(moving_objs, background, VID_PATH, fps)

    # Write the final video
    output_path = write_output_video(final_video, background, fps)

    return logs, output_path

def associate_moving_objects(all_conts, fps):
    moving_objs = []
    for curr_time, new_boxes in enumerate(all_conts):
        if len(new_boxes) != 0:
            new_assocs = [None] * len(new_boxes)
            obj_coords = np.array([obj.last_coords() for obj in moving_objs if obj.age(curr_time) < CONTINUITY_THRESHOLD])
            unexp_idx = -1  # Index of unexpired objects
            for obj_idx, obj in enumerate(moving_objs):
                if obj.age(curr_time) < CONTINUITY_THRESHOLD:
                    unexp_idx += 1
                    nearest_new = get_nearest(obj.last_coords(), new_boxes)
                    nearest_obj = get_nearest(new_boxes[nearest_new], obj_coords)

                    if nearest_obj == unexp_idx:
                        new_assocs[nearest_new] = obj_idx

        for new_idx, new_coords in enumerate(new_boxes):
            new_assoc = new_assocs[new_idx] if 'new_assocs' in locals() else None
            new_box = Box(new_coords, curr_time)

            if new_assoc is not None:
                moving_objs[new_assoc].add_box(new_box)
            else:
                new_moving_obj = MovingObject(new_box)
                moving_objs.append(new_moving_obj)

    # Filter out moving objects that are too short
    MIN_FRAMES = MIN_SECONDS * fps
    moving_objs = [obj for obj in moving_objs if (obj.boxes[-1].time - obj.boxes[0].time) >= MIN_FRAMES]

    if not moving_objs:
        detections_in_summary = 0
    else:
        # Update detections_in_summary
        detections_in_summary = sum(len(obj.boxes) for obj in moving_objs)

    return moving_objs, detections_in_summary

def overlay_moving_objects(moving_objs, background, VID_PATH, fps):
    max_orig_len = max(obj.boxes[-1].time for obj in moving_objs) if moving_objs else 0
    max_duration = max((obj.boxes[-1].time - obj.boxes[0].time) for obj in moving_objs) if moving_objs else 0
    start_times = [obj.boxes[0].time for obj in moving_objs]
    N_DIVISIONS = int(max_orig_len / (INTERVAL_BW_DIVISIONS * fps)) if fps else 0

    final_video_length = int(max_duration + N_DIVISIONS * GAP_BW_DIVISIONS * fps + 10) if fps else 0
    final_video = [background.copy() for _ in range(final_video_length)]

    cap = cv2.VideoCapture(VID_PATH)
    if not cap.isOpened():
        logging.error("Failed to open the video file during overlay.")
        return [], [], [], []

    all_texts = []
    logs = []

    for obj_idx, mving_obj in enumerate(moving_objs):
        obj_start_time = mving_obj.boxes[0].time
        obj_end_time = mving_obj.boxes[-1].time
        obj_duration = (obj_end_time - obj_start_time) / fps
        start_timestamp = frame2HMS(obj_start_time, fps)
        logs.append(f"Object {obj_idx + 1}: Start Time = {start_timestamp}, Duration = {obj_duration:.2f} seconds")

        for box_instance in mving_obj.boxes:
            # Read the frame at box_instance.time
            cap.set(cv2.CAP_PROP_POS_FRAMES, box_instance.time)
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame at time {box_instance.time}")
                continue

            division_factor = int(start_times[obj_idx] / (INTERVAL_BW_DIVISIONS * fps))
            final_time = int(box_instance.time - start_times[obj_idx] + division_factor * GAP_BW_DIVISIONS * fps)

            if final_time - 1 < len(final_video):
                overlay(final_video[final_time - 1], frame, box_instance.coords)
                (x, y, w, h) = box_instance.coords
                all_texts.append((final_time - 1, frame2HMS(box_instance.time, fps), (x + int(w / 2), y + int(h / 2))))
            else:
                logging.warning(f"Frame index {final_time - 1} out of bounds.")

    cap.release()
    cv2.destroyAllWindows()

    # Annotate moving objects
    for (t, text, org) in all_texts:
        if t < len(final_video):
            cv2.putText(final_video[t], text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 240, 3), 1)
        else:
            logging.warning(f"Annotation frame index {t} out of bounds.")

    return final_video_length, final_video, all_texts, logs
