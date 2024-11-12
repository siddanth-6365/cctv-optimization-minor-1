import os
import numpy as np
import cv2
import time as tm
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VID_PATH = "/Users/siddanthreddy/Code/projects/minor-1/input/highway_input.mp4"

CONTINUITY_THRESHOLD = 10  # For cutting out boxes
MIN_SECONDS = 2            # Minimum duration of a moving object (in seconds)
INTERVAL_BW_DIVISIONS = 10 # For distributing moving objects over a duration to reduce overlapping (in seconds)
GAP_BW_DIVISIONS = 1.5    # Gap between divisions (in seconds)

logging.info("Starting video processing...")

# Initialize video capture
cap = cv2.VideoCapture(VID_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()
if not ret:
    logging.error("Failed to read the video file.")
    exit()

all_conts = []
avg2 = np.float32(frame)  # For background extraction

logging.info("Loading YOLO model...")
# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Ensure you have the correct model file

# Load class names
YOLO_NAMES_PATH = "coco.names"
with open(YOLO_NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Relevant classes (adjust based on your needs)
relevant_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

logging.info("Extracting bounding boxes using YOLO...")

frame_count = 0
while ret:
    # Background extraction
    try:
        cv2.accumulateWeighted(frame, avg2, 0.01)
    except Exception as e:
        logging.error(f"Error in accumulateWeighted: {e}")
        break

    # Run YOLO object detection
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

    frame_count += 1  # Increment frame count after processing
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
background = cv2.convertScaleAbs(avg2)
logging.info("Finished extracting bounding boxes.")

def get_centres(p1):
    return np.transpose(np.array([p1[:, 0] + p1[:, 2]/2, p1[:, 1] + p1[:, 3]/2]))

def distance(p1, p2):
    p1 = np.expand_dims(p1, 0)
    if p2.ndim == 1:
        p2 = np.expand_dims(p2, 0)
    c1 = get_centres(p1)
    c2 = get_centres(p2)
    return np.linalg.norm(c1 - c2, axis=1)

def get_nearest(p1, points):
    """Returns index of the point in points that is closest to p1."""
    return np.argmin(distance(p1, points))

class box:
    def __init__(self, coords, time):
        self.coords = coords  # Coordinates (x, y, w, h)
        self.time = time      # Frame number/time

class moving_obj:
    def __init__(self, starting_box):
        self.boxes = [starting_box]
    
    def add_box(self, box):
        self.boxes.append(box)
    
    def last_coords(self):
        return self.boxes[-1].coords
    
    def age(self, curr_time):
        last_time = self.boxes[-1].time
        return curr_time - last_time 

logging.info("Associating boxes into moving objects...")

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
        new_box = box(new_coords, curr_time)

        if new_assoc is not None:
            moving_objs[new_assoc].add_box(new_box)
        else:
            new_moving_obj = moving_obj(new_box)
            moving_objs.append(new_moving_obj)

logging.info("Finished associating boxes.")

def cut(image, coords):
    (x, y, w, h) = coords
    return image[y:y+h, x:x+w]

def overlay(frame, image, coords):
    (x, y, w, h) = coords
    # Ensure coordinates are within frame boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    frame[y:y+h, x:x+w] = cut(image, (x, y, w, h))

def sec2HMS(seconds):
    return tm.strftime('%M:%S', tm.gmtime(seconds))

def frame2HMS(n_frame, fps):
    return sec2HMS(float(n_frame) / float(fps))

# Filter out moving objects that are too short
MIN_FRAMES = MIN_SECONDS * fps
moving_objs = [obj for obj in moving_objs if (obj.boxes[-1].time - obj.boxes[0].time) >= MIN_FRAMES]
logging.info(f"Number of significant moving objects: {len(moving_objs)}")

max_orig_len = max(obj.boxes[-1].time for obj in moving_objs)
max_duration = max((obj.boxes[-1].time - obj.boxes[0].time) for obj in moving_objs)
start_times = [obj.boxes[0].time for obj in moving_objs]
N_DIVISIONS = int(max_orig_len / (INTERVAL_BW_DIVISIONS * fps))

final_video_length = int(max_duration + N_DIVISIONS * GAP_BW_DIVISIONS * fps + 10)
final_video = [background.copy() for _ in range(final_video_length)]

logging.info("Overlaying moving objects onto the background...")

cap = cv2.VideoCapture(VID_PATH)
if not cap.isOpened():
    logging.error("Failed to open the video file during overlay.")
    exit()

all_texts = []

for obj_idx, mving_obj in enumerate(moving_objs):
    for box_instance in mving_obj.boxes:
        # Read the frame at box_instance.time
        cap.set(cv2.CAP_PROP_POS_FRAMES, box_instance.time)
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Failed to read frame at time {box_instance.time}")
            continue

        division_factor = int(start_times[obj_idx] / (INTERVAL_BW_DIVISIONS * fps))
        final_time = box_instance.time - start_times[obj_idx] + int(division_factor * GAP_BW_DIVISIONS * fps)

        if final_time - 1 < len(final_video):
            overlay(final_video[final_time - 1], frame, box_instance.coords)
            (x, y, w, h) = box_instance.coords
            all_texts.append((final_time - 1, frame2HMS(box_instance.time, fps), (x + int(w / 2), y + int(h / 2))))
        else:
            logging.warning(f"Frame index {final_time - 1} out of bounds.")

cap.release()
cv2.destroyAllWindows()
logging.info("Finished overlaying moving objects.")

# Annotate moving objects
logging.info("Annotating moving objects in the final video...")
for (t, text, org) in all_texts:
    if t < len(final_video):
        cv2.putText(final_video[t], text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 240, 3), 1)
    else:
        logging.warning(f"Annotation frame index {t} out of bounds.")

filename = os.path.basename(VID_PATH).split('.')[0]
output_path = f'{filename}_summary.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed to 'mp4v' for compatibility
out = cv2.VideoWriter(output_path, fourcc, fps, (background.shape[1], background.shape[0]))

logging.info(f"Writing the final video to {output_path}...")

for idx, frame in enumerate(final_video):
    out.write(frame)
    # Optional: Display the frame
    # cv2.imshow('Final Video', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        logging.info("Video writing interrupted by user.")
        break

out.release()
cv2.destroyAllWindows()
logging.info("Video processing completed successfully.")
