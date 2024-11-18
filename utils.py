import numpy as np
import time as tm
import cv2
import tempfile
import logging

def get_centres(p1):
    return np.transpose(np.array([p1[:, 0] + p1[:, 2] / 2, p1[:, 1] + p1[:, 3] / 2]))

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

def sec2HMS(seconds):
    return tm.strftime('%H:%M:%S', tm.gmtime(seconds))

def frame2HMS(n_frame, fps):
    return sec2HMS(float(n_frame) / float(fps))

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

def write_output_video(final_video, background, fps):
    filename = 'summary_video.mp4'

    # Use a temporary file to store the video
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_video_file.name

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' for H.264 encoding
    out = cv2.VideoWriter(output_path, fourcc, fps, (background.shape[1], background.shape[0]))

    for idx, frame in enumerate(final_video):
        # Ensure frame is in uint8 format
        frame_uint8 = frame.astype('uint8')
        out.write(frame_uint8)

    out.release()
    cv2.destroyAllWindows()

    return output_path
