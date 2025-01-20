## CCTV Footage Optimization
This project Summarize hours of footage captured by static CCTV cameras into a concise clip that presents all events as if they are happening simultaneously, complete with timestamps. It utilizes video processing techniques, including YOLO-based object detection, custom tracking algorithms, and video summarization methods, to identify, track, and summarize moving objects efficiently

## Problem
In today's world, many CCTV cameras are increasingly being used in homes for security. However, reviewing many hours of footage to spot suspicious activity can be a time-consuming and tedious process. Storing these lengthy videos also requires significant memory, often leading to reduced video resolution to save space. There is a need for a solution that can condense long footage into short clips while preserving key events.

## Working of Project:
1. Video Preprocessing
- The video is read using OpenCV, and the frames per second (fps) are extracted.
A background model is created using cv2.accumulateWeighted for overlaying tracked objects.
Frames are processed one by one until the video ends.
2. Object Detection
- The YOLOv8 model detects objects in each frame, filtering those of interest (e.g., people, cars) with a confidence score > 0.5.
Bounding box coordinates and object labels are extracted for further processing.
3. Object Tracking
- Objects are tracked across frames using spatial proximity and temporal continuity.
A custom algorithm links bounding boxes, tracks movements, and handles object disappearance.
Classes like Box and MovingObject store tracking data.
4. Video Summarization
- A summarized video is created by overlaying tracked objects with labels and timestamps on the background frame.
Frames are compiled into a short video using OpenCV's cv2.VideoWriter.
5. Logging and Reporting
- A log tracks each object's type, appearance time, disappearance time, and duration.
Logs are displayed in a UI and stored for analysis, with error handling for failed detections or frame reads.

## Results :
![demo](https://github.com/user-attachments/assets/3f52e0d5-faab-4693-b12d-0de6767351e7)

### Workflow
![image](https://github.com/user-attachments/assets/915bca55-faf4-421f-beb8-c9a0564c298a)
