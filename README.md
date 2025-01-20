The Adaptive Surveillance System project processes videos to detect, track, and summarize moving objects using a combination of YOLO-based object detection, custom tracking algorithms, and video summarization techniques. Below is a detailed breakdown of the current code’s functionality.

Key Components of the Code
	1.	Video Preprocessing
	2.	Object Detection
	3.	Object Tracking
	4.	Video Summarization
	5.	Logging and Reporting

1. Video Preprocessing
	•	File Reading:
	•	The video is read using OpenCV’s cv2.VideoCapture.
	•	The fps (frames per second) is extracted for time-based calculations.
	•	Running Average for Background Creation:
	•	cv2.accumulateWeighted is used to compute a background model by averaging pixel values across frames. This helps in creating a static background for overlaying tracked objects.
	•	Frame Handling:
	•	Frames are processed one at a time in a while loop until the video ends.

2. Object Detection
	•	YOLO Model Integration:
	•	The YOLOv8 model is loaded using the Ultralytics library with a pre-trained weights file (yolov8n.pt).
	•	Relevant Object Filtering:
	•	YOLO detects all objects in each frame, but only objects of interest (e.g., person, car, bicycle) with a confidence score above 0.5 are retained.
	•	Bounding Box Extraction:
	•	For each detected object:
	•	Bounding box coordinates (x1, y1, x2, y2) are extracted.
	•	These coordinates are converted to a structured format for further processing.
	•	The bounding box data is stored along with the detected object’s label.

3. Object Tracking
	•	Purpose:
	•	To associate detected objects across frames and identify continuous movements.
	•	Nearest Neighbor Association:
	•	The algorithm uses the spatial proximity of bounding boxes to link objects across consecutive frames.
	•	A custom function (get_nearest) calculates the Euclidean distance between the center points of bounding boxes.
	•	Temporal Continuity:
	•	Objects are tracked only if they remain visible for at least MIN_SECONDS (converted to frames).
	•	Objects disappearing for more than CONTINUITY_THRESHOLD frames are considered “lost.”
	•	Custom Classes:
	•	Box: Represents a single detected object’s bounding box, coordinates, and label.
	•	MovingObject: Represents a tracked object across multiple frames, storing its history of bounding boxes and timestamps.

4. Video Summarization
	•	Background Frame:
	•	The averaged background is used as the base frame for summarization.
	•	Overlaying Objects:
	•	Detected and tracked objects are overlayed onto the background. Each object is annotated with:
	•	Label: The object’s class (e.g., car, person).
	•	Timestamps: The time when the object appeared and disappeared.
	•	Compilation of Summary Video:
	•	OpenCV’s cv2.VideoWriter compiles the annotated frames into a summarized video.
	•	Frames are written in sequence, maintaining the original video’s resolution and frame rate.

5. Logging and Reporting
	•	Object Logs:
	•	A log is maintained for each tracked object. This includes:
	•	Label: The type of object (e.g., bus, truck).
	•	Start Time: The timestamp of the first appearance.
	•	End Time: The timestamp of the last appearance.
	•	Duration: The total time the object was visible.
	•	Usage of Logs:
	•	Logs are displayed in the Streamlit UI and stored for further analysis.
	•	Error Handling:
	•	OpenCV’s read operations and YOLO detections are monitored, and errors are logged when frames or detections fail.

Code Structure

The code is divided into modular sections for maintainability and clarity:
	1.	Main Processing Function:
	•	process_video(VID_PATH):
	•	Coordinates the entire pipeline:
	•	Reads the video.
	•	Detects objects.
	•	Tracks objects across frames.
	•	Generates the summarized video and logs.
	2.	Supporting Functions:
	•	associate_moving_objects(all_conts, fps):
	•	Handles object tracking and continuity checks.
	•	overlay_moving_objects(moving_objs, background, VID_PATH, fps):
	•	Adds tracked objects and annotations to the background and compiles the summary video.
	•	write_output_video(final_video, background, fps):
	•	Writes the final summary video to disk.
	3.	Custom Utility Functions:
	•	get_centres(p1): Computes the center points of bounding boxes.
	•	distance(p1, p2): Calculates Euclidean distance between two points.
	•	get_nearest(p1, points): Finds the closest object to a given point.
	•	frame2HMS(n_frame, fps): Converts frame numbers to timestamps (HH:MM:SS format).
	•	overlay(frame, image, coords): Overlays a bounding box onto a frame with annotations.

Parameters and Constants
	•	YOLO_MODEL_PATH: Path to the YOLOv8 weights file.
	•	YOLO_NAMES_PATH: Path to the COCO class labels file.
	•	CONTINUITY_THRESHOLD: Maximum gap (in frames) for tracking continuity.
	•	MIN_SECONDS: Minimum time (in seconds) for an object to be considered significant.
	•	INTERVAL_BW_DIVISIONS: Interval between divisions in the summary video.
	•	GAP_BW_DIVISIONS: Time gap between divisions in the summary video.

Workflow
	1.	Initialization:
	•	Load YOLO model and set parameters.
	•	Read video and initialize data structures for tracking.
	2.	Frame-by-Frame Processing:
	•	Detect objects using YOLO.
	•	Filter and store relevant detections.
	•	Track objects using spatial proximity and continuity checks.
	3.	Summarization:
	•	Compile detected objects into a summary video with annotations.
	•	Log information about each tracked object.
	4.	Output:
	•	Save and display the summary video.
	•	Provide detailed logs for analysis.

Key Features
	•	Real-Time Object Detection: YOLOv8 provides fast and accurate detection of relevant objects.
	•	Custom Tracking: Tracks objects across frames using a lightweight and efficient algorithm.
	•	Video Summarization: Outputs a concise, annotated summary video highlighting significant movements.
	•	User-Friendly Interface: Streamlit UI allows users to upload videos, view logs, and download results seamlessly.

Challenges Addressed
	•	Overlapping Objects: Handled using nearest-neighbor matching and continuity thresholds.
	•	Low-Quality Videos: Optimized YOLO confidence thresholds and applied filtering to minimize false detections.
	•	Performance Optimization: Reduced processing time by minimizing redundant computations and leveraging efficient libraries like OpenCV and NumPy.

Conclusion

This project effectively combines state-of-the-art object detection (YOLO) with custom tracking and summarization logic to create a robust surveillance system. Its modular design ensures scalability, and the integration with Streamlit makes it accessible for non-technical users.
