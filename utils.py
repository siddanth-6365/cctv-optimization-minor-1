import cv2
import os

def save_frame(frame, frame_count, relevant_count, irrelevant_count, frame_type):
    folder = 'relevant' if frame_type == 'relevant' else 'irrelevant'
    output_folder = os.path.join('/Users/siddanthreddy/Code/demotry/Adaptive-survillence-system/output', folder)
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

    # Set the filename and full path for the frame
    filename = f"frame_{relevant_count if frame_type == 'relevant' else irrelevant_count}.jpg"
    file_path = os.path.join(output_folder, filename)

    # Save the frame
    success = cv2.imwrite(file_path, frame)
    if success:
        print(f"Frame {frame_count}: {frame_type.capitalize()} - Saved to {file_path}")
    else:
        print(f"Error saving Frame {frame_count} as {file_path}")

def log_detection(log_data, frame_count, frame_type, timestamp):
    log_data.append({
        'Frame': frame_count,
        'Type': frame_type,
        'Timestamp': timestamp
    })
