import streamlit as st
import cv2
import easyocr
import numpy as np
import tempfile
import os
import ultralytics
from ultralytics import YOLO
from norfair import Detection, Tracker
import os
os.environ["STREAMLIT_ENV"] = "production"

# Load YOLO model
model = YOLO('best.pt')

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def euclidean(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

# Initialize NorFair Tracker
tracker = Tracker(distance_function=euclidean, distance_threshold=50)

# Define colors for each class
colors = {
    0: (0, 165, 255),   # Ball - Orange
    1: (0, 0, 255),     # Hoop - Red
    2: (128, 0, 128),   # Jersey - Purple
    3: (0, 255, 255),   # Referee - Cyan
    4: (255, 255, 0),   # Team 2 with Ball - Yellow
    5: (255, 0, 0),     # Team 1 - Blue
    6: (0, 255, 0),     # Team 1 with Ball - Green
    7: (255, 165, 0)    # Team 2 - Orange
}

# Function to extract text (jersey number) using EasyOCR
def extract_jersey_number(jersey_crop):
    gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    results = reader.readtext(gray)

    for (bbox, text, prob) in results:
        if prob > 0.35:
            text = ''.join(filter(str.isdigit, text))
            if text:
                return text
    return None

# Streamlit Page Config
st.set_page_config(page_title="🏀 Basketball Analyzer", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Home", "Video Analyzer"])

# Home Page
if page == "Home":
    st.title("🏀 Basketball Video Analyzer")
    st.markdown("""
    Welcome to the **Basketball Video Analyzer**! 🏀🎥

    This tool allows you to:
    - Upload and analyze basketball game videos
    - Detect ball possession for each team
    - Extract jersey numbers using OCR
    - Track player movement with real-time object detection and tracking

    Built using:
    - **YOLO** for object detection
    - **EasyOCR** for reading jersey numbers
    - **Norfair** for tracking player movement
    - **Streamlit** for interactive UI

    Get started by selecting the **Video Analyzer** page from the sidebar.
    """)

# Video Analyzer Page
elif page == "Video Analyzer":
    st.title("🏀 Basketball Video Analyzer")
    st.write("Upload a video to detect ball possession and count team possession times.")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save uploaded file to temp path
        temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_input_path)

        cap = cv2.VideoCapture(temp_input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        jersey_numbers = {}
        team1_with_ball_count = 0
        team2_with_ball_count = 0
        previously_counted = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            detections = []
            jersey_bboxes = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls == 2:
                        jersey_crop = frame[y1:y2, x1:x2]
                        jersey_number = extract_jersey_number(jersey_crop)
                        jersey_bboxes.append((x1, y1, x2, y2, jersey_number))

                    detections.append(Detection(points=np.array([(x1 + x2) / 2, (y1 + y2) / 2]), data={"cls": cls, "bbox": (x1, y1, x2, y2)}))

            tracked_objects = tracker.update(detections)

            for track in tracked_objects:
                center_point = track.estimate
                x, y = center_point[0]
                data = track.last_detection.data
                x1, y1, x2, y2 = data["bbox"]
                cls = data["cls"]
                track_id = track.id

                if cls in [4, 6]:  # Team with ball
                    if track_id not in previously_counted:
                        if cls == 4:
                            team2_with_ball_count += 1
                        elif cls == 6:
                            team1_with_ball_count += 1
                        previously_counted.add(track_id)

                if cls in [4, 5, 6, 7]:
                    jersey_number = None
                    for (jx1, jy1, jx2, jy2, jnum) in jersey_bboxes:
                        if x1 < jx1 < x2 and y1 < jy1 < y2 and jnum is not None:
                            jersey_number = jnum
                            break

                    if track_id in jersey_numbers:
                        prev_number = jersey_numbers[track_id]
                        if jersey_number is not None:
                            jersey_numbers[track_id] = jersey_number
                        else:
                            jersey_number = prev_number
                    else:
                        jersey_numbers[track_id] = jersey_number

                    final_number = jersey_numbers.get(track_id, " ")
                    color = colors.get(cls, (255, 255, 255))
                    label = f"{model.names[cls]} - {final_number}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = colors.get(cls, (255, 255, 255))
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

        cap.release()
        out.release()

        # Display possession in columns
        total_possession = team1_with_ball_count + team2_with_ball_count
        if total_possession == 0:
            team1_percent = 0
            team2_percent = 0
        else:
            team1_percent = (team1_with_ball_count / total_possession) * 100
            team2_percent = (team2_with_ball_count / total_possession) * 100

        col1, col2 = st.columns(2)
        col1.metric(label="Team 1 Possession %", value=f"{team1_percent:.2f}%")
        col2.metric(label="Team 2 Possession %", value=f"{team2_percent:.2f}%")

        # Provide download button
        with open(temp_output_path, "rb") as f:
            st.download_button("📥 Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")

        # Clean up temp files
        os.remove(temp_input_path)
        os.remove(temp_output_path)
