import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access camera. Try restarting.")
    exit()

# Create Face Detection Model
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Resize for better performance
    frame = cv2.resize(frame, (640, 480))

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Face Detection
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Draw Bounding Box around Face
            mp_drawing.draw_detection(frame, detection)

    # Show the Output
    cv2.imshow("MediaPipe Face Detection", frame)

    # Exit on 'q' or Close Button (❌)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("MediaPipe Face Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
