import cv2
import mediapipe as mp

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands and Drawing Utilities/Styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Optionally flip the frame horizontally for a mirror-view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as MediaPipe requires RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and connections using default styles
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

    # Display the frame
    cv2.imshow("Hand Detection", frame)

    # Exit on pressing 'q'
 if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("MediaPipe Face Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
