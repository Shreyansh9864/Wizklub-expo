import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# OpenCV Camera Capture
cap = cv2.VideoCapture(0)

# Screen Size
screen_width, screen_height = pyautogui.size()

# Hand Detection Model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Smoothening Variables
prev_x, prev_y = 0, 0
smooth_factor = 5  # Adjust for smoother movement

# FPS Calculation
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Flip frame and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)

    # Get frame size
    frame_height, frame_width, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw Hand Landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get Finger Tip Landmarks
            index_tip = hand_landmarks.landmark[8]   # Index Finger Tip
            thumb_tip = hand_landmarks.landmark[4]   # Thumb Tip
            middle_tip = hand_landmarks.landmark[12] # Middle Finger Tip

            # Convert Index Finger to Screen Coordinates
            x, y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

            # Smooth movement
            smooth_x = (prev_x + x) / smooth_factor
            smooth_y = (prev_y + y) / smooth_factor

            # Move Mouse Cursor
            screen_x = screen_width * (smooth_x / frame_width)
            screen_y = screen_height * (smooth_y / frame_height)
            pyautogui.moveTo(screen_x, screen_y)

            # Update previous coordinates
            prev_x, prev_y = smooth_x, smooth_y

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on Screen
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show Output
    cv2.imshow("Finger Mouse (Pinch to Click)", frame)

    # Detect Window Close Event
    if cv2.getWindowProperty("Finger Mouse (Pinch to Click)", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed!")
        break

cap.release()
cv2.destroyAllWindows()
