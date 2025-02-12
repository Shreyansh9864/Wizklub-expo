import cv2 as cv
import mediapipe as mp
import time

# Initialize MediaPipe Hands and Gesture Recognizer
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Gesture Recognition Modules
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Function to initialize hand gesture recognizer and display results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    gesture_name = result.gestures[0].category_name if result.gestures else "No Gesture Detected"
    print(f'Gesture recognized: {gesture_name}')
    # Add gesture name on the output image
    cv.putText(output_image, f'Gesture: {gesture_name}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Create Gesture Recognizer instance
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=r"C:\Users\rahul\PycharmProjects\Noob to pro opencv\gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    # Initialize VideoCapture and MediaPipe Hands
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        exit()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # Convert the frame to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # If hand landmarks are detected, draw them
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(thickness=2, color=(0, 255, 0), circle_radius=3),
                    connection_drawing_spec=mp_draw.DrawingSpec(thickness=2, color=(0, 0, 255)),
                )

        # Recognize gestures and update the frame with the gesture name
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Get the current timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # Pass both image and timestamp to the recognizer
        try:
            recognizer.recognize_async(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error during gesture recognition: {e}")
            continue

        # Display the frame
        cv.imshow("Hand Landmark Detection with Gesture", frame)

        # Exit the loop when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()
