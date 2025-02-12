import cv2
import mediapipe as mp

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe FaceMesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create FaceMesh instance with iris refinement
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define colors
WHITE = (255, 255, 255)  # Face Mesh
RED = (0, 0, 255)    # Left Eye Iris
GREEN = (0, 255, 0)  # Right Eye Iris
BLUE = (255, 0, 0)   # Iris Connections

# Landmark indices for irises
left_iris_indices = [468, 469, 470, 471, 472]  # Left Iris
right_iris_indices = [473, 474, 475, 476, 477]  # Right Iris

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            # Draw Face Mesh (White)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            # Draw Face Contours (White)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

            # Draw Iris Connections (Blue)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

            # Draw Left Iris (Red)
            for idx in left_iris_indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                cv2.circle(frame, (x, y), 2, RED, -1)

            # Draw Right Iris (Green)
            for idx in right_iris_indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                cv2.circle(frame, (x, y), 2, GREEN, -1)

    # Display the frame
    cv2.imshow("Face Mesh & Iris Tracking", frame)

    # Exit on pressing 'q' and pressing x
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("MediaPipe Face Detection", cv2.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv2.destroyAllWindows()
