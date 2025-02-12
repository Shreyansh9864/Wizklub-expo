import cv2
import face_recognition as fg
import os

# Directory containing reference images
directory = "Shreyansh"
files = os.listdir(directory)

# Ensure there is at least one image file
if not files:
    print("Error: No images found in the directory.")
    exit()

# Load and encode all the images in the directory
known_face_encodings = []
known_face_names = []

for file in files:
    image_path = os.path.join(directory, file)
    fac = fg.load_image_file(image_path)
    fac_encodings = fg.face_encodings(fac)

    if fac_encodings:
        known_face_encodings.append(fac_encodings[0])  # Add encoding for the first detected face
        known_face_names.append(os.path.splitext(file)[0])  # Use the filename as the name

# Check if any faces were encoded
if not known_face_encodings:
    print("Error: No faces detected in any of the reference images.")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the webcam frame
    face_locations = fg.face_locations(rgb_frame)
    face_encodings = fg.face_encodings(rgb_frame, face_locations)

    # Compare detected faces with reference faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compute face distances
        face_distances = fg.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        threshold = 0.6  # Lower means stricter match

        if face_distances[best_match_index] < threshold:
            match_text = known_face_names[best_match_index]  # Display the matched name
            color = (0, 255, 0)  # Green for match
        else:
            match_text = "Unknown"
            color = (0, 0, 255)  # Red for unknown

        # Draw a rectangle around the detected face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, match_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
