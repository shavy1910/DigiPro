import cv2
import os
import numpy as np

# Paths
dataset_path = "faces_dataset"
model_path = "face_model.yml"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()

def register_face():
    """Register a new person's face."""
    person_name = input("Enter the name of the person: ")
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    cap = cv2.VideoCapture(0)
    print("Press 's' to save an image, 'q' to quit.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Registering Face", gray)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save image
            img_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(img_path, gray)
            print(f"Image {count} saved.")
            count += 1
        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

def train_model():
    """Train the face recognition model."""
    faces = []
    labels = []
    label_map = {}

    for label, person_name in enumerate(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_name)
        label_map[label] = person_name
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if face is not None:  # Ensure valid image
                faces.append(face)
                labels.append(label)

    if not faces:
        print("No faces found. Please register faces first.")
        return None

    labels = np.array(labels, dtype=np.int32)
    recognizer.train(faces, labels)
    recognizer.write(model_path)
    print("Model trained successfully and saved.")
    return label_map

def load_model():
    """Load the trained model and label map."""
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first.")
        return None
    recognizer.read(model_path)
    label_map = {label: person_name for label, person_name in enumerate(os.listdir(dataset_path))}
    return label_map

def recognize_face(label_map):
    """Recognize faces in real-time."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected_faces = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            
            try:
                label, confidence = recognizer.predict(face_resized)
                if confidence < 50:  # Confidence threshold
                    name = label_map.get(label, "Registered Face")
                    text = f"{name} ({int(confidence)}%)"
                else:
                    text = "Registered Face"
            except Exception as e:
                text = "Registered Face"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Recognizing Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1. Register a new face")
    print("2. Train model")
    print("3. Recognize faces")
    choice = int(input("Enter your choice: "))

    if choice == 1:
        register_face()
    elif choice == 2:
        label_map = train_model()
        if label_map is not None:
            print("Label map:", label_map)
    elif choice == 3:
        label_map = load_model()
        if label_map is not None:
            recognize_face(label_map)
    else:
        print("Invalid choice.")
