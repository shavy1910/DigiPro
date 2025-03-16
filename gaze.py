import cv2
import time

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables
gaze_start_time = None
interaction_time = 0
screen_gaze_threshold = 0.5  # Adjust threshold for gaze calculation

def calculate_gaze_direction(eyes):
    """
    Determine if the gaze direction indicates interaction with the screen.
    """
    if len(eyes) >= 2:  # Ensure both eyes are detected
        eye_centers = [(ex + ew // 2, ey + eh // 2) for ex, ey, ew, eh in eyes]
        average_x = sum(center[0] for center in eye_centers) / len(eye_centers)
        if average_x > screen_gaze_threshold:
            return "Screen"
        else:
            return "Away"
    return "Unknown"

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    gaze_direction = "Unknown"
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the ROI for eyes within the face area
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        
        # Draw rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

        # Calculate gaze direction
        gaze_direction = calculate_gaze_direction(eyes)
        
        # Track interaction time based on gaze direction
        if gaze_direction == "Screen":
            if gaze_start_time is None:
                gaze_start_time = time.time()
        else:
            if gaze_start_time is not None:
                interaction_time += time.time() - gaze_start_time
                gaze_start_time = None

    # Display gaze direction
    cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display interaction time
    cv2.putText(frame, f"Interaction Time: {interaction_time:.2f} seconds", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Gaze and Interaction Time', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
