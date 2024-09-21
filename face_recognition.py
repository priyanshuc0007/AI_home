import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('facefeatures_new_model.h5')

# Define class labels (based on your training dataset structure)
class_labels = {0: 'amit', 1:'aryan'}  # Replace with actual labels based on your dataset

# Function to preprocess frames from the webcam
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to match model input
    frame_array = np.array(frame_resized, dtype="float32")
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array /= 255.0  # Normalize pixel values
    return frame_array

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Preprocess the frame for prediction
    processed_frame = preprocess_frame(frame)

    # Make a prediction
    prediction = model.predict(processed_frame)
    class_index = np.argmax(prediction, axis=1)[0]

    # Map the class index to the class label
    predicted_label = class_labels[class_index]

    # Display the result on the frame
    cv2.putText(frame, f"Class: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-Time Prediction', frame)

    # Press 'q' to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
