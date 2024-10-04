# Face Tracking System

This project implements a face tracking system using OpenCV, MediaPipe, and Arduino. The system detects faces in real-time using a camera and adjusts pan-tilt servo motors to keep the detected face centered in the frame.

## Requirements

To run this project, you need to have the following libraries installed:

- Python >=3.8
- OpenCV
- MediaPipe
- PyFirmata

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

# Additional Requirements
- An Arduino board connected to your computer.
- Servo motors connected to the Arduino for pan and tilt movements.
- A camera that is compatible with OpenCV.

# Installation
1. Clone the repository:

```bash
git clone https://github.com/dzaja123/face-tracking-pan-tilt.git
cd face-tracking-system
```

2. Connect your Arduino board and upload the necessary firmware to enable communication with the Python script.

3. Connect the servo motors to the specified pins on the Arduino:
Pin Connections
Pan Servo Motor: Connect to the pin specified by PAN_PIN (e.g., Pin 9).
Tilt Servo Motor: Connect to the pin specified by TILT_PIN (e.g., Pin 10).

Power: Ensure that the servo motors are connected to an appropriate external 5V power source.
Ground: Connect the ground of the servos to the ground pin on the Arduino and to the ground of the external 5V source.

# Install the required libraries as mentioned above.

- Connect your Arduino board and upload the necessary firmware to enable communication with the Python script.
- Connect the servo motors to the specified pins on the Arduino.

# Usage

Make sure your camera and Arduino are connected.
Modify the parameters in the main() function of face_tracking.py if needed:

- CAMERA_INDEX: Index of the camera (0 for the default camera).
- WIDTH: Desired width of the video feed.
- HEIGHT: Desired height of the video feed.
- PORT: COM port where the Arduino is connected (e.g., "COM4" for Windows).
- PAN_PIN: Pin number for the pan servo motor.
- TILT_PIN: Pin number for the tilt servo motor.

# Run the program:

```bash
python face_tracking.py
```

Press 'q' to quit the application.

# Code Description

## FaceTrackingSystem Class:

Initializes the camera, Arduino connection, and face detector.
Captures frames from the camera and detects faces.
Controls the pan and tilt servo motors based on the face's position in the frame.
Draws the interface on the video feed with bounding boxes and servo positions.

## FaceDetector Class:

Utilizes MediaPipe for face detection.
Processes images to detect faces and returns bounding box coordinates and scores.
Logging:

Integrated logging provides real-time feedback and error messages.
