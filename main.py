import cv2
import mediapipe as mp
import pyfirmata


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        # Initialize the FaceDetector with minimum detection confidence and model selection
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        # Import face detection and drawing utilities from MediaPipe
        self.mp_face_detection = mp.solutions.face_detection

        # Create a face detection model with the given parameters
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
            model_selection=self.model_selection,
        )

    def find_faces(self, image):
        """
        Find faces in the image using the MediaPipe face detection model.
        """

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect faces
        self.results = self.face_detection.process(img_rgb)

        # List to store bounding boxes and face information
        face_bboxes = []

        if self.results.detections:
            for detection_id, detection in enumerate(self.results.detections):
                # Only process if detection score is above the threshold
                if detection.score[0] > self.min_detection_confidence:
                    # Get the bounding box data in a relative format
                    bounding_box = detection.location_data.relative_bounding_box
                    image_height, image_width, _ = image.shape

                    # Convert the relative bounding box to pixel values
                    bbox = (
                        int(bounding_box.xmin * image_width),
                        int(bounding_box.ymin * image_height),
                        int(bounding_box.width * image_width),
                        int(bounding_box.height * image_height),
                    )

                    # Calculate the center of the bounding box
                    center_x = bbox[0] + (bbox[2] // 2)
                    center_y = bbox[1] + (bbox[3] // 2)

                    # Create a dictionary to store face information
                    face_info = {
                        "id": detection_id,
                        "bbox": bbox,
                        "score": detection.score,
                        "center": (center_x, center_y),
                    }

                    # Append the face information to the list
                    face_bboxes.append(face_info)

        # Return the image (with optional drawings) and the list of detected faces
        return image, face_bboxes


class FaceTrackingSystem:
    def __init__(self, camera_index=0, width=1280, height=720, port="COM4", pan_pin=9, tilt_pin=10):
        # Initialize camera capture with given index
        self.cap = cv2.VideoCapture(camera_index)
        
        self.frame_width = width
        self.frame_height = height

        # Set camera frame dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Check if camera is accessible
        if not self.cap.isOpened():
            print("Unable to access camera!")
            exit()

        # Initialize Arduino board and set pins for pan and tilt servos
        self.board = pyfirmata.Arduino(port)
        
        # Get the pins for the pan and tilt servos
        self.pan_servo_pin = self.board.get_pin(f"d:{pan_pin}:s")
        self.tilt_servo_pin = self.board.get_pin(f"d:{tilt_pin}:s")

        # Initialize face detector with default confidence and model selection
        self.face_detector = FaceDetector(min_detection_confidence=0.5, model_selection=1)

        # Initial servo positions (90 degrees - neutral position)
        self.servo_positions = [90, 90]

        # Calculate screen center
        self.screen_center_x = self.frame_width // 2
        self.screen_center_y = self.frame_height // 2

        # Threshold for face position tolerance before moving servos
        self.face_center_threshold = 50

        # Define font for drawing text
        self.font = cv2.FONT_HERSHEY_SIMPLEX    

        # Define text color
        self.text_color = (255, 0, 0)

        # Define bounding box color
        self.bbox_color = (0, 0, 255)

    def move_servos(self, x_diff, y_diff):
        """
        Adjust the pan and tilt servos based on the face position differences.
        """

        # Adjust pan servo (X-axis) if face is outside the threshold
        if abs(x_diff) > self.face_center_threshold:
            if x_diff > 0:
                self.servo_positions[0] -= 1  # Move left
            else:
                self.servo_positions[0] += 1  # Move right

        # Adjust tilt servo (Y-axis) if face is outside the threshold
        if abs(y_diff) > self.face_center_threshold:
            if y_diff > 0:
                self.servo_positions[1] -= 1  # Move down
            else:
                self.servo_positions[1] += 1  # Move up

        # Clamp servo positions to the range [0, 180]
        self.servo_positions[0] = max(0, min(self.servo_positions[0], 180))
        self.servo_positions[1] = max(0, min(self.servo_positions[1], 180))

        # Send the servo positions to the Arduino
        self.pan_servo_pin.write(self.servo_positions[0])
        self.tilt_servo_pin.write(self.servo_positions[1])

    def draw_interface(self, image, face_bboxes):
        """
        Draw targeting interface and crosshairs on the image.
        """

        # Get the score of the detection
        score_value = round(face_bboxes[0]["score"][0] * 100)
        score = f"{score_value}%"

        # Get the bounding box coordinates
        bbox = face_bboxes[0]["bbox"]

        # Draw the bounding box and confidence score on the image
        image = cv2.rectangle(image, bbox, self.bbox_color, 2)

        # Draw the confidence score above the bounding box
        cv2.putText(image, score, (bbox[0], bbox[1] - 20), self.font, 2, self.bbox_color, 2)

        # Indicate target lock
        cv2.putText(image, "TARGET LOCKED", (780, 50), self.font, 2, self.bbox_color, 2)

    def draw_servo_positions(self, image):
        """
        Draw the current servo positions on the image.
        """

        # Draw the servo positions
        cv2.putText(image, f"Servo X: {int(self.servo_positions[0])} deg", (50, 50), self.font, 1, self.text_color, 2)
        cv2.putText(image, f"Servo Y: {int(self.servo_positions[1])} deg", (50, 100), self.font, 1, self.text_color, 2)

    def cleanup(self):
        """
        Clean up resources.
        """

        # Release the camera and close the Arduino conection    
        cv2.destroyAllWindows()
        self.cap.release()
        self.board.exit()

    def run(self):
        """
        Main loop to process video frames, detect faces, and control servos.
        """

        while True:
            # Capture the frame from the camera
            success, image = self.cap.read()
            if not success:
                print("Failed to capture image")
                break

            # Detect faces in the frame
            image, face_bboxes = self.face_detector.find_faces(image)

            if face_bboxes:
                # Get the center of the first detected face
                face_x, face_y = face_bboxes[0]["center"]

                # Calculate the difference between face center and screen center
                x_diff = face_x - self.screen_center_x
                y_diff = face_y - self.screen_center_y

                # Move servos based on the face position
                self.move_servos(x_diff, y_diff)

                # Draw the targeting interface on the frame
                self.draw_interface(image, face_bboxes)
            
            else:
                # If no face is detected, display a default "No Target" message
                cv2.putText(image, "NO TARGET", (850, 50), self.font, 2, (0, 0, 255), 2)

            # Display the current servo angles on the frame
            self.draw_servo_positions(image)

            # Display the frame with interface
            cv2.imshow("Face Tracking System", image)

            # Exit the loop if 'q' key is pressed
            if cv2.waitKey(1) == ord("q"):
                break

        # Cleanup: Close the camera and windows
        self.cleanup()


def main():
    """
    Main function to initialize and run the face tracking system.
    """

    # Set camera index, width, height, port, pan pin, and tilt pin
    CAMERA_INDEX = 1
    WIDTH = 1280
    HEIGHT = 720
    PORT = "COM4"
    PAN_PIN = 9
    TILT_PIN = 10

    # Initialize and run the face tracking system
    face_tracking_system = FaceTrackingSystem(camera_index=CAMERA_INDEX, width=WIDTH, height=HEIGHT, port=PORT, pan_pin=PAN_PIN, tilt_pin=TILT_PIN)
    face_tracking_system.run()


if __name__ == "__main__":
    main()
