import cv2
import pyfirmata
from face_detector import FaceDetector


class FaceTrackingSystem:
    def __init__(self, camera_index=0, width=1280, height=720, port="COM4", pan_pin=9, tilt_pin=10):
        # Initialize the camera capture with the given index
        self.cap = cv2.VideoCapture(camera_index)
        
        # Set the dimensions of the camera frame
        self.frame_width = width
        self.frame_height = height

        # Set the camera frame dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Check if the camera is accessible
        if not self.cap.isOpened():
            print("Unable to access the camera!")
            exit()

        # Initialize the Arduino board and set up pins for pan and tilt servos
        self.board = pyfirmata.Arduino(port)
        
        # Define the pins for pan and tilt servos
        self.pan_servo_pin = self.board.get_pin(f"d:{pan_pin}:s")
        self.tilt_servo_pin = self.board.get_pin(f"d:{tilt_pin}:s")

        # Initialize the face detector with set confidence and model selection
        self.face_detector = FaceDetector(min_detection_confidence=0.5, model_selection=1)

        # Initial positions of the servo motors (90 degrees - neutral position)
        self.servo_positions = [90, 90]

        # Calculate the screen center
        self.screen_center_x = self.frame_width // 2
        self.screen_center_y = self.frame_height // 2

        # Tolerance for face position before moving the servo motors
        self.face_center_threshold = 35

        # Define font for drawing text
        self.font = cv2.FONT_HERSHEY_SIMPLEX    

        # Define text color
        self.text_color = (255, 0, 0)

        # Define bounding box color
        self.bbox_color = (0, 0, 255)

    def move_servos(self, x_diff, y_diff):
        """
        Control the pan and tilt servo motors based on the face position differences.
        """

        # Adjust the pan servo motor (X-axis) if the face is outside the allowed threshold
        if abs(x_diff) > self.face_center_threshold:
            if x_diff > 0:
                self.servo_positions[0] -= 1  # Move left
            else:
                self.servo_positions[0] += 1  # Move right

        # Adjust the tilt servo motor (Y-axis) if the face is outside the allowed threshold
        if abs(y_diff) > self.face_center_threshold:
            if y_diff > 0:
                self.servo_positions[1] -= 1  # Move down
            else:
                self.servo_positions[1] += 1  # Move up

        # Limit servo positions to the range [0, 180]
        self.servo_positions[0] = max(0, min(self.servo_positions[0], 180))
        self.servo_positions[1] = max(0, min(self.servo_positions[1], 180))

        # Send servo positions to the Arduino
        self.pan_servo_pin.write(self.servo_positions[0])
        self.tilt_servo_pin.write(self.servo_positions[1])

    def draw_interface(self, image, face_bboxes):
        """
        Draw the interface on the image.
        """

        # Get detection results
        score_value = round(face_bboxes[0]["score"][0] * 100)
        score = f"{score_value}%"

        # Get bounding box coordinates
        bbox = face_bboxes[0]["bbox"]

        # Draw the bounding box and confidence score on the image
        image = cv2.rectangle(image, bbox, self.bbox_color, 2)

        # Draw the confidence score above the bounding box
        cv2.putText(image, score, (bbox[0], bbox[1] - 20), self.font, 2, self.bbox_color, 2)

        # Indicate that the target is locked
        cv2.putText(image, "TARGET LOCKED", (780, 50), self.font, 2, self.bbox_color, 2)

    def draw_servo_positions(self, image):
        """
        Draw the current servo motor positions on the image.
        """

        # Draw servo motor positions
        cv2.putText(image, f"Servo X: {int(self.servo_positions[0])} deg", (50, 50), self.font, 1, self.text_color, 2)
        cv2.putText(image, f"Servo Y: {int(self.servo_positions[1])} deg", (50, 100), self.font, 1, self.text_color, 2)

    def cleanup(self):
        """
        Clean up resources.
        """

        # Release the camera and close the connection to Arduino
        cv2.destroyAllWindows()
        self.cap.release()
        self.board.exit()

    def run(self):
        """
        Main loop for processing the camera stream, detecting faces, and controlling servo motors.
        """

        while True:
            # Capture image from the camera
            success, image = self.cap.read()
            if not success:
                print("Unable to capture image")
                break

            # Detect faces in the image
            image, face_bboxes = self.face_detector.find_faces(image)

            if face_bboxes:
                # Get the center of the first detected face
                face_x, face_y = face_bboxes[0]["center"]

                # Calculate the difference between the face center and the screen center
                x_diff = face_x - self.screen_center_x
                y_diff = face_y - self.screen_center_y

                # Move the servo motors based on the difference
                self.move_servos(x_diff, y_diff)

                # Draw the interface on the image
                self.draw_interface(image, face_bboxes)
                
            else:
                cv2.putText(image, "NO TARGET", (850, 50), self.font, 2, (0, 0, 255), 2)

            # Draw the servo positions
            self.draw_servo_positions(image)

            # Display the image in a window
            cv2.imshow("Pan-Tilt Face Tracker", image)

            # Stop the program when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up resources after finishing
        self.cleanup()