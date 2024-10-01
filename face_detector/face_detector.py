import mediapipe as mp
import cv2
import logging


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        # Initialize the FaceDetector with minimum detection confidence and model selection
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        # Define the MediaPipe model for face detection
        self.mp_face_detection = mp.solutions.face_detection

        # Create the face detection model with given parameters
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
            model_selection=self.model_selection,
        )
        logging.info("FaceDetector initialized with min_detection_confidence=%s, model_selection=%s",
                     self.min_detection_confidence, self.model_selection)

    def find_faces(self, image):
        """
        Find faces in the image using the MediaPipe face detection model.
        """

        logging.info("Starting face detection")

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect faces
        self.results = self.face_detection.process(img_rgb)

        # List to store bounding boxes and face information
        face_bboxes = []

        if self.results.detections:
            logging.info("Detected %d faces", len(self.results.detections))
            for detection_id, detection in enumerate(self.results.detections):
                # Process only if the detection result is above the threshold
                if detection.score[0] > self.min_detection_confidence:
                    # Get the bounding box coordinates
                    bounding_box = detection.location_data.relative_bounding_box
                    image_height, image_width, _ = image.shape

                    # Convert bounding box coordinates to pixel values
                    bbox = (
                        int(bounding_box.xmin * image_width),
                        int(bounding_box.ymin * image_height),
                        int(bounding_box.width * image_width),
                        int(bounding_box.height * image_height),
                    )

                    # Calculate the center of the bounding box
                    center_x = bbox[0] + (bbox[2] // 2)
                    center_y = bbox[1] + (bbox[3] // 2)

                    # Create a dictionary to store detected face information
                    face_info = {
                        "id": detection_id,
                        "bbox": bbox,
                        "score": detection.score,
                        "center": (center_x, center_y),
                    }

                    # Add face information to the list
                    face_bboxes.append(face_info)

                    logging.info("Face ID %d detected with score %.2f and bounding box %s", 
                                 detection_id, detection.score[0], bbox)
                else:
                    logging.warning("Face ID %d detection score %.2f below threshold", 
                                    detection_id, detection.score[0])
        else:
            logging.info("No faces detected")

        # Return the image and the list of detected faces
        return image, face_bboxes
