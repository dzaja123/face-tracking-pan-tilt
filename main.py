import logging

from face_tracking import FaceTrackingSystem


def main():
    """
    Main function to initialize and run the tracking system.
    """

    # Set up logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the face tracking system")

    # Define parameters for initializing the tracking system
    CAMERA_INDEX = 0
    WIDTH = 1280
    HEIGHT = 720
    PORT = "COM4"
    PAN_PIN = 9
    TILT_PIN = 10

    # Initialize and run the system
    face_tracking_system = FaceTrackingSystem(camera_index=CAMERA_INDEX, width=WIDTH, height=HEIGHT, port=PORT, pan_pin=PAN_PIN, tilt_pin=TILT_PIN)
    face_tracking_system.run()
    logging.info("Face tracking system has been initialized and is running.")


if __name__ == "__main__":
    main()
