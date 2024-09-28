import cv2
import mediapipe as mp
import pyfirmata


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        # Иницијализација FaceDetector-а са минималном детекцијом поузданости и избором модела
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        # Дефинисање модела из MediaPipe-а за детекцију лица
        self.mp_face_detection = mp.solutions.face_detection

        # Креирање модела за детекцију лица са датим параметрима
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
            model_selection=self.model_selection,
        )

    def find_faces(self, image):
        """
        Пронађи лица на слици користећи MediaPipe модел за детекцију лица.
        """

        # Конвертуј слику у RGB формат
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Обради слику да детектује лица
        self.results = self.face_detection.process(img_rgb)

        # Листа за чување граница и информација о лицу
        face_bboxes = []

        if self.results.detections:
            for detection_id, detection in enumerate(self.results.detections):
                # Обрада само ако је резултат детекције изнад прага
                if detection.score[0] > self.min_detection_confidence:
                    # Добијање података о координатама оквира лица
                    bounding_box = detection.location_data.relative_bounding_box
                    image_height, image_width, _ = image.shape

                    # Претварање координата оквира у пиксел вредности
                    bbox = (
                        int(bounding_box.xmin * image_width),
                        int(bounding_box.ymin * image_height),
                        int(bounding_box.width * image_width),
                        int(bounding_box.height * image_height),
                    )

                    # Израчунавање центра оквира
                    center_x = bbox[0] + (bbox[2] // 2)
                    center_y = bbox[1] + (bbox[3] // 2)

                    # Креирање речника за чување информација о детектованом лицу
                    face_info = {
                        "id": detection_id,
                        "bbox": bbox,
                        "score": detection.score,
                        "center": (center_x, center_y),
                    }

                    # Додавање информација о лицу у листу
                    face_bboxes.append(face_info)

        # Враћање слике и листе детектованих лица
        return image, face_bboxes


class FaceTrackingSystem:
    def __init__(self, camera_index=0, width=1280, height=720, port="COM4", pan_pin=9, tilt_pin=10):
        # Иницијализација снимања камере са датим индексом
        self.cap = cv2.VideoCapture(camera_index)
        
        # Подешавање димензија кадра камере
        self.frame_width = width
        self.frame_height = height

        # Сетовање димензија кадра камере
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Провера да ли је камера доступна
        if not self.cap.isOpened():
            print("Није могуће приступити камери!")
            exit()

        # Иницијализација Ардуино плоче и подешавање пинова за пан и тилт сервое
        self.board = pyfirmata.Arduino(port)
        
        # Дефинисање пинова за пан и тилт сервое
        self.pan_servo_pin = self.board.get_pin(f"d:{pan_pin}:s")
        self.tilt_servo_pin = self.board.get_pin(f"d:{tilt_pin}:s")

        # Иницијализација детектора лица са подешеном поузданошћу и избором модела
        self.face_detector = FaceDetector(min_detection_confidence=0.5, model_selection=1)

        # Почетне позиције серво мотора (90 степени - неутрална позиција)
        self.servo_positions = [90, 90]

        # Израчунавање центра екрана
        self.screen_center_x = self.frame_width // 2
        self.screen_center_y = self.frame_height // 2

        # Толеранција за позицију лица пре померања серво мотора
        self.face_center_threshold = 35

        # Дефинисање фонта за цртање текста
        self.font = cv2.FONT_HERSHEY_SIMPLEX    

        # Дефинисање боје текста
        self.text_color = (255, 0, 0)

        # Дефинисање боје оквира
        self.bbox_color = (0, 0, 255)

    def move_servos(self, x_diff, y_diff):
        """
        Управљање пан и тилт серво мотора на основу промене позиције лица.
        """

        # Подешавање пан серво мотора (X-ос) ако је лице ван дозвољеног прага
        if abs(x_diff) > self.face_center_threshold:
            if x_diff > 0:
                self.servo_positions[0] -= 1  # Померање улево
            else:
                self.servo_positions[0] += 1  # Померање удесно

        # Подешавање тилт серво мотора (Y-ос) ако је лице ван дозвољеног прага
        if abs(y_diff) > self.face_center_threshold:
            if y_diff > 0:
                self.servo_positions[1] -= 1  # Померање надоле
            else:
                self.servo_positions[1] += 1  # Померање навише

        # Ограничење позиција серво мотора на опсег [0, 180]
        self.servo_positions[0] = max(0, min(self.servo_positions[0], 180))
        self.servo_positions[1] = max(0, min(self.servo_positions[1], 180))

        # Слање позиција серво мотора на Ардуино
        self.pan_servo_pin.write(self.servo_positions[0])
        self.tilt_servo_pin.write(self.servo_positions[1])

    def draw_interface(self, image, face_bboxes):
        """
        Цртање интерфејса на слици.
        """

        # Добијање резултата детекције
        score_value = round(face_bboxes[0]["score"][0] * 100)
        score = f"{score_value}%"

        # Добијање координата оквира
        bbox = face_bboxes[0]["bbox"]

        # Цртање оквира и резултата поузданости на слици
        image = cv2.rectangle(image, bbox, self.bbox_color, 2)

        # Цртање резултата поузданости изнад оквира
        cv2.putText(image, score, (bbox[0], bbox[1] - 20), self.font, 2, self.bbox_color, 2)

        # Индикација да је мета закључана
        cv2.putText(image, "TARGET LOCKED", (780, 50), self.font, 2, self.bbox_color, 2)

    def draw_servo_positions(self, image):
        """
        Цртање тренутних позиција серво мотора на слици.
        """

        # Цртање позиција серво мотора
        cv2.putText(image, f"Servo X: {int(self.servo_positions[0])} deg", (50, 50), self.font, 1, self.text_color, 2)
        cv2.putText(image, f"Servo Y: {int(self.servo_positions[1])} deg", (50, 100), self.font, 1, self.text_color, 2)

    def cleanup(self):
        """
        Чишћење ресурса.
        """

        # Ослобађање камере и затварање везе са Ардуином
        cv2.destroyAllWindows()
        self.cap.release()
        self.board.exit()

    def run(self):
        """
        Главни циклус за обраду стрима са камере, детекцију лица и контролу серво мотора.
        """

        while True:
            # Преузимање слике са камере
            success, image = self.cap.read()
            if not success:
                print("Није могуће преузети слику")
                break

            # Детекција лица на слици
            image, face_bboxes = self.face_detector.find_faces(image)

            if face_bboxes:
                # Добијање центра првог детектованог лица
                face_x, face_y = face_bboxes[0]["center"]

                # Израчунавање разлике између центра лица и центра екрана
                x_diff = face_x - self.screen_center_x
                y_diff = face_y - self.screen_center_y

                # Померање серво мотора на основу разлике
                self.move_servos(x_diff, y_diff)

                # Цртање интерфејса на слици
                self.draw_interface(image, face_bboxes)
                
            else:
                # Уколико нема детектованог лица обавести се да нема мете
                cv2.putText(image, "NO TARGET", (850, 50), self.font, 2, (0, 0, 255), 2)

            # Цртање позиција серво мотора
            self.draw_servo_positions(image)

            # Приказ слике у прозору
            cv2.imshow("Pan-Tilt Face Tracker", image)

            # Заустављање програма када се притисне тастер 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Чишћење ресурса након завршетка рада
        self.cleanup()


def main():
    """
    Main функција за иницијализацију и покретање система
    """
    
    # Дефинисање параметара за иницијализацију система за праћење
    CAMERA_INDEX = 0
    WIDTH = 1280
    HEIGHT = 720
    PORT = "COM4"
    PAN_PIN = 9
    TILT_PIN = 10

    # Иницијализација и покретање система
    face_tracking_system = FaceTrackingSystem(camera_index=CAMERA_INDEX, width=WIDTH, height=HEIGHT, port=PORT, pan_pin=PAN_PIN, tilt_pin=TILT_PIN)
    face_tracking_system.run()


if __name__ == "__main__":
    main()
