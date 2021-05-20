from frame_decorator import FrameDecorator
import cv2

BACKGROUND_RESET_COUNT = 50
MIN_AREA = 500


class MotionDetectionDecorator(FrameDecorator):
    background = None
    count = 0

    def decorate(self, frame):
        self.count = self.count + 1

        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        if self.background is None or self.count == BACKGROUND_RESET_COUNT:
            self.background = gray
            self.count = 0

        delta = cv2.absdiff(self.background, gray)
        _, threshold = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
        threshold = cv2.dilate(threshold, None, iterations=2)
        countours, _ = cv2.findContours(
            threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in countours:
            if cv2.contourArea(c) < MIN_AREA:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return jpg.tobytes()
