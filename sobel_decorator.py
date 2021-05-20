from frame_decorator import FrameDecorator
import cv2

class SobelDecorator(FrameDecorator):
    def decorate(self, frame):
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        grad_x = cv2.convertScaleAbs(grad_x)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
            
        _, jpg = cv2.imencode('.jpg', grad, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return jpg.tobytes()
