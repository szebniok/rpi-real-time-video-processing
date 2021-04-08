from flask import Flask, render_template, Response
import cv2

app = Flask(__name__, template_folder = "", static_folder="")
cap = cv2.VideoCapture(0)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def frames():
    while True:
        ret, frame = cap.read()
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        grad_x = cv2.convertScaleAbs(grad_x)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
            
        _, jpg = cv2.imencode('.jpg', grad, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        jpg = jpg.tobytes()
            
        yield (b"--frame\r\n" +
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")


if __name__ == "__main__":
    app.run(debug = True)
    cap.release()
    cv2.destroyAllWindows()