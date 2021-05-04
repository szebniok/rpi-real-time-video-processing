from flask import Flask, render_template, Response
import cv2
from sobel_decorator import SobelDecorator
from object_detection_decorator import ObjectDetectionDecorator 
from motion_detection_decorator import MotionDetectionDecorator
import time

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

@app.route("/video/<decorator_id>")
def video(decorator_id):
    return Response(frames(apply_decorator(decorator_id)), mimetype="multipart/x-mixed-replace; boundary=frame")

def frames(decorator):
    counter = 1
    while True:
        _, frame = cap.read()
        
        data = decorator.decorate(frame)
        counter += 1
            
        yield (b"--frame\r\n" +
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")


def apply_decorator(decorator_id):
    if decorator_id == "object_detection":
        return ObjectDetectionDecorator()
    if decorator_id == "sobel":
        return SobelDecorator()
    if decorator_id == "motion_detection":
        return MotionDetectionDecorator()
    raise RuntimeError(f"Could not find filter name: {decorator_id}") 

if __name__ == "__main__":
    app.run(debug = True)
    cap.release()
    cv2.destroyAllWindows()