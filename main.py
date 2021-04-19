from flask import Flask, render_template, Response
import cv2
from sobel_decorator import SobelDecorator
from object_detection_decorator import ObjectDetectionDecorator 
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

@app.route("/video/<decorator>")
def video(decorator):
    return Response(frames(apply_decorator(decorator)), mimetype="multipart/x-mixed-replace; boundary=frame")

def frames(decorator):
    counter = 1
    while True:
        _, frame = cap.read()
        
        data = decorator.decorate(frame)
        counter += 1
            
        yield (b"--frame\r\n" +
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")


def apply_decorator(decorator):
    if decorator == "detection":
        return ObjectDetectionDecorator()
    if decorator == "sobel":
        return SobelDecorator()
    raise RuntimeError(f"Could not find filter name: {filter}") 

if __name__ == "__main__":
    app.run(debug = True)
    cap.release()
    cv2.destroyAllWindows()