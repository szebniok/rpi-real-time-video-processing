import cv2
import sys
import os
import msvcrt

if sys.platform == "win32":
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    grad_x = cv2.convertScaleAbs(grad_x)

    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    # cv2.imshow('original', frame)
    # cv2.imshow('blur', blur)
    # cv2.imshow('grayscale', gray)
    # cv2.imshow('grad_x', grad_x)
    # cv2.imshow('grad_y', grad_y)
    # cv2.imshow('grad', grad)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    sys.stdout.buffer.write(frame.tobytes())