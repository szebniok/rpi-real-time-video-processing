import cv2

class ObjectDetectionDecorator():

    def __init__(self):
        print("Initializing object detector...")

        self.classes = []
        with open('detection_data/coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        config_file = 'detection_data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weights_file = 'detection_data/frozen_inference_graph.pb'
        net = cv2.dnn_DetectionModel(weights_file, config_file)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        
        self.network = net
        self.threshold = 0.6
        
        
        
    def decorate(self, frame):
        class_ids, confidences, bbox = self.network.detect(
            frame, confThreshold=self.threshold)
        if len(class_ids) > 0:
            for id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bbox):
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, self.classes[id - 1].upper(), (box[0]+10,
                            box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str(round(
                    confidence*100)), (box[2]-50, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        _, data = cv2.imencode('.jpg', frame)
        return data.tobytes()
