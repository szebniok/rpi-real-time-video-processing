from abc import abstractmethod

# class interface for applying different types of frame decorations e.g:
# - edge detection filters
# - object detection and recognition
class FrameDecorator():
    @abstractmethod
    def decorate(self, frame):
        raise NotImplementedError

    
    