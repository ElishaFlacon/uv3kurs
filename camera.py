import cv2 as cv
import numpy as np
import threading
from typing import Optional
from settings import CameraSettings


class AsyncCamera:
    def __init__(self, src=0, camera_settings: Optional[CameraSettings] = None):
        settings = camera_settings or CameraSettings()
        self.src = src
        self.cap = cv.VideoCapture(src)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, settings.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, settings.height)
        self.grabbed, self.frame = self.cap.read()
        self.processing = False

        self.thread = None
        self.read_lock = threading.Lock()

    def update(self):
        while self.processing:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def start(self):
        if self.processing:
            print("[!] Камера уже начала съемку.")
            return
        self.processing = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def read(self):
        with self.read_lock:
            grabbed = self.grabbed
            self.frame: np.ndarray = self.frame
            frame = self.frame.copy()
        return grabbed, frame

    def stop(self):
        self.processing = False
        self.thread: threading.Thread = self.thread
        self.thread.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
