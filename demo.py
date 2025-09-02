import cv2
import numpy as np
import time
from recognizer import FaceRecognitionSystem

face_system = FaceRecognitionSystem()
imgs = np.load('TestImages.npy').tolist()

for img in imgs:
   print(f"treating: {img}")
   st = time.perf_counter()
   img = cv2.imread(f"./testImage/{img}")
   res = face_system.recognize(img,k=5,threshold=0.46)
   end = time.perf_counter() - st
   print("inf Time face_system:",end)
   print(res,"\n") 