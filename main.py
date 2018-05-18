import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    smoothened_frame = cv2.GaussianBlur(frame , (5 , 5) , 0)
    edge_detect_frame = cv2.Canny(smoothened_frame , 50 , 120)
    mask_white = np.ndarray(shape=(edge_detect_frame.shape[0], edge_detect_frame.shape[1]), dtype=np.uint8)
    mask_white.fill(255)
    frame_after_masking = cv2.bitwise_xor(edge_detect_frame , mask_white)
    cv2.imshow('first video' , frame_after_masking)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()