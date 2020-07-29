import numpy as np
import cv2
from mss import mss

def grab_screen(region=None):

    sct=mss()
    sct_img = sct.grab(region)
    return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2RGB)
