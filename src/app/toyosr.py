from PIL import Image
import sys
import cv2
#import math
from pyzbar.pyzbar import decode, ZBarSymbol
import numpy
import pdf2image


class OneCharRecognizer:
    def __init__(self):
        self.model_file = 'dnn/mnist_100.onnx'
        self.net = None

    def detect_char(self,box_img):
        if self.net == None:
            self.net = cv2.dnn.readNetFromONNX(self.model_file)
        target_img = cv2.resize(box_img,(28,28))
        target_img = cv2.bitwise_not(target_img)
        target_img =  cv2.dnn.blobFromImage(target_img)
        self.net.setInput(target_img)
        pred = numpy.squeeze(self.net.forward())
        ind = numpy.argsort(pred)
        return ind[-1]
