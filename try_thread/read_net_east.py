# from threading import Thread
import threading
import cv2

def readNetEAST(args, blob, layerNames):
    net = cv2.dnn.readNet(args["east"])
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    return (scores, geometry)
