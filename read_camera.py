# coding=utf-8
import cv2 as cv


URL='rtsp://admin:abcd1234@192.168.8.108:8555/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif'
# URL='rtsp://192.168.8.88:554/av0_0'
cap = cv.VideoCapture(URL)
assert cap.isOpened(), 'Failed to open'
a =1