#!/usr/bin/env python 
# coding:utf-8 

import cv2 as cv
import numpy as np 
import threading
import queue


# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

if __name__ == '__main__':
    # camera capture 
    URL='rtsp://192.168.8.88:554/av0_0'
    capture = VideoCapture(URL)

    # initialize the camera node
    
    # define the publisher 

    # create image index
    img_index = 0

    # initialize the rospy rate 

    # loop 
    while True:
        #start = time.time()
        #while ret:
        frame = capture.read()
        #    frame_buffer = frame
        #frame = frame_buffer
        # frame = cv.flip(frame, 0)
        #frame = cv.flip(frame, 1)

        size = frame.shape
        width  = size[1] 
        height = size[0]
        #print(size[1], size[0])
        #width = 720 
        #height = 540 
        #new_frame = cv.resize(frame,dsize = (width,height),interpolation = cv.INTER_LINEAR)
        #new_frame = cv.cvtColor(new_frame,cv.COLOR_BGR2BGRA)

        # publish the image msg

        if img_index == 0:
            print("The image topic is published...\n")
            img_index = 1
        #end = time.time()
        #print("cost time:", end - start)
        
        
    capture.cap.release()
    print('Quit successfully')