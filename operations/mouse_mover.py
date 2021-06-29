import win32api,win32con
import os
import time as tm
import numpy as np

def move(x,y):
	win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,x,y,0,0)



try:
	while True:
		move(int(np.random.random()*10000),int(np.random.random()*10000))
		tm.sleep(20)
except KeyboardInterrupt:
	print('script terminated')
	pass




