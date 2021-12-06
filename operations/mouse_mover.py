import win32api,win32con
import os
import time as tm
import numpy as np

def move(x,y):
	win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,x,y,0,0)
	win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
	tm.sleep(0.01)
	win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
	tm.sleep(2)


try:
	while True:
		move(int(np.random.random()*10000),int(np.random.random()*10000))
		tm.sleep(20)
except KeyboardInterrupt:
	print('script terminated')
	pass




