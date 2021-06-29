import win32api,win32con
import os
import time as tm
import numpy as np

def click(x,y):
	win32api.SetCursorPos((x,y))
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def move(x,y):
	win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,x,y,0,0)


file_path = 'D:\\IRVB\\2021-05-28'
print('Monitoring '+file_path)

f = []
for (dirpath,dirnames,filenames) in os.walk(file_path):
	f.append(filenames)
	break
min_to_wait = 2
old_number_of_files = len(f[0])
print(str(old_number_of_files)+' initial files')

try:
	while True:
		move(int(np.random.random()*10000),int(np.random.random()*10000))
		f = []
		for (dirpath,dirnames,filenames) in os.walk(file_path):
			f.append(filenames)
			break
	
		new_number_of_files = len(f[0])
		print(str(new_number_of_files)+' files present')
		tm.sleep(min_to_wait*60)
		# tm.sleep(5*3)
		if new_number_of_files!=old_number_of_files:
			print('clicking')
			click(360,55)
			print('just clicked')
			old_number_of_files = new_number_of_files
		else:
			print('no need to click')

except KeyboardInterrupt:
	print('script terminated')
	pass




