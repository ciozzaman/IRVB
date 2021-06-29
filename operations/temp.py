import win32api,win32con
import os
import time as tm
import numpy as np
import shutil
from PIL import ImageChops # $ pip install pillow
from pyscreenshot import grab # $ pip install pyscreenshot
import subprocess
import copy as cp
os.chdir('D:\\IRVB')

def click(x,y):
	win32api.SetCursorPos((x,y))
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def move(x,y):
	win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,x,y,0,0)


# file_path = 'D:\\IRVB\\2021-06-03-altair'
upper_file_path = 'D:\\IRVB'
target_upper_file_path = 'F:\\work\\irvb\\MAST-U'

f = []
for (dirpath,dirnames,filenames) in os.walk(upper_file_path):
	f.append(dirnames)
	break

file_path = upper_file_path+'\\'+f[0][-1]
print('Monitoring '+ file_path)

target_file_path = target_upper_file_path+'\\'+f[0][-1]
if not os.path.exists(target_file_path):
	os.mkdir(target_file_path)

f = []
for (dirpath,dirnames,filenames) in os.walk(file_path):
	f.append(filenames)
	break

min_to_wait = 2
old_number_of_files = len(f[0])
old_files = f[0]
print(str(old_number_of_files)+' initial files')


# the state of the camera before starting this should be with recording activated and internal clocks in free running
# checking that the camera is in external trigger
im1 = grab();tm.sleep(0.1);im2 = grab()
diff = ImageChops.difference(im1, im2)
diff1 = diff.getdata()
changed_pixels = 0
for i in range(np.shape(diff1)[0]):
	if diff1[i]!=(0,0,0):
		changed_pixels+=1
if changed_pixels>1000:	# arbitrary threshold for camera in internal clocks
	click(195,1125)	# select the "camera" sheet
	tm.sleep(0.1)
	click(185,1070)	# select external clocks
# this will leave the camera in external clocks in the camera page, but with recording activated

# read last pulse
subprocess.call([r'D:\\IRVB\\getshotno.bat'])
f = open('mshot.dat', 'r')
last_mshot = str(f.read())
last_pulse = int(last_mshot[1:6])
last_time_updated = last_mshot[8:-1]
next_pulse = cp.deepcopy(last_pulse)


# check if a new shot is initiated
while next_pulse==last_pulse:
	move(int(np.random.random()*10000),int(np.random.random()*10000))
	tm.sleep(10)
	subprocess.call([r'D:\\IRVB\\getshotno.bat'])
	f = open('mshot.dat', 'r')
	next_mshot = str(f.read())
	f.close()
	next_pulse = int(next_mshot[1:6])
	next_time_updated = next_mshot[8:-1]
	print(next_mshot)
