import win32api,win32con
import os
import time as tm
import numpy as np
import shutil

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

try:
	while True:
		move(int(np.random.random()*10000),int(np.random.random()*10000))
		f = []
		for (dirpath,dirnames,filenames) in os.walk(file_path):
			f.append(filenames)
			break

		new_number_of_files = len(f[0])
		new_files = f[0]

		print(str(new_number_of_files)+' files present')
		print('wait '+str(min_to_wait)+'min')
		tm.sleep(min_to_wait*60)
		# tm.sleep(5*3)
		if new_number_of_files!=old_number_of_files:
			print('clicking')
			# click(360,55)	# position for ResearchIR
			# click(80,490)	# position for minimised Altair
			click(550,1070)	# position for full screen Altair
			print('just clicked')
			print('wait '+str(min_to_wait)+'min')
			tm.sleep(min_to_wait*60)
			for file in new_files:
				if not (file in old_files):
					new_file = file
			print('copying '+new_file)
			shutil.copyfile(file_path+'\\'+new_file, target_file_path+'\\'+new_file)
			print('just copied')
			old_number_of_files = new_number_of_files
			old_files = new_files
		else:
			print('no action required')

except KeyboardInterrupt:
	print('script terminated')
	pass
