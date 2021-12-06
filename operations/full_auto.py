import win32api,win32con
import os
import time as tm
import numpy as np
import shutil
from PIL import ImageChops # $ pip install pillow
from pyscreenshot import grab # $ pip install pyscreenshot
import subprocess
import copy as cp
# to manage a timeout
from threading import Thread
import functools
os.chdir('D:\\IRVB')

def click(x,y):
	win32api.SetCursorPos((x,y))
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
	tm.sleep(0.01)
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
	tm.sleep(0.01)
	win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
	tm.sleep(0.01)
	win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
	tm.sleep(2)

def move(x,y):
	win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,x,y,0,0)
	win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
	tm.sleep(0.01)
	win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
	tm.sleep(2)

def wait_while_moving_mouse(minutes,wait_for_move=20):
	loops_left = int((minutes*60)//wait_for_move)
	while loops_left>=0:
		move(int(np.random.random()*10000),int(np.random.random()*10000))
		loops_left-=1
		tm.sleep(wait_for_move)

# file_path = 'D:\\IRVB\\2021-06-03-altair'
upper_file_path = 'D:\\IRVB'
target_upper_file_path = 'F:\\work\\irvb\\MAST-U'

f = []
for (dirpath,dirnames,filenames) in os.walk(upper_file_path):
	f.append(dirnames)
	break

target_folder = f[0][-1]
file_path = upper_file_path+'\\'+f[0][-1]
print('Monitoring '+ file_path)

test = input('is this the right folder? if OK enter "k" otherwise the right folder')
if test!='k':
	target_folder = test
	file_path = upper_file_path + '\\' + test
	target_file_path = target_upper_file_path + '\\' + test
	print('Monitoring '+ file_path)

target_file_path = target_upper_file_path+'\\'+target_folder
if False:
	if not os.path.exists(target_file_path):
		os.mkdir(target_file_path)
else:
	def timeout(timeout):
	    def deco(func):
	        @functools.wraps(func)
	        def wrapper(*args, **kwargs):
	            res = [Exception('function timeout [%s seconds] exceeded!' % (timeout))]
	            def newFunc():
	                try:
	                    res[0] = func(*args, **kwargs)
	                except Exception as e:
	                    res[0] = e
	            t = Thread(target=newFunc)
	            t.daemon = True
	            try:
	                t.start()
	                t.join(timeout)
	            except Exception as je:
	                print ('error starting thread')
	                raise je
	            ret = res[0]
	            if isinstance(ret, BaseException):
	                raise ret
	            return ret
	        return wrapper
	    return deco

	def create_folder():
		if not os.path.exists(target_file_path):
			os.mkdir(target_file_path)
			print('freia folder created')

	func = timeout(timeout=1*60)(create_folder)
	try:
		func()
	except Exception as exc:
		print('creation of the folder on freia failed')
		print(exc)

f = []
for (dirpath,dirnames,filenames) in os.walk(file_path):
	f.append(filenames)
	break

min_to_wait = 2
old_number_of_files = len(f[0])
old_files = f[0]
print(str(old_number_of_files)+' initial files')

def check_screen_change():
	#	checking how many pixels change in two screenshots taken shortly apart
	im1 = grab();tm.sleep(0.5);im2 = grab()
	diff = ImageChops.difference(im1, im2)
	diff1 = diff.getdata()
	changed_pixels = 0
	for i in range(np.shape(diff1)[0]):
		if diff1[i]!=(0,0,0):
			changed_pixels+=1
	return changed_pixels
# the state of the camera before starting this should be with recording activated and internal clocks in free running
# checking that the camera is in external trigger
tm.sleep(0.1)
print('reset of visual range')
click(640,60)	# reset the range of the frame visualisation
print('visual range reset')
changed_pixels = check_screen_change()
if changed_pixels>1000:	# arbitrary threshold for camera in internal clocks
	print(str(changed_pixels)+' pixels changed\ncamera assumed to be in internal clocks\ntherefore changed to external')
	click(195,1125)	# select the "camera" sheet
	tm.sleep(0.1)
	click(185,1070)	# select external clocks
else:
	print('only '+str(changed_pixels)+' pixels changed, so no action taken')
# this will leave the camera in external clocks in the camera page, but with recording activated

if False:	# using a location deep into MAST-U system that I should really not do. This also does not return the machine states
	# read last pulse
	subprocess.call([r'D:\\IRVB\\getshotno.bat'])
	f = open('mshot.dat', 'r')
	last_mshot = str(f.read())
	f.close()
	last_pulse = int(last_mshot[1:6])
	# last_time_updated = last_mshot[8:-1]
	next_pulse = cp.deepcopy(last_pulse)
	print('last pulse '+str(last_pulse))
else:
	DAProxy_path = 'D:\\IRVB\\mastda\\DAProxy'
	DAProxy_file = 'prx' + target_folder.translate({ord('-'):None})[2:] + '.log'
	DAProxy_full_path = DAProxy_path+'\\log\\'+DAProxy_file
	print('checking file '+DAProxy_full_path)
	while not os.path.exists(DAProxy_full_path):
		print('checking file '+DAProxy_full_path)
		test = input('the log file '+DAProxy_full_path+" doesn't exists. r to rety or input the full path to the log file (including log file name)")
		if test == 'r':
			continue
		else:
			DAProxy_full_path = test
	def return_shot_and_state(DAProxy_full_path=DAProxy_full_path):
		done = 0
		while done==0:
			try:
				f = open(DAProxy_full_path,'r')
				for row in f:
					None
				last_pulse = int(row[row.find('shot=') + len('shot='):row.find('&state')])
				MASTU_state = int(row[row.find('state=') + len('state='):row.find('&)\n')])
				done = 1
			except:
				print('shot info missing in log')
				wait_while_moving_mouse(10/60,wait_for_move=2)
		return last_pulse,MASTU_state
	last_pulse,MASTU_state = return_shot_and_state()
	next_pulse = cp.deepcopy(last_pulse)
	print('last pulse '+str(last_pulse))

try:
	while tm.gmtime().tm_hour+1<21:#18:
		# check if a new shot is initiated
		waiting_for_new_shot = True
		useless_counter = 0
		while waiting_for_new_shot:
			move(int(np.random.random()*10000),int(np.random.random()*10000))
			tm.sleep(1)
			useless_counter +=1
			if False:
				subprocess.call([r'D:\\IRVB\\getshotno.bat'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
				f = open('mshot.dat', 'r')
				next_mshot = str(f.read())
				f.close()
				next_pulse = int(next_mshot[1:6])
				# next_time_updated = next_mshot[8:-1]
			else:
				next_pulse,MASTU_state = return_shot_and_state()

			if next_pulse!=last_pulse:
				waiting_for_new_shot = False
			else:
				if useless_counter%10==0:
					print('still on '+str(next_pulse))
		print('next pulse '+str(next_pulse))
		# last_pulse = cp.deepcopy(next_pulse)
		# exit when new pulse initiated
		print('start switch to internal trigger')
		click(195,1125)	# select the "camera" sheet
		tm.sleep(0.1)
		click(185,1070)	# select internal clocks
		print('back to internal trigger')

		waiting_for_new_file = True
		abort_detected = False
		useless_counter = 0
		while waiting_for_new_file:
			useless_counter +=1
			move(int(np.random.random()*10000),int(np.random.random()*10000))
			f = []
			for (dirpath,dirnames,filenames) in os.walk(file_path):
				f.append(filenames)
				break
			new_number_of_files = len(f[0])
			new_files = f[0]

			if useless_counter%10==0:
				print(str(new_number_of_files)+' files present')
			wait_while_moving_mouse(2/60,wait_for_move=1)
			trash,MASTU_state = return_shot_and_state()
			if MASTU_state==11:
				abort_detected = True
				break

			# tm.sleep(5*3)
			if new_number_of_files!=old_number_of_files:
				print(str(new_number_of_files)+' files present')
				waiting_for_new_file = False
			else:
				if useless_counter%10==0:
					print('no action required')
		if 	abort_detected==False:
			tm.sleep(10)	# time to allow to finish finish saving files
			print('clicking start recording')
			# click(360,55)	# position for ResearchIR
			# click(80,490)	# position for minimised Altair
			click(320,1125)	# select the "recorder" sheet
			tm.sleep(0.1)
			click(550,1070)	# position for full screen Altair, start record
			print('new recording started')
			# print('wait '+str(min_to_wait)+'min')

			tm.sleep(0.1)
			print('reset of visual range')
			click(640,60)	# reset the range of the frame visualisation
			print('visual range reset')

			if False:	# I'm not sure I still need to do it, but I keep it
				# I need to reset the last pulse here to adapt to aborts
				subprocess.call([r'D:\\IRVB\\getshotno.bat'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
				f = open('mshot.dat', 'r')
				next_mshot = str(f.read())
				f.close()
				last_pulse = cp.deepcopy(int(next_mshot[1:6]))
			else:
				last_pulse,MASTU_state = return_shot_and_state()

			tm.sleep(10)	# time to allow to finish resetting the range
			changed_pixels = check_screen_change()
			if changed_pixels>1000:	# arbitrary threshold for camera in internal clocks
				print(str(changed_pixels)+' pixels changed\ncamera assumed to be in internal clocks\ntherefore changed to external')
				click(195,1125)	# select the "camera" sheet
				tm.sleep(0.1)
				click(185,1070)	# select external clocks
			else:
				print('only '+str(changed_pixels)+' pixels changed, so no action taken')
			# camera left in record activated and external clocks

			tm.sleep(0.1)
			for file in new_files:
				if not (file in old_files):
					new_file = file
			try:
				print('copying '+new_file)
				shutil.copyfile(file_path+'\\'+new_file, target_file_path+'\\'+new_file)
				print('just copied')
			except:
				print('WARNING: Copy failed. continuing')

			try:
				if not os.path.exists(target_upper_file_path+'\\'+'last_pulse.npz'):
					last_pulse_dict = dict([])
					last_pulse_dict['location'] = [target_folder]
					last_pulse_dict['filename'] = [new_file]
				else:
					last_pulse_dict = dict(np.load(target_upper_file_path+'\\'+'last_pulse.npz'))
					temp = list(last_pulse_dict['location'])
					temp.append(target_folder)
					last_pulse_dict['location'] = temp
					temp = list(last_pulse_dict['filename'])
					temp.append(new_file)
					last_pulse_dict['filename'] = temp

				# actually don't do it for now
				if True:
					np.savez_compressed(target_upper_file_path+'\\'+'last_pulse',**last_pulse_dict)
					print(target_upper_file_path+'\\'+'last_pulse.npz updated')

					os.system('putty -ssh ffederic@freia023.hpc.l -pw Ooup313313 -m F:\\work\\analysis_scripts\\scripts\\operations\\pulse_processor_launcher.sh')
					print('Freia job submitted')
			except:
				print('something is wrong with launching the FREIA job')
			old_number_of_files = new_number_of_files
			old_files = new_files
		else:
			print('Abort detected')
			print('clicking stop recording')
			click(320,1125)	# select the "recorder" sheet
			tm.sleep(0.1)
			click(600,1070)	# position for full screen Altair, stop record
			print('recording stopped')

			# now the number of the shot inside altair should advance by one

			print('clicking start recording')
			click(320,1125)	# select the "recorder" sheet
			tm.sleep(0.1)
			click(550,1070)	# position for full screen Altair, start record
			print('new recording started')
			# print('wait '+str(min_to_wait)+'min')

			tm.sleep(0.1)
			print('reset of visual range')
			click(640,60)	# reset the range of the frame visualisation
			print('visual range reset')

			tm.sleep(10)
			changed_pixels = check_screen_change()
			if changed_pixels>1000:	# arbitrary threshold for camera in internal clocks
				print(str(changed_pixels)+' pixels changed\ncamera assumed to be in internal clocks\ntherefore changed to external')
				click(195,1125)	# select the "camera" sheet
				tm.sleep(0.1)
				click(185,1070)	# select external clocks
			else:
				print('only '+str(changed_pixels)+' pixels changed, so no action taken')
			# camera left in record activated and external clocks

			last_pulse,MASTU_state = return_shot_and_state()
			old_number_of_files = new_number_of_files
			old_files = new_files

	print('Now is\n'+tm.ctime()+'\nso operations, and this script, terminate')
except KeyboardInterrupt:
	print('script terminated')
	pass
