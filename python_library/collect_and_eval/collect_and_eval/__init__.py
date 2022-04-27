import numpy as np
from scipy.optimize import curve_fit
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import statistics as s
import csv
import os,sys
from astropy.io import fits
import matplotlib.animation as animation
import pandas
import scipy.stats
import peakutils
import collections
from scipy.ndimage import rotate
from skimage.transform import resize
from uncertainties import correlated_values,ufloat
from uncertainties.unumpy import nominal_values,std_devs,uarray
import time as tm
import concurrent.futures as cf
import copy as cp

import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

import pyuda
# client=pyuda.Client()

#
# This is a collection of script that can be usefull in the interpretation of IR camera files
#



# SOMETHING USEFULL FOR PROFILING
#
#
# import cProfile
# import re
# cProfile.run(' SCRIPT OR SINGLE INSTRUCTION THAT I WANT TO PROFILE ')
#
#


def is_number(s):
	""" Returns True is string is a number. """
	try:
		float(s)
		return True
	except ValueError:
		return False

def rsquared(x, y):
	""" Return R^2 where x and y are array-like."""

	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	return r_value**2


#This function will generate a polinomial of n order GOOD FOR WHOLE FRAMES
def polygen(n):
	def polyadd(x, *params):
		params=np.array(params)
		shape=np.shape(x)
		temp=np.zeros(shape)
		# if (len(shape)>1) & (len(np.shape(params))==1):
		# 	params=np.reshape(params,(shape[-2],shape[-1],n))
		for i in range(n):
			# print('np.shape(temp),np.shape(params),np.shape(x)',temp,params,x)
			x2=np.power(x,i)
			# print('params[:][:][i]',params[:][:][i])
			# print('params',params)
			para=np.multiply(params[:,:,i],x2)
			temp=np.add(temp,para)
		return temp
	return polyadd

# #This function will generate a polinomial of n order GOOD FOR background fitting BUT NOT WORKING!! 2018/03/30
# def polygen2(n,shape):
# 	def polyadd(x, *params):
# 		params=np.array(params)
# 		x=np.reshape(x,shape)
# 		temp=np.zeros(shape)
# 		# if (len(shape)>1) & (len(np.shape(params))==1):
# 		params=np.reshape(params,(shape[-2],shape[-1],n))
# 		for i in range(n):
# 			# print('np.shape(temp),np.shape(params),np.shape(x)',temp,params,x)
# 			x2=np.power(x,i)
# 			# print('params[:][:][i]',params[:][:][i])
# 			# print('params',params)
# 			para=np.multiply(params[:,:,i],x2)
# 			temp=np.add(temp,para)
# 		return temp
# 	return polyadd

#This function will generate a polinomial of n order
def polygen3(n):
	def polyadd(x, *params):
		temp=0
		for i in range(n):
			temp+=params[i]*x**i
		return temp
	return polyadd


#############################################################################################

def order_filenames(filenames):

	# 13/05/2018 THIS FUNCTION IS INTODUCED TO FIX A BUG IN THE CASE filenames CONTAINS .npy FILES AND NOT .csv

	extention=filenames[0][-4:]
	if ((extention=='.csv') or (extention=='.CSV')):
		return order_filenames_csv(filenames)
	else:
		numbers = []
		for filename in filenames:
			temp = '0'
			for digit in filename:
				if digit.isdigit() == True:
					temp = temp+digit
			numbers.append(np.int(temp))
		filenames = [__ for _, __ in sorted(zip(numbers, filenames))]
		# return sorted(filenames, key=str.lower)
		return filenames

######################################################################################################

def order_filenames_csv(filenames):

	# section to be sure that the files are ordered in the proper numeric order
	# CAUTION!! THE FOLDER MUST CONTAIN MAX 9999999 FILES!

	reference=[]
	filenamescorr=[]
	for i in range(len(filenames)):
		# print(filenames[i])
		start=0
		end=len(filenames[i])
		for j in range(len(filenames[i])):
			if ((filenames[i][j]=='-') or (filenames[i][j]=='_')): #modified 12/08/2018 from "_" to "-"
				start=j
			elif filenames[i][j]=='.':
				end=j
		index=filenames[i][start+1:end]
		#print('index',index)

		# Commented 21/08/2018
		# if int(index)<10:
		# 	temp=filenames[i][:start+1]+'000000'+filenames[i][end-1:]
		# 	filenamescorr.append(temp)
		# elif int(index)<100:
		# 	temp=filenames[i][:start+1]+'00000'+filenames[i][end-2:]
		# 	filenamescorr.append(temp)
		# 	# print(filenames[i],temp)
		# elif int(index)<1000:
		# 	temp=filenames[i][:start+1]+'0000'+filenames[i][end-3:]
		# 	filenamescorr.append(temp)
		# elif int(index)<10000:
		# 	temp=filenames[i][:start+1]+'000'+filenames[i][end-4:]
		# 	filenamescorr.append(temp)
		# elif int(index)<100000:
		# 	temp=filenames[i][:start+1]+'00'+filenames[i][end-5:]
		# 	filenamescorr.append(temp)
		# elif int(index)<1000000:
		# 	temp=filenames[i][:start+1]+'0'+filenames[i][end-6:]
		# 	filenamescorr.append(temp)
		# else:
		# 	filenamescorr.append(filenames[i])
		reference.append(index)
	reference=np.array(reference)
	filenamesnew=np.array([filenames for _,filenames in sorted(zip(reference,filenames))])

	# Commented 21/08/2018
	# filenamescorr=sorted(filenamescorr, key=str.lower)
	# filenamesnew=[]
	# for i in range(len(filenamescorr)):
	# 	# print(filenamescorr[i])
	# 	for j in range(len(filenamescorr[i])):
	# 		if ((filenames[i][j]=='-') or (filenames[i][j]=='_')): #modified 12/08/2018 from "_" to "-"
	# 			start=j
	# 		elif filenamescorr[i][j]=='.':
	# 			end=j
	# 	index=filenamescorr[i][start+1:end]
	# 	# if int(index)<10:
	# 	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	# 	filenamesnew.append(temp)
	# 	# elif int(index)<100:
	# 	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	# 	filenamesnew.append(temp)
	# 	# 	# print(filenames[i],temp)
	# 	# if int(index)<1000000:
	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	filenamesnew.append(temp)
	# 	# 	# print(filenamescorr[i],temp)
	# 	# else:
	# 	# 	filenamesnew.append(filenamescorr[i])
	filenames=filenamesnew

	return(filenames)

#############################################################################################


def collect_subfolder(extpath):

	# THIS FUNCTION GENERATE THE .npy FILE CORRESPONDING TO A SINGLE FOLDER STARTING FROM ALL THE .csv FILES IN IT

	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)


	# path=os.getcwd()
	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)


	temp=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='csv':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
	filenames=temp

	filenames=order_filenames(filenames)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen

	data=np.zeros((1,numfiles,sizey,sizex))

	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						data[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1

	np.save(os.path.join(extpath,lastpath),data)

	# plt.pcolor(data[0,20,:,:])
	# plt.show()


##############################################################################################

def collect_subfolderfits(extpath,start='auto',stop='auto'):

	# 2018/01/16 Upgraded to handle, as optional, a limited range of frames
	# extpath: folder the containing the file .fts

	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)


	# path=os.getcwd()
	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)

	filefits=[]
	temp=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='csv':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
		elif filenames[index][-3:]=='fts':
			filefits.append(filenames[index])
	filenames=temp
	filenames=sorted(filenames, key=str.lower)

	numfiles=len(filenames)

	if numfiles>0:
		filename=filenames[0]
		# print('os.path.join(path,filename)',os.path.join(path,filename))
		firstrow=-1
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			pointer=0
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if (is_number(temp)) & (firstrow==-1):
					firstrow=pointer
					rowlen=np.shape(row)[0]
					pointer+=1
				elif is_number(temp) & firstrow>-1:
					pointer+=1
				else:
					ponter=0
		lastrow=pointer
		sizey=lastrow-firstrow
		sizex=rowlen


	datafit=fits.open(os.path.join(path,filefits[0]))
	if numfiles==0:
		sizey = datafit[0].shape[1]
		sizex = datafit[0].shape[2]
	lenfits=datafit[0].shape[0]
	zero_level = 2**(datafit[0].header['BITPIX']-1)
	datafit=datafit[0].data

	datafit=datafit+zero_level

	datatest=np.zeros((1,numfiles,sizey,sizex))
	# data=np.zeros((1,lenfits,sizey,sizex))
	# index=0
	# for frame in datafit:
	# 	# plt.pcolor(frame)
	# 	# plt.show()
	# 	frame=np.flip(frame,0)
	# 	# plt.pcolor(frame)
	# 	# plt.show()
	# 	data[0,index,:,:]=frame
	# 	index+=1
	data = np.array([np.flip(datafit,axis=1)])

	if numfiles>0:
		print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)
	else:
		print('sizey,sizex',sizey,sizex)

	if numfiles>0:
		file=0
		for filename in filenames:
			with open(os.path.join(path,filename),'r') as csvfile:
				reader = csv.reader(csvfile)
				# print('reader',reader)
				tempdata=[]
				pointer=sizey-1
				for row in reader:
			#		print('shape',np.shape(row))
					# print('row',row)
					if not row:
						temp='empty'
					else:
						temp=row[0]
					if is_number(temp):
						for k in range(len(row)):
							# print('j,i,pointer,k',j,i,pointer,k)
							datatest[0,file,pointer,k]=(float(row[k]))
							# print('float(k)',float(row[k]))
						pointer-=1
						# print('row',row)
					else:
						ponter=0
				file+=1


	# 2018/01/16 Section added to avoid the need of a separate "collect_subfolderfits_limited" function
	if start=='auto':
		force_start=0
	elif (start<0 or start>lenfits):
		print('The initial limit is out of range (a number of frame)')
		print('0 will be used instead of '+str(start))
		force_start=0
	else:
		force_start=start
	if stop=='auto':
		force_end=lenfits
	elif ((stop<-lenfits+1) or stop>lenfits or stop<=force_start):
		print('The final limit to search for the oscillation ad erase it is out of range (a number of frame)')
		print(str(lenfits)+'s will be used instead of '+str(stop))
		force_end=lenfits
	else:
		force_end=stop

	# if start<0:
	# 	start=0
	# if start>lenfits:
	# 	start=lenfits
	# if stop>lenfits:
	# 	stop=lenfits
	# if stop<start:
	# 	print('there must be something wrong, you are giving start frame higher than stop one')
	# 	exit()
	datacropped=data[:,force_start:force_end,:,:]


	# if (not (np.array_equal(datatest[0,0],data[0,0]))):
	# 	print('there must be something wrong, the first frame of the FITS file do not match with the first csv files')
	# 	exit()
	if (numfiles>0 and (not (np.array_equal(datatest[0,-1],data[0,-1])))):
		print('there must be something wrong, the last frame of the FITS file do not match with the last csv files')
		# exit()
	else:
		# np.save(os.path.join(extpath,lastpath),datacropped)
		np.savez_compressed(os.path.join(extpath,lastpath),datacropped=datacropped)

	# plt.pcolor(data[0,20,:,:])
	# plt.show()

##############################################################################################

def collect_subfolderfits_limited(extpath,start,stop):

	# 10/05/2018 THIS FUNCTION IS EXACLT LIKE collect_subfolderfits BUT LIMIT THE FILE CREATED FROM THE start FRAME TO THE  finish ONE
	print('Use collect_subfolderfits instead of collect_subfolderfits_limited')
	exit()


	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)


	# path=os.getcwd()
	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)

	filefits=[]
	temp=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='csv':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
		elif filenames[index][-3:]=='fts':
			filefits.append(filenames[index])
	filenames=temp
	filenames=sorted(filenames, key=str.lower)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	datafit=fits.open(os.path.join(path,filefits[0]))
	datafit=datafit[0].data

	datafit=datafit+32767
	lenfits=len(datafit)

	data=np.zeros((1,lenfits,sizey,sizex))
	datatest=np.zeros((1,numfiles,sizey,sizex))
	index=0
	for frame in datafit:
		# plt.pcolor(frame)
		# plt.show()
		frame=np.flip(frame,0)
		# plt.pcolor(frame)
		# plt.show()
		data[0,index,:,:]=frame
		index+=1


	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						datatest[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1

	if (np.array_equal(datatest[0,0],data[0,0]))&(np.array_equal(datatest[0,-1],data[0,-1])):
		print('there must be something wrong, the first or last frame of the TITS file do not match with the two csv files')
		exit()


	if start<0:
		start=0
	if start>lenfits:
		start=lenfits
	if stop>lenfits:
		stop=lenfits
	if stop<start:
		print('there must be something wrong, you are giving start frame higher than stop one')
		exit()
	datacropped=data[:,start:stop,:,:]

	np.save(os.path.join(extpath,lastpath),datacropped)

	# plt.pcolor(data[0,20,:,:])
	# plt.show()

##############################################################################################

# make movie

def movie(extpath,framerate,integration,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',timesteps='auto',extvmin='auto',extvmax='auto'):

	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)

	filefits=[]
	temp=[]
	filemovie=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='npy':
			temp.append(filenames[index])

	filenames=temp
	filenames=sorted(filenames, key=str.lower)

	numfiles=len(filenames)


	filename=filenames[0]

	data=np.load(os.path.join(path,filename))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#	 return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data[0])
	frames[0]=data[0,0]

	for i in range(len(data[0])):
		# x	   += 1
		# curVals  = f(x, y)
		frames[i]=(data[0,i])

	cv0 = frames[0]
	im = ax.imshow(cv0,cmap, origin='lower') # Here make an AxesImage rather than contour
	cb = fig.colorbar(im).set_label(barlabel)
	cb = ax.set_xlabel(xlabel)
	cb = ax.set_ylabel(ylabel)
	tx = ax.set_title('Frame 0')



	if timesteps=='auto':
		def animate(i):
			arr = frames[i]
			if extvmax=='auto':
				vmax = np.max(arr)
			elif extvmax=='allmax':
				vmax = np.max(data)
			else:
				vmax = extvmax

			if extvmin=='auto':
				vmin = np.min(arr)
			elif extvmin=='allmin':
				vmin = np.min(data)
			else:
				vmin = extvmin

			im.set_data(arr)
			im.set_clim(vmin, vmax)
			tx.set_text('Frame {0}'.format(i)+', FR '+str(framerate)+'Hz, t '+str(np.around(0+i/framerate,decimals=3))+'s int '+str(integration)+'ms')
			# In this version you don't have to do anything to the colorbar,
			# it updates itself when the mappable it watches (im) changes
	else:
		def animate(i):
			arr = frames[i]
			if extvmax=='auto':
				vmax = np.max(arr)
			elif extvmax=='allmax':
				vmax = np.max(data)
			else:
				vmax = extvmax

			if extvmin=='auto':
				vmin = np.min(arr)
			elif extvmin=='allmin':
				vmin = np.min(data)
			else:
				vmin = extvmin

			im.set_data(arr)
			im.set_clim(vmin, vmax)
			tx.set_text('Frame {0}'.format(i)+', t '+str(timesteps[i])+'s int '+str(integration)+'ms')

	ani = animation.FuncAnimation(fig, animate, frames=len(data[0]))

	ani.save(os.path.join(extpath,lastpath)+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


######################################################################################################
EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
foil_size = [0.07,0.09]

exec(open("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval/collect_and_eval/MASTU_structure.py").read())

# functions to draw the x-point on images or movies
def point_toroidal_to_cartesian(coords):	# r,z,teta deg	to	x,y,z
	out = np.zeros_like(coords).astype(float)
	out.T[0]=coords.T[0] * np.cos(coords.T[2]*2*np.pi/360)
	out.T[1]=coords.T[0] * np.sin(coords.T[2]*2*np.pi/360)
	out.T[2]=coords.T[1]
	return out

stand_off_length = 0.045	# m
# Rf=1.54967	# m	radius of the centre of the foil
Rf=1.48967 + 0.01 + 0.003 + 0.002 + stand_off_length	# m	radius of the centre of the foil
plane_equation = np.array([1,-1,0,2**0.5 * Rf])	# plane of the foil
centre_of_front_plate = np.array([1.48967+0.002,-0.7,135])	# R,z,teta deg
pinhole_offset = np.array([-0.0198,-0.0198])	# toroidal direction parallel to the place surface, z

def locate_pinhole(centre_of_front_plate=centre_of_front_plate,pinhole_offset=pinhole_offset):
	pinhole_location_toroidal = np.array([(centre_of_front_plate[0]**2 + pinhole_offset[0]**2)**0.5,centre_of_front_plate[1]+pinhole_offset[1],centre_of_front_plate[2]+360/(2*np.pi)*np.arctan(pinhole_offset[0]/centre_of_front_plate[0])])	# R,z,teta deg
	return point_toroidal_to_cartesian(pinhole_location_toroidal)	# x,y,z

# pinhole_location = np.array([-1.04087,1.068856,-0.7198])	# x,y,z
pinhole_location = locate_pinhole(pinhole_offset=pinhole_offset)
centre_of_foil = np.array([-Rf/(2**0.5), Rf/(2**0.5), -0.7])	# x,y,z
foil_size = [0.07,0.09]
R_centre_column = 0.261	# m
pinhole_relative_location = np.array(foil_size)/2 + 0.0198
pinhole_radious = 0.004/2	# m


def find_location_on_foil(point_coord,plane_equation=plane_equation,pinhole_location=pinhole_location):
	t = (-plane_equation[-1] -np.sum(plane_equation[:-1]*point_coord,axis=-1)) / np.sum(plane_equation[:-1]*(pinhole_location-point_coord),axis=-1)
	out = point_coord + ((pinhole_location-point_coord).T*t).T
	return out

def absolute_position_on_foil_to_foil_coord(coords,centre_of_foil=centre_of_foil,foil_size=foil_size):	# out in [x,z]
	out = np.zeros((np.shape(coords)[0],np.shape(coords)[1]-1))
	out.T[1] = foil_size[1]/2 -(coords.T[2] - centre_of_foil[2])
	out.T[1][np.logical_or(out.T[1]>foil_size[1],out.T[1]<0)] = np.nan
	out.T[0] = np.sign((coords.T[0]-centre_of_foil[0]))*((coords.T[0]-centre_of_foil[0])**2 + (coords.T[1]-centre_of_foil[1])**2)**0.5 + foil_size[0]/2
	out.T[0][np.logical_or(out.T[0]>foil_size[0],out.T[0]<0)] = np.nan
	return out

structure_point_location_on_foil = []
for time in range(len(stucture_r)):
	point_location = np.array([stucture_r[time],stucture_z[time],stucture_t[time]]).T
	point_location = point_toroidal_to_cartesian(point_location)
	point_location = find_location_on_foil(point_location)
	structure_point_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location))
structure_point_location_on_foil.append(np.array([pinhole_relative_location[0] + np.arange(-pinhole_radious,+pinhole_radious+pinhole_radious/10/2,pinhole_radious/10),pinhole_relative_location[1] + np.abs(pinhole_radious**2-np.arange(-pinhole_radious,+pinhole_radious+pinhole_radious/10/2,pinhole_radious/10)**2)**0.5]).T)
structure_point_location_on_foil.append(np.array([pinhole_relative_location[0] + np.arange(-pinhole_radious,+pinhole_radious+pinhole_radious/10/2,pinhole_radious/10),pinhole_relative_location[1] - np.abs(pinhole_radious**2-np.arange(-pinhole_radious,+pinhole_radious+pinhole_radious/10/2,pinhole_radious/10)**2)**0.5]).T)

def return_structure_point_location_on_foil(plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	structure_point_location_on_foil = []
	for time in range(len(stucture_r)):
		point_location = np.array([stucture_r[time],stucture_z[time],stucture_t[time]]).T
		point_location = point_toroidal_to_cartesian(point_location)
		point_location = find_location_on_foil(point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
		structure_point_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location,centre_of_foil=centre_of_foil))
	structure_point_location_on_foil.append(np.array([pinhole_relative_location[0] + np.arange(-pinhole_radious,+pinhole_radious+pinhole_radious/10/2,pinhole_radious/10),pinhole_relative_location[1] + np.abs(pinhole_radious**2-np.arange(-pinhole_radious,+pinhole_radious+pinhole_radious/10/2,pinhole_radious/10)**2)**0.5]).T)
	structure_point_location_on_foil.append(np.array([pinhole_relative_location[0] + np.arange(-pinhole_radious,+pinhole_radious+pinhole_radious/10/2,pinhole_radious/10),pinhole_relative_location[1] - np.abs(pinhole_radious**2-np.arange(-pinhole_radious,+pinhole_radious+pinhole_radious/10/2,pinhole_radious/10)**2)**0.5]).T)
	return structure_point_location_on_foil

structure_radial_profile = [_MASTU_CORE_GRID_POLYGON]
structure_radial_profile.append(FULL_MASTU_CORE_GRID_POLYGON)
structure_radial_profile.append(np.array([[1.554,1.554,1.746,1.746,1.554],[-0.270,-0.452,-0.452,-0.270,-0.270]]).T)	# P5
structure_radial_profile.append(np.array([[1.516,1.516,1.788,1.788,1.516],[-0.227,-0.495,-0.495,-0.227,-0.227]]).T)	# P5 support
structure_radial_profile.append(np.array([[1.554,1.554,1.746,1.746,1.554],[0.270,0.452,0.452,0.270,0.270]]).T)	# P5
structure_radial_profile.append(np.array([[1.516,1.516,1.788,1.788,1.516],[0.227,0.495,0.495,0.227,0.227]]).T)	# P5 support
structure_radial_profile.append(np.array([[1.309,1.268,1.272,1.369,1.370,1.309],[-0.879,-0.95,-1.017,-1.017,-0.879,-0.879]]).T)	# P6
structure_radial_profile.append(np.array([[1.406,1.306,1.194,1.208,1.309,1.413,1.406],[-0.822,-0.822,-1.002,-1.002,-0.842,-0.842,-0.842]]).T)	# P6 cover
structure_radial_profile.append(np.array([[1.309,1.268,1.272,1.369,1.370,1.309],[0.879,0.95,1.017,1.017,0.879,0.879]]).T)	# P6
structure_radial_profile.append(np.array([[1.406,1.306,1.194,1.208,1.309,1.413,1.406],[0.822,0.822,1.002,1.002,0.842,0.842,0.842]]).T)	# P6 cover
structure_radial_profile.append(np.array([[1.468,1.395,1.446,1.518,1.468],[-0.495,-0.776,-0.788,-0.508,-0.495]]).T)	# ELM coil
structure_radial_profile.append(np.array([[1.468,1.395,1.446,1.518,1.468],[0.495,0.776,0.788,0.508,0.495]]).T)	# ELM coil
structure_radial_profile.append(np.array([[2,1.490,1.490,2],[-0.616,-0.616,-0.784,-0.784]]).T)	# IRVB tube
def return_structure_radial_profile():
	return structure_radial_profile
res_bolo_radial_profile = [np.array([[R_centre_column,(core_tangential_common_point[0]**2+core_tangential_common_point[1]**2)**0.5],[0,0]]).T]
for core_poloidal_arrival_ in core_poloidal_arrival:
	res_bolo_radial_profile.append(np.array([[core_poloidal_arrival_[0],core_poloidal_common_point[0]],[core_poloidal_arrival_[1],core_poloidal_common_point[1]]]).T)
for divertor_poloidal_arrival_ in divertor_poloidal_arrival:
	res_bolo_radial_profile.append(np.array([[divertor_poloidal_arrival_[0],divertor_poloidal_common_point[0]],[divertor_poloidal_arrival_[1],divertor_poloidal_common_point[1]]]).T)

fueling_point_location_on_foil = []
for time in range(len(fueling_r)):
	point_location = np.array([fueling_r[time],fueling_z[time],fueling_t[time]]).T
	point_location = point_toroidal_to_cartesian(point_location)
	point_location = find_location_on_foil(point_location)
	fueling_point_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location))

def return_fueling_point_location_on_foil(plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	fueling_point_location_on_foil = []
	for time in range(len(fueling_r)):
		point_location = np.array([fueling_r[time],fueling_z[time],fueling_t[time]]).T
		point_location = point_toroidal_to_cartesian(point_location)
		point_location = find_location_on_foil(point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
		fueling_point_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location,centre_of_foil=centre_of_foil))
	return fueling_point_location_on_foil

def return_all_time_x_point_location(efit_reconstruction,resolution = 1000,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	all_time_x_point_location = []
	for time in range(len(efit_reconstruction.time)):
		x_point_location = np.array([[efit_reconstruction.lower_xpoint_r[time]]*resolution,[efit_reconstruction.lower_xpoint_z[time]]*resolution,np.linspace(0,360,resolution)]).T
		x_point_location = point_toroidal_to_cartesian(x_point_location)
		x_point_location = find_location_on_foil(x_point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
		all_time_x_point_location.append(absolute_position_on_foil_to_foil_coord(x_point_location,centre_of_foil=centre_of_foil))
	all_time_x_point_location = np.array(all_time_x_point_location)
	return all_time_x_point_location

def return_all_time_mag_axis_location(efit_reconstruction,resolution = 1000,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	all_time_mag_axis_location = []
	for time in range(len(efit_reconstruction.time)):
		mag_axis_location = np.array([[efit_reconstruction.mag_axis_r[time]]*resolution,[efit_reconstruction.mag_axis_z[time]]*resolution,np.linspace(0,360,resolution)]).T
		mag_axis_location = point_toroidal_to_cartesian(mag_axis_location)
		mag_axis_location = find_location_on_foil(mag_axis_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
		all_time_mag_axis_location.append(absolute_position_on_foil_to_foil_coord(mag_axis_location,centre_of_foil=centre_of_foil))
	all_time_mag_axis_location = np.array(all_time_mag_axis_location)
	return all_time_mag_axis_location

def return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,resolution = 1000,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	from scipy.signal import find_peaks, peak_prominences as get_proms
	temp_R = np.ones((len(efit_reconstruction.time),20))*np.nan
	temp_Z = np.ones((len(efit_reconstruction.time),20))*np.nan
	for time in range(len(efit_reconstruction.time)):
		temp = np.array([efit_reconstruction.strikepointR[time],efit_reconstruction.strikepointZ[time]]).T
		if temp.max()<1e-1 and len(flatten(all_time_sep_r[time]))>0:
			try:
				a=np.concatenate([r_fine[all_time_sep_r[time][0]],r_fine[all_time_sep_r[time][2]]])
				b=np.concatenate([z_fine[all_time_sep_z[time][0]],z_fine[all_time_sep_z[time][2]]])
			except:
				try:
					a=r_fine[all_time_sep_r[time][0]]
					b=z_fine[all_time_sep_z[time][0]]
				except:
					a=r_fine[all_time_sep_r[time][2]]
					b=z_fine[all_time_sep_z[time][2]]
			# c=np.abs(a-R_centre_column)
			c=np.abs(a-R_centre_column_interpolator(b))
			peaks = find_peaks(-c)[0]
			peaks = peaks[c[peaks]<4e-3]
			# efit_reconstruction.strikepointR[time][:min(len(efit_reconstruction.strikepointR[time]),len(peaks))] = a[peaks][:min(len(efit_reconstruction.strikepointR[time]),len(peaks))]
			# efit_reconstruction.strikepointZ[time][:min(len(efit_reconstruction.strikepointZ[time]),len(peaks))] = b[peaks][:min(len(efit_reconstruction.strikepointZ[time]),len(peaks))]
			temp_R[time][:min(20,len(peaks))] = a[peaks][:min(20,len(peaks))]
			temp_Z[time][:min(20,len(peaks))] = b[peaks][:min(20,len(peaks))]
		else:
			temp_R[time][:min(20,len(efit_reconstruction.strikepointR[time]))] = efit_reconstruction.strikepointR[time][:min(20,len(efit_reconstruction.strikepointR[time]))]
			temp_Z[time][:min(20,len(efit_reconstruction.strikepointZ[time]))] = -np.abs(efit_reconstruction.strikepointZ[time][:min(20,len(efit_reconstruction.strikepointZ[time]))])
	all_time_strike_points_location = []
	for time in range(len(efit_reconstruction.time)):
		# strike_point_location = np.array([efit_reconstruction.strikepointR[time],-np.abs(efit_reconstruction.strikepointZ[time]),[60]*len(efit_reconstruction.strikepointZ[time])]).T
		strike_point_location = np.array([temp_R[time],temp_Z[time],[60]*len(temp_Z[time])]).T
		strike_point_location = point_toroidal_to_cartesian(strike_point_location)
		strike_point_location = find_location_on_foil(strike_point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
		all_time_strike_points_location.append(absolute_position_on_foil_to_foil_coord(strike_point_location,centre_of_foil=centre_of_foil))
	all_time_strike_points_location_rot = []
	for time in range(len(efit_reconstruction.time)):
		temp = []
		for __i in range(len(temp_R[time])):
			# strike_point_location = np.array([[efit_reconstruction.strikepointR[time][__i]]*resolution,[-np.abs(efit_reconstruction.strikepointZ[time][__i])]*resolution,np.linspace(0,360,resolution)]).T
			strike_point_location = np.array([[temp_R[time][__i]]*resolution,[temp_Z[time][__i]]*resolution,np.linspace(0,360,resolution)]).T
			strike_point_location = point_toroidal_to_cartesian(strike_point_location)
			strike_point_location = find_location_on_foil(strike_point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
			temp.append(absolute_position_on_foil_to_foil_coord(strike_point_location,centre_of_foil=centre_of_foil))
		all_time_strike_points_location_rot.append(temp)
	return all_time_strike_points_location,all_time_strike_points_location_rot

def return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,resolution = 1000):
	from scipy.signal import find_peaks, peak_prominences as get_proms
	temp_R = np.ones((len(efit_reconstruction.time),20))*np.nan
	temp_Z = np.ones((len(efit_reconstruction.time),20))*np.nan
	for time in range(len(efit_reconstruction.time)):
		temp = np.array([efit_reconstruction.strikepointR[time],efit_reconstruction.strikepointZ[time]]).T
		if temp.max()<1e-1 and len(flatten(all_time_sep_r[time]))>0:
			try:
				a=np.concatenate([r_fine[all_time_sep_r[time][0]],r_fine[all_time_sep_r[time][2]]])
				b=np.concatenate([z_fine[all_time_sep_z[time][0]],z_fine[all_time_sep_z[time][2]]])
			except:
				try:
					a=r_fine[all_time_sep_r[time][0]]
					b=z_fine[all_time_sep_z[time][0]]
				except:
					a=r_fine[all_time_sep_r[time][2]]
					b=z_fine[all_time_sep_z[time][2]]
			# c=np.abs(a-R_centre_column)
			c=np.abs(a-R_centre_column_interpolator(b))
			peaks = find_peaks(-c)[0]
			peaks = peaks[c[peaks]<4e-3]
			# efit_reconstruction.strikepointR[time][:min(len(efit_reconstruction.strikepointR[time]),len(peaks))] = a[peaks][:min(len(efit_reconstruction.strikepointR[time]),len(peaks))]
			# efit_reconstruction.strikepointZ[time][:min(len(efit_reconstruction.strikepointZ[time]),len(peaks))] = b[peaks][:min(len(efit_reconstruction.strikepointZ[time]),len(peaks))]
			temp_R[time][:min(20,len(peaks))] = a[peaks][:min(20,len(peaks))]
			temp_Z[time][:min(20,len(peaks))] = b[peaks][:min(20,len(peaks))]
		else:
			temp_R[time][:min(20,len(efit_reconstruction.strikepointR[time]))] = efit_reconstruction.strikepointR[time][:min(20,len(efit_reconstruction.strikepointR[time]))]
			temp_Z[time][:min(20,len(efit_reconstruction.strikepointZ[time]))] = -np.abs(efit_reconstruction.strikepointZ[time][:min(20,len(efit_reconstruction.strikepointZ[time]))])
	return np.transpose([temp_R,temp_Z], axes=(1,0,2))

def return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,ref_angle=60,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	all_time_separatrix = []
	for time in range(len(efit_reconstruction.time)):
		separatrix = []
		for i in range(len(all_time_sep_r[time])):
			try:
				point_location = np.array([r_fine[all_time_sep_r[time][i]],z_fine[all_time_sep_z[time][i]],[ref_angle]*len(all_time_sep_z[time][i])]).T
			except:
				point_location = np.array([[0],[0],[ref_angle]]).T
			point_location = point_toroidal_to_cartesian(point_location)
			point_location = find_location_on_foil(point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
			separatrix.append(absolute_position_on_foil_to_foil_coord(point_location,centre_of_foil=centre_of_foil))
		all_time_separatrix.append(separatrix)
	return all_time_separatrix

def return_all_time_separatrix_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine):
	all_time_separatrix = []
	for time in range(len(efit_reconstruction.time)):
		separatrix = []
		for i in range(len(all_time_sep_r[time])):
			try:
				separatrix.append([r_fine[all_time_sep_r[time][i]],z_fine[all_time_sep_z[time][i]]])
			except:
				separatrix.append([[0],[0]])
		all_time_separatrix.append(separatrix)
	return all_time_separatrix

def return_core_tangential_location_on_foil(resolution = 10000,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	core_tangential_location_on_foil = []
	for i in range(len(core_tangential_arrival)):
		point_location = np.array([np.linspace(core_tangential_arrival[i][0],core_tangential_common_point[0],resolution),np.linspace(core_tangential_arrival[i][1],core_tangential_common_point[1],resolution),[core_tangential_arrival[i][2]]*resolution]).T
		point_location = find_location_on_foil(point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
		core_tangential_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location,centre_of_foil=centre_of_foil))
	return core_tangential_location_on_foil

def return_core_poloidal_location_on_foil(angle=60,resolution = 10000,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	core_poloidal_location_on_foil = []
	for i in range(len(core_poloidal_arrival)):
		point_location = np.array([np.linspace(core_poloidal_arrival[i][0],core_poloidal_common_point[0],resolution),np.linspace(core_poloidal_arrival[i][1],core_poloidal_common_point[1],resolution),[angle]*resolution]).T
		point_location = point_toroidal_to_cartesian(point_location)
		point_location = find_location_on_foil(point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
		core_poloidal_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location,centre_of_foil=centre_of_foil))
	return core_poloidal_location_on_foil

def return_core_poloidal(resolution = 100):
	core_poloidal = []
	for arrival in core_poloidal_arrival:
		if np.sum(np.isnan(arrival))>0:
			core_poloidal.append([])
		else:
			interp1 = interp1d([arrival[0],core_poloidal_common_point[0]],[arrival[1],core_poloidal_common_point[1]],fill_value="extrapolate",bounds_error=False)
			core_poloidal.append(np.array([np.linspace(arrival[0],core_poloidal_common_point[0],resolution),interp1(np.linspace(arrival[0],core_poloidal_common_point[0],resolution))]).T)
	return core_poloidal

def return_divertor_poloidal_location_on_foil(angle=60,resolution = 10000,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil):
	divertor_poloidal_location_on_foil = []
	for i in range(len(divertor_poloidal_arrival)):
		point_location = np.array([np.linspace(divertor_poloidal_arrival[i][0],divertor_poloidal_common_point[0],resolution),np.linspace(divertor_poloidal_arrival[i][1],divertor_poloidal_common_point[1],resolution),[angle]*resolution]).T
		point_location = point_toroidal_to_cartesian(point_location)
		point_location = find_location_on_foil(point_location,plane_equation=plane_equation,pinhole_location=pinhole_location)
		divertor_poloidal_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location,centre_of_foil=centre_of_foil))
	return divertor_poloidal_location_on_foil

def return_divertor_poloidal(resolution = 100):
	divertor_poloidal = []
	for arrival in divertor_poloidal_arrival:
		if np.sum(np.isnan(arrival))>0:
			divertor_poloidal.append([])
		else:
			interp1 = interp1d([arrival[0],divertor_poloidal_common_point[0]],[arrival[1],divertor_poloidal_common_point[1]],fill_value="extrapolate",bounds_error=False)
			divertor_poloidal.append(np.array([np.linspace(arrival[0],divertor_poloidal_common_point[0],resolution),interp1(np.linspace(arrival[0],divertor_poloidal_common_point[0],resolution))]).T)
	return divertor_poloidal

def movie_from_data(data,framerate,integration=1,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',form_factor_size=15,timesteps='auto',extvmin='auto',extvmax='auto',image_extent=[],mask=[0],mask_alpha=0.2,time_offset=0,prelude='',vline=None,hline=None,EFIT_path=EFIT_path_default,include_EFIT=False,efit_reconstruction=None,EFIT_output_requested = False,pulse_ID=None,overlay_x_point=False,overlay_mag_axis=False,overlay_structure=False,overlay_strike_points=False,overlay_separatrix=False,structure_alpha=0.5,foil_size=foil_size,additional_polygons_dict = dict([])):
	import matplotlib.animation as animation
	import numpy as np

	if len(image_extent)==4:
		form_factor = (image_extent[1]-image_extent[0])/(image_extent[3]-image_extent[2])
	else:
		form_factor = (np.shape(data[0][0])[1])/(np.shape(data[0][0])[0])
	fig = plt.figure(figsize=(form_factor_size*form_factor, form_factor_size))
	ax = fig.add_subplot(111)

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#	 return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	if len(image_extent)==4:
		ver = np.arange(0,np.shape(data[0][0])[0])
		up = np.abs(image_extent[3]-ver).argmin()
		down = np.abs(image_extent[2]-ver).argmin()
		hor = np.arange(0,np.shape(data[0][0])[1])
		left = np.abs(image_extent[0]-hor).argmin()
		right = np.abs(image_extent[1]-hor).argmin()
		data[0][:,:,:left] = np.nan
		data[0][:,:,right+1:] = np.nan
		data[0][:,:down] = np.nan
		data[0][:,up+1:] = np.nan

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data[0])
	frames[0]=data[0,0]

	for i in range(len(data[0])):
		# x	   += 1
		# curVals  = f(x, y)
		frames[i]=(data[0,i])

	cv0 = frames[0]
	im = ax.imshow(cv0,cmap, origin='lower', interpolation='none') # Here make an AxesImage rather than contour
	if len(image_extent)==4:
		ax.set_ylim(top=image_extent[3],bottom=image_extent[2])
		ax.set_xlim(right=image_extent[1],left=image_extent[0])


	if len(np.shape(mask))==2:
		if np.shape(mask)!=np.shape(data[0][0]):
			print('The shape of the mask '+str(np.shape(mask))+' does not match with the shape of the record '+str(np.shape(data[0][0])))
		masked = np.ma.masked_where(mask == 0, mask)
		im2 = ax.imshow(masked, 'gray', interpolation='none', alpha=mask_alpha,origin='lower',extent = [0,np.shape(cv0)[1]-1,0,np.shape(cv0)[0]-1])

	if include_EFIT:
		try:
			if efit_reconstruction == None:
				print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc')
				efit_reconstruction = mclass(EFIT_path+'/epm0'+str(pulse_ID)+'.nc',pulse_ID=pulse_ID)
			else:
				print('EFIT reconstruction externally supplied')
			EFIT_dt = np.median(np.diff(efit_reconstruction.time))
		except Exception as e:
			print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc failed')
			logging.exception('with error: ' + str(e))
			include_EFIT=False
			overlay_x_point=False
			overlay_mag_axis=False
			overlay_separatrix=False
			overlay_strike_points=False
			overlay_separatrix=False
			efit_reconstruction = None
		if overlay_x_point:
			all_time_x_point_location = return_all_time_x_point_location(efit_reconstruction)
			plot1 = ax.plot(0,0,'-r', alpha=1)[0]
		if overlay_mag_axis:
			all_time_mag_axis_location = return_all_time_mag_axis_location(efit_reconstruction)
			plot2 = ax.plot(0,0,'--r', alpha=1)[0]
		if overlay_separatrix or overlay_strike_points:
			all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
		if overlay_strike_points:
			all_time_strike_points_location,all_time_strike_points_location_rot = return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			plot3 = ax.plot(0,0,'xr',markersize=20, alpha=1)[0]
			plot4 = []
			for __i in range(len(all_time_strike_points_location[0])):
				plot4.append(ax.plot(0,0,'-r', alpha=1)[0])
		if overlay_separatrix:
			all_time_separatrix = return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			plot5 = []
			for __i in range(len(all_time_separatrix[0])):
				plot5.append(ax.plot(0,0,'--b', alpha=1)[0])
	if overlay_structure:
		for i in range(len(fueling_point_location_on_foil)):
			ax.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
			ax.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
		for i in range(len(structure_point_location_on_foil)):
			ax.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
	if len(list(additional_polygons_dict.keys()))!=0:
		plot7 = []
		for __i in range(additional_polygons_dict['number_of_polygons']):
			plot7.append(ax.plot(0,0,additional_polygons_dict['marker'][__i], alpha=1)[0])

	# if len(np.shape(mask)) == 2:
	# im = ax.imshow(mask,'gray',interpolation='none',alpha=1)
	if np.sum(vline == None)==0:
		if np.shape(vline)==():
			vline = max(0,min(vline,np.shape(cv0)[1]-1))
			axvline = ax.axvline(x=vline,linestyle='--',color='k')
		else:
			for i in range(len(vline)):
				vline[i] = max(0,min(vline[i],np.shape(cv0)[1]-1))
				axvline = ax.axvline(x=vline[i],linestyle='--',color='k')
	if np.sum(hline == None)==0:
		if np.shape(hline)==():
			hline = max(0,min(hline,np.shape(cv0)[0]-1))
			axhline = ax.axhline(y=hline,linestyle='--',color='k')
		else:
			for i in range(len(hline)):
				hline[i] = max(0,min(hline[i],np.shape(cv0)[0]-1))
				axhline = ax.axhline(y=hline[i],linestyle='--',color='k')

	cb = fig.colorbar(im).set_label(barlabel)
	cb = ax.set_xlabel(xlabel)
	cb = ax.set_ylabel(ylabel)
	tx = ax.set_title('Frame 0')


	# if timesteps=='auto':
	# 	timesteps_int = time_offset+np.arange(len(data[0])+1)/framerate
	# else:
	# 	timesteps_int = cp.deepcopy(timesteps)
	def animate(i):
		arr = frames[i]
		if extvmax=='auto':
			vmax = np.nanmax(arr)
		elif extvmax=='allmax':
			vmax = np.nanmax(data)
		else:
			vmax = extvmax

		if extvmin=='auto':
			vmin = np.nanmin(arr)
		elif extvmin=='allmin':
			vmin = np.nanmin(data)
		else:
			vmin = extvmin
		im.set_data(arr)
		im.set_clim(vmin, vmax)
		if timesteps=='auto':
			time_int = time_offset+i/framerate
			tx.set_text(prelude + 'Frame {0}'.format(i)+', FR %.3gHz, t %.3gs, int %.3gms' %(framerate,time_int,integration))
		else:
			time_int = timesteps[i]
			tx.set_text(prelude + 'Frame {0}'.format(i)+', t %.3gs, int %.3gms' %(time_int,integration))
		if include_EFIT:
			if np.min(np.abs(time_int-efit_reconstruction.time))>EFIT_dt:	# means that the reconstruction is not available for that time
				if overlay_x_point:
					plot1.set_data(([],[]))
				if overlay_mag_axis:
					plot2.set_data(([],[]))
				if overlay_strike_points:
					plot3.set_data(([],[]))
					for __i in range(len(plot4)):
						plot4[__i].set_data(([],[]))
				if overlay_separatrix:
					for __i in range(len(plot5)):
						plot5[__i].set_data(([],[]))
			else:
				i_time = np.abs(time_int-efit_reconstruction.time).argmin()
				if overlay_x_point:
					if np.sum(np.isnan(all_time_x_point_location[i_time]))>=len(all_time_x_point_location[i_time]):	# means that all the points calculated are outside the foil
						plot1.set_data(([],[]))
					else:
						plot1.set_data((all_time_x_point_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
				if overlay_mag_axis:
					# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
					# 	plot2.set_data(([],[]))
					# else:
					plot2.set_data((all_time_mag_axis_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
				if overlay_strike_points:
					# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
					# 	plot3.set_data(([],[]))
					# 	for __i in range(len(plot4)):
					# 		plot4[__i].set_data(([],[]))
					# else:
					plot3.set_data((all_time_strike_points_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
					for __i in range(len(plot4)):
						plot4[__i].set_data((all_time_strike_points_location_rot[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
				if overlay_separatrix:
					# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
					for __i in range(len(plot5)):
						plot5[__i].set_data((all_time_separatrix[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
		if len(list(additional_polygons_dict.keys()))!=0:
			i_time2 = np.abs(time_int-additional_polygons_dict['time']).argmin()
			# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
			for __i in range(additional_polygons_dict['number_of_polygons']):
				plot7[__i].set_data((additional_polygons_dict[str(__i)][i_time2][0],additional_polygons_dict[str(__i)][i_time2][1]))
			# 	masked = np.ma.masked_where(mask == 0, mask)
			# 	ax.imshow(masked, 'gray', interpolation='none', alpha=0.2,origin='lower',extent = [0,np.shape(data)[0]-1,0,np.shape(data)[1]-1])

			# In this version you don't have to do anything to the colorbar,
			# it updates itself when the mappable it watches (im) changes
	# else:
	# 	def animate(i):
	# 		arr = frames[i]
	# 		if extvmax=='auto':
	# 			vmax = np.max(arr)
	# 		elif extvmax=='allmax':
	# 			vmax = np.max(data)
	# 		else:
	# 			vmax = extvmax
	#
	# 		if extvmin=='auto':
	# 			vmin = np.min(arr)
	# 		elif extvmin=='allmin':
	# 			vmin = np.min(data)
	# 		else:
	# 			vmin = extvmin
	# 		im.set_data(arr)
	# 		im.set_clim(vmin, vmax)
	# 		tx.set_text(prelude + 'Frame {0}'.format(i)+', t %.3gs, int %.3gms' %(timesteps[i],integration))
	# 		# if len(np.shape(mask)) == 2:
	# 		# 	# 	im.imshow(mask,'binary',interpolation='none',alpha=0.3)
	# 		# 	masked = np.ma.masked_where(mask == 0, mask)
	# 		# 	ax.imshow(masked, 'gray', interpolation='none', alpha=0.2,origin='lower',extent = [0,np.shape(data)[0]-1,0,np.shape(data)[1]-1])

	ani = animation.FuncAnimation(fig, animate, frames=len(data[0]))

	if EFIT_output_requested == False:
		return ani
	else:
		return ani,efit_reconstruction

def image_from_data(data,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',form_factor_size=15,ref_time=None,extvmin='auto',extvmax='auto',image_extent=[],mask=[0],mask_alpha=0.2,prelude='',vline=None,hline=None,EFIT_path=EFIT_path_default,include_EFIT=False,efit_reconstruction=None,EFIT_output_requested = False,pulse_ID=None,overlay_x_point=False,overlay_mag_axis=False,overlay_structure=False,overlay_strike_points=False,overlay_separatrix=False,overlay_res_bolo=False,structure_alpha=0.5,foil_size=foil_size):
	import matplotlib.animation as animation
	import numpy as np
	from matplotlib import cm	# to print nan as white

	if len(image_extent)==4:
		form_factor = (image_extent[1]-image_extent[0])/(image_extent[3]-image_extent[2])
	else:
		form_factor = (np.shape(data[0][0])[1])/(np.shape(data[0][0])[0])
	fig = plt.figure(figsize=(form_factor_size*form_factor, form_factor_size))
	ax = fig.add_subplot(111)

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#	 return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	if len(image_extent)==4:
		ver = np.arange(0,np.shape(data[0][0])[0])
		up = np.abs(image_extent[3]-ver).argmin()
		down = np.abs(image_extent[2]-ver).argmin()
		hor = np.arange(0,np.shape(data[0][0])[1])
		left = np.abs(image_extent[0]-hor).argmin()
		right = np.abs(image_extent[1]-hor).argmin()
		data = data.astype(float)
		data[0][:,:,:left] = np.nan
		data[0][:,:,right+1:] = np.nan
		data[0][:,:down] = np.nan
		data[0][:,up+1:] = np.nan

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data[0])
	frames[0]=data[0,0]

	for i in range(len(data[0])):
		# x	   += 1
		# curVals  = f(x, y)
		frames[i]=(data[0,i])

	cv0 = frames[0]
	masked_array = np.ma.array (cv0, mask=np.isnan(cv0))
	# exec('cmap=cm.' + cmap)
	cmap=cm.rainbow
	cmap.set_bad('white',1.)
	im = ax.imshow(masked_array,cmap=cmap, origin='lower', interpolation='none') # Here make an AxesImage rather than contour
	if len(image_extent)==4:
		ax.set_ylim(top=image_extent[3],bottom=image_extent[2])
		ax.set_xlim(right=image_extent[1],left=image_extent[0])


	if len(np.shape(mask))==2:
		if np.shape(mask)!=np.shape(data[0][0]):
			print('The shape of the mask '+str(np.shape(mask))+' does not match with the shape of the record '+str(np.shape(data[0][0])))
		masked = np.ma.masked_where(mask == 0, mask)
		im2 = ax.imshow(masked, 'gray', interpolation='none', alpha=mask_alpha,origin='lower',extent = [0,np.shape(cv0)[1]-1,0,np.shape(cv0)[0]-1])

	if ref_time==None:
		include_EFIT = False

	if include_EFIT:
		try:
			if efit_reconstruction == None:
				print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc')
				efit_reconstruction = mclass(EFIT_path+'/epm0'+str(pulse_ID)+'.nc',pulse_ID=pulse_ID)
			else:
				print('EFIT reconstruction externally supplied')
			EFIT_dt = np.median(np.diff(efit_reconstruction.time))
		except Exception as e:
			print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc failed')
			logging.exception('with error: ' + str(e))
			include_EFIT=False
			overlay_x_point=False
			overlay_mag_axis=False
			overlay_separatrix=False
			overlay_strike_points=False
			overlay_separatrix=False
			efit_reconstruction = None
		if overlay_x_point:
			all_time_x_point_location = return_all_time_x_point_location(efit_reconstruction)
			plot1 = ax.plot(0,0,'-r', alpha=1)[0]
		if overlay_mag_axis:
			all_time_mag_axis_location = return_all_time_mag_axis_location(efit_reconstruction)
			plot2 = ax.plot(0,0,'--r', alpha=1)[0]
		if overlay_separatrix or overlay_strike_points:
			all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
		if overlay_strike_points:
			all_time_strike_points_location,all_time_strike_points_location_rot = return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			plot3 = ax.plot(0,0,'xr',markersize=20, alpha=1)[0]
			plot4 = []
			for __i in range(len(all_time_strike_points_location[0])):
				plot4.append(ax.plot(0,0,'-r', alpha=1)[0])
		if overlay_separatrix:
			all_time_separatrix = return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			plot5 = []
			for __i in range(len(all_time_separatrix[0])):
				plot5.append(ax.plot(0,0,'--b', alpha=1)[0])
	if overlay_structure:
		for i in range(len(fueling_point_location_on_foil)):
			ax.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
			ax.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
		for i in range(len(structure_point_location_on_foil)):
			ax.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)

	# if len(np.shape(mask)) == 2:
	# im = ax.imshow(mask,'gray',interpolation='none',alpha=1)
	if np.sum(vline == None)==0:
		if np.shape(vline)==():
			vline = max(0,min(vline,np.shape(cv0)[1]-1))
			axvline = ax.axvline(x=vline,linestyle='--',color='k')
		else:
			for i in range(len(vline)):
				vline[i] = max(0,min(vline[i],np.shape(cv0)[1]-1))
				axvline = ax.axvline(x=vline[i],linestyle='--',color='k')
	if np.sum(hline == None)==0:
		if np.shape(hline)==():
			hline = max(0,min(hline,np.shape(cv0)[0]-1))
			axhline = ax.axhline(y=hline,linestyle='--',color='k')
		else:
			for i in range(len(hline)):
				hline[i] = max(0,min(hline[i],np.shape(cv0)[0]-1))
				axhline = ax.axhline(y=hline[i],linestyle='--',color='k')

	cb = fig.colorbar(im).set_label(barlabel)
	cb = ax.set_xlabel(xlabel)
	cb = ax.set_ylabel(ylabel)
	tx = ax.set_title('Frame 0')


	arr = frames[0]
	if extvmax=='auto':
		vmax = np.nanmax(arr)
	elif extvmax=='allmax':
		vmax = np.nanmax(data)
	else:
		vmax = extvmax

	if extvmin=='auto':
		vmin = np.nanmin(arr)
	elif extvmin=='allmin':
		vmin = np.nanmin(data)
	else:
		vmin = extvmin
	masked_array = np.ma.array (arr, mask=np.isnan(arr))
	im.set_data(masked_array)
	im.set_clim(vmin, vmax)
	tx.set_text(prelude)
	if include_EFIT:
		if np.min(np.abs(ref_time-efit_reconstruction.time))>EFIT_dt:	# means that the reconstruction is not available for that time
			if overlay_x_point:
				plot1.set_data(([],[]))
			if overlay_mag_axis:
				plot2.set_data(([],[]))
			if overlay_strike_points:
				plot3.set_data(([],[]))
				for __i in range(len(plot4)):
					plot4[__i].set_data(([],[]))
			if overlay_separatrix:
				for __i in range(len(plot5)):
					plot5[__i].set_data(([],[]))
		else:
			i_time = np.abs(ref_time-efit_reconstruction.time).argmin()
			if overlay_x_point:
				if np.sum(np.isnan(all_time_x_point_location[i_time]))>=len(all_time_x_point_location[i_time]):	# means that all the points calculated are outside the foil
					plot1.set_data(([],[]))
				else:
					plot1.set_data((all_time_x_point_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
			if overlay_mag_axis:
				# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
				# 	plot2.set_data(([],[]))
				# else:
				plot2.set_data((all_time_mag_axis_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
			if overlay_strike_points:
				# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
				# 	plot3.set_data(([],[]))
				# 	for __i in range(len(plot4)):
				# 		plot4[__i].set_data(([],[]))
				# else:
				plot3.set_data((all_time_strike_points_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
				for __i in range(len(plot4)):
					plot4[__i].set_data((all_time_strike_points_location_rot[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))
			if overlay_separatrix:
				# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
				for __i in range(len(plot5)):
					plot5[__i].set_data((all_time_separatrix[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1]))

	if EFIT_output_requested == False:
		return fig
	else:
		return fig,efit_reconstruction



######################################################################################################


def collect_stat(extpath):

	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)


	# path=os.getcwd()
	path=extpath
	print('path =', path)


	filenames=all_file_names(path,'stat.npy')
	#
	# position=[]
	# for i in range(len(path)):
	# 	if path[i]=='/':
	# 		position.append(i)
	# position=max(position)
	# lastpath=path[position+1:]
	# # print('lastpath',lastpath)
	#
	# f = []
	# for (dirpath, dirnames, filenames) in os.walk(path):
	# 	f.extend(filenames)
	# #	break
	#
	# filenames=f
	# # print('filenames',filenames)
	#
	# filefits=[]
	# temp=[]
	# print('len(filenames)',len(filenames))
	# for index in range(len(filenames)):
	# 	# print(filenames[index])
	# 	if filenames[index][-8:]=='stat.npy':
	# 		temp.append(filenames[index])
	# 		# filenames=np.delete(filenames,index)
	# 		# print('suca')
	# filenames=temp
	# filenames=sorted(filenames, key=str.lower)
	#
	# numfiles=len(filenames)


	filename=filenames[0]

	data=np.load(os.path.join(extpath,filename))

	return data

########################################################################################################


def evaluate_back(extpath):

	# THIS FUNCTION PREPARE THE FILES THAT WILL BE USED FOR GETTING THE COUNTS TO TEMPERATURE CALIBRATION
	#
	# UPDATE 10/05/2018 I FOUND AN ERROR AND CHANGED datastat[0]=np.mean(data,axis=(-1,-2))  TO   datastat[0]=np.mean(data,axis=(0,1))

	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)

	path=extpath
	# path=os.getcwd()
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=sorted(filenames, key=str.lower)
	temp=[]
	# print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='npy' and filenames[index][-3-6:] != '_stat.npy':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')

	filenames=temp

	#filenames=np.delete(filenames,0)
	numfiles=len(filenames)


	print('filenames',filenames)


	data=np.load(os.path.join(path,filenames[0]))
	datashape=np.shape(data)[-2:]

	datastatshape=[2]
	datastatshape.extend(datashape)
	datastat=np.zeros(datastatshape)

	print('np.shape(datastatshape)',np.shape(datastat))

	datastat[0]=np.mean(data,axis=(0,1))
	datastat[1]=np.std(data,axis=(0,1))

	# for i in range(datashape[0]):
	# 	for j in range(datashape[1]):
	# 		datastat[0,i,j]=np.mean(data[0,:,i,j])
	# 		datastat[1,i,j]=np.std(data[0,:,i,j])

	# datastat.extend([np.mean(data[0,:,:,:]),np.std(data[0,:,:,:])])
	# datastat[2]=np.std(data[0,:,:,:])

	# plt.pcolor(datastat[0])
	# plt.colorbar()
	# plt.figure()
	# plt.pcolor(datastat[1])
	# plt.colorbar()
	# plt.show()

	datastat=datastat.tolist()
	datastat.append([np.mean(data[0,:,:,:]),np.std(data[0,:,:,:])])

	np.save(os.path.join(extpath,lastpath+'_stat'),datastat)

######################################################################################################################

def all_file_names(extpath,type):

	# This utility returns a list with all filenames of a defined type in a given folder, orderen alphabetically and in number

	# path=os.getcwd()
	path=extpath
	print('path =', path)

	# position=[]
	# for i in range(len(path)):
	# 	if path[i]=='/':
	# 		position.append(i)
	# position=max(position)
	# lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)


	temp=[]

	typelen=len(type)
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-typelen:]==type:
			temp.append(filenames[index])
		elif filenames[index][:typelen]==type:
			temp.append(filenames[index])

			# filenames=np.delete(filenames,index)
			# print('suca')
	filenames=temp

	if len(filenames)==0:
		print('ERROR - there are no files of type '+type+' in path '+path)
		return ()

	if len(filenames)==1:
		print('len(filenames)',len(filenames))
		return filenames

	filenames=order_filenames(filenames)

	print('len(filenames)',len(filenames))
	return filenames

######################################################################################################

def read_csv(extpath,filenames):

	# SLOW VERSION OF READ_CSV WITH csv.reader

	path=extpath
	print('path =', path)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	data=np.zeros((1,numfiles,sizey,sizex))

	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						data[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1

	return data

###########################################################################################################

def read_csv2(extpath,filenames):

	# FAST VERSION OF READ_CVS WITH PANDA READER

	path=extpath
	print('path =', path)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	csvfile=path+'/'+filename
	reader = pandas.read_csv(csvfile,sep='\r')
	reader=reader.values
	# print('reader',reader)
	pointer=0
	for row in reader:
#		print('shape',np.shape(row))
		# print('row',row)
		# if np.fromstring(row[0],sep=',').size>0:
			# print('sti cazzi2')
			# print(row)
		# print((np.fromstring(row[0],sep=',')).size>0)
		if (((np.fromstring(row[0],sep=',')).size>0) & (firstrow==-1)):
			# print((np.fromstring(row[0],sep=',')).size>0)
			# print('pointer,firstrow',pointer,firstrow)
			firstrow=pointer
			rowlen=np.shape(np.fromstring(row[0],sep=','))[0]
			pointer+=1
		# elif ((np.fromstring(row[0],sep=',')).size>0) & (firstrow>-1):
		# 	pointer+=1
		else:
			pointer+=1

	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	data=np.zeros((1,numfiles,sizey,sizex))

	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		csvfile=path+'/'+filename
		# print('csvfile',csvfile)
		reader = pandas.read_csv(csvfile,sep='\r')
		reader=reader.values
		reader=np.delete(reader,(np.linspace(0,firstrow-1,firstrow).astype(int)))
		# print('reader',reader)
		tempdata=[]
		pointer=sizey-1
		# pointer=sizey-1
		for row in reader:
			# tempdata.append(row[0])
			# tempdata=np.array(tempdata[firstrow:])
			data[0,file,pointer]=np.fromstring(row,sep=',')
			# print('np.fromstring(row[0],sep=',')',row)
			pointer-=1
		file+=1
	# data=data.astype(float)
	# print('suca')
	return data

###########################################################################################################


def read_fits(extpath,filenames,filefits):

	if len(filenames)>2:
		print('this is wrong! to make this function work toy must provide as second agrument the first and last frame of the movie as .csv')
		exit()

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	datafit=fits.open(os.path.join(path,filefits[0]))
	datafit=datafit[0].data

	datafit=datafit+32767
	lenfits=len(datafit)

	data=np.zeros((1,lenfits,sizey,sizex))
	datatest=np.zeros((1,numfiles,sizey,sizex))
	index=0
	for frame in datafit:
		# plt.pcolor(frame)
		# plt.show()
		frame=np.flip(frame,0)
		# plt.pcolor(frame)
		# plt.show()
		data[0,index,:,:]=frame
		index+=1


	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						datatest[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1

	if (np.array_equal(datatest[0,0],data[0,0]))&(np.array_equal(datatest[0,-1],data[0,-1])):
		print('there must be something wrong, the first or last frame of the TITS file do not match with the two csv files')
		exit()

	return data

################################################################################################

def count_to_temp_poly(data,params,errparams,errdata=(0,0)):
	print('ERROR: count_to_temp_poly was deleted 2018/04/02 because count_to_temp_poly2 has a better formulation for the error')
	exit()

	polygrade=len(params[0,0])

	shape=np.shape(data)
	datatemp=np.zeros(shape)

	if np.max(errdata)==0:
		errdata=np.zeros(shape)

	errdatatemp=np.zeros(shape)
	numframes=shape[1]
	sizey=shape[-2]
	sizex=shape[-1]
	sumi2=sum(np.power(np.linspace(1,polygrade-1,polygrade-1),2))

	# OLD VERSION WHEN POLYGEN WAS ABLE TO PRECESSO ONLY SCALARS 2018/03/29
	# for i in range(numframes):
	# 	for j in range(sizey):
	# 		index=0
	# 		for count in data[0,i,j]:
	# 			datatemp[0,i,j,index]=polygen(polygrade)(count,*params[j,index])
	# 			errdatatemp[0,i,j,index]=count*np.sqrt(sum([sumi2*errdata[0,i,j,index]**2/count**2]+np.power(np.divide(errparams[j,index],params[j,index]),2)))
	# 			index+=1

	paramerr=np.sum(np.power(np.divide(errparams,params),2),axis=-1)
	index=0
	for frame in data[0]:
		# print('np.shape(frame)',np.shape(frame))
		# print('np.shape(params)',np.shape(params))
		datatemp[0,index]=polygen(polygrade)(frame,*params)
		errdatatemp[0,index]=np.multiply(datatemp[0,index],np.sqrt(np.add(sumi2*np.divide(np.power(errdata[0,index],2),np.power(frame,2)),paramerr)))
		index+=1

	return datatemp,errdatatemp

################################################################################################

def count_to_temp_poly2(data,params,errparams,errdata=(0,0),averaged_params=False):

	polygrade=len(params[0,0])

	shape=np.shape(data)
	datatemp=np.zeros(shape)

	if np.max(errdata)==0:
		errdata=np.zeros(shape)

	errdatatemp=np.zeros(shape)
	numframes=shape[1]
	sizey=shape[-2]
	sizex=shape[-1]
	#sumi2=sum(np.power(np.linspace(1,polygrade-1,polygrade-1),2))

	# paramerr=np.sum(np.power(np.divide(errparams,params),2),axis=-1)

	if averaged_params==True:
		params_mean=np.mean(params,axis=(0,1))
		params_to_use=np.ones(np.shape(params))*params_mean
		errparams_mean=np.mean(errparams,axis=(0,1))
		errparams_to_use=np.ones(np.shape(errparams))*errparams_mean
	else:
		params_to_use=params
		errparams_to_use=errparams

	index=0
	for frame in data[0]:
		# print('np.shape(frame)',np.shape(frame))
		# print('np.shape(params)',np.shape(params))
		datatemp[0,index]=polygen(polygrade)(frame,*params_to_use)
		temp=0
		for i in range(polygrade):
			temp+=np.power(np.multiply(errparams_to_use[:,:,i],np.power(frame,i)),2)
			if i>0:
				temp+=np.power(np.multiply(np.multiply(np.multiply(params_to_use[:,:,i],np.power(frame,i-1)),i),errdata[0,index]),2)
		# errdatatemp[0,index]=np.sqrt(np.add(sumi2*np.divide(np.power(errdata[0,index],2),np.power(frame,2)),paramerr))
		errdatatemp[0,index]=np.sqrt(temp)
		index+=1




	return datatemp,errdatatemp

################################################################################################

def build_poly_coeff(temperature,files,int,path,nmax):
	# modified 2018-10-08 to build the coefficient only for 1 degree of polinomial
	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)
	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)
	# nmax=5
	shapex=np.shape(meancounttot[0])[0]
	shapey=np.shape(meancounttot[0])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	#for n in range(2,nmax+1):
	n=nmax
	guess=np.ones(n)
	guess,temp2=curve_fit(polygen3(n), meancounttot[:,0,0],temperature, p0=guess, maxfev=100000000)

	coeff=np.zeros((shapex,shapey,n))
	errcoeff=np.zeros((shapex,shapey,n))

	for j in range(shapex):
		for k in range(shapey):
			x=np.array(meancounttot[:,j,k])
			xerr=np.array(meancountstdtot[:,j,k])
			temp1,temp2=curve_fit(polygen3(n),x, temperature, p0=guess, maxfev=100000000)
			yerr=(polygen3(n)((x+xerr),*temp1)-polygen3(n)((x-xerr),*temp1))/2
			temp1,temp2=curve_fit(polygen3(n),x, temperature, p0=temp1, sigma=yerr, maxfev=100000000)
			# yerr=(polygen3(n)((x+xerr),*temp1)-polygen3(n)((x-xerr),*temp1))/2
			guess=temp1
			coeff[j,k,:]=temp1
			errcoeff[j,k,:]=np.sqrt(np.diagonal(temp2))
			score[n-2,j,k]=rsquared(temperature,polygen3(n)(x,*temp1))
	np.save(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'ms'),coeff)
	np.save(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'ms'),errcoeff)

	print('for a polinomial of degree '+str(n-1)+' the R^2 score is '+str(np.sum(score[n-2])))


###############################################################################################################

def build_poly_coeff2(temperature,files,int,path,nmax):
	print('ERROR: builf_poly_coeff2 was deleted 2018/04/02 because curve_fit must be used without the flag absolute_sigma=True on for significant parameters covariance matrix')
	exit()

	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)
	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)
	# nmax=5
	shapex=np.shape(meancounttot[0])[0]
	shapey=np.shape(meancounttot[0])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	for n in range(2,nmax+1):
		guess=np.ones(n)
		guess,temp2=curve_fit(polygen3(n), meancounttot[:,0,0],temperature, p0=guess, maxfev=100000000)

		coeff=np.zeros((shapex,shapey,n))
		errcoeff=np.zeros((shapex,shapey,n))

		for j in range(shapex):
			for k in range(shapey):
				x=np.array(meancounttot[:,j,k])
				xerr=np.array(meancountstdtot[:,j,k])
				temp1,temp2=curve_fit(polygen3(n),x, temperature, p0=guess,absolute_sigma=True)
				yerr=(polygen3(n)((x+xerr),*temp1)-polygen3(n)((x-xerr),*temp1))/2
				temp1,temp2=curve_fit(polygen3(n),x, temperature, p0=temp1, sigma=yerr, maxfev=100000000,absolute_sigma=True)
				yerr=(polygen3(n)((x+xerr),*temp1)-polygen3(n)((x-xerr),*temp1))/2
				guess=temp1
				coeff[j,k,:]=temp1
				errcoeff[j,k,:]=np.sqrt(np.diagonal(temp2))
				score[n-2,j,k]=rsquared(temperature,polygen3(n)(x,*temp1))
		np.save(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'ms'),coeff)
		np.save(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'ms'),errcoeff)

		print('for a polinomial with '+str(n)+' coefficients the R^2 score is '+str(np.sum(score[n-2])))

################################################################################################

def build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,nmax,function_to_use = build_poly_coeff):
	# 08/10/2018 THIS CALCULATE FROM MULTIPLE HOT>ROOM AND COLD >ROOM CYCLES THE COEFFICIENTS FOR ALL THE POSSIBLE COMBINATIONS

	for i in range(len(temperaturehot)):
		temperaturehot[i]=flatten_full(temperaturehot[i])
	for i in range(len(temperaturecold)):
		temperaturecold[i]=flatten_full(temperaturecold[i])
	for i in range(len(fileshot)):
		fileshot[i]=flatten_full(fileshot[i])
	for i in range(len(filescold)):
		filescold[i]=flatten_full(filescold[i])

	for i in range(len(temperaturehot)):
		if len(temperaturehot[i])!=len(fileshot[i]):
			print('Error, temperaturehot'+str(i)+' and fileshot'+str(i)+' length is different')
			exit()
	for i in range(len(temperaturecold)):
		if len(temperaturecold[i])!=len(filescold[i]):
			print('Error, temperaturecold'+str(i)+' and filescold'+str(i)+' length is different')
			exit()

	lengthhot=len(temperaturehot)
	lengthcold=len(temperaturecold)

	# This lies must be placed outside of the function to make it work
	# fileshot=np.array([fileshot1,fileshot2])
	# temperaturehot=np.array([temperaturehot1,temperaturehot2])
	# filescold=np.array([filescold1,filescold2])
	# temperaturecold=np.array([temperaturecold1,temperaturecold2])

	# THIS COMPUTE THE 1-1, 1-2, 2-1, 2-2 PARAMETERS
	for i in range(lengthhot):
		for j in range(lengthcold):
			path=pathparam+'/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(nmax)+'/'+str(i+1)+'-'+str(j+1)
			if not os.path.exists(path):
				os.makedirs(path)
			temperature=[temperaturehot[i],temperaturecold[j]]
			files=[fileshot[i],filescold[j]]
			function_to_use(temperature,files,inttime,path,nmax)

################################################################################################

def build_average_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,nmax):
	# 08/10/2018 THIS MAKES THE AVERAGE OF COEFFICIENTS FROM MULTIPLE HOT>ROOM AND COLD >ROOM CYCLES THE COEFFICIENTS

	lengthhot=len(temperaturehot)
	lengthcold=len(temperaturecold)

	first=True
	for i in range(lengthhot):
		for j in range(lengthcold):
			path=pathparam+'/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(nmax)+'/'+str(i+1)+'-'+str(j+1)
			params=np.load(os.path.join(path,'coeffpolydeg'+str(nmax)+'int'+str(inttime)+'ms'+'.npy'))
			if first==True:
				shape=np.shape(params)
				shape=np.concatenate(((lengthhot,lengthcold),shape))
				parameters=np.zeros(shape)
				first=False
			parameters[i,j]=params

	meanparameters=np.mean(parameters,axis=(0,1))
	stdparameters=np.std(parameters,axis=(0,1))

	path=pathparam+'/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(nmax)+'/average'
	if not os.path.exists(path):
		os.makedirs(path)
	np.save(os.path.join(path,'coeffpolydeg'+str(nmax)+'int'+str(inttime)+'ms'),meanparameters)
	np.save(os.path.join(path,'errcoeffpolydeg'+str(nmax)+'int'+str(inttime)+'ms'),stdparameters)


#####################################################################################################

def average_frame(frame,pixelmean,extremedelete=False):

	# Does the average over pixelmean x pixelmean pixels of frame
	#
	# If the flag extremedelete is True for every mean the maximuna and minimum values are deleted
	#
	if pixelmean==1:
		return frame
	shapeorig=np.shape(frame)
	shapeaver=(np.divide(shapeorig,pixelmean)).astype(int)
	frameaver=np.zeros(shapeaver)
	for i in range(shapeaver[0]):
		if ((i+1)*pixelmean)>shapeorig[0]:
			indexi=shapeorig[0]-1
		else:
			indexi=(i+1)*pixelmean
		for j in range(shapeaver[1]):
			if ((j+1)*pixelmean)>shapeorig[1]:
				indexj=shapeorig[1]-1
			else:
				indexj=(j+1)*pixelmean
			flat=np.ravel(frame[i*pixelmean:int(indexi),j*pixelmean:int(indexj)])
			if extremedelete:
				if len(flat)>3:
					flat=np.delete(flat,np.argmax(flat))
					flat=np.delete(flat,np.argmin(flat))
			frameaver[i,j]=np.mean(flat)
			# print(i*pixelmean,indexi,j*pixelmean,indexj,flat)

	return frameaver

#####################################################################################################

def average_multiple_frames(frames,pixelmean,timemean=1,extremedelete=False):

	# Does the average over pixelmean x pixelmean pixels of multiple frames
	#
	# If the flag extremedelete is True for every mean the maximuna and minimum values are deleted
	#

	shapeorig=np.shape(frames)
	nframes=shapeorig[1]
	# framesaver=[]
	# framesaver.append([None] * nframes)
	# framesaver=np.array(framesaver)
	framesaver = [None]*nframes
	for i in range(nframes):
		framesaver[i]=average_frame(frames[0,i],pixelmean,extremedelete)

	if timemean>1:
		timemean = int(timemean)
		reduced_frames = int(nframes/timemean)
		# temp = []
		# temp.append([None] * reduced_frames)
		# temp = np.array(temp)
		temp=[None] * reduced_frames
		for index in range(reduced_frames):
			temp[index] = np.mean(framesaver[index*timemean:(index+1)*timemean],axis=(0))
		return np.array([temp])

	return np.array([framesaver])


#####################################################################################################

def average_multiple_frames2(frames,pixelmean,timemean=1):

	# Created to use the faster function skimage.transform.resize
	# Does the average over pixelmean x pixelmean pixels of multiple frames
	#
	import numpy as np
	from skimage.transform import resize

	shapeorig=np.shape(frames)
	nframes=shapeorig[1]
	shapeaver=(np.divide(shapeorig[-2:],pixelmean)).astype(int)
	# framesaver=[]
	# framesaver.append([None] * nframes)
	# framesaver=np.array(framesaver)
	framesaver = [None]*nframes
	for i in range(nframes):
		# framesaver[i]=average_frame(frames[0,i],pixelmean,extremedelete)
		framesaver[i] =resize(frames[0,i], shapeaver,order=1)
	framesaver = np.array(framesaver)

	if timemean>1:
		timemean = int(timemean)
		reduced_frames = int(nframes/timemean)
		# temp = []
		# temp.append([None] * reduced_frames)
		# temp = np.array(temp)
		temp=[None] * reduced_frames
		for index in range(reduced_frames):
			temp[index] = np.mean(framesaver[index*timemean:(index+1)*timemean],axis=(0))
		return np.array([temp])

	return np.array([framesaver])


#####################################################################################################

def flatten(array):

	# 11/06/2018
	# This function flatten any array of one level.
	# If it is already in one level it return is the way it is
	#
	# array=np.array(array)
	length=len(array)

	done=0
	for item in array:
		if np.shape(item)!=():
			done+=1
	if done==0:
		return array

	temp=[]
	lengthinside=np.zeros(length)
	for i in range(length):
		temp2=np.array(array[i])
		lengthinside=np.shape(temp2)
		if lengthinside==():
			temp.append(array[i])
		else:
			for j in range(lengthinside[0]):
				temp.append(temp2[j])
	try:
		temp=np.array(temp)
	except:
		print('partial flattening obtained')
	return temp

#####################################################################################################

def flatten_full(array):
	# this function flatten an array fully

	while np.shape(array[0])!=():
		array=flatten(array)

	return array

#####################################################################################################

def ddx(array,dx,axis,otheraxis=(),howcropotheraxis=0):

	# this function makes the central difference derivative on the axis AXIS. it reduces the size of the array in that direction by two pixels
	# at the same time it can reduce the size ao the array in other directions too of any number of pixels.

	if axis==0:
		temp=np.divide(array[2:]-array[:-2],2*dx)
	elif axis==1:
		temp=np.divide(array[:,2:]-array[:,:-2],2*dx)
	elif axis==2:
		temp=np.divide(array[:,:,2:]-array[:,:,:-2],2*dx)
	elif axis==3:
		temp=np.divide(array[:,:,:,2:]-array[:,:,:,:-2],2*dx)
	elif axis==4:
		temp=np.divide(array[:,:,:,:,2:]-array[:,:,:,:,:-2],2*dx)
	elif axis==5:
		temp=np.divide(array[:,:,:,:,:,2:]-array[:,:,:,:,:,:-2],2*dx)
	elif axis==6:
		temp=np.divide(array[:,:,:,:,:,:,2:]-array[:,:,:,:,:,:,:-2],2*dx)


	if howcropotheraxis==0:
		return temp

	if otheraxis==():
		print('if you specify the number of pixels to crop tou must specify too the axis where to do that')
		exit()

	numaxis=len(np.shape(array))
	for i in range(len(otheraxis)):
		if otheraxis[i]<0:
			otheraxis[i]=numaxis-otheraxis[i]

	if howcropotheraxis/2-howcropotheraxis//2!=0:
		print('the amount of pixels you want to crop the array must be even, to crop of half on one side and half on the other')
		exit()
	htc=howcropotheraxis//2
	if 0 in otheraxis:
		temp=temp[htc:-htc]
	if 1 in otheraxis:
		temp=temp[:,htc:-htc]
	if 2 in otheraxis:
		temp=temp[:,:,htc:-htc]
	if 3 in otheraxis:
		temp=temp[:,:,:,htc:-htc]
	if 4 in otheraxis:
		temp=temp[:,:,:,:,htc:-htc]
	if 5 in otheraxis:
		temp=temp[:,:,:,:,:,htc:-htc]
	if 6 in otheraxis:
		temp=temp[:,:,:,:,:,:,htc:-htc]

	return temp

#####################################################################################################

def d2dx2(array,dx,axis,otheraxis=(),howcropotheraxis=0):

	# this function makes the tentral difference second derivative on the axis AXIS. it reduces the size of the array in that direction by two pixels
	# at the same time it can reduce the size ao the array in other directions too of any number of pixels.

	if axis==0:
		temp=np.divide(array[2:]-np.multiply(2,array[1:-1])+array[:-2],dx**2)
	elif axis==1:
		temp=np.divide(array[:,2:]-np.multiply(2,array[:,1:-1])+array[:,:-2],dx**2)
	elif axis==2:
		temp=np.divide(array[:,:,2:]-np.multiply(2,array[:,:,1:-1])+array[:,:,:-2],dx**2)
	elif axis==3:
		temp=np.divide(array[:,:,:,2:]-np.multiply(2,array[:,:,:,1:-1])+array[:,:,:,:-2],dx**2)
	elif axis==4:
		temp=np.divide(array[:,:,:,:,2:]-np.multiply(2,array[:,:,:,:,1:-1])+array[:,:,:,:,:-2],dx**2)
	elif axis==5:
		temp=np.divide(array[:,:,:,:,:,2:]-np.multiply(2,array[:,:,:,:,:,1:-1])+array[:,:,:,:,:,:-2],dx**2)
	elif axis==6:
		temp=np.divide(array[:,:,:,:,:,:,2:]-np.multiply(2,array[:,:,:,:,:,:,1:-1])+array[:,:,:,:,:,:,:-2],dx**2)


	if howcropotheraxis==0:
		return temp
	if howcropotheraxis/2-howcropotheraxis//2!=0:
		print('the amount of pixels you want to crop the array must be even, to crop of half on one side and half on the other')
		exit()

	numaxis=len(np.shape(array))
	for i in range(len(otheraxis)):
		if otheraxis[i]<0:
			otheraxis[i]=numaxis-otheraxis[i]

	if howcropotheraxis/2-howcropotheraxis//2!=0:
		print('the amount of pixels you want to crop the array must be even, to crop of half on one side and half on the other')
		exit()
	htc=howcropotheraxis//2
	if 0 in otheraxis:
		temp=temp[htc:-htc]
	if 1 in otheraxis:
		temp=temp[:,htc:-htc]
	if 2 in otheraxis:
		temp=temp[:,:,htc:-htc]
	if 3 in otheraxis:
		temp=temp[:,:,:,htc:-htc]
	if 4 in otheraxis:
		temp=temp[:,:,:,:,htc:-htc]
	if 5 in otheraxis:
		temp=temp[:,:,:,:,:,htc:-htc]
	if 6 in otheraxis:
		temp=temp[:,:,:,:,:,:,htc:-htc]

	return temp


#####################################################################################################

def save_timestamp(extpath):

	# 09/08/2018 This function looks at the CSV files and saves the timestamp of the firs and last one in a _timestamp.npy file

	# path=os.getcwd()

	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)

	filefits=[]
	temp=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='csv':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
		# elif filenames[index][-3:]=='fts':
		# 	filefits.append(filenames[index])
	filenames=temp
	filenames=sorted(filenames, key=str.lower)

	timestamp=[]
	# file=0
	for filename in [filenames[0],filenames[-1]]:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			for row in reader:
				if not not row:
				# else:
					if row[0][0:4]=='Time':
						time=row[0][7:]
						ampm=int(time[0:3])
						hh=int(time[4:6])
						mm=int(time[7:9])
						ss=float(time[10:])
						timess=ss+60*(mm+60*(hh+ampm*12))
						timestamp.append(timess)



	timestamp.append(np.mean(timestamp))
	timestamp=np.array(timestamp)
	np.save(os.path.join(extpath,lastpath+'_timestamp'),timestamp)


####################################################################################################

def search_background_timestamp(extpath,ref_directory):

	# 09/08/2018 This function lools at the timestamp in "extpath" and compare it with all the timestamps in the directories indicated by "ref_directory"
	# then I pick the two that are closer and interpolate in between the two to get the proper background


	ref_directories=[]
	for (dirpath, dirnames, filenames) in os.walk(ref_directory):
		ref_directories.extend(dirnames)


	ref_directories=order_filenames(ref_directories)

	type="_timestamp.npy"
	time=[]
	for directory in ref_directories:
		filename=all_file_names(os.path.join(ref_directory,directory),type)[0]
		timestamp=np.load(os.path.join(ref_directory,directory,filename))
		time.append(timestamp[2])

	time=np.array(time)
	if np.sum(time!=np.sort(time))>0:
		print('Something is wrong, the order of the files does not follows a cronological order, as it should')
		exit()


	filename=all_file_names(os.path.join(extpath),type)[0]
	specific_time=np.load(os.path.join(extpath,filename))[2]
	specific_filename=filename[:-14]

	index=np.searchsorted(time,specific_time)-1

	print('found '+str(specific_time)+' between '+str(time[index])+' and '+str(time[index+1]))

	type="_stat.npy"
	filename=all_file_names(os.path.join(ref_directory,ref_directories[index]),type)[0]
	pre_ref=np.load(os.path.join(ref_directory,ref_directories[index],filename))[0]

	filename=all_file_names(os.path.join(ref_directory,ref_directories[index+1]),type)[0]
	post_ref=np.load(os.path.join(ref_directory,ref_directories[index+1],filename))[0]

	dt=time[index+1]-time[index]
	reference=np.multiply(pre_ref,(time[index+1]-specific_time)/dt)+np.multiply(post_ref,(specific_time-time[index])/dt)

	np.save(os.path.join(extpath,specific_filename+'_reference'),reference)


####################################################################################################

def find_nearest_index(array,value):

	# 14/08/2018 This function returns the index of the closer value to "value" inside an array
	# 08/01/2021 This function is completely reduntant, but I have it in so many places I keep it

	if False:
		array_shape=np.shape(array)
		index = np.abs(np.add(array,-value)).argmin()
		residual_index=index
		cycle=1
		done=0
		position_min=np.zeros(len(array_shape),dtype=int)
		while done!=1:
			length=array_shape[-cycle]
			if residual_index<length:
				position_min[-cycle]=residual_index
				done=1
			else:
				position_min[-cycle]=round(((residual_index/length) %1) *length +0.000000000000001)
				residual_index=residual_index//length
				cycle+=1
		return position_min[0]
	else:
		return np.abs(np.array(array)-value).argmin()


###################################################################################################

def track_change_from_baseframe(data,poscentred,basecounts):

	# Created 27/12/2018 to clean up some legacy data. Now this function is not necessay anymore
	# This function evaluate the average counts on a limited number of locations [poscentered] and in a small window around them [a].
	# Then this is compared with same locations and window for a reference frame, and the whole record will need be scaled of the difference [base_counts_correction]


	data_mean_difference = []
	data_mean_difference_std = []

	for pos in poscentred:
		for a in [5, 10]:
			temp1 = np.mean(data[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2, -3))
			temp1std = np.std(np.mean(data[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2)))
			temp2 = np.mean(basecounts[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2, -3))
			temp2std = np.std(
				np.mean(basecounts[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2)))
			data_mean_difference.append(temp1 - temp2)
			data_mean_difference_std.append(temp1std + temp2std)
	data_mean_difference = np.array(data_mean_difference)
	data_mean_difference_std = np.array(data_mean_difference_std)
	guess = [1]
	base_counts_correction, temp2 = curve_fit(costant,
											  np.linspace(1, len(data_mean_difference), len(data_mean_difference)),
											  data_mean_difference, sigma=data_mean_difference_std, p0=guess,
											  maxfev=100000000)
	print('background correction = ' + str(int(base_counts_correction * 1000) / 1000))

	return base_counts_correction







###################################################################################################

def clear_oscillation_central(data,framerate,oscillation_search_window_begin='auto',oscillation_search_window_end='auto',plot_conparison=False):

	# Created 01/11/2018
	# This function take the raw counts. analyse the fast fourier transform in a selectable interval IN THE CENTER OF THE FRAME.
	# Then search for the peak between 20Hz and 34Hz  and substract the oscillation found to the counts
	print('Use clear_oscillation_central2 instead of clear_oscillation_central')
	exit()


	print('shape of data array is '+str(np.shape(data))+', it should be (x,frames,v pixel,h pixel)')

	data=data[0]
	if oscillation_search_window_begin=='auto':	# in seconds
		force_start=0
	elif (oscillation_search_window_begin<0 or oscillation_search_window_begin>len(data)*framerate):
		print('The initial limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print('0s will be used instead')
		force_start=0
	else:
		force_start=int(oscillation_search_window_begin/framerate)

	if oscillation_search_window_end=='auto':	# in seconds
		force_end=len(data)
	elif (oscillation_search_window_end<0 or oscillation_search_window_end>(len(data)*framerate) or oscillation_search_window_end<=(force_start*framerate)):
		print('The final limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print(str(int(len(data)//(2*framerate)))+'s will be used instead')
		force_end=int(len(data)//(2*framerate))
	else:
		force_end=int(oscillation_search_window_end*framerate)

	window = 10
	datasection = data

	if plot_conparison==True:
		poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]

		spectra_orig = np.fft.fft(data, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		phase = np.angle(spectra_orig)
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

		color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
		for i in range(len(poscentred)):
			pos = poscentred[i]
			y = np.mean(magnitude[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=(-1, -2))
			# y=magnitude[:,pos[0],pos[1]]
			y = np.array([y for _, y in sorted(zip(freq, y))])
			x = np.sort(freq)
			plt.plot(x, y, color[i], label='original data at the point ' + str(pos))
		# plt.title()

		plt.figure(1)
		plt.title('Amplitued from fast Fourier transform for different groups of ' + str(window * 2) + 'x' + str(
			window * 2) + ' pixels, framerate ' + str(framerate)+'Hz' )
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.legend()
	# plt.show()


	sections = 31  # number found with practice, no specific mathematical reasons
	max_time = 5  # seconds of record that I can use to filter the signal. I assume to start from zero
	poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]
	record_magnitude = []
	record_phase = []
	record_freq = []
	peak_freq_record = []
	peak_value_record = []
	section_frames_record = []

	# I restrict the window over which I search for the oscillation
	datarestricted=data[force_start:force_end]

	if oscillation_search_window_end == 'auto':
		if (len(datarestricted) / framerate) <= 1:
			max_start = int(sections // 2)
		else:
			max_start = min(int(1 + sections / 2), int(
				max_time * framerate / (len(datarestricted) / sections)))  # I can use only a part of the record to filter the signal
	else:
		max_start = int(oscillation_search_window_end * framerate / (len(datarestricted) / sections))

	if (len(datarestricted) / framerate) <= 1:
		min_start = max(1, int(0.2 * framerate / (len(datarestricted) / sections)))
	else:
		extra = 0
		while ((max_start - int(max_start / (5 / 2.5)) + extra) < 7):	# 7 is just a try. this is in a way to have enough fitting to compare
			extra+=1
		min_start = max(1,int(max_start / (5 / 2.5)) - extra)
		# min_start = max(1, int(max_start / (5 / 2.5)) )  # with too little intervals it can interpret noise for signal

	for i in range(min_start, max_start):
		section_frames = (i) * (len(datarestricted) // sections)
		section_frames_record.append(section_frames)
		datasection = datarestricted[0:section_frames, poscentre[0] - window:poscentre[0] + window, poscentre[1] - window:poscentre[1] + window]
		spectra = np.fft.fft(datasection, axis=0)
		magnitude = 2 * np.abs(spectra) / len(spectra)
		record_magnitude.append(magnitude[0:len(magnitude) // 2])
		phase = np.angle(spectra)
		record_phase.append(phase[0:len(magnitude) // 2])
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		record_freq.append(freq[0:len(magnitude) // 2])
		magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
		y = np.array(
			[magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
		x = np.sort(freq)
		if plot_conparison == True:
			plt.figure(2)
			plt.plot(x, y, label='size of the analysed window ' + str(section_frames / framerate))
		index_20 = int(find_nearest_index(x, 20))  # I restric the window over which I do the peak search
		index_34 = int(find_nearest_index(x, 34))
		index_7 = int(find_nearest_index(x, 7))
		index_n7 = int(find_nearest_index(x, -7))
		index_n20 = int(find_nearest_index(x, -20))
		index_n34 = int(find_nearest_index(x, -34))
		index_0 = int(find_nearest_index(x, 0))
		noise = np.mean(np.array(
			y[3:index_n34].tolist() + y[index_n20:index_n7].tolist() + y[index_7:index_20].tolist() + y[
																									  index_34:-3].tolist()),
						axis=(-1))
		temp = peakutils.indexes(y[index_20:index_34], thres=noise + np.abs(magnitude.min()),
								 min_dist=(index_34 - index_20) // 2)
		if len(temp) == 1:
			peak_index = index_20 + int(temp)
			peak_freq_record.append(x[peak_index])
			peak_value = float(y[peak_index])
			peak_value_record.append(peak_value)
	record_magnitude = np.array(record_magnitude)
	record_phase = np.array(record_phase)
	record_freq = np.array(record_freq)
	peak_freq_record = np.array(peak_freq_record)
	peak_value_record = np.array(peak_value_record)
	section_frames_record = np.array(section_frames_record)
	if plot_conparison==True:
		plt.figure(2)
		plt.title('Amplitued from fast Fourier transform averaged in a wondow of ' + str(window) + 'pixels around ' + str(
			poscentre) + ', framerate ' + str(framerate) + 'Hz')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.legend()


	# I find the highest peak and that will be the one I use
	index = int(find_nearest_index(peak_value_record, max(peak_value_record)+1))
	section_frames = section_frames_record[index]
	datasection = datarestricted[0:section_frames]
	spectra = np.fft.fft(datasection, axis=0)
	# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
	magnitude = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	freq_to_erase = peak_freq_record[index]
	freq_to_erase_index = int(find_nearest_index(freq, freq_to_erase))
	framenumber = np.linspace(0, len(data) - 1, len(data)) - force_start
	data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))

	if plot_conparison==True:
		plt.figure(1)
		datasection2 = data2
		spectra = np.fft.fft(datasection2, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude2 = 2 * np.abs(spectra) / len(spectra)
		phase2 = np.angle(spectra)
		freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
		for i in range(len(poscentred)):
			pos = poscentred[i]
			y = np.mean(magnitude2[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=(-1, -2))
			# y=magnitude[:,pos[0],pos[1]]
			y = np.array([y for _, y in sorted(zip(freq, y))])
			x = np.sort(freq)
			plt.plot(x, y, color[i] + '--',
					 label='data at the point ' + str(pos) + ', ' + str(freq_to_erase) + 'Hz oscillation substracted')
		# plt.title()


		plt.grid()
		plt.semilogy()
		plt.legend()
		plt.pause(0.0001)




	print('stats of the oscillation removal')
	print('with window of size '+str(section_frames_record[index]/framerate)+'s of '+str(len(data)/framerate)+'s of record')
	print('found oscillation of frequency '+str(freq_to_erase)+'Hz')

	return np.array([data2])


###################################################################################################

def clear_oscillation_central2(data,framerate,oscillation_search_window_begin='auto',oscillation_search_window_end='auto',min_frequency_to_erase=20,max_frequency_to_erase=34,plot_conparison=False,which_plot=[1,2,3],ROI='auto',window=2,force_poscentre='auto',output_noise=False,multiple_frequencies_cleaned=1):

	# Created 15/02/2019
	# This function take the raw counts. analyse the fast fourier transform in a selectable interval IN THE CENTER OF THE FRAME.
	# Then search for the peak between 20Hz and 34Hz  and substract the oscillation found to the counts
	# The difference from clear_oscillation_central is that instead of splitting the interval of interest in smaller chunks and analyse them I analyse the full window available and shift the first and last point.
	# This aproach seems much more efficient

	print('shape of data array is '+str(np.shape(data))+', it should be (x,frames,v pixel,h pixel)')
	# figure_index = plt.gcf().number

	data=data[0]
	if oscillation_search_window_begin=='auto':
		force_start=0
	elif (oscillation_search_window_begin<0 or oscillation_search_window_begin>len(data)*framerate):
		print('The initial limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print('0s will be used instead')
		force_start=0
	else:
		force_start=int(oscillation_search_window_begin*framerate)

	if oscillation_search_window_end=='auto':
		force_end=len(data)
	elif (oscillation_search_window_end<0 or oscillation_search_window_end>(len(data)/framerate) or oscillation_search_window_end<=force_start/framerate):
		print('The final limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print(str(int(len(data)//(2*framerate)))+'s will be used instead')
		force_end=int(len(data)//(2*framerate))
	else:
		force_end=int(oscillation_search_window_end*framerate)


	central_freq_for_search = (max_frequency_to_erase-min_frequency_to_erase)/2+min_frequency_to_erase
	if (framerate<2*central_freq_for_search):
		print('There is a problem. The framerate is too low to try to extract the oscillation')
		print('The minimum framrate for doing it is 2*oscillation frequency to detect. Therefore '+str(np.around(2*central_freq_for_search,decimals=1))+'Hz, in this case.')
		print('See http://www.skillbank.co.uk/SignalConversion/rate.htm')
		exit()

	#window = 2	# Previously found that as long as the fft is averaged over at least 4 pixels the peak shape and location does not change
	datasection = data

	if plot_conparison==True:
		# plt.figure()
		# plt.pause(0.01)
		plt.figure()
		figure_index = plt.gcf().number
		if 1 in which_plot:
			data_shape = np.shape(data)
			poscentred = [[int(data_shape[1]*1/5), int(data_shape[2]*1/5)], [int(data_shape[1]*1/2), int(data_shape[2]*1/5)], [int(data_shape[1]*4/5), int(data_shape[2]*1/5)], [int(data_shape[1]*4/5), int(data_shape[2]*1/2)], [int(data_shape[1]*4/5), int(data_shape[2]*4/5)], [int(data_shape[1]*1/2), int(data_shape[2]*1/2)]]

			# spectra_orig = np.fft.fft(data, axis=0)
			# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
			# magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
			# phase = np.angle(spectra_orig)
			# freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

			color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
			for i in range(len(poscentred)):
				pos = poscentred[i]
				spectra_orig = np.fft.fft(np.mean(data[:, pos[0] - window:pos[0] + window+1, pos[1] - window:pos[1] + window+1],axis=(-1,-2)), axis=0)
				magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
				freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
				# y = np.mean(magnitude, axis=(-1, -2))
				y = magnitude
				# y=magnitude[:,pos[0],pos[1]]
				y = np.array([y for _, y in sorted(zip(freq, y))])
				x = np.sort(freq)
				plt.plot(x, y*100, color[i], label='original data at the point ' + str(pos) + ' x100')
			# plt.title()


			plt.title('Amplitued from fast Fourier transform in the whole time interval\nfor different groups of ' + str(window * 2+1) + 'x' + str(
				window * 2+1) + ' pixels, framerate %.3gHz' %(framerate) )
			plt.xlabel('Frequency [Hz]')
			plt.ylabel('Amplitude [au]')
			plt.grid()
			plt.semilogy()
			plt.legend(loc='best',fontsize='xx-small')
		else:
			plt.close(figure_index)
	# plt.show()



	frames_for_oscillation = framerate//central_freq_for_search

	number_of_waves = 3
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	if fft_window_move<10:
		number_of_waves=10//frames_for_oscillation	# I want to scan for at least 10 frame shifts
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	step = 1
	while int(fft_window_move/step)>80:
		step+=1		#if framerate is too high i will skip some of the shifts to limit the number of Fourier transforms to 100


	# I restrict the window over which I search for the oscillation
	datarestricted = data[force_start:force_end]#
	len_data_restricted = len(datarestricted)
	if force_poscentre == 'auto':
		poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]
	else:
		poscentre = force_poscentre

	if (oscillation_search_window_begin=='auto' and oscillation_search_window_end=='auto'):
		while fft_window_move>(len_data_restricted/5):
			fft_window_move-=1		# I want that the majority of the data I analyse remains the same

	if oscillation_search_window_end == 'auto':
		if (len(datarestricted) / framerate) <= 1:
			# max_start = int(sections // 2)
			max_start = int(len(datarestricted) // 2)
		else:
			max_start = int(5*framerate)  # I use 5 seconds of record
	else:
		max_start = len(datarestricted)	# this is actually ineffective, as datarestricted is already limited to force_start:force_end

	if oscillation_search_window_begin == 'auto':
		min_start = 0
	else:
		# min_start = force_start	# alreay enforced through force_start
		min_start = 0

	section_frames = max_start - min_start-fft_window_move

	if ROI=='auto':
		# datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(1,2))
		datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datarestricted2 = np.mean(datarestricted[:, select],axis=(-1))

	record_magnitude = []
	record_phase = []
	record_freq = []
	peak_freq_record = []
	peak_value_record = []
	peak_index_record = []
	shift_record = []

	for i in range(int(fft_window_move/step)):
		shift=i*step
		datasection = datarestricted2[min_start:max_start-fft_window_move+shift]
		spectra = np.fft.fft(datasection, axis=0)
		magnitude = 2 * np.abs(spectra) / len(spectra)
		record_magnitude.append(magnitude[0:len(magnitude) // 2])
		phase = np.angle(spectra)
		record_phase.append(phase[0:len(magnitude) // 2])
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		record_freq.append(freq[0:len(magnitude) // 2])
		# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
		magnitude_space_averaged = cp.deepcopy(magnitude)
		y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
		x = np.sort(freq)

		index_min_freq = int(find_nearest_index(x, min_frequency_to_erase))  # I restric the window over which I do the peak search
		index_max_freq = int(find_nearest_index(x, max_frequency_to_erase))
		index_7 = int(find_nearest_index(x, 7))
		index_n7 = int(find_nearest_index(x, -7))
		index_min_freq_n = int(find_nearest_index(x, -min_frequency_to_erase))
		index_max_freq_n = int(find_nearest_index(x, -max_frequency_to_erase))
		index_0 = int(find_nearest_index(x, 0))
		# noise = np.mean(np.array(
		# 	y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[
		# 																									index_max_freq:-3].tolist()),
		# 				axis=(-1))
		# temp = peakutils.indexes(y[index_min_freq:index_max_freq], thres=noise + np.abs(magnitude.min()),
		# 						 min_dist=(index_max_freq - index_min_freq) // 2)
		if len(y[index_min_freq:index_max_freq])==0:
			continue
		if plot_conparison == True and (2 in which_plot):
			plt.figure(figure_index+1)
			plt.plot(x, y, label='Applied shift of ' + str(shift))
		temp = int(find_nearest_index(y[index_min_freq:index_max_freq],(y[index_min_freq:index_max_freq]).max()))
		# if len(temp) == 1:
		peak_index = index_min_freq + int(temp)
		peak_freq_record.append(x[peak_index])
		peak_value = float(y[peak_index])
		peak_value_record.append(peak_value)
		peak_index_record.append(peak_index)
		shift_record.append(shift)
	record_magnitude = np.array(record_magnitude)
	record_phase = np.array(record_phase)
	record_freq = np.array(record_freq)
	peak_freq_record = np.array(peak_freq_record)
	peak_value_record = np.array(peak_value_record)
	peak_index_record = np.array(peak_index_record)
	shift_record = np.array(shift_record)
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		# # plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged in a window of ' %(force_start/framerate,(force_start+max_start)/framerate)+ str(window*2+1) + ' pixels around ' + str(
		# # 	poscentre) + ', framerate %.3gHz' %(framerate))
		# if ROI=='auto':
		# 	plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged in a window of ' %(force_start/framerate,(force_start+max_start)/framerate)+ str([window * 2+1,window * 2+1]) + ' pixels around ' + str(poscentre) + ', framerate %.3gHz' %(framerate)+'\n'+str(multiple_frequencies_cleaned)+' frequencies around ')
		# else:
		# 	plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged ouside the ROI ' %(force_start/framerate,(force_start+max_start)/framerate)+ str(ROI) + ' pixels around ' + str(ROI) + ', framerate %.3gHz' %(framerate))
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.xlim(left=min_frequency_to_erase*0.8,right=max_frequency_to_erase*1.2)
		plt.ylim(bottom=np.median(np.sort(y)[:int(len(y)/4)])*1e-1)
		plt.legend(loc='best',fontsize='xx-small')


	if False:	# this method fails if the oscillation is actually composed of 2 separate oscillations very close to each other. it tries to remove a frequency in between that does not cause any problem. better to just not do it
		try:	# considering that the variation of the peak value of FFT is like a wave this methid seems more fair.
			shift_peaks = shift_record[1:-1][np.logical_and((peak_value_record[1:-1]-peak_value_record[:-2])>=0,(peak_value_record[1:-1]-peak_value_record[2:])>=0)]
			shift_through = shift_record[1:-1][np.logical_and((peak_value_record[1:-1]-peak_value_record[:-2])<=0,(peak_value_record[1:-1]-peak_value_record[2:])<=0)]
			while np.diff(shift_peaks).min()<=2:	# to avoid detection of adiacent "fake" peak or through
				target = np.diff(shift_peaks).argmin()
				shift_through = shift_through[np.logical_not(np.logical_and(shift_through>shift_peaks[target],shift_through<shift_peaks[target+1]))]
				shift_peaks = np.array(shift_peaks[:target].tolist() + [np.mean(shift_peaks[target:target+2])] + shift_peaks[target+2:].tolist())
			while np.diff(shift_through).min()<=2:	# to avoid detection of adiacent "fake" peak or through
				target = np.diff(shift_through).argmin()
				shift_peaks = shift_peaks[np.logical_not(np.logical_and(shift_peaks>shift_through[target],shift_peaks<shift_through[target+1]))]
				shift_through = np.array(shift_through[:target].tolist() + [np.mean(shift_through[target:target+2])] + shift_through[target+2:].tolist())
			shift_roots = np.sort(np.append(shift_peaks,shift_through,0))
			fit = np.polyfit(np.arange(len(shift_roots)),shift_roots,1)
			shift = int(round(np.polyval(fit,[0,1])[np.abs(np.polyval(fit,[0,1])-shift_peaks[0]).argmin()]))
			if peak_value_record[0]>peak_value_record[np.abs(shift_record-shift).argmin()]:
				shift = shift_record[0]
			index = shift//step
			gaussian = lambda x,A,sig,x0,q : A*np.exp(-0.5*(((x-x0)/sig)**2)) + q
			bds = [[0,0,peak_freq_record.min(),-np.inf],[np.inf,np.inf,peak_freq_record.max(),peak_value_record.min()]]
			guess = [peak_value_record.max()-peak_value_record.min(),0.3,peak_freq_record[peak_value_record.argmax()],peak_value_record.min()]
			fit = curve_fit(gaussian, peak_freq_record,peak_value_record, p0=guess, bounds = bds, maxfev=100000000)
			freq_to_erase = fit[0][2]
			if plot_conparison==True:
				if 2 in which_plot:
					plt.figure(figure_index+1)
					plt.plot(np.linspace(peak_freq_record.min(),peak_freq_record.max()),gaussian(np.linspace(peak_freq_record.min(),peak_freq_record.max()),*fit[0]),':k',label='fit')
				if 3 in which_plot:
					plt.figure(figure_index+2)
					plt.plot(shift_record,peak_value_record)
					plt.plot([shift]*2,[peak_value_record.min(),peak_value_record.max()],'--k')
					# plt.title('Amplitude from fast Fourier transform based on window shift\naveraged in a wondow of ' + str(window+1) + ' pixels around ' + str(
					# 	poscentre) + ', framerate %.3gHz' %(framerate))
					if ROI=='auto':
						plt.title('Amplitude from fast Fourier transform based on window shift\naveraged in a window of ' + str([window * 2+1,window * 2+1]) + ' pixels around ' + str(poscentre) + ', framerate %.3gHz' %(framerate))
					else:
						plt.title('Amplitude from fast Fourier transform based on window shift\naveraged ouside the ROI ' + str(ROI) + ' pixels around ' + str(ROI) + ', framerate %.3gHz' %(framerate))
					plt.xlabel('Shift [au]')
					plt.ylabel('Amplitude [au]')
					plt.grid()
					plt.grid()
					plt.semilogy()
			# fit = np.polyfit(peak_freq_record,peak_value_record,2)
			# freq_to_erase = -fit[1]/(fit[0]*2)
		except:	# I find the highest peak and that will be the one I use
			print('search of the best interval shift via the linear peak method failed')
			index = int(find_nearest_index(peak_value_record, max(peak_value_record)+1))
			shift = index * step
			freq_to_erase = peak_freq_record[index]
	else:
		index = int(find_nearest_index(peak_value_record, max(peak_value_record)+1))
		shift = index * step
		freq_to_erase = peak_freq_record[index]
	datasection = datarestricted[min_start:max_start-fft_window_move+shift]
	spectra = np.fft.fft(datasection, axis=0)
	# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
	magnitude = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	freq_to_erase_index = int(find_nearest_index(freq, freq_to_erase))
	freq_to_erase_index_multiple = np.arange(-(multiple_frequencies_cleaned-1)//2,(multiple_frequencies_cleaned-1)//2+1) + freq_to_erase_index

	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		# plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged in a window of ' %(force_start/framerate,(force_start+max_start)/framerate)+ str(window*2+1) + ' pixels around ' + str(
		# 	poscentre) + ', framerate %.3gHz' %(framerate))
		if ROI=='auto':
			plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged in a window of ' %(force_start/framerate,(force_start+max_start)/framerate)+ str([window * 2+1,window * 2+1]) + ' pixels around ' + str(poscentre) + ', framerate %.3gHz' %(framerate)+'\n'+str(multiple_frequencies_cleaned)+' frequencies around %.5gHz erased' %(freq[freq_to_erase_index]))
		else:
			plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged ouside the ROI ' %(force_start/framerate,(force_start+max_start)/framerate)+ str(ROI) + ' pixels around ' + str(ROI) + ', framerate %.3gHz' %(framerate)+'\n'+str(multiple_frequencies_cleaned)+' frequencies around %.5gHz erased' %(freq[freq_to_erase_index]))

	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		# plt.plot([freq_to_erase]*2,[peak_value_record.min(),peak_value_record.max()],':k')
		# plt.plot([freq[freq_to_erase_index]]*2,[peak_value_record.min(),peak_value_record.max()],'--k')
		plt.axvline(x=freq_to_erase,color='k',linestyle=':')
		plt.axvline(x=freq[freq_to_erase_index],color='k',linestyle='--')
		if len(freq_to_erase_index_multiple)>1:
			for i_freq in freq_to_erase_index_multiple:
				if i_freq!=freq_to_erase_index:
					plt.axvline(x=freq[i_freq],color='r',linestyle='--')
		plt.ylim(top=peak_value_record.max()*2)
	framenumber = np.linspace(0, len(data) - 1, len(data)) -force_start- min_start
	data2 = cp.deepcopy(data)
	for i_freq in freq_to_erase_index_multiple:
		# print(i_freq)
		data2 -= np.multiply(magnitude[i_freq], np.cos(np.repeat(np.expand_dims(phase[i_freq], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq[i_freq] * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))
	# data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))


	# added to visualize the goodness of the result
	if ROI=='auto':
		datasection = np.mean(data2[force_start:force_end][min_start:max_start-fft_window_move+shift][:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data2[force_start:force_end][min_start:max_start-fft_window_move+shift][:, select],axis=(-1))
	# datasection = data2[force_start:force_end][min_start:max_start-fft_window_move+shift, poscentre[0] -1 :poscentre[0] + window+1, poscentre[1] - 1:poscentre[1] + window+1]
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.plot(x, y,'--k',label='subtracted')
		# plt.grid()


	# section only for stats
	# datasection = data[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	if ROI=='auto':
		datasection = np.mean(data[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data[:, select],axis=(-1))
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	index_min_freq = int(find_nearest_index(x, min_frequency_to_erase))  # I restric the window over which I do the peak search
	index_max_freq = int(find_nearest_index(x, max_frequency_to_erase))
	index_7 = int(find_nearest_index(x, 7))
	index_n7 = int(find_nearest_index(x, -7))
	index_min_freq_n = int(find_nearest_index(x, -min_frequency_to_erase))
	index_max_freq_n = int(find_nearest_index(x, -max_frequency_to_erase))
	index_0 = int(find_nearest_index(x, 0))
	noise = (np.array(y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[index_max_freq:-3].tolist()))
	# temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	# peak_index = index_min_freq + int(temp)
	peak_index = index_min_freq + y[index_min_freq:index_max_freq].argmax()
	peak_value_pre_filter = float(y[peak_index])

	# datasection = data2[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	if ROI=='auto':
		datasection = np.mean(data2[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data2[:, select],axis=(-1))
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	# temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	# peak_index = index_min_freq + int(temp)
	peak_index = index_min_freq + y[index_min_freq:index_max_freq].argmax()
	peak_value_post_filter = float(y[peak_index])
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.axhline(y=np.max(noise),linestyle='--',color='k',label='max noise')
		plt.axhline(y=np.median(noise),linestyle='--',color='b',label='median noise')
		plt.axhline(y=peak_value_pre_filter,linestyle='--',color='r',label='peak pre')
		plt.axhline(y=peak_value_post_filter,linestyle='--',color='g',label='peak post')
		plt.legend(loc='best',fontsize='xx-small')

	if plot_conparison==True:
		if 1 in which_plot:
			plt.figure(figure_index)
			datasection2 = data2
			# spectra = np.fft.fft(datasection2, axis=0)
			# # magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
			# magnitude2 = 2 * np.abs(spectra) / len(spectra)
			# phase2 = np.angle(spectra)
			# freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
			for i in range(len(poscentred)):
				pos = poscentred[i]
				spectra = np.fft.fft(np.mean(datasection2[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window],axis=(-1,-2)), axis=0)
				magnitude2 = 2 * np.abs(spectra) / len(spectra)
				freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
				# y = np.mean(magnitude2, axis=(-1, -2))
				y = magnitude2
				# y=magnitude[:,pos[0],pos[1]]
				y = np.array([y for _, y in sorted(zip(freq, y))])
				x = np.sort(freq)
				plt.plot(x, y, color[i] + '--',label='data at the point ' + str(pos) + ', ' + str(np.around(freq_to_erase,decimals=2)) + 'Hz oscillation substracted')
				plt.axvline(x=freq_to_erase,color='k',linestyle=':')
				plt.axvline(x=freq[np.abs(freq-freq_to_erase).argmin()],color='k',linestyle='--')
			# plt.title()


			# plt.grid()
			plt.semilogy()
			plt.xlim(left=0)
			plt.ylim(top=y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)].max()*2e3)
			plt.legend(loc='best',fontsize='xx-small')
		plt.pause(0.0001)




	print('stats of the oscillation removal')
	print('with window of size '+str(np.around(section_frames/framerate,decimals=5))+'s of '+str(len(data)/framerate)+'s of record')
	print('found oscillation of frequency '+str(freq_to_erase)+'Hz')
	print('On the ROI oscillation magnitude reduced from %.5g[au] to %.5g[au]' %(peak_value_pre_filter,peak_value_post_filter)+'\nwith an approximate maximum noise of %.5g[au] and median of %.5g[au]' %(np.max(noise),np.median(noise)))

	if output_noise:
		return np.array([data2]),peak_value_pre_filter,peak_value_post_filter,np.max(noise),np.median(noise)
	else:
		return np.array([data2])

###################################################################################################

def clear_oscillation_central3(data,time,framerate,oscillation_search_window_begin=6,oscillation_search_window_end=2.5,min_frequency_to_erase=20,max_frequency_to_erase=34,plot_conparison=False,which_plot=[1,2,3],ROI='auto',window=2,force_poscentre='auto'):

	# Created 17/06/2021
	# I try a different approach.
	# I find the oscillation acress the whole foil and subtract it from the whole record without looking at it pixel by pixel

	plt.figure()
	full_average = np.mean(data[0],axis=(-1,-2))
	full_spectra = np.fft.fft(full_average)
	full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
	full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / framerate)
	plt.plot(full_freq,full_magnitude,label='full')
	after_pulse_average = np.mean(data[0][int(oscillation_search_window_begin*framerate):],axis=(-1,-2))
	after_pulse_spectra = np.fft.fft(after_pulse_average)
	after_pulse_magnitude = 2 * np.abs(after_pulse_spectra) / len(full_spectra)
	after_pulse_freq = np.fft.fftfreq(len(after_pulse_magnitude), d=1 / framerate)
	plt.plot(after_pulse_freq,after_pulse_magnitude,label='after pulse')
	plt.legend(loc='best',fontsize='x-small')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [au]')
	plt.grid()
	plt.semilogy()
	plt.pause(0.01)

	from pynfft import NFFT, Solver
	time_cropped = time[np.logical_or(time<oscillation_search_window_end,time>oscillation_search_window_begin)]
	temp1 = data[0][time<oscillation_search_window_end]
	temp1 = np.mean(temp1,axis=(-1,-2))
	# temp1 -= np.mean(temp1)
	temp2 = data[0][time>oscillation_search_window_begin]
	temp2 = np.mean(temp2,axis=(-1,-2))
	# temp2 -= np.mean(temp2)
	data_cropped = np.array(temp1.tolist() + temp2.tolist())
	# data_cropped = np.mean(data[0],axis=(1,2))
	data_cropped -= generic_filter(data_cropped,np.mean,size=[int(framerate/30*2)])
	plt.figure()
	plt.plot(time_cropped,data_cropped)
	plt.pause(0.01)

	temp1 = data[0][time<oscillation_search_window_end]
	temp1 = np.mean(temp1,axis=(-1,-2))
	temp1 -= generic_filter(temp1,np.nanmean,size=[int(framerate/min_frequency_to_erase*10)])
	temp1 -= np.mean(temp1)
	temp2 = data[0][time>oscillation_search_window_begin]
	temp2 = np.mean(temp2,axis=(-1,-2))
	temp2 -= generic_filter(temp2,np.nanmean,size=[int(framerate/min_frequency_to_erase*10)])
	temp2 -= np.mean(temp2)
	data_cropped = np.array(temp1.tolist() + temp2.tolist())
	# plt.figure()
	plt.plot(time_cropped,data_cropped)
	plt.pause(0.01)

	M=len(data_cropped)
	N=M//2
	f = np.empty(M, dtype=np.complex128)
	f_hat = np.empty(N, dtype=np.complex128)


	if True:	# even if simple this seems to wiork well
		this_nfft = NFFT(N=N, M=M)
		this_nfft.x = (time_cropped/time_cropped.max() - 0.5)
		this_nfft.precompute()
		this_nfft.f = data_cropped
		ret2=this_nfft.adjoint()
		magnitude=2*np.abs(ret2)/len(ret2)
		freq = np.arange(-N/2,+N/2,1)/time_cropped.max()
		select = np.logical_and(magnitude>=magnitude.max(),freq>=0)
		# select = magnitude>=magnitude.max()
		peak_freq = freq[select]
		peak_angle = np.angle(ret2[select])
		plt.figure()
		plt.plot(np.abs(freq),magnitude)
		plt.plot(np.abs(freq)[select],magnitude[select],'+')
		plt.axvline(x=peak_freq,color='k',linestyle='--')
		plt.pause(0.01)
		angle = np.angle(ret2)
		plt.figure()
		plt.plot(freq,angle)
		plt.plot(freq[select],angle[select],'+')
		plt.pause(0.01)

		ret2[abs(freq)<20] = 0
		check_nfft = NFFT(N=N, M=M)
		check_nfft.x = (time_cropped/time_cropped.max() - 0.5)
		check_nfft.precompute()
		check_nfft.f_hat = ret2
		f=check_nfft.trafo()

		plt.figure()
		plt.plot((time_cropped/time_cropped.max() - 0.5),data_cropped)
		plt.plot((time_cropped/time_cropped.max() - 0.5),f*data_cropped.max()/f.max(),'--')
		plt.pause(0.01)
		plt.figure()
		plt.plot((time_cropped/time_cropped.max() - 0.5),data_cropped)
		plt.plot((time_cropped/time_cropped.max() - 0.5),f,'--')
		plt.pause(0.01)

		# ret2[freq<0] = 0
		# ret2[abs(freq)<29] = 0
		# ret2[abs(freq)>45] = 0
		ret2[::2]=0
		# ret2[np.logical_not(select)] = 0
		check_nfft = NFFT(N=N, M=len(time))
		check_nfft.x = (time/time.max() - 0.5)
		check_nfft.precompute()
		check_nfft.f_hat = ret2
		f=check_nfft.trafo()
		plt.figure()
		plt.plot((time_cropped/time_cropped.max() - 0.5),data_cropped)
		plt.plot((time/time.max() - 0.5),f*data_cropped.max()/f.max(),'--')
		plt.pause(0.01)

		def fun_cos(x,a,f,p):
			print([a,f,p])
			return a*np.cos(p+2*np.pi*x*f)

		for value in np.linspace(29,30,num=10):
			guess = [1,np.abs(peak_freq),peak_angle]
			# bds = [[0.5,0,-np.pi],[np.inf,np.inf,np.pi]]
			bds = [[1,28,-np.inf],[np.inf,32,np.inf]]
			x_scale = [0.1,30,0.1]
			fit = curve_fit(fun_cos,time_cropped,data_cropped,p0=guess,bounds=bds,ftol=1e-13,xtol=1e-13,verbose=2)
			print(value)
			print(fit)
		plt.figure()
		plt.plot(time_cropped,data_cropped)
		temp = fun_cos(time,*fit[0])
		plt.plot(time,temp,'--')
		temp = fun_cos(time_cropped,*fit[0])
		plt.plot(time_cropped,data_cropped-temp)
		plt.pause(0.01)


		# freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		# select = np.logical_and(magnitude>0.1,freq>=0)
		# plt.figure()
		# plt.plot(freq,magnitude)
		# plt.plot(freq[select],magnitude[select],'+')
		# plt.pause(0.01)
	elif False:	# the iterative process seems to diverge
		this_solver = Solver(this_nfft)
		this_solver.y = data_cropped
		this_solver.f_hat_iter = ret2
		this_solver.before_loop()
		while not np.all(this_solver.r_iter < 1e-2):
			this_solver.loop_one_step()
		magnitude=2*np.abs(this_solver.f_hat_iter)/len(this_solver.f_hat_iter)
		plt.figure()
		plt.plot(np.abs(np.arange(-N/2,+N/2,1)),magnitude)
		plt.pause(0.01)
		angle = np.angle(this_solver.f_hat_iter)
		plt.figure()
		plt.plot(np.abs(np.arange(-N/2,+N/2,1)),angle)
		plt.plot(np.abs(np.arange(-N/2,+N/2,1)),magnitude>np.sort(magnitude)[-10],anglemagnitude>np.sort(magnitude)[-10],'+')
		magnitude>np.sort(magnitude)[-10]
		plt.pause(0.01)

	data2 = cp.deepcopy(data[0].T)
	for i_freq,freq_ in enumerate(freq):
		if select[i_freq]==True:
			# break
			data2 -= magnitude[i_freq]*np.cos(angle[i_freq] + 2 * np.pi * freq_ * time)
	data2 = data2.T

	neg_nfft = NFFT(N=N, M=np.shape(data)[1])
	neg_nfft.x = time/time.max() - 0.5
	neg_nfft.precompute()
	neg_nfft.f_hat = ret2
	f = neg_nfft.trafo()

	plt.figure()
	plt.plot(time,np.mean(data[0],axis=(-1,-2)))
	plt.plot(time,np.mean(data2,axis=(-1,-2)),'--')
	plt.pause(0.01)


	cos_fun = lambda x,a,x0,p0,x1,p1,x2,p2: a*np.cos(x0 + 2*np.pi*p0*x)*np.cos(x1 + 2*np.pi*p1*x)*np.cos(x2 + 2*np.pi*p2*x)
	bds = [[1,-np.pi,24,-np.pi,0,-np.pi,0],[np.inf,np.pi,34,np.pi,5,np.pi,15]]
	guess=[1.5,0,29,0,1,]
	x_scale = [1,0.1,0.1,0.1,0.1]
	fit = curve_fit(cos_fun, time_cropped, data_cropped, p0=guess,bounds=bds,x_scale=x_scale,maxfev=int(1e6),verbose=2,ftol=1e-15)
	plt.plot(time,cos_fun(time,*fit[0]))
	plt.figure()
	plt.plot(time,np.mean(data[0],axis=(-1,-2)))
	plt.plot(time,np.mean(data[0],axis=(-1,-2))-cos_fun(time,*fit[0]),'--')
	plt.pause(0.01)

	# path abandoned

###################################################################################################


def clear_oscillation_central4(data,framerate,oscillation_search_window_begin='auto',oscillation_search_window_end='auto',min_frequency_to_erase=20,max_frequency_to_erase=34,plot_conparison=False,which_plot=[1,2,3],ROI='auto',window=2,force_poscentre='auto',output_noise=False,multiple_frequencies_cleaned=1):
	from scipy.signal import find_peaks, peak_prominences as get_proms

	# Created 29/07/2021
	# Function created starting from clear_oscillation_central2, but instead of repeating it over and ofer externally I do an internal loop
	# this is to remove oscillations that involve more than one single frequency (as it always is) in a reliable way

	print('shape of data array is '+str(np.shape(data))+', it should be (x,frames,v pixel,h pixel)')
	# figure_index = plt.gcf().number

	data=data[0]
	if oscillation_search_window_begin=='auto':
		force_start=0
	elif (oscillation_search_window_begin<0 or oscillation_search_window_begin>len(data)*framerate):
		print('The initial limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print('0s will be used instead')
		force_start=0
	else:
		force_start=int(oscillation_search_window_begin*framerate)

	if oscillation_search_window_end=='auto':
		force_end=len(data)
	elif (oscillation_search_window_end<0 or oscillation_search_window_end>(len(data)/framerate) or oscillation_search_window_end<=force_start/framerate):
		print('The final limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print(str(int(len(data)//(2*framerate)))+'s will be used instead')
		force_end=int(len(data)//(2*framerate))
	else:
		force_end=int(oscillation_search_window_end*framerate)


	central_freq_for_search = (max_frequency_to_erase-min_frequency_to_erase)/2+min_frequency_to_erase
	if (framerate<2*central_freq_for_search):
		print('There is a problem. The framerate is too low to try to extract the oscillation')
		print('The minimum framrate for doing it is 2*oscillation frequency to detect. Therefore '+str(np.around(2*central_freq_for_search,decimals=1))+'Hz, in this case.')
		print('See http://www.skillbank.co.uk/SignalConversion/rate.htm')
		exit()

	#window = 2	# Previously found that as long as the fft is averaged over at least 4 pixels the peak shape and location does not change
	datasection = data

	if plot_conparison==True:
		# plt.figure()
		# plt.pause(0.01)
		plt.figure()
		figure_index = plt.gcf().number
		if 1 in which_plot:
			data_shape = np.shape(data)
			poscentred = [[int(data_shape[1]*1/5), int(data_shape[2]*1/5)], [int(data_shape[1]*1/5), int(data_shape[2]*4/5)], [int(data_shape[1]*1/2), int(data_shape[2]*1/5)], [int(data_shape[1]*4/5), int(data_shape[2]*1/5)], [int(data_shape[1]*4/5), int(data_shape[2]*1/2)], [int(data_shape[1]*4/5), int(data_shape[2]*4/5)], [int(data_shape[1]*1/2), int(data_shape[2]*1/2)]]

			# spectra_orig = np.fft.fft(data, axis=0)
			# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
			# magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
			# phase = np.angle(spectra_orig)
			# freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

			color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
			for i in range(len(poscentred)):
				pos = poscentred[i]
				spectra_orig = np.fft.fft(np.mean(data[:, pos[0] - window:pos[0] + window+1, pos[1] - window:pos[1] + window+1],axis=(-1,-2)), axis=0)
				magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
				freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
				# y = np.mean(magnitude, axis=(-1, -2))
				y = magnitude
				# y=magnitude[:,pos[0],pos[1]]
				y = np.array([y for _, y in sorted(zip(freq, y))])
				x = np.sort(freq)
				plt.plot(x, y*100, color[i], label='original data at the point ' + str(pos) + ' x100')
			# plt.title()


			plt.title('Amplitued from fast Fourier transform in the whole time interval\nfor different groups of ' + str(window * 2+1) + 'x' + str(
				window * 2+1) + ' pixels, framerate %.3gHz' %(framerate) )
			plt.xlabel('Frequency [Hz]')
			plt.ylabel('Amplitude [au]')
			plt.grid()
			plt.semilogy()
			plt.legend(loc='best',fontsize='xx-small')
		else:
			plt.close(figure_index)
	# plt.show()



	frames_for_oscillation = framerate//central_freq_for_search

	number_of_waves = 3
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	if fft_window_move<10:
		number_of_waves=10//frames_for_oscillation	# I want to scan for at least 10 frame shifts
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	step = 1
	while int(fft_window_move/step)>80:
		step+=1		#if framerate is too high i will skip some of the shifts to limit the number of Fourier transforms to 100


	# I restrict the window over which I search for the oscillation
	datarestricted = data[force_start:force_end]#
	len_data_restricted = len(datarestricted)
	if force_poscentre == 'auto':
		poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]
	else:
		poscentre = force_poscentre

	if (oscillation_search_window_begin=='auto' and oscillation_search_window_end=='auto'):
		while fft_window_move>(len_data_restricted/5):
			fft_window_move-=1		# I want that the majority of the data I analyse remains the same

	if oscillation_search_window_end == 'auto':
		if (len(datarestricted) / framerate) <= 1:
			# max_start = int(sections // 2)
			max_start = int(len(datarestricted) // 2)
		else:
			max_start = int(5*framerate)  # I use 5 seconds of record
	else:
		max_start = len(datarestricted)	# this is actually ineffective, as datarestricted is already limited to force_start:force_end

	if oscillation_search_window_begin == 'auto':
		min_start = 0
	else:
		# min_start = force_start	# alreay enforced through force_start
		min_start = 0

	section_frames = max_start - min_start-fft_window_move

	if ROI=='auto':
		# datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(1,2))
		datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datarestricted2 = np.mean(datarestricted[:, select],axis=(-1))


	spectra = np.fft.fft(datarestricted2, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	freq = np.fft.fftfreq(len(magnitude_space_averaged), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	peaks_1 = find_peaks(y)[0]
	peaks = peaks_1[np.logical_and(x[peaks_1]<max_frequency_to_erase+15,x[peaks_1]>max_frequency_to_erase)]
	if len(peaks)==0:
		peaks = np.array(peaks_1[np.logical_and(x[peaks_1]>min_frequency_to_erase-15,x[peaks_1]<min_frequency_to_erase)])
		fit = [0,0,np.mean(np.log(y[peaks]))]	# I do this because I want to avoid that the fit only on the left goes too loo and I filter too much
	else:
		peaks = np.concatenate([peaks_1[np.logical_and(x[peaks_1]>min_frequency_to_erase-15,x[peaks_1]<min_frequency_to_erase)] , peaks_1[np.logical_and(x[peaks_1]<max_frequency_to_erase+15,x[peaks_1]>max_frequency_to_erase)]])
		fit = np.polyfit(x[peaks],np.log(y[peaks]),2)
	# poly2deg = lambda freq,a2,a1,a0 : a0 + a1*freq + a2*(freq**2)
	# guess = [min(fit[0],0),min(fit[1],0),fit[2]]
	# bds = [[-np.inf,-np.inf,-np.inf],[0,0,np.inf]]
	# fit = curve_fit(poly2deg, x[peaks],np.log(y[peaks]), p0=guess, bounds=bds, maxfev=100000000)[0]
	y_reference = np.exp(np.polyval(fit,x[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]))
	if min_frequency_to_erase>35:	# this is in place mainly only for the filter around 90Hz
		min_of_fit = x[np.logical_and(x>min_frequency_to_erase-15,x<min_frequency_to_erase)]
		min_of_fit = np.exp(np.polyval(fit,min_of_fit)).min()
		y_reference[y_reference > min_of_fit] = min_of_fit	# this is to avoid that when the parabola is fitted only on the left the fit goes up in the area of interest. the reference is limited to the minimum value of the fit outside of it
	y_reference = (3*np.std(y[peaks]/np.exp(np.polyval(fit,x[peaks])))+1)*y_reference

	y_test = y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.plot(x, y,'--r',label='original')
		plt.grid()
		plt.semilogy()
		plt.plot(x[peaks_1],y[peaks_1],'x')
		plt.axvline(x=min_frequency_to_erase,color='k',linestyle='--')
		plt.axvline(x=max_frequency_to_erase,color='k',linestyle='--')
		plt.plot(x[peaks],y[peaks],'o')
		plt.plot(x,np.exp(np.polyval(fit,x)))
		plt.plot(x[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)],y_reference,'--')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.xlim(left=min_frequency_to_erase-5,right=max_frequency_to_erase+5)
		plt.ylim(bottom=np.median(np.sort(y)[:int(len(y)/4)])*1e-1,top = np.max(y[peaks])*2)

	frequencies_removed_all = []
	first_pass = True
	data2 = cp.deepcopy(data)
	while np.sum((y_test-y_reference)>0)>0:

		record_magnitude = []
		record_phase = []
		record_freq = []
		peak_freq_record = []
		peak_value_record = []
		peak_index_record = []
		shift_record = []

		for i in range(int(fft_window_move/step)):
			shift=i*step
			datasection = datarestricted2[min_start:max_start-fft_window_move+shift]
			spectra = np.fft.fft(datasection, axis=0)
			magnitude = 2 * np.abs(spectra) / len(spectra)
			record_magnitude.append(magnitude[0:len(magnitude) // 2])
			phase = np.angle(spectra)
			record_phase.append(phase[0:len(magnitude) // 2])
			freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
			record_freq.append(freq[0:len(magnitude) // 2])
			# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
			y = np.array([value for _, value in sorted(zip(freq, magnitude))])
			x = np.sort(freq)

			index_min_freq = np.abs(x-min_frequency_to_erase).argmin()  # I restric the window over which I do the peak search
			index_max_freq = np.abs(x-max_frequency_to_erase).argmin()
			index_7 = np.abs(x-7).argmin()
			index_n7 = np.abs(x-(-7)).argmin()
			index_min_freq_n = np.abs(x-(-min_frequency_to_erase)).argmin()
			index_max_freq_n = np.abs(x-(-max_frequency_to_erase)).argmin()
			index_0 = np.abs(x-0).argmin()
			# noise = np.mean(np.array(
			# 	y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[
			# 																									index_max_freq:-3].tolist()),
			# 				axis=(-1))
			# temp = peakutils.indexes(y[index_min_freq:index_max_freq], thres=noise + np.abs(magnitude.min()),
			# 						 min_dist=(index_max_freq - index_min_freq) // 2)
			if len(y[index_min_freq:index_max_freq])==0:
				continue
			# if plot_conparison == True and (2 in which_plot):
			# 	plt.figure(figure_index+1)
			# 	plt.plot(x, y, label='Applied shift of ' + str(shift))
			temp = y[index_min_freq:index_max_freq].argmax()
			# if len(temp) == 1:
			peak_index = index_min_freq + int(temp)
			peak_freq_record.append(x[peak_index])
			peak_value = float(y[peak_index])
			peak_value_record.append(peak_value)
			peak_index_record.append(peak_index)
			shift_record.append(shift)
		record_magnitude = np.array(record_magnitude)
		record_phase = np.array(record_phase)
		record_freq = np.array(record_freq)
		peak_freq_record = np.array(peak_freq_record)
		peak_value_record = np.array(peak_value_record)
		peak_index_record = np.array(peak_index_record)
		shift_record = np.array(shift_record)
		# index = np.array(peak_value_record).argmax()	# this is wrong in the case that the baseline noise is strongly decreasing
		index = (np.array(peak_value_record)/np.exp(np.polyval(fit,peak_freq_record))).argmax()	# here I look at the strongest deviation from the noise baseline
		shift = index * step
		freq_to_erase = peak_freq_record[index]
		datasection = datarestricted[min_start:max_start-fft_window_move+shift]
		spectra = np.fft.fft(datasection, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude = 2 * np.abs(spectra) / len(spectra)
		phase = np.angle(spectra)
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		freq_to_erase_index = np.abs(freq-freq_to_erase).argmin()
		frequencies_removed_all.append(freq[freq_to_erase_index])
		freq_to_erase_index_multiple = np.arange(-(multiple_frequencies_cleaned-1)//2,(multiple_frequencies_cleaned-1)//2+1) + freq_to_erase_index
		print(freq_to_erase_index_multiple)

		if plot_conparison==True and (2 in which_plot):
			plt.figure(figure_index+1)
			# plt.plot([freq_to_erase]*2,[peak_value_record.min(),peak_value_record.max()],':k')
			# plt.plot([freq[freq_to_erase_index]]*2,[peak_value_record.min(),peak_value_record.max()],'--k')
			# plt.axvline(x=freq_to_erase,color='k',linestyle=':')
			plt.plot(freq[freq_to_erase_index],record_magnitude[index][np.abs(record_freq[index]-freq[freq_to_erase_index]).argmin()],'rx',markersize=10)
			if len(freq_to_erase_index_multiple)>1:
				for i_freq in freq_to_erase_index_multiple:
					if i_freq!=freq_to_erase_index:
						plt.plot(freq[i_freq],record_magnitude[index][np.abs(record_freq[index]-freq[i_freq]).argmin()],'yx',markersize=10)
			if first_pass:
				plt.ylim(top=peak_value_record.max()*2)
		if plot_conparison==True and (1 in which_plot):
			plt.figure(figure_index)
			plt.plot(freq[freq_to_erase_index],100*record_magnitude[index][np.abs(record_freq[index]-freq[freq_to_erase_index]).argmin()],'rx',markersize=10)
			if len(freq_to_erase_index_multiple)>1:
				for i_freq in freq_to_erase_index_multiple:
					if i_freq!=freq_to_erase_index:
						plt.plot(freq[i_freq],100*record_magnitude[index][np.abs(record_freq[index]-freq[i_freq]).argmin()],'yx',markersize=10)
		framenumber = np.linspace(0, len(data) - 1, len(data)) -force_start- min_start
		for i_freq in freq_to_erase_index_multiple:
			# print(i_freq)
			data2 -= np.multiply(magnitude[i_freq], np.cos(np.repeat(np.expand_dims(phase[i_freq], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq[i_freq] * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))
		# data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))

		datarestricted = data2[force_start:force_end]#
		if ROI=='auto':
			# datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(1,2))
			datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
		else:
			horizontal_coord = np.arange(np.shape(datarestricted)[2])
			vertical_coord = np.arange(np.shape(datarestricted)[1])
			horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
			select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
			datarestricted2 = np.mean(datarestricted[:, select],axis=(-1))

		spectra = np.fft.fft(datarestricted2, axis=0)
		magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
		freq = np.fft.fftfreq(len(magnitude_space_averaged), d=1 / framerate)
		# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
		y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
		x = np.sort(freq)
		y_test = y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]
		first_pass = False

	frequencies_removed_all = np.round(np.array(frequencies_removed_all)*100)/100
	# added to visualize the goodness of the result
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.plot(x, y,'--k',label='subtracted')
		# plt.grid()
		plt.legend(loc='best',fontsize='xx-small')
		if ROI=='auto':
			plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged in a window of ' %(force_start/framerate,(force_start+max_start)/framerate)+ str([window * 2+1,window * 2+1]) + ' pixels around ' + str(poscentre) + ', framerate %.3gHz' %(framerate)+'\nremoved freq: '+str(frequencies_removed_all)+' Hz')
		else:
			plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged ouside the ROI ' %(force_start/framerate,(force_start+max_start)/framerate)+ str(ROI) + ' pixels around ' + str(ROI) + ', framerate %.3gHz' %(framerate)+'\nremoved freq: '+str(frequencies_removed_all)+' Hz')


	# section only for stats
	# datasection = data[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	if ROI=='auto':
		datasection = np.mean(data[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data[:, select],axis=(-1))
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	index_min_freq = np.abs(x-(min_frequency_to_erase)).argmin()	# I restric the window over which I do the peak search
	index_max_freq = np.abs(x-(max_frequency_to_erase)).argmin()
	index_7 = np.abs(x-(7)).argmin()
	index_n7 = np.abs(x-(-7)).argmin()
	index_min_freq_n = np.abs(x-(-min_frequency_to_erase)).argmin()
	index_max_freq_n = np.abs(x-(-max_frequency_to_erase)).argmin()
	index_0 = np.abs(x-(0)).argmin()
	noise = (np.array(y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[index_max_freq:-3].tolist()))
	# temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	# peak_index = index_min_freq + int(temp)
	peak_index = index_min_freq + y[index_min_freq:index_max_freq].argmax()
	peak_value_pre_filter = float(y[peak_index])

	# datasection = data2[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	if ROI=='auto':
		datasection = np.mean(data2[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data2[:, select],axis=(-1))
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	# temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	# peak_index = index_min_freq + int(temp)
	peak_index = index_min_freq + y[index_min_freq:index_max_freq].argmax()
	peak_value_post_filter = float(y[peak_index])
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.axhline(y=np.max(noise),linestyle='--',color='k',label='max noise')
		plt.axhline(y=np.median(noise),linestyle='--',color='b',label='median noise')
		plt.axhline(y=peak_value_pre_filter,linestyle='--',color='r',label='peak pre')
		plt.axhline(y=peak_value_post_filter,linestyle='--',color='g',label='peak post')
		plt.legend(loc='best',fontsize='xx-small')

	if plot_conparison==True:
		if 1 in which_plot:
			plt.figure(figure_index)
			datasection2 = data2
			# spectra = np.fft.fft(datasection2, axis=0)
			# # magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
			# magnitude2 = 2 * np.abs(spectra) / len(spectra)
			# phase2 = np.angle(spectra)
			# freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
			for i in range(len(poscentred)):
				pos = poscentred[i]
				spectra = np.fft.fft(np.mean(datasection2[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window],axis=(-1,-2)), axis=0)
				magnitude2 = 2 * np.abs(spectra) / len(spectra)
				freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
				# y = np.mean(magnitude2, axis=(-1, -2))
				y = magnitude2
				# y=magnitude[:,pos[0],pos[1]]
				y = np.array([y for _, y in sorted(zip(freq, y))])
				x = np.sort(freq)
				plt.plot(x, y, color[i] + '--',label='data at the point ' + str(pos) + ', oscillation substracted')
			# plt.title()


			# plt.grid()
			plt.semilogy()
			plt.xlim(left=0)
			plt.ylim(top=y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)].max()*2e3)
			plt.legend(loc='best',fontsize='xx-small')
		plt.pause(0.0001)




	print('stats of the oscillation removal')
	print('with window of size '+str(np.around(section_frames/framerate,decimals=5))+'s of '+str(len(data)/framerate)+'s of record')
	try:
		print('found oscillation of frequency '+str(frequencies_removed_all)+'Hz')
	except:
		print('no frequency needed removing')
	print('On the ROI oscillation magnitude reduced from %.5g[au] to %.5g[au]' %(peak_value_pre_filter,peak_value_post_filter)+'\nwith an approximate maximum noise of %.5g[au] and median of %.5g[au]' %(np.max(noise),np.median(noise)))

	if output_noise:
		return np.array([data2]),peak_value_pre_filter,peak_value_post_filter,np.max(noise),np.median(noise)
	else:
		return np.array([data2])


def real_mean_filter_agent(datasection,freq_to_erase_frames):
	filter_agent = cp.deepcopy(datasection)
	padded_by = int(np.ceil(freq_to_erase_frames/2))+1
	temp = np.pad(filter_agent, ((padded_by,padded_by),(0,0),(0,0)), mode='reflect')
	for ii in range(-int(np.floor((freq_to_erase_frames-1)/2)),int(np.floor((freq_to_erase_frames-1)/2))+1):
		if ii!=0:
			filter_agent += temp[padded_by+ii:-padded_by+ii]
	filter_agent += temp[padded_by-int(np.floor((freq_to_erase_frames-1)/2))-1:-padded_by-int(np.floor((freq_to_erase_frames-1)/2))-1]*((freq_to_erase_frames-1)/2-int(np.floor((freq_to_erase_frames-1)/2)))
	filter_agent += temp[padded_by+int(np.floor((freq_to_erase_frames-1)/2))+1:-padded_by+int(np.floor((freq_to_erase_frames-1)/2))+1]*((freq_to_erase_frames-1)/2-int(np.floor((freq_to_erase_frames-1)/2)))
	filter_agent/=freq_to_erase_frames
	return filter_agent

def clear_oscillation_central5(data,framerate,oscillation_search_window_begin='auto',oscillation_search_window_end='auto',min_frequency_to_erase=20,max_frequency_to_erase=34,plot_conparison=False,which_plot=[1,2,3],ROI='auto',window=2,force_poscentre='auto',output_noise=False,multiple_frequencies_cleaned=1):
	from scipy.signal import find_peaks, peak_prominences as get_proms

	# Created 04/08/2021
	# Function created starting from clear_oscillation_central4, the mean filter is applied before each frequency subtraction

	print('shape of data array is '+str(np.shape(data))+', it should be (x,frames,v pixel,h pixel)')
	# figure_index = plt.gcf().number

	data=data[0]
	if oscillation_search_window_begin=='auto':
		force_start=0
	elif (oscillation_search_window_begin<0 or oscillation_search_window_begin>len(data)*framerate):
		print('The initial limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print('0s will be used instead')
		force_start=0
	else:
		force_start=int(oscillation_search_window_begin*framerate)

	if oscillation_search_window_end=='auto':
		force_end=len(data)
	elif (oscillation_search_window_end<0 or oscillation_search_window_end>(len(data)/framerate) or oscillation_search_window_end<=force_start/framerate):
		print('The final limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print(str(int(len(data)//(2*framerate)))+'s will be used instead')
		force_end=int(len(data)//(2*framerate))
	else:
		force_end=int(oscillation_search_window_end*framerate)


	central_freq_for_search = (max_frequency_to_erase-min_frequency_to_erase)/2+min_frequency_to_erase
	if (framerate<2*central_freq_for_search):
		print('There is a problem. The framerate is too low to try to extract the oscillation')
		print('The minimum framrate for doing it is 2*oscillation frequency to detect. Therefore '+str(np.around(2*central_freq_for_search,decimals=1))+'Hz, in this case.')
		print('See http://www.skillbank.co.uk/SignalConversion/rate.htm')
		exit()

	#window = 2	# Previously found that as long as the fft is averaged over at least 4 pixels the peak shape and location does not change
	datasection = data

	if plot_conparison==True:
		# plt.figure()
		# plt.pause(0.01)
		plt.figure()
		figure_index = plt.gcf().number
		if 1 in which_plot:
			data_shape = np.shape(data)
			poscentred = [[int(data_shape[1]*1/5), int(data_shape[2]*1/5)], [int(data_shape[1]*1/5), int(data_shape[2]*4/5)], [int(data_shape[1]*1/2), int(data_shape[2]*1/5)], [int(data_shape[1]*4/5), int(data_shape[2]*1/5)], [int(data_shape[1]*4/5), int(data_shape[2]*1/2)], [int(data_shape[1]*4/5), int(data_shape[2]*4/5)], [int(data_shape[1]*1/2), int(data_shape[2]*1/2)]]

			# spectra_orig = np.fft.fft(data, axis=0)
			# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
			# magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
			# phase = np.angle(spectra_orig)
			# freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

			color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
			for i in range(len(poscentred)):
				pos = poscentred[i]
				spectra_orig = np.fft.fft(np.mean(data[:, pos[0] - window:pos[0] + window+1, pos[1] - window:pos[1] + window+1],axis=(-1,-2)), axis=0)
				magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
				freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
				# y = np.mean(magnitude, axis=(-1, -2))
				y = magnitude
				# y=magnitude[:,pos[0],pos[1]]
				y = np.array([y for _, y in sorted(zip(freq, y))])
				x = np.sort(freq)
				plt.plot(x, y*100, color[i], label='original data at the point ' + str(pos) + ' x100')
			# plt.title()


			plt.title('Amplitued from fast Fourier transform in the whole time interval\nfor different groups of ' + str(window * 2+1) + 'x' + str(
				window * 2+1) + ' pixels, framerate %.3gHz' %(framerate) )
			plt.xlabel('Frequency [Hz]')
			plt.ylabel('Amplitude [au]')
			plt.grid()
			plt.semilogy()
			plt.legend(loc='best',fontsize='xx-small')
		else:
			plt.close(figure_index)
	# plt.show()



	frames_for_oscillation = framerate//central_freq_for_search

	number_of_waves = 3
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	if fft_window_move<10:
		number_of_waves=10//frames_for_oscillation	# I want to scan for at least 10 frame shifts
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	step = 1
	while int(fft_window_move/step)>80:
		step+=1		#if framerate is too high i will skip some of the shifts to limit the number of Fourier transforms to 100


	# I restrict the window over which I search for the oscillation
	datarestricted = data[force_start:force_end]#
	len_data_restricted = len(datarestricted)
	if force_poscentre == 'auto':
		poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]
	else:
		poscentre = force_poscentre

	if (oscillation_search_window_begin=='auto' and oscillation_search_window_end=='auto'):
		while fft_window_move>(len_data_restricted/5):
			fft_window_move-=1		# I want that the majority of the data I analyse remains the same

	if oscillation_search_window_end == 'auto':
		if (len(datarestricted) / framerate) <= 1:
			# max_start = int(sections // 2)
			max_start = int(len(datarestricted) // 2)
		else:
			max_start = int(5*framerate)  # I use 5 seconds of record
	else:
		max_start = len(datarestricted)	# this is actually ineffective, as datarestricted is already limited to force_start:force_end

	if oscillation_search_window_begin == 'auto':
		min_start = 0
	else:
		# min_start = force_start	# alreay enforced through force_start
		min_start = 0

	section_frames = max_start - min_start-fft_window_move

	if ROI=='auto':
		# datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(1,2))
		datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datarestricted2 = np.mean(datarestricted[:, select],axis=(-1))


	spectra = np.fft.fft(datarestricted2, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	freq = np.fft.fftfreq(len(magnitude_space_averaged), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	peaks_1 = find_peaks(y)[0]
	peaks = peaks_1[np.logical_and(x[peaks_1]<max_frequency_to_erase+15,x[peaks_1]>max_frequency_to_erase)]
	if len(peaks)==0:
		peaks = np.array(peaks_1[np.logical_and(x[peaks_1]>min_frequency_to_erase-15,x[peaks_1]<min_frequency_to_erase)])
		fit = [0,0,np.mean(np.log(y[peaks]))]	# I do this because I want to avoid that the fit only on the left goes too loo and I filter too much
	else:
		peaks = np.concatenate([peaks_1[np.logical_and(x[peaks_1]>min_frequency_to_erase-15,x[peaks_1]<min_frequency_to_erase)] , peaks_1[np.logical_and(x[peaks_1]<max_frequency_to_erase+15,x[peaks_1]>max_frequency_to_erase)]])
		fit = np.polyfit(x[peaks],np.log(y[peaks]),2)
	# poly2deg = lambda freq,a2,a1,a0 : a0 + a1*freq + a2*(freq**2)
	# guess = [min(fit[0],0),min(fit[1],0),fit[2]]
	# bds = [[-np.inf,-np.inf,-np.inf],[0,0,np.inf]]
	# fit = curve_fit(poly2deg, x[peaks],np.log(y[peaks]), p0=guess, bounds=bds, maxfev=100000000)[0]
	y_reference = np.exp(np.polyval(fit,x[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]))
	if min_frequency_to_erase>35:	# this is in place mainly only for the filter around 90Hz
		min_of_fit = x[np.logical_and(x>min_frequency_to_erase-15,x<min_frequency_to_erase)]
		min_of_fit = np.exp(np.polyval(fit,min_of_fit)).min()
		y_reference[y_reference > min_of_fit] = min_of_fit	# this is to avoid that when the parabola is fitted only on the left the fit goes up in the area of interest. the reference is limited to the minimum value of the fit outside of it
	y_reference = (3*np.std(y[peaks]/np.exp(np.polyval(fit,x[peaks])))+1)*y_reference

	y_test = y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.plot(x, y,'--r',label='original')
		plt.grid()
		plt.semilogy()
		plt.plot(x[peaks_1],y[peaks_1],'x')
		plt.axvline(x=min_frequency_to_erase,color='k',linestyle='--')
		plt.axvline(x=max_frequency_to_erase,color='k',linestyle='--')
		plt.plot(x[peaks],y[peaks],'o')
		plt.plot(x,np.exp(np.polyval(fit,x)))
		plt.plot(x[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)],y_reference,'--')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.xlim(left=min_frequency_to_erase-5,right=max_frequency_to_erase+5)
		plt.ylim(bottom=np.median(np.sort(y)[:int(len(y)/4)])*1e-1,top = np.max(y[peaks])*2)

	frequencies_removed_all = []
	first_pass = True
	data2 = cp.deepcopy(data)
	while np.sum((y_test-y_reference)>0)>0:

		record_magnitude = []
		record_phase = []
		record_freq = []
		peak_freq_record = []
		peak_value_record = []
		peak_index_record = []
		shift_record = []

		for i in range(int(fft_window_move/step)):
			shift=i*step
			datasection = datarestricted2[min_start:max_start-fft_window_move+shift]
			spectra = np.fft.fft(datasection, axis=0)
			magnitude = 2 * np.abs(spectra) / len(spectra)
			record_magnitude.append(magnitude[0:len(magnitude) // 2])
			phase = np.angle(spectra)
			record_phase.append(phase[0:len(magnitude) // 2])
			freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
			record_freq.append(freq[0:len(magnitude) // 2])
			# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
			y = np.array([value for _, value in sorted(zip(freq, magnitude))])
			x = np.sort(freq)

			index_min_freq = np.abs(x-min_frequency_to_erase).argmin()  # I restric the window over which I do the peak search
			index_max_freq = np.abs(x-max_frequency_to_erase).argmin()
			index_7 = np.abs(x-7).argmin()
			index_n7 = np.abs(x-(-7)).argmin()
			index_min_freq_n = np.abs(x-(-min_frequency_to_erase)).argmin()
			index_max_freq_n = np.abs(x-(-max_frequency_to_erase)).argmin()
			index_0 = np.abs(x-0).argmin()
			# noise = np.mean(np.array(
			# 	y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[
			# 																									index_max_freq:-3].tolist()),
			# 				axis=(-1))
			# temp = peakutils.indexes(y[index_min_freq:index_max_freq], thres=noise + np.abs(magnitude.min()),
			# 						 min_dist=(index_max_freq - index_min_freq) // 2)
			if len(y[index_min_freq:index_max_freq])==0:
				continue
			# if plot_conparison == True and (2 in which_plot):
			# 	plt.figure(figure_index+1)
			# 	plt.plot(x, y, label='Applied shift of ' + str(shift))
			temp = y[index_min_freq:index_max_freq].argmax()
			# if len(temp) == 1:
			peak_index = index_min_freq + int(temp)
			peak_freq_record.append(x[peak_index])
			peak_value = float(y[peak_index])
			peak_value_record.append(peak_value)
			peak_index_record.append(peak_index)
			shift_record.append(shift)
		record_magnitude = np.array(record_magnitude)
		record_phase = np.array(record_phase)
		record_freq = np.array(record_freq)
		peak_freq_record = np.array(peak_freq_record)
		peak_value_record = np.array(peak_value_record)
		peak_index_record = np.array(peak_index_record)
		shift_record = np.array(shift_record)
		# index = np.array(peak_value_record).argmax()	# this is wrong in the case that the baseline noise is strongly decreasing
		index = (np.array(peak_value_record)/np.exp(np.polyval(fit,peak_freq_record))).argmax()	# here I look at the strongest deviation from the noise baseline
		shift = index * step
		freq_to_erase = peak_freq_record[index]
		datasection = datarestricted[min_start:max_start-fft_window_move+shift]
		print('filtering '+str([freq_to_erase,framerate/freq_to_erase,index]))
		filter_agent = real_mean_filter_agent(datasection,framerate/freq_to_erase)	# added to make sure to dynamically remove only the wanted frequency
		spectra = np.fft.fft(datasection-filter_agent, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude = 2 * np.abs(spectra) / len(spectra)
		phase = np.angle(spectra)
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		freq_to_erase_index = np.abs(freq-freq_to_erase).argmin()
		frequencies_removed_all.append(freq[freq_to_erase_index])
		freq_to_erase_index_multiple = np.arange(-(multiple_frequencies_cleaned-1)//2,(multiple_frequencies_cleaned-1)//2+1) + freq_to_erase_index
		print(freq_to_erase_index_multiple)

		if plot_conparison==True and (2 in which_plot):
			plt.figure(figure_index+1)
			# plt.plot([freq_to_erase]*2,[peak_value_record.min(),peak_value_record.max()],':k')
			# plt.plot([freq[freq_to_erase_index]]*2,[peak_value_record.min(),peak_value_record.max()],'--k')
			# plt.axvline(x=freq_to_erase,color='k',linestyle=':')
			plt.plot(freq[freq_to_erase_index],record_magnitude[index][np.abs(record_freq[index]-freq[freq_to_erase_index]).argmin()],'rx',markersize=10)
			if len(freq_to_erase_index_multiple)>1:
				for i_freq in freq_to_erase_index_multiple:
					if i_freq!=freq_to_erase_index:
						plt.plot(freq[i_freq],record_magnitude[index][np.abs(record_freq[index]-freq[i_freq]).argmin()],'yx',markersize=10)
			if first_pass:
				plt.ylim(top=peak_value_record.max()*2)
		if plot_conparison==True and (1 in which_plot):
			plt.figure(figure_index)
			plt.plot(freq[freq_to_erase_index],100*record_magnitude[index][np.abs(record_freq[index]-freq[freq_to_erase_index]).argmin()],'rx',markersize=10)
			if len(freq_to_erase_index_multiple)>1:
				for i_freq in freq_to_erase_index_multiple:
					if i_freq!=freq_to_erase_index:
						plt.plot(freq[i_freq],100*record_magnitude[index][np.abs(record_freq[index]-freq[i_freq]).argmin()],'yx',markersize=10)
		framenumber = np.linspace(0, len(data) - 1, len(data)) -force_start- min_start
		for i_freq in freq_to_erase_index_multiple:
			# print(i_freq)
			data2 -= np.multiply(magnitude[i_freq], np.cos(np.repeat(np.expand_dims(phase[i_freq], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq[i_freq] * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))
		# data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))

		datarestricted = data2[force_start:force_end]#
		if ROI=='auto':
			# datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(1,2))
			datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
		else:
			horizontal_coord = np.arange(np.shape(datarestricted)[2])
			vertical_coord = np.arange(np.shape(datarestricted)[1])
			horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
			select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
			datarestricted2 = np.mean(datarestricted[:, select],axis=(-1))

		spectra = np.fft.fft(datarestricted2, axis=0)
		magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
		freq = np.fft.fftfreq(len(magnitude_space_averaged), d=1 / framerate)
		# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
		y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
		x = np.sort(freq)
		y_test = y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]
		first_pass = False

	frequencies_removed_all = np.round(np.array(frequencies_removed_all)*100)/100
	# added to visualize the goodness of the result
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.plot(x, y,'--k',label='subtracted')
		# plt.grid()
		plt.legend(loc='best',fontsize='xx-small')
		if ROI=='auto':
			plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged in a window of ' %(force_start/framerate,(force_start+max_start)/framerate)+ str([window * 2+1,window * 2+1]) + ' pixels around ' + str(poscentre) + ', framerate %.3gHz' %(framerate)+'\nremoved freq: '+str(frequencies_removed_all)+' Hz')
		else:
			plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged ouside the ROI ' %(force_start/framerate,(force_start+max_start)/framerate)+ str(ROI) + ' pixels around ' + str(ROI) + ', framerate %.3gHz' %(framerate)+'\nremoved freq: '+str(frequencies_removed_all)+' Hz')


	# section only for stats
	# datasection = data[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	if ROI=='auto':
		datasection = np.mean(data[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data[:, select],axis=(-1))
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	index_min_freq = np.abs(x-(min_frequency_to_erase)).argmin()	# I restric the window over which I do the peak search
	index_max_freq = np.abs(x-(max_frequency_to_erase)).argmin()
	index_7 = np.abs(x-(7)).argmin()
	index_n7 = np.abs(x-(-7)).argmin()
	index_min_freq_n = np.abs(x-(-min_frequency_to_erase)).argmin()
	index_max_freq_n = np.abs(x-(-max_frequency_to_erase)).argmin()
	index_0 = np.abs(x-(0)).argmin()
	noise = (np.array(y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[index_max_freq:-3].tolist()))
	# temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	# peak_index = index_min_freq + int(temp)
	peak_index = index_min_freq + y[index_min_freq:index_max_freq].argmax()
	peak_value_pre_filter = float(y[peak_index])

	# datasection = data2[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	if ROI=='auto':
		datasection = np.mean(data2[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data2[:, select],axis=(-1))
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	# temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	# peak_index = index_min_freq + int(temp)
	peak_index = index_min_freq + y[index_min_freq:index_max_freq].argmax()
	peak_value_post_filter = float(y[peak_index])
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.axhline(y=np.max(noise),linestyle='--',color='k',label='max noise')
		plt.axhline(y=np.median(noise),linestyle='--',color='b',label='median noise')
		plt.axhline(y=peak_value_pre_filter,linestyle='--',color='r',label='peak pre')
		plt.axhline(y=peak_value_post_filter,linestyle='--',color='g',label='peak post')
		plt.legend(loc='best',fontsize='xx-small')

	if plot_conparison==True:
		if 1 in which_plot:
			plt.figure(figure_index)
			datasection2 = data2
			# spectra = np.fft.fft(datasection2, axis=0)
			# # magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
			# magnitude2 = 2 * np.abs(spectra) / len(spectra)
			# phase2 = np.angle(spectra)
			# freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
			for i in range(len(poscentred)):
				pos = poscentred[i]
				spectra = np.fft.fft(np.mean(datasection2[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window],axis=(-1,-2)), axis=0)
				magnitude2 = 2 * np.abs(spectra) / len(spectra)
				freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
				# y = np.mean(magnitude2, axis=(-1, -2))
				y = magnitude2
				# y=magnitude[:,pos[0],pos[1]]
				y = np.array([y for _, y in sorted(zip(freq, y))])
				x = np.sort(freq)
				plt.plot(x, y, color[i] + '--',label='data at the point ' + str(pos) + ', oscillation substracted')
			# plt.title()


			# plt.grid()
			plt.semilogy()
			plt.xlim(left=0)
			plt.ylim(top=y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)].max()*2e3)
			plt.legend(loc='best',fontsize='xx-small')
		plt.pause(0.0001)




	print('stats of the oscillation removal')
	print('with window of size '+str(np.around(section_frames/framerate,decimals=5))+'s of '+str(len(data)/framerate)+'s of record')
	try:
		print('found oscillation of frequency '+str(frequencies_removed_all)+'Hz')
	except:
		print('no frequency needed removing')
	print('On the ROI oscillation magnitude reduced from %.5g[au] to %.5g[au]' %(peak_value_pre_filter,peak_value_post_filter)+'\nwith an approximate maximum noise of %.5g[au] and median of %.5g[au]' %(np.max(noise),np.median(noise)))

	if output_noise:
		return np.array([data2]),peak_value_pre_filter,peak_value_post_filter,np.max(noise),np.median(noise)
	else:
		return np.array([data2])

###################################################################################################


def abs_angle(angle,deg=False):
	import numpy as np

	out = angle
	if deg==False:
		if out<=0:
			out=angle+2*np.pi
	elif deg==True:
		if out<=0:
			out=angle+360

	return out



###################################################################################################



def split_fixed_length(A,length):
	A=np.array(A)
	length = int(length)
	B=[]
	while len(A)>length:
		B.append(A[:length].tolist())
		A=A[length:]
	if len(A)>0:
		B.append(A.tolist())
	return np.array(B)



###################################################################################################


# function to print on screen a whole array even if it is very large

def fullprint(*args, **kwargs):
	from pprint import pprint
	import numpy as np
	opt = np.get_printoptions()
	np.set_printoptions(threshold=np.inf)
	pprint(*args, **kwargs)
	np.set_printoptions(**opt)



###################################################################################################


def find_dead_pixels(data, start_interval='auto',end_interval='auto', framerate='auto',from_data=True,treshold_for_bad_low_std=0,treshold_for_bad_std=10,treshold_for_bad_difference=13,verbose=2):

	# Created 26/02/2019
	# function that finds the dead pixels. 'data' can be a string with the path of the record or the data itself.
	# if the path is given the oscillation filtering is done
	# default tresholds for std and difference found 25/02/2019

	# Output legend:
		# 3 = treshold for difference with neighbouring pixels trepassed
		# 6 = treshold for std trepassed
		# 9 = both treshold trepassed

	if from_data==False:
		if framerate=='auto':
			print('if you load from file you must specify the framerate')
			exit()
		path=data
		print('Loading a recod from '+path)
		filenames = all_file_names(path, '.npy')[0]
		data = np.load(os.path.join(path, filenames))
		data2 = clear_oscillation_central2(data, framerate, plot_conparison=False)
		data = np.array(data2)
	else:
		data = np.array(data)

	if end_interval=='auto':
		end_interval=np.shape(data)[1]
	else:
		if framerate=='auto':
			print('if you specify an end time you must specify the framerate')
			exit()
		else:
			end_interval=end_interval*framerate
			if (end_interval>np.shape(data)[1]):
				end_interval = np.shape(data)[1]

	if start_interval=='auto':
		start_interval=0
	else:
		if framerate=='auto':
			print('if you specify a start time you must specify the framerate')
			exit()
		else:
			start_interval=start_interval*framerate
			if (start_interval>end_interval):
				print("You set the start interval after the end one, I'm going to use zero seconds")
				start_interval = 0


	data=data[0,start_interval:end_interval]

	# in /home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000018
	# I found that some frames can be completely missing, filled with 0 or 64896
	# and this messes with calculating std and mean, so I need to remove this frames
	min_data = np.nanmin(data,axis=(-1,-2))
	# max_data = np.nanmin(data,axis=-1,-2)		# I don't want to use this because saturation could cause this too
	data = data[min_data>0]

	mean=np.mean(data,axis=(0))
	std=np.std(data,axis=(0))
	flag_check=np.zeros(np.shape(std))
	flag = np.ones(np.shape(std))
	# for i in range(np.shape(std)[0]):
	# 	for j in range(np.shape(std)[1]):
	# 		if std[i,j]>treshold_for_bad_std:
	# 			flag_check[i,j]=6
	# 			flag[i, j] = 0
	flag_check[std>treshold_for_bad_std] = 6
	flag[std>treshold_for_bad_std] = 0
	flag_check[std<=treshold_for_bad_low_std] = 6
	flag[std<=treshold_for_bad_low_std] = 0
	for repeats in [0, 1]:
		for i in range(1, np.shape(std)[0] - 1):
			for j in range(1, np.shape(std)[1] - 1):
				if flag_check[i,j] in [3,9]:
					continue
				temp = (mean[i - 1, j - 1:j + 2] * flag[i - 1, j - 1:j + 2]).tolist() + [
					(mean[i, j - 1] * flag[i, j - 1]).tolist()] + [(mean[i, j + 1] * flag[i, j + 1]).tolist()] + (
							   mean[i + 1, j - 1:j + 2] * flag[i + 1, j - 1:j + 2]).tolist()
				if len(temp) == 0:
					# flag_check[i, j] += 3
					# flag[i, j] = 0
					continue
				else:
					temp2 = [x for x in temp if x != 0]
					if len(temp2) == 0:
						# flag_check[i, j] += 3
						# flag[i, j] = 0
						continue
					else:
						if (mean[i, j] > max(temp2) + treshold_for_bad_difference or mean[i, j] < min(temp2) - treshold_for_bad_difference):
						# if np.abs(mean[i, j] - np.median(temp2))> treshold_for_bad_difference:
							flag_check[i, j] += 3
							flag[i, j] = 0
	# follows a slower version that checks also the edges
	# i_indexes = (np.ones_like(std).T*np.linspace(0,np.shape(std)[0]-1,np.shape(std)[0])).T
	# j_indexes = np.ones_like(std) * np.linspace(0, np.shape(std)[1] - 1, np.shape(std)[1])
	# # gna=np.zeros_like(std)
	# for repeats in [0, 1]:
	# 	for i in range(np.shape(std)[0] ):
	# 		for j in range(np.shape(std)[1] ):
	# 			if flag_check[i,j] in [3,9]:
	# 				continue
	# 			temp = mean[np.logical_and(flag , np.logical_and(np.abs(i_indexes-i)<=1 , np.logical_and( np.abs(j_indexes-j)<=1 , np.logical_or(i_indexes!=i , j_indexes!=j))))]
	# 			# gna[i,j] = np.std(temp)
	# 			if (mean[i, j] > max(temp) + treshold_for_bad_difference or mean[i, j] < min(temp) - treshold_for_bad_difference):
	# 				flag_check[i, j] += 3
	# 				flag[i, j] = 0


	if verbose>1:
		counter = collections.Counter(flatten_full(flag_check))
		print('Number of pixels that trepass '+str(treshold_for_bad_difference)+' counts difference with neighbours: '+str(counter.get(3)))
		print('Number of pixels with standard deviation > '+str(treshold_for_bad_std)+' counts: '+str(counter.get(6)))
		print('Number of pixels that trepass both limits: '+str(counter.get(9)))
	elif verbose>0:
		counter = collections.Counter(flatten_full(flag_check))
		print('diff>'+str(treshold_for_bad_difference)+': '+str(counter.get(3)) + ' , ' + 'std>' +str(treshold_for_bad_std)+': '+str(counter.get(6)) + ' , ' + 'both: '+str(counter.get(9)))
	return flag_check


###################################################################################################

def find_dead_pixels_data_acquisition_stage(data,treshold_for_bad_difference=30,verbose=2):
	from scipy.ndimage import median_filter
	import collections
	# Created 17/12/2021
	# function that finds the dead pixels looking at the 8 around it. it is used only to check if during the recording the order of the digitizer switches
	# if the path is given the oscillation filtering is done
	# default tresholds for std and difference found 25/02/2019

	data = np.array(data)
	median=median_filter(data,footprint=[[1,1,1],[1,0,1],[1,1,1]])
	flag_check = (np.abs(data-median)>treshold_for_bad_difference)*3

	if verbose>1:
		counter = collections.Counter(flag_check.flatten())
		print('Number of pixels that trepass '+str(treshold_for_bad_difference)+' counts difference with neighbours: '+str(counter.get(3)))
	elif verbose>0:
		counter = collections.Counter(flag_check.flatten())
		print('diff>'+str(treshold_for_bad_difference)+': '+str(counter.get(3)))
	return flag_check


###################################################################################################


def replace_dead_pixels(data,flag, framerate='auto',from_data=True):

	# Created 26/02/2019
	# function replace dead pixels with the average from their neighbours

	if from_data==False:
		if framerate=='auto':
			print('if you load from file you must specify the framerate')
			exit()
		path=data
		print('Loading a recod from '+path)
		filenames = all_file_names(path, '.npy')[0]
		data = np.load(os.path.join(path, filenames))
		data2 = clear_oscillation_central2(data, framerate, plot_conparison=False)
		data = np.array(data2)[0]
		del data2
	else:
		data = np.array(data)[0]

	flag=np.array(flag)

	if np.shape(data[0])!=np.shape(flag):
		print('There is something wrong, the shape of the data '+str(np.shape(data[0]))+' and the dead pixels map '+str(np.shape(flag))+' you want to use are not the same')
		bla=gna-sblu	# I want it to fail here, so I know where it did
		exit()

	data2=np.array([data])

	for i in range(np.shape(data)[-2]):
		for j in range(np.shape(data)[-1]):
			if flag[i,j]!=0:
				temp=[]
				for i_ in [-1,0,1]:
					for j_ in [-1, 0, 1]:
						if (i_!=0 and j_!=0):
							if flag[i+i_,j+j_]==0:
								temp.append(data[:,i+i_, j +j_])
				data2[0,:,i,j]=np.median(temp,axis=(0))

	return data2


###################################################################################################


def record_rotation(data,rotangle, foilcenter= [160, 133],	foilhorizw = 0.09, foilvertw = 0.07,foilhorizwpixel = 240,precisionincrease = 10):

	# Created 26/02/2019
	# function created to rotate and crop a whole record given central point and rotation angle

	mean = data[0,0]

	testrot = mean
	foilrot = rotangle * 2 * np.pi / 360
	foilrotdeg = rotangle
	foilvertwpixel = np.int(np.around((foilhorizwpixel * foilvertw) / foilhorizw))
	r = ((foilhorizwpixel ** 2 + foilvertwpixel ** 2) ** 0.5) / 2  # HALF DIAGONAL
	a = foilvertwpixel / np.cos(foilrot)
	tgalpha = np.tan(foilrot)
	delta = -(a ** 2) / 4 + (1 + tgalpha ** 2) * (r ** 2)
	foilx = np.add(foilcenter[0], [(-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
								   (-0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
								   (0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
								   (0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
								   (-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2)])
	foily = np.add(foilcenter[1] - tgalpha * foilcenter[0],
				   [tgalpha * foilx[0] + a / 2, tgalpha * foilx[1] + a / 2, tgalpha * foilx[2] - a / 2,
					tgalpha * foilx[3] - a / 2, tgalpha * foilx[0] + a / 2])
	foilxint = (np.rint(foilx)).astype(int)
	foilyint = (np.rint(foily)).astype(int)


	dummy = np.ones(np.multiply(np.shape(testrot), precisionincrease))
	dummy[foilcenter[1] * precisionincrease, foilcenter[0] * precisionincrease] = 2
	dummy[int(foily[0] * precisionincrease), int(foilx[0] * precisionincrease)] = 3
	dummy[int(foily[1] * precisionincrease), int(foilx[1] * precisionincrease)] = 4
	dummy[int(foily[2] * precisionincrease), int(foilx[2] * precisionincrease)] = 5
	dummy[int(foily[3] * precisionincrease), int(foilx[3] * precisionincrease)] = 6
	dummy2 = rotate(dummy, foilrotdeg, axes=(-1, -2), order=0)
	foilcenterrot = (
		np.rint(
			[np.where(dummy2 == 2)[1][0] / precisionincrease, np.where(dummy2 == 2)[0][0] / precisionincrease])).astype(
		int)
	foilxrot = (
		np.rint([np.where(dummy2 == 3)[1][0] / precisionincrease, np.where(dummy2 == 4)[1][0] / precisionincrease,
				 np.where(dummy2 == 5)[1][0] / precisionincrease, np.where(dummy2 == 6)[1][0] / precisionincrease,
				 np.where(dummy2 == 3)[1][0] / precisionincrease])).astype(int)
	foilyrot = (
		np.rint([np.where(dummy2 == 3)[0][0] / precisionincrease, np.where(dummy2 == 4)[0][0] / precisionincrease,
				 np.where(dummy2 == 5)[0][0] / precisionincrease, np.where(dummy2 == 6)[0][0] / precisionincrease,
				 np.where(dummy2 == 3)[0][0] / precisionincrease])).astype(int)

	foillx = min(foilxrot)
	foilrx = max(foilxrot)
	foilhorizwpixel = foilrx - foillx
	foildw = min(foilyrot)
	foilup = max(foilyrot)
	foilvertwpixel = foilup - foildw

	datarot = rotate(data, foilrotdeg, axes=(-1, -2))
	datacrop = datarot[:, :, foildw:foilup, foillx:foilrx]

	return datacrop


#####################################################################################################

def sensitivities_matrix_averaging_foil_pixels(sensitivities,h_pixels,pixelmean_h,pixelmean_v):

	# Created 27/02/2019
	# Does the average of a homogeneous sensitivity matrix on foil pixels
	# It can return a non homogeneous foil, and loose resolution due to the residual of h_pixels/pixelmean_h and v_pixels/pixelmean_v
	#
	#


	shapeorig=np.shape(sensitivities)
	npixels=shapeorig[0]
	nvoxels=shapeorig[1]

	v_pixels=npixels//h_pixels

	h_end_pixels = h_pixels//pixelmean_h
	v_end_pixels = v_pixels // pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('the loss of pixels is (vertical,horizontal)'+str([v_pixels-v_end_pixels*pixelmean_v,h_pixels-h_end_pixels*pixelmean_h]))

	h_shift = (h_pixels-h_end_pixels*pixelmean_h)//2
	v_shift = (v_pixels-v_end_pixels*pixelmean_v)//2

	sensitivities_averaged=[]
	for voxel in range(nvoxels):
		voxel_sensitivity=[]
		foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
		for i in range(v_end_pixels):
			for j in range(h_end_pixels):
				voxel_sensitivity.append(np.mean(foil_image[v_shift+i*pixelmean_v:v_shift+(i+1)*pixelmean_v,h_shift+j*pixelmean_h:h_shift+(j+1)*pixelmean_h]))
		sensitivities_averaged.append(voxel_sensitivity)

	sensitivities_averaged=np.array(sensitivities_averaged)

	return sensitivities_averaged.T


#####################################################################################################

def sensitivities_matrix_averaging_foil_pixels_extra(sensitivities,h_pixels,pixelmean_h):

	# Created 27/02/2019
	# Does the average of a homogeneous sensitivity matrix on foil pixels
	# It can loose resolution due to the residual of h_pixels/pixelmean_h and v_pixels/pixelmean_v
	#
	# Additionally from sensitivities_matrix_averaging_foil_pixels this function calculate also the pixels obtained from shifting the area of the average, one by one
	# the additional constrain is that the foil grid will be homogeneous


	pixelmean_v=pixelmean_h
	shapeorig=np.shape(sensitivities)
	npixels=shapeorig[0]
	nvoxels=shapeorig[1]

	v_pixels=npixels//h_pixels

	h_end_pixels = h_pixels//pixelmean_h
	v_end_pixels = v_pixels // pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('the loss of pixels is (vertical,horizontal)' + str([v_pixels - v_end_pixels * pixelmean_v, h_pixels - h_end_pixels * pixelmean_h]))
	print('but the sensityvity matrix generated will be sized '+str([nvoxels,v_end_pixels*h_end_pixels+(v_end_pixels-1)*(h_end_pixels-1)*(pixelmean_h-1)]))
	print('instead of (vertical,horizontal)'+str([nvoxels,v_end_pixels*h_end_pixels]))

	h_shift = (h_pixels-h_end_pixels*pixelmean_h)//2
	v_shift = (v_pixels-v_end_pixels*pixelmean_v)//2


	sensitivities_averaged=[]
	for voxel in range(nvoxels):
		voxel_sensitivity=[]
		foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
		for i in range(v_end_pixels):
			for j in range(h_end_pixels):
				voxel_sensitivity.append(np.mean(foil_image[v_shift+i*pixelmean_v:v_shift+(i+1)*pixelmean_v,h_shift+j*pixelmean_h:h_shift+(j+1)*pixelmean_h]))
		sensitivities_averaged.append(voxel_sensitivity)
	intermediate=(np.array(sensitivities_averaged).T).tolist()

	print('check 1')
	print('intermediate matrix size is ' + str(np.shape(intermediate)))


	for extra_shift in range(1,pixelmean_h,1):
		sensitivities_averaged = []
		for voxel in range(nvoxels):
			voxel_sensitivity=[]
			foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
			for i in range(v_end_pixels-1):
				for j in range(h_end_pixels-1):
					voxel_sensitivity.append(np.mean(foil_image[extra_shift+v_shift+i*pixelmean_v:extra_shift+v_shift+(i+1)*pixelmean_v,extra_shift+h_shift+j*pixelmean_h:extra_shift+h_shift+(j+1)*pixelmean_h]))
			sensitivities_averaged.append(voxel_sensitivity)
		print('check ' + str(extra_shift + 1))
		print('sensitivities_averaged matrix size is ' + str(np.shape(sensitivities_averaged)))
		intermediate.extend((np.array(sensitivities_averaged).T).tolist())

	intermediate=np.array(intermediate)

	print('sensitivity matrix size is '+str(np.shape(intermediate)))

	return intermediate

#####################################################################################################

def foil_measurement_averaging_foil_pixels_extra(d,h_pixels,pixelmean_h):

	# Created 27/02/2019
	# Does the average of a homogeneous sensitivity matrix on foil pixels
	# It can loose resolution due to the residual of h_pixels/pixelmean_h and v_pixels/pixelmean_v
	#
	# Additionally, this function calculate also the pixels obtained from shifting the area of the average, one by one
	# the additional constrain is that the grid foil will be homogeneous




	pixelmean_v=pixelmean_h
	shapeorig=np.shape(d)
	npixels=shapeorig[0]

	v_pixels=npixels//h_pixels

	h_end_pixels = h_pixels//pixelmean_h
	v_end_pixels = v_pixels // pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('but the foil reading matrix generated will be sized '+str([1,v_end_pixels*h_end_pixels+(v_end_pixels-1)*(h_end_pixels-1)*(pixelmean_h-1)]))
	print('the loss of pixels is (vertical,horizontal)'+str([v_pixels-v_end_pixels*pixelmean_v,h_pixels-h_end_pixels*pixelmean_h]))

	h_shift = (h_pixels-h_end_pixels*pixelmean_h)//2
	v_shift = (v_pixels-v_end_pixels*pixelmean_v)//2

	foil_image = np.array(split_fixed_length(d, h_pixels))

	averaged_foil_reading=[]
	for i in range(v_end_pixels):
		for j in range(h_end_pixels):
			averaged_foil_reading.append(np.mean(foil_image[v_shift+i*pixelmean_v:v_shift+(i+1)*pixelmean_v,h_shift+j*pixelmean_h:h_shift+(j+1)*pixelmean_h]))



	for extra_shift in range(1,pixelmean_h,1):
		for i in range(v_end_pixels-1):
			for j in range(h_end_pixels-1):
				averaged_foil_reading.append(np.mean(foil_image[extra_shift+v_shift+i*pixelmean_v:extra_shift+v_shift+(i+1)*pixelmean_v,extra_shift+h_shift+j*pixelmean_h:extra_shift+h_shift+(j+1)*pixelmean_h]))

	averaged_foil_reading=np.array(averaged_foil_reading)

	print('power on the foil array size is '+str(np.shape(averaged_foil_reading)))

	return averaged_foil_reading



#####################################################################################################

def sensitivities_matrix_averaging_foil_pixels_loseless(sensitivities,h_pixels,pixelmean_h,pixelmean_v):

	# Created 28/03/2019
	# Does the average of a homogeneous sensitivity matrix on foil pixels
	# Different from sensitivities_matrix_averaging_foil_pixels because I do it loseless. I do not discard the last pixels
	#
	#


	shapeorig=np.shape(sensitivities)
	npixels=shapeorig[0]
	nvoxels=shapeorig[1]

	v_pixels=npixels//h_pixels

	h_end_pixels = np.ceil(h_pixels/pixelmean_h).astype('int')
	v_end_pixels = np.ceil(v_pixels/pixelmean_v).astype('int')
	pixels_averaged = pixelmean_h*pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))

	sensitivities_averaged=[]
	for voxel in range(nvoxels):
		voxel_sensitivity=[]
		foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
		for i in range(v_end_pixels):
			for j in range(h_end_pixels):
				voxel_sensitivity.append(np.sum(foil_image[i*pixelmean_v:(i+1)*pixelmean_v,j*pixelmean_h:(j+1)*pixelmean_h])/pixels_averaged)
		sensitivities_averaged.append(voxel_sensitivity)

	sensitivities_averaged=np.array(sensitivities_averaged)


	return sensitivities_averaged.T


#####################################################################################################

def foil_measurement_averaging_foil_pixels_loseless(d,h_pixels,pixelmean_h,pixelmean_v):

	# Created 01/04/2019
	# Does the average of a regular sensitivity matrix on foil pixels
	#


	shapeorig=np.shape(d)
	npixels=shapeorig[0]

	v_pixels=npixels//h_pixels

	h_end_pixels = np.ceil(h_pixels/pixelmean_h).astype('int')
	v_end_pixels = np.ceil(v_pixels/pixelmean_v).astype('int')
	pixels_averaged = pixelmean_h*pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))

	foil_image = np.array(split_fixed_length(d, h_pixels))

	averaged_foil_reading=[]
	for i in range(v_end_pixels):
		for j in range(h_end_pixels):
			averaged_foil_reading.append(np.sum(foil_image[i*pixelmean_v:(i+1)*pixelmean_v,j*pixelmean_h:(j+1)*pixelmean_h])/pixels_averaged)


	averaged_foil_reading=np.array(averaged_foil_reading)

	print('power on the foil array size is '+str(np.shape(averaged_foil_reading)))

	return averaged_foil_reading


#####################################################################################################

def sensitivities_matrix_averaging_foil_pixels_extra_loseless(sensitivities,h_pixels,pixelmean_h,pixelmean_v):

	# Created 30/03/2019
	# Does the average of a regular sensitivity matrix on foil pixels
	# Additionally from sensitivities_matrix_averaging_foil_pixels this function calculate also the pixels obtained from shifting the area of the average, one by one
	#
	# Different from sensitivities_matrix_averaging_foil_pixels_extra because I do it loseless. I do not discard the last pixels
	#
	#


	shapeorig=np.shape(sensitivities)
	npixels=shapeorig[0]
	nvoxels=shapeorig[1]

	v_pixels=npixels//h_pixels

	h_end_pixels = np.ceil(h_pixels/pixelmean_h).astype('int')
	v_end_pixels = np.ceil(v_pixels/pixelmean_v).astype('int')
	pixels_averaged = pixelmean_h*pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('but the sensityvity matrix generated will be sized '+str([nvoxels,v_end_pixels*h_end_pixels+(v_end_pixels-1)*(h_end_pixels-1)*(pixelmean_h-1)*(pixelmean_v-1)]))
	print('instead of (vertical,horizontal)'+str([nvoxels,v_end_pixels*h_end_pixels]))


	sensitivities_averaged=[]
	for voxel in range(nvoxels):
		voxel_sensitivity=[]
		foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
		for i in range(v_end_pixels):
			for j in range(h_end_pixels):
				voxel_sensitivity.append(np.sum(foil_image[i*pixelmean_v:(i+1)*pixelmean_v,j*pixelmean_h:(j+1)*pixelmean_h])/pixels_averaged)
		sensitivities_averaged.append(voxel_sensitivity)
	intermediate=(np.array(sensitivities_averaged).T).tolist()

	print('check 1')
	print('intermediate matrix size is ' + str(np.shape(intermediate)))

	for extra_v_shift in range(0, pixelmean_v, 1):
		for extra_h_shift in range(0,pixelmean_h,1):
			if (extra_v_shift+extra_v_shift)==0:
				continue
			sensitivities_averaged = []
			for voxel in range(nvoxels):
				voxel_sensitivity=[]
				foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
				if extra_v_shift==0:
					for i in range(v_end_pixels):
						for j in range(h_end_pixels-1):
							voxel_sensitivity.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
				elif extra_h_shift == 0:
					for i in range(v_end_pixels-1):
						for j in range(h_end_pixels):
							voxel_sensitivity.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
				else:
					for i in range(v_end_pixels-1):
						for j in range(h_end_pixels-1):
							voxel_sensitivity.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
				sensitivities_averaged.append(voxel_sensitivity)
			print('check ' + str(extra_v_shift + extra_h_shift + 1))
			print('sensitivities_averaged matrix size is ' + str(np.shape(sensitivities_averaged)))
			intermediate.extend((np.array(sensitivities_averaged).T).tolist())

	intermediate=np.array(intermediate)

	print('sensitivity matrix size is '+str(np.shape(intermediate)))

	return intermediate

#####################################################################################################

def foil_measurement_averaging_foil_pixels_extra_loseless(d,h_pixels,pixelmean_h,pixelmean_v):

	# Created 30/03/2019
	# Does the average of a rgular sensitivity matrix on foil pixels
	#
	# Additionally, this function calculate also the pixels obtained from shifting the area of the average, one by one



	shapeorig=np.shape(d)
	npixels=shapeorig[0]

	v_pixels=npixels//h_pixels

	h_end_pixels = np.ceil(h_pixels/pixelmean_h).astype('int')
	v_end_pixels = np.ceil(v_pixels/pixelmean_v).astype('int')
	pixels_averaged = pixelmean_h*pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('but the foil reading matrix generated will be sized '+str([1,v_end_pixels*h_end_pixels+(v_end_pixels-1)*(h_end_pixels-1)*(pixelmean_h-1)*(pixelmean_v-1)]))

	foil_image = np.array(split_fixed_length(d, h_pixels))

	averaged_foil_reading=[]
	for i in range(v_end_pixels):
		for j in range(h_end_pixels):
			averaged_foil_reading.append(np.sum(foil_image[i*pixelmean_v:(i+1)*pixelmean_v,j*pixelmean_h:(j+1)*pixelmean_h])/pixels_averaged)

	for extra_v_shift in range(0, pixelmean_v, 1):
		for extra_h_shift in range(0,pixelmean_h,1):
			if (extra_v_shift+extra_v_shift)==0:
				continue
			if extra_v_shift == 0:
				for i in range(v_end_pixels):
					for j in range(h_end_pixels-1):
						averaged_foil_reading.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
			elif extra_h_shift == 0:
				for i in range(v_end_pixels-1):
					for j in range(h_end_pixels):
						averaged_foil_reading.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
			else:
				for i in range(v_end_pixels-1):
					for j in range(h_end_pixels-1):
						averaged_foil_reading.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)

	averaged_foil_reading=np.array(averaged_foil_reading)

	print('power on the foil array size is '+str(np.shape(averaged_foil_reading)))

	return averaged_foil_reading




#####################################################################################################

def record_binning(data,time_averaging,spatial_averaging_h,spatial_averaging_v,flag,plot_info=False):

	if np.shape(flag) != np.shape(data[0,0]):
		print('ERROR\nThe shape of the flags and record is not equal\nflag is ('+str(np.shape(flag))+') while record is ('+str(np.shape(data))+')')
		exit()

	pixel_t, pixel_h, pixel_v = np.shape(data[0])
	pixel_h_new = np.floor(pixel_h / spatial_averaging_h).astype('int')
	pixel_v_new = np.floor(pixel_v / spatial_averaging_v).astype('int')
	pixel_t_new = np.floor(pixel_t / time_averaging).astype('int')
	print('record will be binned to (txhxv) ' + str(pixel_t_new) + 'x' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')

	# if (time_averaging==1 and spatial_averaging_h==1 and spatial_averaging_v==1) :
	# 	return data

	binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging_h).astype('int')
	binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
	binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging_v).astype('int')
	binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
	binning_t = np.abs(np.linspace(0, pixel_t - 1, pixel_t) // time_averaging).astype('int')
	binning_t[binning_t > pixel_t_new - 1] = pixel_t_new - 1
	# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
	pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

	# temp = np.zeros((len(data[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
	temp = np.zeros((len(data[0]), np.max(pixels_to_bin) + 1))
	for bin_index in range(np.max(pixels_to_bin) + 1):
		temp[:,bin_index] = np.mean(data[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
	# data_to_check_not_binned = copy.deepcopy(data)
	data_spatially_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))


	pixels_to_bin = binning_t.astype('int')

	pixels_to_bin = binning_t.astype('int')
	temp = np.zeros((np.max(pixels_to_bin) + 1, *np.shape(data_spatially_binned[0])))
	for bin_index in range(np.max(pixels_to_bin) + 1):
		temp[bin_index] = np.mean(data_spatially_binned[pixels_to_bin == bin_index], axis=0)
	# data_to_check_not_binned = copy.deepcopy(data_to_check)
	# data_binned = cp.deepcopy(temp)

	return np.array([temp])



###################################################################################################


def replace_dead_pixels_2(data_, framerate='auto',from_data=True):

	import copy as cp
	# Created 14/10/2019
	# function replace dead pixels with the average from their neighbours
	# different from# replace_dead_pixels because here you don't use a flag, but the dead pixels are the one that are nan

	# if from_data==False:
	# 	if framerate=='auto':
	# 		print('if you load from file you must specify the framerate')
	# 		exit()
	# 	path=data_
	# 	print('Loading a recod from '+path)
	# 	filenames = all_file_names(path, '.npy')[0]
	# 	data_ = np.load(os.path.join(path, filenames))
	# 	data2 = clear_oscillation_central2(data_, framerate, plot_conparison=False)
	# 	data_ = np.array(data2)
	# else:
	# 	data_ = np.array(data_)


	if np.sum(np.isnan(data_))==0:
		print('there are no "not a number" pixels')
		return data_
	else:
		pixel_t, pixel_h, pixel_v = np.shape(data_[0])
		while  np.sum(np.isnan(data_))!=0:
			sample = cp.deepcopy(data_[0,0])
			while np.sum(np.isnan(sample))!=0:
				where_is = np.isnan(sample).argmax()
				i = int(np.floor(where_is/pixel_v))
				j = int(where_is-i*pixel_v)
				positions = np.array([[i-1,j-1],[i,j-1],[i+1,j-1],[i-1,j],[i,j],[i+1,j],[i-1,j+1],[i,j+1],[i+1,j+1]]).astype('int')
				temp = []
				for index in range(len(positions)):
					if (positions[index][0]>=0 and positions[index][1]>=0 and positions[index,0]<pixel_h and positions[index,1]<pixel_v) :
						# print(str(positions[index])+' is good')
						temp.append(data_[0,:,positions[index][0],positions[index][1]])
				sample[i, j] = 0
				if np.sum(np.isfinite(temp))>0:
					# print('fixing ' + str([i, j]))
					data_[0,:,i,j] = np.nanmean(temp,axis=(-1,-2))
		return data_


###################################################################################################


def log_log_fit_derivative(x,y):
	# 22/10/2019 calculates the derivative of a function by fitting its log-log plot and coming back to the original coordinates
	import numpy as np
	from scipy.interpolate import splrep, splev

	x=np.array(x)
	y = np.array(y)

	x_hat = np.log(x)
	y_hat = np.log(y)

	f_hat = splrep(x_hat,y_hat,s=0)

	f_derivative = np.exp(y_hat)*splev(x_hat,f_hat,der=1)*(1/x)

	return f_derivative

####################################################################################################

# set of functions that allow to extract all information I need directly from the .ats file of the FLIR camera

def hex8_to_int(hex):
	temp = hex[-2:]+hex[4:6]+hex[2:4]+hex[0:2]
	return int(temp,16)

def hex16_to_int(hex):
	temp = hex[-2:]+hex[-4:-2]+hex[-6:-4]+hex[-8:-6]+hex[-10:-8]+hex[-12:-10]+hex[-14:-12]+hex[-16:-14]
	return int(temp,16)

def hex4_to_int(hex):
	temp = hex[-2:]+hex[0:2]
	return int(temp,16)

def hex8_to_float(hex):
	import struct
	temp = hex[-2:]+hex[4:6]+hex[2:4]+hex[0:2]
	return struct.unpack('!f', bytes.fromhex(temp))[0]

def FLIR_record_header_decomposition(header):
	# There is certainly a field for the filter used, but I don't have files with different filters on to compare.
	camera_type = bytearray.fromhex(header[678:724]).decode()
	width = hex4_to_int(header[814:818])
	height = hex4_to_int(header[818:822])
	camera_SN = int(bytearray.fromhex(header[1034:1050]).decode())
	lens = bytearray.fromhex(header[1066:1074]).decode()
	return dict([('camera_type',camera_type),('width',width),('height',height),('camera_SN',camera_SN),('lens',lens)])

def FLIR_frame_header_decomposition(header):
	def return_requested_output(request):
		if request=='time':
			return hex16_to_int(header[16:32])	# microseconds
		elif request=='frame_counter':
			return hex8_to_int(header[32:40])
		elif request=='NUCpresetUsed':
			return hex4_to_int(header[60:64])	# ID
		elif request=='DetectorTemp':
			return hex8_to_float(header[64:72])	# K
		elif request=='SensorTemp_0':
			return hex8_to_float(header[72:80])	# K
		elif request=='SensorTemp_1':
			return hex8_to_float(header[80:88])	# K
		elif request=='SensorTemp_2':
			return hex8_to_float(header[88:96])	# K
		elif request=='SensorTemp_3':
			return hex8_to_float(header[96:104])	# K
		elif request=='MasterClock':
			return hex8_to_float(header[104:112])	# unknown
		elif request=='IntegrationTime':
			return hex8_to_float(header[112:120])	# ns
		elif request=='FrameRate':
			return hex8_to_float(header[120:128])	# Hz
		elif request=='ExternalTrigger':
			return int(header[130:132])
		elif request=='PageIndex':
			# return hex8_to_int(header[134:142])	# index of the digitiser used # wrong, this was valod only in external clocks
			return int(header[134:136])	# index of the digitiser used
		elif request=='ClockType':
			return int(header[140:142])	# 1=internal clock, 0=external clock
	return return_requested_output



def raw_to_image(raw_digital_level,width,height,digital_level_bytes):
	import textwrap
	pixels = width*height
	# raw_digital_level_splitted = textwrap.wrap(raw_digital_level, 4)
	# iterator=map(hex4_to_int,raw_digital_level_splitted)
	# return np.flip(np.array(list(iterator)).reshape((height,width)),axis=0)
	counts_digital_level = []
	for i in range(pixels):
		counts_digital_level.append(hex4_to_int(raw_digital_level[i*digital_level_bytes:(i+1)*digital_level_bytes]))
	return np.flip(np.array(counts_digital_level).reshape((height,width)),axis=0)


def ats_to_dict(full_path,digital_level_bytes=4,header_marker = '4949'):
	data = open(full_path,'rb').read()
	hexdata = data.hex()
	# raw_for_digitizer = b'\x18K\x00\x00zD\x00\x00 A\x00\x00\x00'
	# header_marker = '4949'
	length_header_marker = len(header_marker)
	header_length = 142
	# raw_pointer_after_string = 11 + len(hex_for_digitizer)
	hex_pointer_after_string = 15 + length_header_marker
	header = FLIR_record_header_decomposition(hexdata)
	width = header['width']
	height = header['height']
	camera_SN = header['camera_SN']
	# digital_level_bytes = 4
	data_length = width*height*digital_level_bytes
	digitizer_ID = []
	data = []
	time_of_measurement = []
	frame_counter = []
	DetectorTemp = []
	SensorTemp_0 = []
	SensorTemp_3 = []
	# last = 0
	present_header_marker_position = hexdata.find(header_marker)
	next_header_marker_position = hexdata[present_header_marker_position+length_header_marker+header_length:].find(header_marker)+header_length
	# import time as tm
	# value = data.find(string_for_digitizer)
	# value = hexdata.find(header_marker)
	# last+=value+len(hex_for_digitizer)	# the first one is part of the neader of the whole file
	# value = hexdata[last:].find(hex_for_digitizer)
	# while len(hexdata)-last>header_length:
	while True:
		# start_time = tm.time()
		# print(len(time_of_measurement))
		# header = hexdata[last+value+length_header_marker:last+value+header_length+length_header_marker]
		if next_header_marker_position!=header_length-1:
			header_length = next_header_marker_position-data_length
		header = hexdata[present_header_marker_position+length_header_marker:present_header_marker_position+length_header_marker+header_length]
		# print(header[134:136])
		header = FLIR_frame_header_decomposition(header)
		digitizer_ID.append(header('PageIndex'))
		time_of_measurement.append(header('time'))	# time in microseconds from camera startup
		frame_counter.append(header('frame_counter'))
		DetectorTemp.append(header('DetectorTemp'))
		SensorTemp_0.append(header('SensorTemp_0'))
		SensorTemp_3.append(header('SensorTemp_3'))
		# time_lapsed = tm.time()-start_time
		# print(time_lapsed)
		# raw_digital_level = hexdata[last+value-data_length:last+value]
		raw_digital_level = hexdata[present_header_marker_position-data_length:present_header_marker_position]
		# time_lapsed = tm.time()-start_time-time_lapsed
		# print(time_lapsed)
		data.append(raw_to_image(raw_digital_level,width,height,digital_level_bytes))
		# time_lapsed = tm.time()-start_time-time_lapsed
		# print(time_lapsed)
		# last+=value+header_length+data_length
		if next_header_marker_position==header_length-1:
			break
		present_header_marker_position += next_header_marker_position + length_header_marker
		next_header_marker_position = hexdata[present_header_marker_position+length_header_marker+header_length:present_header_marker_position+length_header_marker+header_length+data_length*2].find(header_marker)+header_length
		if len(time_of_measurement)<=1:	# the spacing between separators seems constant, and take very long, so I do it once
			# value = hexdata[last:].find(header_marker)
			IntegrationTime = header('IntegrationTime')
			FrameRate = header('FrameRate')
			ExternalTrigger = header('ExternalTrigger')
			NUCpresetUsed = header('NUCpresetUsed')
		# print(str(present_header_marker_position)+' / '+str(next_header_marker_position))
		# print(value)
	data = np.array(data)
	data_median = int(np.median(data))
	if np.abs(data-data_median).max()<2**8/2-1:
		data_minus_median = (data-data_median).astype(np.int8)
	elif np.abs(data-data_median).max()<2**16/2-1:
		data_minus_median = (data-data_median).astype(np.int16)
	elif np.abs(data-data_median).max()<2**32/2-1:
		data_minus_median = (data-data_median).astype(np.int32)
	digitizer_ID = np.array(digitizer_ID)
	time_of_measurement = np.array(time_of_measurement)
	frame_counter = np.array(frame_counter)
	DetectorTemp = np.array(DetectorTemp)
	SensorTemp_0 = np.array(SensorTemp_0)
	SensorTemp_3 = np.array(SensorTemp_3)
	out = dict([])
	# out['data'] = data
	out['data_median'] = data_median
	out['data'] = data_minus_median	# I do this to save memory, also because the change in counts in a single recording is always small
	out['digitizer_ID']=digitizer_ID
	out['time_of_measurement']=time_of_measurement
	out['IntegrationTime']=IntegrationTime
	out['FrameRate']=FrameRate
	out['ExternalTrigger'] = ExternalTrigger
	out['NUCpresetUsed'] = NUCpresetUsed
	out['SensorTemp_0'] = SensorTemp_0
	out['SensorTemp_3'] = SensorTemp_3
	out['DetectorTemp'] = DetectorTemp
	out['width'] = width
	out['height'] = height
	out['camera_SN'] = camera_SN
	out['frame_counter'] = frame_counter
	data_per_digitizer,uniques_digitizer_ID = separate_data_with_digitizer(out)
	out['data_time_avg_counts'] = np.array([(np.mean(data,axis=0)) for data in data_per_digitizer])
	out['data_time_avg_counts_std'] = np.array([(np.std(data,axis=0)) for data in data_per_digitizer])
	out['data_time_space_avg_counts'] = np.array([(np.mean(data,axis=(0,1,2))) for data in data_per_digitizer])
	out['data_time_space_avg_counts_std'] = np.array([(np.std(data,axis=(0,1,2))) for data in data_per_digitizer])
	out['uniques_digitizer_ID'] = uniques_digitizer_ID
	# return data,digitizer_ID,time_of_measurement,IntegrationTime,FrameRate,ExternalTrigger,SensorTemp_0,DetectorTemp,width,height,camera_SN,frame_counter
	return out

def ptw_to_dict(full_path,max_time_s = np.inf):
	os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
	import pyradi.ryptw as ryptw
	import datetime
	from scipy.ndimage import median_filter
	import collections

	header = ryptw.readPTWHeader(full_path)
	width = header.h_Cols
	height = header.h_Rows
	camera_SN = header.h_CameraSerialNumber
	NUCpresetUsed = header.h_NucTable
	FrameRate = 1/header.h_CEDIPAquisitionPeriod # Hz
	IntegrationTime = round(header.h_CEDIPIntegrationTime*1e6,0) # microseconds
	ExternalTrigger = None	# I couldn't find this signal in the header

	digitizer_ID_fake = []
	data = []
	time_of_measurement = []
	frame_counter = []
	DetectorTemp = []
	SensorTemp_0 = []
	SensorTemp_3 = []
	AtmosphereTemp = []
	last = 0
	for i in range(header.h_firstframe,header.h_lastframe+1):
		frame = ryptw.getPTWFrames(header, [i])[0][0]
		if i == header.h_firstframe:
			new_shape = np.shape(frame.T)
		frame_header = ryptw.getPTWFrames(header, [i])[1][0]
		# # 2021-12-09 I checked a few recordings and this seems to actually be true, the "dead" pixels flip in a very regular and consistent manner
		# yyyy = frame_header.h_FileSaveYear
		# mm = frame_header.h_FileSaveMonth
		# dd = frame_header.h_FileSaveDay
		hh = frame_header.h_frameHour
		minutes = frame_header.h_frameMinute
		ss_sss = frame_header.h_frameSecond
		temp = datetime.datetime(frame_header.h_FileSaveYear-30,frame_header.h_FileSaveMonth,frame_header.h_FileSaveDay).timestamp() + hh*60*60 + minutes*60 + ss_sss
		if len(time_of_measurement)>=1:
			if temp-time_of_measurement[0]>max_time_s:
				continue
		data.append(np.flip(frame.flatten().reshape(new_shape).T,axis=0))
		digitizer_ID_fake.append(frame_header.h_framepointer%2)	# I couldn't find this so as a proxy, given the digitisers are always alternated, I only use if the frame is even or odd
		time_of_measurement.append(temp*1e6)	# I leave if as a true timestamp, t=0 is 1970
		frame_counter.append(None)	# I couldn't fine this in the header
		DetectorTemp.append(frame_header.h_detectorTemp)
		SensorTemp_0.append(frame_header.h_detectorTemp)	# this data is missing so I use the more similar again
		SensorTemp_3.append(frame_header.h_sensorTemp4)
		AtmosphereTemp.append(frame_header.h_AtmosphereTemp)	# additional data not present in the .ats format
	#	 section added to actually find the digitizer
	if FrameRate>60:	# I can't see dead pixels in this configuration
		bad_pixels_marker = []
		for i in range(int(min(1*FrameRate,len(data)/5))):
			bad_pixels_flag = find_dead_pixels_data_acquisition_stage(data[i]-data[i+1],treshold_for_bad_difference=50,verbose=1).flatten()
			bad_pixels_marker.extend(np.arange(len(bad_pixels_flag))[bad_pixels_flag>0])
		bad_pixels_marker = np.unique(bad_pixels_marker)
		shape = np.shape(data[0])
		bad_pixels_marker_2 = []
		bad_pixels_marker_str = []
		for i in range(int(min(1*FrameRate,len(data)/5))):
			temp = []
			median = median_filter(data[i],footprint=[[1,1,1],[1,0,1],[1,1,1]])
			for i_ in bad_pixels_marker:
				i__ = np.unravel_index(i_,shape)
				temp.append(np.abs(data[i][i__]-median[i__]))
			bad_pixels_marker_2.append(bad_pixels_marker[np.array(temp)>25])
			bad_pixels_marker_str.append(str(bad_pixels_marker_2[-1]))
		# bad_pixels_marker_2 = np.array(bad_pixels_marker_2)
		counter = collections.Counter(bad_pixels_marker_str)
		most_common_str = [counter.most_common()[0][0],counter.most_common()[1][0]]
		dark_bad_pixels_marker = [bad_pixels_marker_2[(np.array(bad_pixels_marker_str)==most_common_str[0]).argmax()],bad_pixels_marker_2[(np.array(bad_pixels_marker_str)==most_common_str[1]).argmax()]]
		temp=0
		while len(dark_bad_pixels_marker[0])*0.8 < np.sum([value in dark_bad_pixels_marker[1] for value in dark_bad_pixels_marker[0]]):	# safety clause in case by mistake it is used twice the same pattern of dewad pixels for both digitizers
			most_common_str = [counter.most_common()[0][0],counter.most_common()[temp+1+1][0]]
			dark_bad_pixels_marker = [bad_pixels_marker_2[(np.array(bad_pixels_marker_str)==most_common_str[0]).argmax()],bad_pixels_marker_2[(np.array(bad_pixels_marker_str)==most_common_str[1]).argmax()]]
			temp+=1
			print('pattern of dead pixels shifted of '+str(temp))
		print('dead pixel markers:\n'+str(dark_bad_pixels_marker[0])+'\n'+str(dark_bad_pixels_marker[1]))
		digitizer_ID = []
		discarded_frames = []
		for i in range(len(data)):
			median = median_filter(data[i],footprint=[[1,1,1],[1,0,1],[1,1,1]])
			bad_pixels_marker = [[] , []]
			for i_ in range(len(dark_bad_pixels_marker)):
				for i__ in dark_bad_pixels_marker[i_]:
					i__ = np.unravel_index(i__,shape)
					bad_pixels_marker[i_].append(np.abs(data[i][i__]-median[i__]))
			for i_ in range(len(dark_bad_pixels_marker)):
				bad_pixels_marker[i_] = np.mean(bad_pixels_marker[i_])
			if bad_pixels_marker[0]>bad_pixels_marker[1]:
				digitizer_ID.append(0)
			elif bad_pixels_marker[1]>bad_pixels_marker[0]:
				digitizer_ID.append(1)
			else:
				print('frame n'+str(i)+' discarded')
				discarded_frames.append(i)
		if len(discarded_frames)>0:
			print('discarded frames are '+str(discarded_frames))
			if np.sum(np.array(discarded_frames)>10)>0:
				print('error, this should not have happened')
				exit()
			else:	# it can happen that the first few frames are messed up
				digitizer_ID = np.flip(digitizer_ID,axis=0).tolist()
				len_discarded_frames = len(discarded_frames)
				while len_discarded_frames>0:
					if digitizer_ID[-1] == 0:
						digitizer_ID.append(1)
					else:
						digitizer_ID.append(0)
					len_discarded_frames -=1
				digitizer_ID = np.flip(digitizer_ID,axis=0).tolist()
		print('digitizer inversion at frames '+str(np.arange(len(data)-1)[np.diff(digitizer_ID)==0]))
	else:
		digitizer_ID = digitizer_ID_fake
		discarded_frames = []
	# back to the normal process
	data = np.array(data)
	data_median = int(np.median(data))
	if np.abs(data-data_median).max()<2**8/2-1:
		data_minus_median = (data-data_median).astype(np.int8)
	elif np.abs(data-data_median).max()<2**16/2-1:
		data_minus_median = (data-data_median).astype(np.int16)
	elif np.abs(data-data_median).max()<2**32/2-1:
		data_minus_median = (data-data_median).astype(np.int32)
	digitizer_ID = np.array(digitizer_ID)
	time_of_measurement = np.array(time_of_measurement)
	frame_counter = np.array(frame_counter)
	DetectorTemp = np.array(DetectorTemp)
	SensorTemp_0 = np.array(SensorTemp_0)
	SensorTemp_3 = np.array(SensorTemp_3)
	AtmosphereTemp = np.array(AtmosphereTemp)
	out = dict([])
	# out['data'] = data
	out['data_median'] = data_median
	out['data'] = data_minus_median	# I do this to save memory, also because the change in counts in a single recording is always small
	out['digitizer_ID']=digitizer_ID
	out['time_of_measurement']=time_of_measurement
	out['IntegrationTime']=IntegrationTime
	out['FrameRate']=FrameRate
	out['ExternalTrigger'] = ExternalTrigger
	out['NUCpresetUsed'] = NUCpresetUsed
	out['SensorTemp_0'] = SensorTemp_0
	out['SensorTemp_3'] = SensorTemp_3
	out['AtmosphereTemp'] = AtmosphereTemp
	out['DetectorTemp'] = DetectorTemp
	out['width'] = width
	out['height'] = height
	out['camera_SN'] = camera_SN
	out['frame_counter'] = frame_counter
	data_per_digitizer,uniques_digitizer_ID = separate_data_with_digitizer(out)
	out['data_time_avg_counts'] = np.array([(np.mean(data,axis=0)) for data in data_per_digitizer])
	out['data_time_avg_counts_std'] = np.array([(np.std(data,axis=0)) for data in data_per_digitizer])
	out['data_time_space_avg_counts'] = np.array([(np.mean(data,axis=(0,1,2))) for data in data_per_digitizer])
	out['data_time_space_avg_counts_std'] = np.array([(np.std(data,axis=(0,1,2))) for data in data_per_digitizer])
	out['uniques_digitizer_ID'] = uniques_digitizer_ID
	out['discarded_frames'] = discarded_frames
	# return data,digitizer_ID,time_of_measurement,IntegrationTime,FrameRate,ExternalTrigger,SensorTemp_0,DetectorTemp,width,height,camera_SN,frame_counter
	return out

def separate_data_with_digitizer(full_saved_file_dict):
	try:
		data_median = full_saved_file_dict['data_median']
		data = full_saved_file_dict['data']
		data = data.astype(np.int)
		data += data_median
	except:
		data = full_saved_file_dict['data']
	digitizer_ID = full_saved_file_dict['digitizer_ID']
	uniques_digitizer_ID = np.sort(np.unique(digitizer_ID))
	data_per_digitizer = []
	for ID in uniques_digitizer_ID:
		data_per_digitizer.append(data[digitizer_ID==ID])
	return data_per_digitizer,uniques_digitizer_ID

def generic_separate_with_digitizer(data,digitizer_ID):
	uniques_digitizer_ID = np.sort(np.unique(digitizer_ID))
	data_per_digitizer = []
	for ID in uniques_digitizer_ID:
		data_per_digitizer.append(data[digitizer_ID==ID])
	return data_per_digitizer,uniques_digitizer_ID

def read_IR_file(file):
	if os.path.exists(file+'.npz'):
		full_saved_file_dict=np.load(file+'.npz')
	else:
		if os.path.exists(file+'.ats'):
			full_saved_file_dict = ats_to_dict(file+'.ats')
		elif os.path.exists(file+'.ptw'):
			full_saved_file_dict = ptw_to_dict(file+'.ptw')
		print(file+'.npz generated')
		np.savez_compressed(file,**full_saved_file_dict)
	return full_saved_file_dict

def build_poly_coeff_multi_digitizer(temperature,files,inttime,pathparam,n):
	# modified 2018-10-08 to build the coefficient only for 1 degree of polinomial
	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)
	sin_fun = lambda x,A,f,p : A*np.sin(x*f*2*np.pi+p)
	meancounttot=[]
	meancountstdtot=[]
	all_SensorTemp_0 = []
	all_DetectorTemp = []
	all_frame_counter = []
	all_time_of_measurement = []
	for i_file,file in enumerate(files):
		full_saved_file_dict=read_IR_file(file)
		data_per_digitizer,uniques_digitizer_ID = separate_data_with_digitizer(full_saved_file_dict)
		if i_file==0:
			digitizer_ID = np.array(uniques_digitizer_ID)
		if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
			print('ERROR: problem with the ID of the digitizer in \n' + file)
			exit()
		meancounttot.append([np.mean(x,axis=0) for x in data_per_digitizer])
		if False:	# what if I'm exaggerating this because of the oscillation and the baseline drift?
			meancountstdtot.append([np.std(x,axis=0) for x in data_per_digitizer])	# what if I'm exaggerating this because of the oscillation and the baseline drift?
		else:	# this tries to remove the effect of the oshillation. it's marginally (std decrease~1%) better. it is still fast so I can keep it
			a = [np.mean(x,axis=(-1,-2)) for x in data_per_digitizer]
			b = []
			for i in digitizer_ID:
				framerate = float(full_saved_file_dict['FrameRate'])
				time_axis = np.arange(len(a[i]))*2*1 / framerate
				lin_fit = np.polyfit(time_axis,a[i],1)
				baseline = np.polyval(lin_fit,time_axis)
				if framerate>300:	# the oscillation will be present only at high frequency
					bds = [[0,20,-4*np.pi],[np.inf,40,4*np.pi]]
					guess = [1,29,max(-4*np.pi,min(4*np.pi,-np.pi*np.trapz((a[i]-baseline)[time_axis<1/29/2])*2/np.trapz(np.abs(a[i]-baseline)[time_axis<1/29])))]
					fit = curve_fit(sin_fun, time_axis,a[i]-baseline, p0=guess, bounds = bds, maxfev=100000000)
					# plt.figure()
					# plt.plot(time_axis,a[i]-baseline)
					# plt.plot(time_axis,sin_fun(time_axis,*fit[0]))
					# plt.plot(time_axis,sin_fun(time_axis,*guess),'--')
					# plt.pause(0.001)
					b.append((data_per_digitizer[i].T-baseline-sin_fun(time_axis,*fit[0])).T)
				else:
					b.append((data_per_digitizer[i].T-baseline).T)
			meancountstdtot.append([np.std(x,axis=0) for x in b])
		all_SensorTemp_0.append(np.mean(full_saved_file_dict['SensorTemp_0']))
		all_DetectorTemp.append(np.mean(full_saved_file_dict['DetectorTemp']))
		all_time_of_measurement.append(np.mean(full_saved_file_dict['time_of_measurement']))
		all_frame_counter.append(np.mean(full_saved_file_dict['frame_counter']))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)
	shapex=np.shape(meancounttot)[-2]
	shapey=np.shape(meancounttot)[-1]
	score=np.zeros((len(digitizer_ID),shapex,shapey))
	score2=np.zeros((len(digitizer_ID),shapex,shapey))

	# WARNING; THIS CREATE COEFFICIENTS INCOMPATIBLE WITH PREVIOUS build_poly_coeff FUNCTION
	coeff=np.zeros((len(digitizer_ID),shapex,shapey,n))
	errcoeff=np.zeros((len(digitizer_ID),shapex,shapey,n,n))
	coeff2=np.zeros((len(digitizer_ID),shapex,shapey,2))
	errcoeff2=np.zeros((len(digitizer_ID),shapex,shapey,2,2))

	def BB_rad_prob_and_gradient(T_,counts,grads=True):
		def int(arg):
			a1=arg[0]
			a2=arg[1]
			lambda_cam_x = np.linspace(1.5,5.1,10)*1e-6	# m, Range of FLIR SC7500
			lambda_cam = np.array([lambda_cam_x.tolist()]*len(T_)).T
			temp1 = np.trapz(2*scipy.constants.h*(scipy.constants.c**2)/(lambda_cam**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam*scipy.constants.k*T_)) -1) ,x=lambda_cam_x,axis=0)
			temp2 = a1*temp1 + a2 - counts
			out = np.sum(temp2**2)
			if grads:
				grad = [np.sum(2*temp2*temp1),np.sum(2*temp2*1)]
				return out,np.array(grad)
			else:
				return out
		return int

	import numdifftools as nd
	def make_standatd_fit_output(function,x_optimal):
		hessian = nd.Hessian(function)
		hessian = hessian(x_optimal)
		covariance = np.linalg.inv(hessian)
		for i in range(len(covariance)):
			covariance[i,i] = np.abs(covariance[i,i])
		fit = [x_optimal,covariance]
		return fit

	BB_rad = lambda T,a1,a2 : a1*2*scipy.constants.h*(scipy.constants.c**2)/((5e-6)**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/((5e-6)*scipy.constants.k*T)) -1) + a2
	def BB_rad(T_,a1,a2):
		lambda_cam_x = np.linspace(1.5,5.1,10)*1e-6	# m, Range of FLIR SC7500
		lambda_cam = np.array([lambda_cam_x.tolist()]*len(T_)).T
		temp1 = np.trapz(2*scipy.constants.h*(scipy.constants.c**2)/(lambda_cam**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam*scipy.constants.k*T_)) -1) ,x=lambda_cam_x,axis=0)
		out = a1*temp1 + a2
		return out

	for j in range(shapex):
		for k in range(shapey):
			for i_z,z in enumerate(digitizer_ID):
				x=np.array(meancounttot[:,z==digitizer_ID,j,k]).flatten()
				xerr=np.array(meancountstdtot[:,z==digitizer_ID,j,k]).flatten()
				# temp1,temp2=np.polyfit(temperature,x,n-1,cov='unscaled')
				temp1=np.polyfit(x,temperature,n-1)	# this correction alone decrease the errors by 2 error by 2 orders of magnitude
				yerr=(np.polyval(temp1,x+xerr)-np.polyval(temp1,x-xerr))/2
				temp1,temp2=np.polyfit(x,temperature,n-1,w=1/yerr,cov='unscaled')
				fit = curve_fit(BB_rad,np.array(temperature)+273,x,sigma=xerr,absolute_sigma=False,p0=[1e4,100])
				# the following is much slower
				# x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(BB_rad_prob_and_gradient(np.array(temperature)+273,x), x0=[1e4,100], iprint=0, factr=1e2, pgtol=1e-6, maxiter=5000)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				# fit = make_standatd_fit_output(BB_rad_prob_and_gradient(np.array(temperature)+273,x,grads=False),x_optimal)
				# plt.figure()
				# plt.errorbar(x,temperature,xerr=xerr,fmt='+')
				# plt.plot(np.sort(x),np.polyval(temp1,np.sort(x)),'--')
				# plt.plot(BB_rad(np.sort(temperature)+273,*fit[0]),np.sort(temperature),':')
				# plt.pause(0.01)
				coeff[i_z,j,k,:]=temp1
				errcoeff[i_z,j,k,:]=temp2
				coeff2[i_z,j,k,:]=fit[0]
				errcoeff2[i_z,j,k,:]=fit[1]
				score[i_z,j,k]=rsquared(temperature,np.polyval(temp1,x))
				score2[i_z,j,k]=rsquared(x,BB_rad(np.array(temperature)+273,*fit[0]))
	np.savez_compressed(os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms'),**dict([('coeff',coeff),('errcoeff',errcoeff),('score',score),('coeff2',coeff2),('errcoeff2',errcoeff2),('score2',score2)]))
	print('for a polinomial of degree '+str(n-1)+' the R^2 score is '+str(np.sum(score[n-2])))

def build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5):
	# modified 2018-10-08 to build the coefficient only for 1 degree of polinomial
	if len(temperature_window)>0:
		while np.shape(temperature_window[0])!=():
			temperature_window=np.concatenate(temperature_window)
			files_window=np.concatenate(files_window)
	temperature_window = np.array(temperature_window)
	files_window = np.array(files_window)
	if len(temperature_no_window)>0:
		while np.shape(temperature_no_window[0])!=():
			temperature_no_window=np.concatenate(temperature_no_window)
			files_no_window=np.concatenate(files_no_window)
	temperature_no_window = np.array(temperature_no_window)
	files_no_window = np.array(files_no_window)
	sin_fun = lambda x,A,f,p : A*np.sin(x*f*2*np.pi+p)
	meancounttot=[]
	meancountstdtot=[]
	all_SensorTemp_0 = []
	all_DetectorTemp = []
	all_frame_counter = []
	all_time_of_measurement = []
	for i_file,file in enumerate(files_window):
		full_saved_file_dict=read_IR_file(file)
		data_per_digitizer,uniques_digitizer_ID = separate_data_with_digitizer(full_saved_file_dict)
		if i_file==0:
			digitizer_ID = np.array(uniques_digitizer_ID)
		if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
			print('ERROR: problem with the ID of the digitizer in \n' + file)
			exit()
		meancounttot.append([np.mean(x,axis=0) for x in data_per_digitizer])
		if False:	# what if I'm exaggerating this because of the oscillation and the baseline drift?
			meancountstdtot.append([np.std(x,axis=0) for x in data_per_digitizer])	# what if I'm exaggerating this because of the oscillation and the baseline drift?
		else:	# this tries to remove the effect of the oshillation. it's marginally (std decrease~1%) better. it is still fast so I can keep it
			a = [np.mean(x,axis=(-1,-2)) for x in data_per_digitizer]
			b = []
			for i in digitizer_ID:
				framerate = float(full_saved_file_dict['FrameRate'])
				time_axis = np.arange(len(a[i]))*2*1 / framerate
				lin_fit = np.polyfit(time_axis,a[i],1)
				baseline = np.polyval(lin_fit,time_axis)
				if framerate>300:	# the oscillation will be present only at high frequency
					bds = [[0,20,-4*np.pi],[np.inf,40,4*np.pi]]
					guess = [1,29,max(-4*np.pi,min(4*np.pi,-np.pi*np.trapz((a[i]-baseline)[time_axis<1/29/2])*2/np.trapz(np.abs(a[i]-baseline)[time_axis<1/29])))]
					fit = curve_fit(sin_fun, time_axis,a[i]-baseline, p0=guess, bounds = bds, maxfev=100000000)
					# plt.figure()
					# plt.plot(time_axis,a[i]-baseline)
					# plt.plot(time_axis,sin_fun(time_axis,*fit[0]))
					# plt.plot(time_axis,sin_fun(time_axis,*guess),'--')
					# plt.pause(0.001)
					b.append((data_per_digitizer[i].T-baseline-sin_fun(time_axis,*fit[0])).T)
				else:
					b.append((data_per_digitizer[i].T-baseline).T)
			meancountstdtot.append([np.std(x,axis=0) for x in b])
		all_SensorTemp_0.append(np.mean(full_saved_file_dict['SensorTemp_0']))
		all_DetectorTemp.append(np.mean(full_saved_file_dict['DetectorTemp']))
		all_time_of_measurement.append(np.mean(full_saved_file_dict['time_of_measurement']))
		all_frame_counter.append(np.mean(full_saved_file_dict['frame_counter']))
	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	meancounttot_no_window=[]
	meancountstdtot_no_window=[]
	all_SensorTemp_0 = []
	all_DetectorTemp = []
	all_frame_counter = []
	all_time_of_measurement = []
	for i_file,file in enumerate(files_no_window):
		full_saved_file_dict=read_IR_file(file)
		data_per_digitizer,uniques_digitizer_ID = separate_data_with_digitizer(full_saved_file_dict)
		if i_file==0:
			digitizer_ID = np.array(uniques_digitizer_ID)
		if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
			print('ERROR: problem with the ID of the digitizer in \n' + file)
			exit()
		meancounttot_no_window.append([np.mean(x,axis=0) for x in data_per_digitizer])
		if False:	# what if I'm exaggerating this because of the oscillation and the baseline drift?
			meancountstdtot_no_window.append([np.std(x,axis=0) for x in data_per_digitizer])	# what if I'm exaggerating this because of the oscillation and the baseline drift?
		else:	# this tries to remove the effect of the oshillation. it's marginally (std decrease~1%) better. it is still fast so I can keep it
			a = [np.mean(x,axis=(-1,-2)) for x in data_per_digitizer]
			b = []
			for i in digitizer_ID:
				framerate = float(full_saved_file_dict['FrameRate'])
				time_axis = np.arange(len(a[i]))*2*1 / framerate
				lin_fit = np.polyfit(time_axis,a[i],1)
				baseline = np.polyval(lin_fit,time_axis)
				if framerate>300:	# the oscillation will be present only at high frequency
					bds = [[0,20,-4*np.pi],[np.inf,40,4*np.pi]]
					guess = [1,29,max(-4*np.pi,min(4*np.pi,-np.pi*np.trapz((a[i]-baseline)[time_axis<1/29/2])*2/np.trapz(np.abs(a[i]-baseline)[time_axis<1/29])))]
					fit = curve_fit(sin_fun, time_axis,a[i]-baseline, p0=guess, bounds = bds, maxfev=100000000)
					# plt.figure()
					# plt.plot(time_axis,a[i]-baseline)
					# plt.plot(time_axis,sin_fun(time_axis,*fit[0]))
					# plt.plot(time_axis,sin_fun(time_axis,*guess),'--')
					# plt.pause(0.001)
					b.append((data_per_digitizer[i].T-baseline-sin_fun(time_axis,*fit[0])).T)
				else:
					b.append((data_per_digitizer[i].T-baseline).T)
			meancountstdtot_no_window.append([np.std(x,axis=0) for x in b])
		all_SensorTemp_0.append(np.mean(full_saved_file_dict['SensorTemp_0']))
		all_DetectorTemp.append(np.mean(full_saved_file_dict['DetectorTemp']))
		all_time_of_measurement.append(np.mean(full_saved_file_dict['time_of_measurement']))
		all_frame_counter.append(np.mean(full_saved_file_dict['frame_counter']))
	meancounttot_no_window=np.array(meancounttot_no_window)
	meancountstdtot_no_window=np.array(meancountstdtot_no_window)

	shapex=np.shape(data_per_digitizer[0])[-2]	# changed for the case in which there is only data for no_window
	shapey=np.shape(data_per_digitizer[0])[-1]
	score=np.zeros((len(digitizer_ID),shapex,shapey))
	score2=np.zeros((len(digitizer_ID),shapex,shapey))
	score3=np.zeros((len(digitizer_ID),shapex,shapey))
	score4=np.zeros((len(digitizer_ID),shapex,shapey))

	# WARNING; THIS CREATE COEFFICIENTS INCOMPATIBLE WITH PREVIOUS build_poly_coeff FUNCTION
	coeff=np.zeros((len(digitizer_ID),shapex,shapey,n))
	errcoeff=np.zeros((len(digitizer_ID),shapex,shapey,n,n))
	coeff2=np.zeros((len(digitizer_ID),shapex,shapey,4))
	errcoeff2=np.zeros((len(digitizer_ID),shapex,shapey,4,4))
	coeff3=np.zeros((len(digitizer_ID),shapex,shapey,2))
	errcoeff3=np.zeros((len(digitizer_ID),shapex,shapey,2,2))
	coeff4=np.zeros((len(digitizer_ID),shapex,shapey,2))
	errcoeff4=np.zeros((len(digitizer_ID),shapex,shapey,2,2))

	# lambda_cam_x = np.linspace(1.5,5.1,10)*1e-6	# m, Range of FLIR SC7500
	lambda_cam_x = np.linspace(wavelength_bottom,wavewlength_top,100)*1e-6	# m, Range of FLIR SC7500
	temperature = temperature_window.tolist() + temperature_no_window.tolist()
	# temperature_range = np.linspace(np.min(temperature),np.max(temperature))
	temperature_range = np.unique(temperature)
	photon_flux = []
	for T_ in temperature_range:
		photon_flux.append(np.trapz(2*scipy.constants.c/(lambda_cam_x**4) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x*scipy.constants.k*(T_+273.15))) -1) ,x=lambda_cam_x,axis=0) * inttime/1000)
	photon_flux = np.array(photon_flux)
	photon_flux_interpolator = interp1d(temperature_range,photon_flux,bounds_error=False,fill_value='extrapolate')


	def BB_rad_prob_and_gradient(T_,counts,lambda_cam_x=lambda_cam_x,grads=True):
		def int(arg):
			a1=arg[0]
			a2=arg[1]
			lambda_cam = np.array([lambda_cam_x.tolist()]*len(T_)).T
			# temp1 = np.trapz(2*scipy.constants.c/(lambda_cam**4) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam*scipy.constants.k*(T_+273.15))) -1) ,x=lambda_cam_x,axis=0) * inttime/1000
			temp1 = photon_flux_interpolator(T_)
			temp2 = a1*temp1 + a2 - counts
			out = np.sum(temp2**2)
			if grads:
				grad = [np.sum(2*temp2*temp1),np.sum(2*temp2*1)]
				return out,np.array(grad)
			else:
				return out
		return int

	import numdifftools as nd
	def make_standatd_fit_output(function,x_optimal):
		hessian = nd.Hessian(function)
		hessian = hessian(x_optimal)
		covariance = np.linalg.inv(hessian)
		for i in range(len(covariance)):
			covariance[i,i] = np.abs(covariance[i,i])
		fit = [x_optimal,covariance]
		return fit

	BB_rad = lambda T,a1,a2 : a1*2*scipy.constants.c/((5e-6)**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/((5e-6)*scipy.constants.k*(T+273.15))) -1) * inttime/1000 + a2

	def BB_rad(number_of_window,lambda_cam_x=lambda_cam_x):
		def int(T_,a1,a2,a3,a4):
			# lambda_cam_x = np.linspace(1.5,5.1,10)*1e-6	# m, Range of FLIR SC7500
			lambda_cam = np.array([lambda_cam_x.tolist()]*len(T_)).T
			# temp1 = np.trapz(2*scipy.constants.c/(lambda_cam**4) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam*scipy.constants.k*(T_+273.15))) -1) ,x=lambda_cam_x,axis=0) * inttime/1000
			temp1 = photon_flux_interpolator(T_)
			# temp1 = 2*scipy.constants.h*(scipy.constants.c**2)/(lambda_cam_x.max()**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x.max()*scipy.constants.k*T_)) -1)
			out = a1*temp1 + a2
			out[:number_of_window] = a1*a3*temp1[:number_of_window] + a2 + a4
			return out
		return int

	def BB_rad2(lambda_cam_x=lambda_cam_x):
		def int(T_,a1,a2):
			# lambda_cam_x = np.linspace(1.5,5.1,10)*1e-6	# m, Range of FLIR SC7500
			lambda_cam = np.array([lambda_cam_x.tolist()]*len(T_)).T
			# temp1 = np.trapz(2*scipy.constants.c/(lambda_cam**4) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam*scipy.constants.k*(T_+273.15))) -1) ,x=lambda_cam_x,axis=0) * inttime/1000
			temp1 = photon_flux_interpolator(T_)
			# temp1 = 2*scipy.constants.h*(scipy.constants.c**2)/(lambda_cam_x.max()**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x.max()*scipy.constants.k*T_)) -1)
			out = a1*temp1 + a2
			return out
		return int

	def deg_2_poly(x,a2,a1,a0):
		out = a2*x**2 + a1*x + a0
		return out

	def BB_rad_counts_to_delta_temp(trash,T_,lambda_cam_x=lambda_cam_x):
		# lambda_cam_x = np.linspace(1.5,5.1,10)*1e-6	# m, Range of FLIR SC7500
		# temp1 = np.trapz(2*scipy.constants.c/(lambda_cam_x**4) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x*scipy.constants.k*(T_+273.15))) -1) ,x=lambda_cam_x) * inttime/1000
		temp1 = photon_flux_interpolator(T_)
		# temp1 = 2*scipy.constants.h*(scipy.constants.c**2)/(lambda_cam_x.max()**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x.max()*scipy.constants.k*T_)) -1)
		return temp1

	def sigma_T_multimpier(T_,lambda_cam_x=lambda_cam_x):
		# lambda_cam_x = np.linspace(1.5,5.1,10)*1e-6	# m, Range of FLIR SC7500
		temp = 2*scipy.constants.c/(lambda_cam_x**4) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x*scipy.constants.k*(T_+273.15))) -1) * inttime/1000
		temp1 = np.trapz(temp* scipy.constants.h*scipy.constants.c/(lambda_cam_x*scipy.constants.k*(T_**2)) ,x=lambda_cam_x)
		# temp1 = 2*scipy.constants.h*(scipy.constants.c**2)/(lambda_cam_x.max()**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x.max()*scipy.constants.k*T_)) -1)
		return temp1

	number_of_window = len(temperature_window)
	bds = [[0,0,0,-np.inf],[np.inf,np.inf,1,np.inf]]
	bds1=np.array(bds)[:,:2]
	x__all = np.array(meancounttot.tolist() + meancounttot_no_window.tolist())
	xerr__all = np.array(meancountstdtot.tolist() + meancountstdtot_no_window.tolist())
	for j in range(shapex):
		for k in range(shapey):
			for i_z,z in enumerate(digitizer_ID):
				if number_of_window>0:
					x=np.array(meancounttot[:,z==digitizer_ID,j,k]).flatten()
					x_=x__all[:,z==digitizer_ID,j,k].flatten()
					xerr=np.array(meancountstdtot[:,z==digitizer_ID,j,k]).flatten()
					xerr_=xerr__all[:,z==digitizer_ID,j,k].flatten()
					# temp1,temp2=np.polyfit(temperature_window,x,n-1,cov='unscaled')
					temp1=np.polyfit(x,temperature_window,n-1)	# this correction alone decrease the errors by 2 error by 2 orders of magnitude
					yerr=(np.polyval(temp1,x+xerr)-np.polyval(temp1,x-xerr))/2
					temp1,temp2=np.polyfit(x,temperature_window,n-1,w=1/yerr,cov='unscaled')
					# fit = curve_fit(deg_2_poly,x,temperature_window,sigma=yerr,absolute_sigma=True,p0=[1,1,1])	# equivalent to np.polyfit, just to check that both return the same uncertainty
					fit = curve_fit(BB_rad(number_of_window),np.array(temperature),x_,sigma=xerr_,absolute_sigma=True,p0=[1e-13,1000,1,100],bounds=bds,x_scale=[1e-13,1,1,1])
					# the following is much slower
					# x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(BB_rad_prob_and_gradient(np.array(temperature_window)+273,x), x0=[1e4,100], iprint=0, factr=1e2, pgtol=1e-6, maxiter=5000)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
					# fit = make_standatd_fit_output(BB_rad_prob_and_gradient(np.array(temperature_window)+273,x,grads=False),x_optimal)
					x__window=np.array(meancounttot[:,z==digitizer_ID,j,k]).flatten()
					xerr__window=np.array(meancountstdtot[:,z==digitizer_ID,j,k]).flatten()
					fit2 = curve_fit(BB_rad2(),np.array(temperature_window),x__window,sigma=xerr__window,absolute_sigma=True,p0=[1e-13,1000],bounds=bds1,x_scale=[1e-13,1])
				x__no_window=np.array(meancounttot_no_window[:,z==digitizer_ID,j,k]).flatten()
				xerr__no_window=np.array(meancountstdtot_no_window[:,z==digitizer_ID,j,k]).flatten()
				fit1 = curve_fit(BB_rad2(),np.array(temperature_no_window),x__no_window,sigma=xerr__no_window,absolute_sigma=True,p0=[1e-13,1000],bounds=bds1,x_scale=[1e-13,1])
				if False:
					plt.figure()
					plt.errorbar(x,temperature_window,xerr=xerr,fmt='+',color='b')
					plt.errorbar(x_[number_of_window:],temperature_no_window,xerr=xerr_[number_of_window:],fmt='+',color='r')
					plt.plot(np.sort(x),np.polyval(temp1,np.sort(x)),'--',label='poly')
					plt.plot(BB_rad2()(np.sort(temperature_window),*[fit[0][0]*fit[0][2],fit[0][1]+fit[0][3]]),np.sort(temperature_window),':b',label='BB window')
					plt.plot(BB_rad2()(np.sort(temperature_window),*fit2[0]),np.sort(temperature_window),'-.b',label='BB window')
					plt.plot(BB_rad2()(np.sort(temperature_no_window),*fit[0][:2]),np.sort(temperature_no_window),':r',label='BB no window')
					plt.plot(BB_rad2()(np.sort(temperature_no_window),*fit1[0]),np.sort(temperature_no_window),'-.r',label='BB no window')
					plt.grid()
					plt.legend()
					plt.pause(0.01)
				if False:	# small piece to check how the uncertainty goes between the 2 models
					delta_counts = 7194-5800
					a1a3 = fit[0][0]*fit[0][2]
					sigma_a1a3 = a1a3 * ((fit[1][0,0]**0.5/fit[0][0])**2 + (fit[1][2,2]**0.5/fit[0][2])**2 + 2*fit[1][0,2]/a1a3)**0.5
					ref = delta_counts/a1a3 + BB_rad_counts_to_delta_temp(1,300)
					sigma_ref = delta_counts/a1a3*(( (estimate_counts_std(5800+delta_counts)*2/delta_counts)**2 + (sigma_a1a3/(a1a3**2))**2 )**0.5)
					# sigma_ref = (sigma_ref**2 + ((sigma_T_multimpier(300)*0.1)**2))**0.5	# I'm not sure if I should consider this
					check = curve_fit(BB_rad_counts_to_delta_temp,1,ref,sigma=[sigma_ref],absolute_sigma=True,p0=[300])
					counts = 5800+delta_counts
					temp = temp1[-1] + temp1[-2] * counts + temp1[-3] * (counts**2)
					counts_std = estimate_counts_std(counts)
					temperature_std = (temp2[2,2] + (counts_std**2)*(temp1[1]**2) + (counts**2+counts_std**2)*temp2[1,1] + (counts_std**2)*(4*counts**2+3*counts_std**2)*(temp1[0]**2) + (counts**4+6*(counts**2)*(counts_std**2)+3*counts_std**4)*temp2[0,0] + 2*counts*temp2[2,1] + 2*(counts**2+counts_std**2)*temp2[2,0] + 2*(counts**3+counts*(counts_std**2))*temp2[1,0])**0.5
				if number_of_window>0:
					coeff[i_z,j,k,:]=temp1
					errcoeff[i_z,j,k,:]=temp2
					coeff2[i_z,j,k,:]=fit[0]
					errcoeff2[i_z,j,k,:]=fit[1]
					coeff4[i_z,j,k,:]=fit2[0]
					errcoeff4[i_z,j,k,:]=fit2[1]
					score[i_z,j,k]=rsquared(temperature_window,np.polyval(temp1,x))
					score2[i_z,j,k]=rsquared(x_,BB_rad(number_of_window)(np.array(temperature),*fit[0]))
					score4[i_z,j,k]=rsquared(x__window,BB_rad2()(temperature_window,*fit2[0]))
				coeff3[i_z,j,k,:]=fit1[0]
				errcoeff3[i_z,j,k,:]=fit1[1]
				score3[i_z,j,k]=rsquared(x__no_window,BB_rad2()(temperature_no_window,*fit1[0]))
	np.savez_compressed(os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms'),**dict([('coeff',coeff),('errcoeff',errcoeff),('score',score),('coeff2',coeff2),('errcoeff2',errcoeff2),('score2',score2),('coeff3',coeff3),('errcoeff3',errcoeff3),('score3',score3),('coeff4',coeff4),('errcoeff4',errcoeff4),('score4',score4)]))
	print('for a polinomial of degree '+str(n-1)+' the R^2 score is '+str(np.sum(score[n-2])))


def build_average_poly_coeff_multi_digitizer(temperaturehot,temperaturecold,fileshot,filescold,int,framerate,pathparam,n):
	# 08/10/2018 THIS MAKES THE AVERAGE OF COEFFICIENTS FROM MULTIPLE HOT>ROOM AND COLD >ROOM CYCLES THE COEFFICIENTS

	lengthhot=len(temperaturehot)
	lengthcold=len(temperaturecold)

	first=True
	for i in range(lengthhot):
		for j in range(lengthcold):
			path=pathparam+'/'+str(int)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)+'/'+str(i+1)+'-'+str(j+1)
			full_saved_file_dict=np.load(os.path.join(path,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int)+'ms'+'.npz'))
			if first==True:
				shape=np.shape(full_saved_file_dict['coeff'])
				shape=np.concatenate(((lengthhot,lengthcold),shape))
				coeff=np.zeros(shape)
				shape=np.shape(full_saved_file_dict['errcoeff'])
				shape=np.concatenate(((lengthhot,lengthcold),shape))
				errcoeff=np.zeros(shape)
				shape=np.shape(full_saved_file_dict['score'])
				shape=np.concatenate(((lengthhot,lengthcold),shape))
				score=np.zeros(shape)
				first=False
			coeff[i,j]=full_saved_file_dict['coeff']
			errcoeff[i,j]=full_saved_file_dict['errcoeff']

	select = np.zeros((n,n)).astype(bool)
	np.fill_diagonal(select,True)
	meancoeff=np.sum(coeff/errcoeff[:,:,:,:,:,select],axis=(0,1))/np.sum(1/errcoeff[:,:,:,:,:,select],axis=(0,1))
	meanerrcoeff=(1/np.sum(1/errcoeff,axis=(0,1)))*1/((lengthhot*lengthcold)**0.5)
	meanerrcoeff[:,:,:,select] = (np.std(coeff,axis=(0,1))**2 + meanerrcoeff[:,:,:,select])
	meanscore = np.mean(score,axis=(0,1))

	path=pathparam+'/'+str(int)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)+'/average'
	if not os.path.exists(path):
		os.makedirs(path)
	np.savez_compressed(os.path.join(path,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int)+'ms'),**dict([('coeff',meancoeff),('errcoeff',meanerrcoeff),('score',meanscore)]))

def count_to_temp_1D_homo_conversion(arg):
	out = np.sum(np.power(np.array([(arg[0]).tolist()]*arg[3]).T,np.arange(arg[3]-1,-1,-1))*correlated_values(arg[1],arg[2]),axis=1)
	out1 = nominal_values(out)
	out2 = std_devs(out)
	return [out1,out2]

def count_to_temp_0D_homo_conversion(arg):
	out = np.sum(np.power(np.array([(uarray(arg[0],arg[1])).tolist()]*arg[4]).T,np.arange(arg[4]-1,-1,-1))*correlated_values(arg[2],arg[3]),axis=0)
	out1 = nominal_values(out)
	out2 = std_devs(out)
	return [out1,out2]


def count_to_temp_poly_multi_digitizer_time_dependent(counts,params,errparams,reference_background,reference_background_std,reference_background_flat,digitizer_ID,number_cpu_available,n,parallelised=True,report=0):
	temperature = []
	temperature_std = []
	with cf.ProcessPoolExecutor(max_workers=number_cpu_available) as executor:
		for i in range(len(digitizer_ID)):
			counts_temp = np.array(counts[i])
			if False:
				temp1 = []
				temp2 = []
				if report>0:
					start_time = tm.time()
				for j in range(counts_temp.shape[1]):
					if report>0:
						start_time_1 = tm.time()
					if parallelised:	# parallel way
						arg = []
						for k in range(counts_temp.shape[2]):
							arg.append([counts_temp[:,j,k],params[i,j,k],errparams[i,j,k],n])
						if report>1:
							print(str(j) + ' , %.5gs , %.3gs' %(tm.time()-start_time,tm.time()-start_time_1))
						out = list(executor.map(count_to_temp_1D_homo_conversion,arg))
					else:	# non parallel way
						if report>1:
							print(str(j) + ' , %.5gs , %.3gs' %(tm.time()-start_time,tm.time()-start_time_1))
						out = []
						for k in range(counts_temp.shape[2]):
							out.append(count_to_temp_1D_homo_conversion([counts_temp[:,j,k],params[i,j,k],errparams[i,j,k],n]))

					if report>1:
						print(str(j) + ' , %.5gs , %.3gs' %(tm.time()-start_time,tm.time()-start_time_1))
					temp1.append([x for x,y in out])
					temp2.append([y for x,y in out])
					if report>0:
						print(str(j) + ' , %.5gs , %.3gs' %(tm.time()-start_time,tm.time()-start_time_1))
			else:	# method brutally simpler, just doing a matrix multiplication
				temp1 = params[i][:,:,2] + counts_temp*params[i][:,:,1] + (counts_temp**2)*params[i][:,:,0]
				if False:
					# this is approximate because it does not account properly for the error matrix but only the diagonal, but it's massively faster
					temp2 = (errparams[i][:,:,2,2] + (counts_temp**2)*errparams[i][:,:,1,1] + (counts_temp**4)*errparams[i][:,:,0,0])**0.5
				else:
					# unfortunately the correct method is necessary, otherwise the std is overestimated by 2 orders of magnitude
					# I use the the uncertainty on the counts as
					counts_temp_std = estimate_counts_std(counts_temp)
					# temp2 = (errparams[i][:,:,2,2] + counts_temp*(params[i][:,:,1]**2) + (counts_temp**2+counts_temp)*errparams[i][:,:,1,1] + (4*counts_temp**3+3*counts_temp**2)*(params[i][:,:,0]**2) + (counts_temp**4+6*counts_temp**3+3*counts_temp**2)*errparams[i][:,:,0,0] + 2*counts_temp*errparams[i][:,:,2,1] + 2*(counts_temp**2+counts_temp)*errparams[i][:,:,2,0] + 2*(counts_temp**3+counts_temp**2)*errparams[i][:,:,1,0])**0.5
					temp2 = (errparams[i][:,:,2,2] + (counts_temp_std**2)*(params[i][:,:,1]**2) + (counts_temp**2+counts_temp_std**2)*errparams[i][:,:,1,1] + (counts_temp_std**2)*(4*counts_temp**2+3*counts_temp_std**2)*(params[i][:,:,0]**2) + (counts_temp**4+6*(counts_temp**2)*(counts_temp_std**2)+3*counts_temp_std**4)*errparams[i][:,:,0,0] + 2*counts_temp*errparams[i][:,:,2,1] + 2*(counts_temp**2+counts_temp_std**2)*errparams[i][:,:,2,0] + 2*(counts_temp**3+counts_temp*(counts_temp_std**2))*errparams[i][:,:,1,0])**0.5
			# temperature.append(np.transpose(temp1,(2,0,1)))
			# temperature_std.append(np.transpose(temp2,(2,0,1)))
			temperature.append(temp1)
			temperature_std.append(temp2)

	return temperature,temperature_std

def count_to_temp_poly_multi_digitizer_stationary(counts,counts_std,params,errparams,digitizer_ID,number_cpu_available,n,parallelised=True,report=0):

	temperature = []
	temperature_std = []
	with cf.ProcessPoolExecutor(max_workers=number_cpu_available) as executor:
		# executor = cf.ProcessPoolExecutor()#max_workers=number_cpu_available)
		for i in range(len(digitizer_ID)):
			counts_temp = np.array(counts[i])
			counts_std_temp = np.array(counts_std[i])
			temp1 = []
			temp2 = []
			if report>0:
				start_time = tm.time()
			for j in range(counts_temp.shape[0]):
				if report>0:
					start_time_1 = tm.time()

				if parallelised:	# parallel way
					arg = []
					for k in range(counts_temp.shape[1]):
						arg.append([counts_temp[j,k],counts_std_temp[j,k],params[i,j,k],errparams[i,j,k],n])
					if report>1:
						print(str(j) + ' , %.5gs , %.3gs' %(tm.time()-start_time,tm.time()-start_time_1))
					out = list(executor.map(count_to_temp_0D_homo_conversion,arg))
				else:	# non parallel way
					if report>1:
						print(str(j) + ' , %.5gs , %.3gs' %(tm.time()-start_time,tm.time()-start_time_1))
					out = []
					for k in range(counts_temp.shape[1]):
						out.append(count_to_temp_0D_homo_conversion([counts_temp[j,k],counts_std_temp[j,k],params[i,j,k],errparams[i,j,k],n]))

				if report>1:
					print(str(j) + ' , %.5gs , %.3gs' %(tm.time()-start_time,tm.time()-start_time_1))
				temp1.append([x for x,y in out])
				temp2.append([y for x,y in out])
				if report>0:
					print(str(j) + ' , %.5gs , %.3gs' %(tm.time()-start_time,tm.time()-start_time_1))
			temperature.append(np.array(temp1))
			temperature_std.append(np.array(temp2))

	return temperature,temperature_std

def count_to_temp_poly_multi_digitizer(counts,params,errparams,digitizer_ID,number_cpu_available,counts_std=[0],reference_background=[0],reference_background_std=[0],reference_background_flat=0,parallelised=True,report=0):

	n = np.shape(params)[-1]
	shape = np.shape(counts)
	if len(shape)==3:
		if np.shape(counts)!=np.shape(counts_std):
			print("for steady state conversion counts std should be supplied, counts shape "+str(np.shape(counts))+" counts std "+str(np.shape(counts_std)))
			exit()
		return count_to_temp_poly_multi_digitizer_stationary(counts,counts_std,params,errparams,digitizer_ID,number_cpu_available,n,parallelised=parallelised,report=report)
	else:
		if (len(np.shape(reference_background))<2 or len(np.shape(reference_background_std))<2) and False:	# this is no longer necessary in count_to_temp_poly_multi_digitizer_time_dependent
			print("you didn't supply the appropriate background counts, requested "+str(np.shape(counts)[-2:]))
			exit()
		return count_to_temp_poly_multi_digitizer_time_dependent(counts,params,errparams,reference_background,reference_background_std,reference_background_flat,digitizer_ID,number_cpu_available,n,parallelised=parallelised,report=report)

def count_to_temp_BB_multi_digitizer(counts,params,errparams,digitizer_ID,counts_std=[0],reference_background=[0],reference_background_std=[0],ref_temperature=20,ref_temperature_std=0,wavewlength_top=5,wavelength_bottom=2.5,inttime=2):
	# I don't think that there is the need of a separate function for stationary and time dependent

	shape = np.shape(counts)
	if len(shape)==3:
		if np.shape(counts)!=np.shape(counts_std):
			print("for steady state conversion counts std should be supplied, counts shape "+str(np.shape(counts))+" counts std "+str(np.shape(counts_std)))
			exit()
		return count_to_temp_BB_multi_digitizer_int(counts,counts_std,params,errparams,digitizer_ID,reference_background,reference_background_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=wavewlength_top,wavelength_bottom=wavelength_bottom,inttime=inttime)
	else:
		if (len(np.shape(reference_background))<2 or len(np.shape(reference_background_std))<2) and False:	# this is no longer necessary in count_to_temp_poly_multi_digitizer_time_dependent
			print("you didn't supply the appropriate background counts, requested "+str(np.shape(counts)[-2:]))
			exit()
		if counts_std==[0]:
			counts_std = []
			for i in range(len(digitizer_ID)):
				counts_std.append(estimate_counts_std(counts[i],int_time=inttime))
		return count_to_temp_BB_multi_digitizer_int(counts,counts_std,params,errparams,digitizer_ID,reference_background,reference_background_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=wavewlength_top,wavelength_bottom=wavelength_bottom,inttime=inttime)


def calc_interpolators_BB(wavewlength_top=5,wavelength_bottom=2.5,inttime=2):
	lambda_cam_x = np.linspace(wavelength_bottom,wavewlength_top,100)*1e-6	# m, Range of FLIR SC7500
	def BB_rad_counts_to_delta_temp(trash,T_,lambda_cam_x=lambda_cam_x):
		temp1 = np.trapz(2*scipy.constants.c/(lambda_cam_x**4) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x*scipy.constants.k*T_)) -1) ,x=lambda_cam_x) * inttime/1000
		return temp1

	temperature_range = np.linspace(0,50,num=100)
	temperature_range = temperature_range+273.15
	photon_flux = []
	for T in temperature_range:
		photon_flux.append(BB_rad_counts_to_delta_temp(1,T))
	photon_flux = np.array(photon_flux)
	photon_flux_interpolator = interp1d(temperature_range-273.15,photon_flux,bounds_error=False,fill_value='extrapolate')	# in degC
	reverse_photon_flux_interpolator = interp1d(photon_flux,temperature_range-273.15,bounds_error=False,fill_value='extrapolate')	# in degC
	photon_flux_over_temperature = photon_flux/temperature_range
	photon_flux_over_temperature_interpolator = interp1d(temperature_range-273.15,photon_flux_over_temperature,bounds_error=False,fill_value='extrapolate')	# in degC
	photon_dict = dict([])
	photon_dict['photon_flux_interpolator'] = photon_flux_interpolator
	photon_dict['reverse_photon_flux_interpolator'] = reverse_photon_flux_interpolator
	photon_dict['temperature_range'] = temperature_range
	photon_dict['photon_flux'] = photon_flux
	photon_dict['photon_flux_over_temperature'] = photon_flux_over_temperature
	photon_dict['photon_flux_over_temperature_interpolator'] = photon_flux_over_temperature_interpolator
	return photon_dict

def calc_BB_coefficients_multi_digitizer(params,errparams,digitizer_ID,reference_background,reference_background_std,ref_temperature=20,ref_temperature_std=0,wavewlength_top=5,wavelength_bottom=2.5,inttime=2):

	photon_dict = calc_interpolators_BB(wavewlength_top=wavewlength_top,wavelength_bottom=wavelength_bottom,inttime=inttime)
	photon_flux_interpolator = photon_dict['photon_flux_interpolator']

	photon_flux_std = np.abs(photon_flux_interpolator(ref_temperature+ref_temperature_std)-photon_flux_interpolator(ref_temperature-ref_temperature_std))/2
	constant_offset = []
	constant_offset_std = []
	BB_proportional = []
	BB_proportional_std = []
	for i in range(len(digitizer_ID)):
		BB_proportional.append(params[i,:,:,0]*params[i,:,:,2])
		BB_proportional_std.append(((errparams[i,:,:,0,0]**0.5 * params[i,:,:,2])**2 + (errparams[i,:,:,2,2]**0.5 * params[i,:,:,0])**2 + 2*params[i,:,:,0]*params[i,:,:,2]*errparams[i,:,:,2,0])**0.5)
		constant_offset.append( reference_background[i]-BB_proportional[-1]*photon_flux_interpolator(ref_temperature) )
		constant_offset_std.append( ((photon_flux_interpolator(ref_temperature)*BB_proportional_std[-1])**2 + (BB_proportional[-1]*photon_flux_std)**2 + reference_background_std[i]**2 )**0.5 )
	BB_proportional = np.array(BB_proportional)
	BB_proportional_std = np.array(BB_proportional_std)
	constant_offset = np.array(constant_offset)
	constant_offset_std = np.array(constant_offset_std)

	return BB_proportional,BB_proportional_std,constant_offset,constant_offset_std,photon_dict


def count_to_temp_BB_multi_digitizer_int(counts,counts_std,params,errparams,digitizer_ID,reference_background,reference_background_std,ref_temperature=20,ref_temperature_std=0,wavewlength_top=5,wavelength_bottom=2.5,inttime=2):
	# by definition resetting the constant value such that it matches ref_temperature means that the reference temperature is a flat ref_temperature, so I'm not sure if this function will ever be usefull

	BB_proportional,BB_proportional_std,constant_offset,constant_offset_std,photon_dict = calc_BB_coefficients_multi_digitizer(params,errparams,digitizer_ID,reference_background,reference_background_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=wavewlength_top,wavelength_bottom=wavelength_bottom,inttime=inttime)
	photon_dict = calc_interpolators_BB(wavewlength_top=wavewlength_top,wavelength_bottom=wavelength_bottom,inttime=inttime)
	reverse_photon_flux_interpolator = photon_dict['reverse_photon_flux_interpolator']
	photon_flux_over_temperature_interpolator = photon_dict['photon_flux_over_temperature_interpolator']
	photon_flux_interpolator = photon_dict['photon_flux_interpolator']

	temperature = []
	temperature_std = []
	for i in range(len(digitizer_ID)):
		photon_flux = (counts[i] - reference_background[i])/BB_proportional[i] + photon_flux_interpolator(ref_temperature)
		temperature.append(reverse_photon_flux_interpolator(photon_flux))	# in degC
		photon_flux_over_temperature = photon_flux_over_temperature_interpolator(temperature[-1])
		temperature_std.append( ( (counts_std[i]/(photon_flux_over_temperature*BB_proportional[i]))**2 + (reference_background_std[i]/(photon_flux_over_temperature_interpolator(ref_temperature)*BB_proportional[i]))**2 + (BB_proportional_std[i]*(counts[i] - reference_background[i])/((BB_proportional[i]**2)*photon_flux_over_temperature))**2 + (ref_temperature_std)**2 )**0.5 )	# in degC
	temperature = np.array(temperature)
	temperature_std = np.array(temperature_std)

	return temperature,temperature_std


##################################################################################################################################################################################################

def print_all_properties(obj):
	# created 29/09/2020 function that prints all properties of an object
	for attr in dir(obj):
		print("object.%s = %r" % (attr, getattr(obj, attr)))

####################################################################################################################################################################################################################################################

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 19:15:37 2021

@author: Federici
"""

# Created from code from James Harrison
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from . import efitData

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from scipy import interpolate
from mastu_exhaust_analysis.read_efit import read_uda
import netCDF4
# import pyuda
# client=pyuda.Client()

def reset_connection(client):
	client.set_property("timeout", 0)
	try:
		_ = client.get('help::ping()','')
	except pyuda.UDAException:
		pass
	client.set_property("timeout", 600)

class mclass:


	def __init__(self,path,pulse_ID=None):
		client_int=pyuda.Client()
		from multiprocessing.dummy import Pool as ThreadPool
		def abortable_worker(func,timeout, *arg):
			p = ThreadPool(1)
			res = p.apply_async(func, args=arg)
			print('Starting read ' + str(arg[0]))
			start = tm.time()
			try:
				out = res.get(timeout)  # Wait timeout seconds for func to complete.
				print("Succesful read of " +str(arg[0])+ " in %.3g min" %((tm.time() - start)/60))
				p.close()
				return out
				# except multiprocessing.TimeoutError:
			except Exception as e:
				if str(e)=='':
					print("External aborting due to timeout " +str(arg[0])+ " in %.3g s" %(tm.time() - start))
				else:
					print('Error '+str(e))
				output = False
				p.close()
				return output
		netCDF4_read_failed = True
		if os.path.exists(path):
			try:
				# f = netCDF4.Dataset(path)
				f = abortable_worker(netCDF4.Dataset,2,path)	# 2 seconds timeout
				if f==False:
					print(path+' reading failed')
					sba=sgna	# I want this to generate error

				try:
					self.psidat = np.transpose(f['output']['profiles2D']['poloidalFlux'], (0, 2, 1))
					file_prefix = ''
				except IndexError:
					self.psidat = np.transpose(f['/epm/output']['profiles2D']['poloidalFlux'], (0, 2, 1))
					file_prefix = '/epm'

				self.q0 = f[file_prefix+'/output']['globalParameters']['q0'][:].data
				self.q95 = f[file_prefix+'/output']['globalParameters']['q95'][:].data

				try:
					self.r = f[file_prefix+'/output']['profiles2D']['r'][:].data
					self.z = f[file_prefix+'/output']['profiles2D']['z'][:].data
					self.R = self.r[0,:]
					self.Z = self.z[0,:]
					self.bVac = f[file_prefix+'/input']['bVacRadiusProduct']['values'][:].data
					self.r_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis'][:]['R']
					self.z_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis'][:]['Z']
					self.shotnumber = f.pulseNumber
					self.strikepointR = f[file_prefix+'/output']['separatrixGeometry']['strikepointCoords'][:]['R']
					self.strikepointZ = f[file_prefix+'/output']['separatrixGeometry']['strikepointCoords'][:]['Z']
				except IndexError:
					self.R = f[file_prefix+'/output']['profiles2D']['r'][:].data
					self.Z = f[file_prefix+'/output']['profiles2D']['z'][:].data
					self.bVac = f[file_prefix+'/input']['bVacRadiusProduct'][:].data
					self.r_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis']['R'][:]
					self.z_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis']['Z'][:]
					self.shotnumber = f.shot
					self.strikepointR = f[file_prefix+'/output']['separatrixGeometry']['strikepointR'][:].data
					self.strikepointZ = f[file_prefix+'/output']['separatrixGeometry']['strikepointZ'][:].data

				self.psi_bnd = f[file_prefix+'/output']['globalParameters']['psiBoundary'][:].data
				self.psi_axis = f[file_prefix+'/output']['globalParameters']['psiAxis'][:].data
				self.cpasma = f[file_prefix+'/output']['globalParameters']['plasmaCurrent'][:].data
				self.time = f[file_prefix+'/time'][:].data

				f.close()
				netCDF4_read_failed = False
			except:
				print('netCDF4_read_failed')
		if netCDF4_read_failed:
			# efit_reconstruction = read_uda(pulse_ID)
			efit_reconstruction = abortable_worker(read_uda,10,pulse_ID)	# 10 seconds timeout
			if efit_reconstruction==False:
				sba=sgna	# I want this to generate error

			self.psidat = efit_reconstruction['psi']
			self.q0 = efit_reconstruction['q0']
			self.q95 = efit_reconstruction['q95']
			self.R = efit_reconstruction['r']
			self.Z = efit_reconstruction['z']
			self.bVac = efit_reconstruction['RBphi']
			self.r_axis = efit_reconstruction['r_axis']
			self.z_axis = efit_reconstruction['z_axis']
			self.shotnumber = efit_reconstruction['shot']
			# self.strikepointR = efit_reconstruction['strikepointR']
			# self.strikepointZ = efit_reconstruction['strikepointZ']
			self.strikepointR = np.array([efit_reconstruction['outer_strikepoint_upper_r'],efit_reconstruction['outer_strikepoint_lower_r'],client_int.get('epm/output/separatrixGeometry/dndXpoint2InnerStrikepointR',source=pulse_ID).data,client_int.get('epm/output/separatrixGeometry/dndXpoint2InnerStrikepointR',source=pulse_ID).data]).T
			self.strikepointR[np.isnan(self.strikepointR)] = 0
			self.strikepointZ = np.array([efit_reconstruction['outer_strikepoint_upper_z'],efit_reconstruction['outer_strikepoint_lower_z'],client_int.get('epm/output/separatrixGeometry/dndXpoint2InnerStrikepointZ',source=pulse_ID).data,client_int.get('epm/output/separatrixGeometry/dndXpoint1InnerStrikepointZ',source=pulse_ID).data]).T
			self.strikepointZ[np.isnan(self.strikepointZ)] = 0
			self.psi_bnd = efit_reconstruction['psi_boundary']
			self.psi_axis = efit_reconstruction['psi_axis']
			self.cpasma = efit_reconstruction['Ip']
			self.time = efit_reconstruction['t']

		reset_connection(client_int)
		del client_int
		# Calculate the time history of some useful quantites, i.e.
		# Inner and outer LCFS positions
		# Strike point positions
		# q0 at some point?

		print('Calculating equilibrium properties...')

		self.lower_xpoint_r = np.zeros(len(self.time))
		self.lower_xpoint_z = np.zeros(len(self.time))
		self.lower_xpoint_p = np.zeros(len(self.time))
		self.upper_xpoint_r = np.zeros(len(self.time))
		self.upper_xpoint_z = np.zeros(len(self.time))
		self.upper_xpoint_p = np.zeros(len(self.time))

		self.mag_axis_r = np.zeros(len(self.time))
		self.mag_axis_z = np.zeros(len(self.time))
		self.mag_axis_p = np.zeros(len(self.time))

		self.inner_sep_r = np.zeros(len(self.time))
		self.outer_sep_r = np.zeros(len(self.time))

		rr, zz = np.meshgrid(self.R, self.Z)

		if self.psidat is not None:
			for i in np.arange(len(self.time)):
				psiarr = np.array((self.psidat[i,:,:]))
				psi_interp = self.interp2d(self.R, self.Z, psiarr)

				if np.sum(np.isfinite(psiarr)) == np.size(psiarr):
					# Find the position of the xpoints
					opoint, xpoint = self.find_critical(rr.T,zz.T, psiarr.T)

					if len(xpoint) > 0:
						xpt1r = xpoint[0][0]
						xpt1z = xpoint[0][1]
						xpt1p = xpoint[0][2]

						if len(xpoint) > 1:
							xpt2r = xpoint[1][0]
							xpt2z = xpoint[1][1]
							xpt2p = xpoint[1][2]
						else:
							xpt2r = None
							xpt2z = None
							xpt2p = None

						self.mag_axis_r[i] = opoint[0][0]
						self.mag_axis_z[i] = opoint[0][1]
						self.mag_axis_p[i] = opoint[0][2]

						if xpt1z < 0:
							self.lower_xpoint_r[i] = xpt1r
							self.lower_xpoint_z[i] = xpt1z
							self.lower_xpoint_p[i] = xpt1p

						if xpt1z > 0:
							self.upper_xpoint_r[i] = xpt1r
							self.upper_xpoint_z[i] = xpt1z
							self.upper_xpoint_p[i] = xpt1p

						if xpt2z and  xpt2z < 0:
							self.lower_xpoint_r[i] = xpt2r
							self.lower_xpoint_z[i] = xpt2z
							self.lower_xpoint_p[i] = xpt2p

						if xpt2z and xpt2z > 0:
							self.upper_xpoint_r[i] = xpt2r
							self.upper_xpoint_z[i] = xpt2z
							self.upper_xpoint_p[i] = xpt2p

					mp_r_arr = np.linspace(np.min(self.R), np.max(self.R),500)
					mp_p_arr = mp_r_arr*0.0

					for j in np.arange(len(mp_p_arr)):
							mp_p_arr[j] = psi_interp(mp_r_arr[j], self.mag_axis_z[i])

					zcr = self.zcd(mp_p_arr-self.psi_bnd[i])

					# if len(zcr) > 2:	# it should literally have no effect
					# 	zcr = zcr[-2:]
					if len(zcr) < 1:
						zcr = [0,0]
					elif len(zcr) < 2:	# made to prevent the error when there is only one zero in (mp_p_arr-self.psi_bnd[i])
						zcr.append(zcr[0])

					self.inner_sep_r[i] = mp_r_arr[zcr[0]]
					self.outer_sep_r[i] = mp_r_arr[zcr[1]]


					# Calculate dr_sep

	def interp2d(self,R,Z,field):
		return RectBivariateSpline(R,Z,np.transpose(field))


	def find_critical(self,R, Z, psi, discard_xpoints=True):
		"""
		Find critical points

		Inputs
		------

		R - R(nr, nz) 2D array of major radii
		Z - Z(nr, nz) 2D array of heights
		psi - psi(nr, nz) 2D array of psi values

		Returns
		-------

		Two lists of critical points

		opoint, xpoint

		Each of these is a list of tuples with (R, Z, psi) points

		The first tuple is the primary O-point (magnetic axis)
		and primary X-point (separatrix)

		"""

		# Get a spline interpolation function
		f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psi)

		# Find candidate locations, based on minimising Bp^2
		Bp2 = (f(R, Z, dx=1, grid=False) ** 2 + f(R, Z, dy=1, grid=False) ** 2) / R ** 2

		# Get grid resolution, which determines a reasonable tolerance
		# for the Newton iteration search area
		dR = R[1, 0] - R[0, 0]
		dZ = Z[0, 1] - Z[0, 0]
		radius_sq = 9 * (dR ** 2 + dZ ** 2)

		# Find local minima

		J = np.zeros([2, 2])

		xpoint = []
		opoint = []

		nx, ny = Bp2.shape
		for i in range(2, nx - 2):
			for j in range(2, ny - 2):
				if (
					(Bp2[i, j] < Bp2[i + 1, j + 1])
					and (Bp2[i, j] < Bp2[i + 1, j])
					and (Bp2[i, j] < Bp2[i + 1, j - 1])
					and (Bp2[i, j] < Bp2[i - 1, j + 1])
					and (Bp2[i, j] < Bp2[i - 1, j])
					and (Bp2[i, j] < Bp2[i - 1, j - 1])
					and (Bp2[i, j] < Bp2[i, j + 1])
					and (Bp2[i, j] < Bp2[i, j - 1])
				):

					# Found local minimum

					R0 = R[i, j]
					Z0 = Z[i, j]

					# Use Newton iterations to find where
					# both Br and Bz vanish
					R1 = R0
					Z1 = Z0

					count = 0
					while True:

						Br = -f(R1, Z1, dy=1, grid=False) / R1
						Bz = f(R1, Z1, dx=1, grid=False) / R1

						if Br ** 2 + Bz ** 2 < 1e-6:
							# Found a minimum. Classify as either
							# O-point or X-point

							dR = R[1, 0] - R[0, 0]
							dZ = Z[0, 1] - Z[0, 0]
							d2dr2 = (psi[i + 2, j] - 2.0 * psi[i, j] + psi[i - 2, j]) / (
								2.0 * dR
							) ** 2
							d2dz2 = (psi[i, j + 2] - 2.0 * psi[i, j] + psi[i, j - 2]) / (
								2.0 * dZ
							) ** 2
							d2drdz = (
								(psi[i + 2, j + 2] - psi[i + 2, j - 2]) / (4.0 * dZ)
								- (psi[i - 2, j + 2] - psi[i - 2, j - 2]) / (4.0 * dZ)
							) / (4.0 * dR)
							D = d2dr2 * d2dz2 - d2drdz ** 2

							if D < 0.0:
								# Found X-point
								xpoint.append((R1, Z1, f(R1, Z1)[0][0]))
							else:
								# Found O-point
								opoint.append((R1, Z1, f(R1, Z1)[0][0]))
							break

						# Jacobian matrix
						# J = ( dBr/dR, dBr/dZ )
						#	 ( dBz/dR, dBz/dZ )

						J[0, 0] = -Br / R1 - f(R1, Z1, dy=1, dx=1)[0][0] / R1
						J[0, 1] = -f(R1, Z1, dy=2)[0][0] / R1
						J[1, 0] = -Bz / R1 + f(R1, Z1, dx=2) / R1
						J[1, 1] = f(R1, Z1, dx=1, dy=1)[0][0] / R1

						d = np.dot(np.linalg.inv(J), [Br, Bz])

						R1 = R1 - d[0]
						Z1 = Z1 - d[1]

						count += 1
						# If (R1,Z1) is too far from (R0,Z0) then discard
						# or if we've taken too many iterations
						if ((R1 - R0) ** 2 + (Z1 - Z0) ** 2 > radius_sq) or (count > 100):
							# Discard this point
							break

		# Remove duplicates
		def remove_dup(points):
			result = []
			for n, p in enumerate(points):
				dup = False
				for p2 in result:
					if (p[0] - p2[0]) ** 2 + (p[1] - p2[1]) ** 2 < 1e-5:
						dup = True  # Duplicate
						break
				if not dup:
					result.append(p)  # Add to the list
			return result

		xpoint = remove_dup(xpoint)
		opoint = remove_dup(opoint)

		if len(opoint) == 0:
			# Can't order primary O-point, X-point so return
			print("Warning: No O points found")
			return opoint, xpoint

		# Find primary O-point by sorting by distance from middle of domain
		Rmid = 0.5 * (R[-1, 0] + R[0, 0])
		Zmid = 0.5 * (Z[0, -1] + Z[0, 0])
		opoint.sort(key=lambda x: (x[0] - Rmid) ** 2 + (x[1] - Zmid) ** 2)

		# Draw a line from the O-point to each X-point. Psi should be
		# monotonic; discard those which are not

		if discard_xpoints:
			Ro, Zo, Po = opoint[0]  # The primary O-point
			xpt_keep = []
			for xpt in xpoint:
				Rx, Zx, Px = xpt

				rline = np.linspace(Ro, Rx, num=50)
				zline = np.linspace(Zo, Zx, num=50)

				pline = f(rline, zline, grid=False)

				if Px < Po:
					pline *= -1.0  # Reverse, so pline is maximum at X-point

				# Now check that pline is monotonic
				# Tried finding maximum (argmax) and testing
				# how far that is from the X-point. This can go
				# wrong because psi can be quite flat near the X-point
				# Instead here look for the difference in psi
				# rather than the distance in space

				maxp = np.amax(pline)
				if (maxp - pline[-1]) / (maxp - pline[0]) > 0.001:
					# More than 0.1% drop in psi from maximum to X-point
					# -> Discard
					continue

				ind = np.argmin(pline)  # Should be at O-point
				if (rline[ind] - Ro) ** 2 + (zline[ind] - Zo) ** 2 > 1e-4:
					# Too far, discard
					continue
				xpt_keep.append(xpt)
			xpoint = xpt_keep

		# Sort X-points by distance to primary O-point in psi space
		psi_axis = opoint[0][2]
		xpoint.sort(key=lambda x: (x[2] - psi_axis) ** 2)

		return opoint, xpoint


	def zcd(self, data):
		sign_array=np.sign(data)
		out=[]
		for i in np.arange(1,len(sign_array)):
			if sign_array[i] != sign_array[i-1]:
				out.append(i)
		return out

############################################################################################################################################################################################################################################################################
# from https://gist.github.com/derricw/95eab740e1b08b78c03f
def bin_ndarray(ndarray, new_shape, operation='sum'):
	"""
	Bins an ndarray in all axes based on the target shape, by summing or
		averaging.

	Number of output dimensions must match number of input dimensions and
		new axes must divide old ones.

	Example
	-------
	>>> m = np.arange(0,100,1).reshape((10,10))
	>>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
	>>> print(n)

	[[ 22  30  38  46  54]
	 [102 110 118 126 134]
	 [182 190 198 206 214]
	 [262 270 278 286 294]
	 [342 350 358 366 374]]

	"""
	# if operation=='np.nanmean':
	# 	operation='mean'
	# if operation=='np.nansum':
	# 	operation='sum'

	operation = operation.lower()
	if not operation in ['sum', 'mean','np.nansum','np.nanmean','np.nanstd']:
		raise ValueError("Operation not supported.")
	if ndarray.ndim != len(new_shape):
		raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
														   new_shape))
	compression_pairs = [(d, c//d) for d,c in zip(new_shape,
												  ndarray.shape)]
	flattened = [l for p in compression_pairs for l in p]
	ndarray = ndarray.reshape(flattened)
	for i in range(len(new_shape)):
		if operation in ['mean','sum']:
			op = getattr(ndarray, operation)
			ndarray = op(-1*(i+1))
		elif operation == 'np.nanmean':
			ndarray = np.nanmean(ndarray, axis=-1*(i+1) )
		elif operation == 'np.nansum':
			ndarray = np.nansum(ndarray, axis=-1*(i+1) )
		elif operation == 'np.nanstd':
			ndarray = np.nanstd(ndarray, axis=-1*(i+1) )
	return ndarray

def proper_homo_binning_t_2D(data,shrink_factor_t,shrink_factor_x,type='np.nanmean'):
	old_shape = np.array(np.shape(data))
	new_shape=np.array([int(np.ceil(old_shape[0]/shrink_factor_t)),int(np.ceil(old_shape[1]/shrink_factor_x)),int(np.ceil(old_shape[2]/shrink_factor_x))]).astype(int)
	to_pad=np.array([(shrink_factor_t-old_shape[0]%shrink_factor_t)*(old_shape[0]%shrink_factor_t>0),(shrink_factor_x-old_shape[1]%shrink_factor_x)*(old_shape[1]%shrink_factor_x>0),(shrink_factor_x-old_shape[2]%shrink_factor_x)*(old_shape[2]%shrink_factor_x>0)]).astype(int)
	to_pad_right = to_pad//2
	to_pad_left = to_pad - to_pad_right
	to_pad = np.array([to_pad_left,to_pad_right]).T
	# data_binned = np.pad(data,to_pad,mode='mean',stat_length=((max(1,shrink_factor_t//2),max(1,shrink_factor_t//2)),(max(1,shrink_factor_x//2),max(1,shrink_factor_x//2)),(max(1,shrink_factor_x//2),max(1,shrink_factor_x//2))))
	data_binned = np.pad(np.array(data).astype(float),to_pad,mode='constant',constant_values=np.nan)
	data_binned = bin_ndarray(data_binned, new_shape=new_shape, operation=type)
	nan_ROI_mask = np.isfinite(np.nanmedian(data_binned[:10],axis=0))
	return data_binned,nan_ROI_mask

def proper_homo_binning_2D(data,shrink_factor_x,type='np.nanmean',shrink_factor_x2=0):
	if shrink_factor_x2==0:
		shrink_factor_x2 = shrink_factor_x
	old_shape = np.array(np.shape(data))
	new_shape=np.array([int(np.ceil(old_shape[0]/shrink_factor_x)),int(np.ceil(old_shape[1]/shrink_factor_x2))]).astype(int)
	to_pad=np.array([(shrink_factor_x-old_shape[0]%shrink_factor_x)*(old_shape[0]%shrink_factor_x>0),(shrink_factor_x2-old_shape[1]%shrink_factor_x2)*(old_shape[1]%shrink_factor_x2>0)]).astype(int)
	to_pad_right = to_pad//2
	to_pad_left = to_pad - to_pad_right
	to_pad = np.array([to_pad_left,to_pad_right]).T
	# data_binned = np.pad(data,to_pad,mode='mean',stat_length=((max(1,shrink_factor_x//2),max(1,shrink_factor_x//2)),(max(1,shrink_factor_x//2),max(1,shrink_factor_x//2))))
	data_binned = np.pad(np.array(data).astype(float),to_pad,mode='constant',constant_values=np.nan)
	data_binned = bin_ndarray(data_binned, new_shape=new_shape, operation=type)
	return data_binned

def proper_homo_binning_1D_1D_1D(data,shrink_factor_x_1,shrink_factor_x_2,shrink_factor_v,type='np.nanmean'):	# v stands for voxel, because this will be used for the sensitivity matrix
	old_shape = np.array(np.shape(data))
	new_shape=np.array([int(np.ceil(old_shape[0]/shrink_factor_x_1)),int(np.ceil(old_shape[1]/shrink_factor_x_2)),int(np.ceil(old_shape[2]/shrink_factor_v))]).astype(int)
	to_pad=np.array([(shrink_factor_x_1-old_shape[0]%shrink_factor_x_1)*(old_shape[0]%shrink_factor_x_1>0),(shrink_factor_x_2-old_shape[1]%shrink_factor_x_2)*(old_shape[1]%shrink_factor_x_2>0),(shrink_factor_v-old_shape[2]%shrink_factor_v)*(old_shape[2]%shrink_factor_v>0)]).astype(int)
	to_pad_right = to_pad//2
	to_pad_left = to_pad - to_pad_right
	to_pad = np.array([to_pad_left,to_pad_right]).T
	# data_binned = np.pad(data,to_pad,mode='mean',stat_length=((max(1,shrink_factor_x_1//2),max(1,shrink_factor_x_1//2)),(max(1,shrink_factor_x_2//2),max(1,shrink_factor_x_2//2)),(max(1,shrink_factor_v//2),max(1,shrink_factor_v//2))))
	data_binned = np.pad(np.array(data).astype(float),to_pad,mode='constant',constant_values=np.nan)
	data_binned = bin_ndarray(data_binned, new_shape=new_shape, operation=type)
	return data_binned

def proper_homo_binning_t(time,shrink_factor_t,type='np.nanmean'):
	old_shape = np.array(np.shape(time))
	new_shape=np.array([int(np.ceil(old_shape[0]/shrink_factor_t))]).astype(int)
	to_pad=np.array([(shrink_factor_t-old_shape[0]%shrink_factor_t)*(old_shape[0]%shrink_factor_t>0)]).astype(int)
	to_pad_right = to_pad//2
	to_pad_left = to_pad - to_pad_right
	to_pad = np.array([to_pad_left,to_pad_right]).T
	# time_binned = np.pad(time,to_pad,mode='mean',stat_length=((max(1,shrink_factor_t//2),max(1,shrink_factor_t//2))))
	time_binned = np.pad(np.array(time).astype(float),to_pad,mode='constant',constant_values=np.nan)
	time_binned = bin_ndarray(time_binned, new_shape=new_shape, operation=type)
	# time_binned[0] = time_binned[1] - np.median(np.diff(time_binned[1:-1]))	# I'm not sure if it's proper to leave this lines, because with the binning it can actually be right that all dt are not the same. I remove it for now
	# time_binned[-1] = time_binned[-2] + np.median(np.diff(time_binned[1:-1]))
	return time_binned

def efit_reconstruction_to_separatrix_on_foil(efit_reconstruction,refinement=1000):
	from scipy.signal import find_peaks, peak_prominences as get_proms
	from scipy.interpolate import interp2d
	all_time_sep_r = []
	all_time_sep_z = []
	r_fine = np.unique(np.linspace(efit_reconstruction.R.min(),efit_reconstruction.R.max(),refinement).tolist() + np.linspace(R_centre_column-0.01,R_centre_column+0.08,refinement).tolist())
	r_fine = r_fine[r_fine>=R_centre_column-0.01]	# this is to avoid ambiguity around the x-point
	z_fine = np.linspace(efit_reconstruction.Z.min(),efit_reconstruction.Z.max(),refinement)
	interp1 = interp1d([1.1,1.5],[-1.5,-1.75],fill_value="extrapolate",bounds_error=False)
	interp2 = interp1d([1.1,1.5],[-1.5,-1.2],fill_value="extrapolate",bounds_error=False)
	for time in range(len(efit_reconstruction.time)):
		gna = efit_reconstruction.psidat[time]
		sep_up = efit_reconstruction.upper_xpoint_p[time]
		sep_low = efit_reconstruction.lower_xpoint_p[time]
		x_point_z_proximity = np.abs(np.nanmin([efit_reconstruction.upper_xpoint_z[time],efit_reconstruction.lower_xpoint_z[time],-0.573-0.2]))-0.2	# -0.573 is an arbitrary treshold in case both are nan

		psi_interpolator = interp2d(efit_reconstruction.R,efit_reconstruction.Z,gna)
		psi = psi_interpolator(r_fine,z_fine)
		psi_up = -np.abs(psi-sep_up)
		psi_low = -np.abs(psi-sep_low)
		all_peaks_up = []
		all_z_up = []
		all_peaks_low = []
		all_z_low = []
		z_array = np.arange(len(z_fine))
		z_array = np.array([np.concatenate((z_array[z_fine>=0],np.flip(z_array[z_fine<0],axis=0))),np.concatenate((z_fine[z_fine>=0],np.flip(z_fine[z_fine<0],axis=0)))]).T
		# for i_z,z in enumerate(z_fine):
		for i_z,z in z_array:
			i_z = int(i_z)
			# psi_z = psi[i_z]
			peaks = find_peaks(psi_up[i_z])[0]
			all_peaks_up.append(peaks)
			all_z_up.append([i_z]*len(peaks))
			peaks = find_peaks(psi_low[i_z])[0]
			all_peaks_low.append(peaks)
			all_z_low.append([i_z]*len(peaks))
		all_peaks_up = np.concatenate(all_peaks_up).astype(int)
		all_z_up = np.concatenate(all_z_up).astype(int)
		found_psi_up = np.abs(psi_up[all_z_up,all_peaks_up])
		all_peaks_low = np.concatenate(all_peaks_low).astype(int)
		all_z_low = np.concatenate(all_z_low).astype(int)
		found_psi_low = np.abs(psi_low[all_z_low,all_peaks_low])

		# plt.figure()
		# plt.plot(z_fine[all_z_up[found_psi_up<(gna.max()-gna.min())/500]],r_fine[all_peaks_up[found_psi_up<(gna.max()-gna.min())/500]],'+b')
		# plt.plot(z_fine[all_z_low[found_psi_low<(gna.max()-gna.min())/500]],r_fine[all_peaks_low[found_psi_low<(gna.max()-gna.min())/500]],'+r')
		all_peaks_up = all_peaks_up[found_psi_up<(gna.max()-gna.min())/500]
		all_z_up = all_z_up[found_psi_up<(gna.max()-gna.min())/500]
		all_peaks_low = all_peaks_low[found_psi_low<(gna.max()-gna.min())/500]
		all_z_low = all_z_low[found_psi_low<(gna.max()-gna.min())/500]

		# plt.figure()
		# plt.plot(z_fine[all_z_up],r_fine[all_peaks_up],'+b')
		# plt.plot(z_fine[all_z_low],r_fine[all_peaks_low],'+r')
		# plt.pause(0.01)
		select = np.logical_or( interp1(r_fine[all_peaks_up])>z_fine[all_z_up] , interp2(r_fine[all_peaks_up])<z_fine[all_z_up] )
		all_peaks_up = all_peaks_up[select]
		all_z_up = all_z_up[select]
		select = np.logical_or( interp1(r_fine[all_peaks_low])>z_fine[all_z_low] , interp2(r_fine[all_peaks_low])<z_fine[all_z_low] )
		all_peaks_low = all_peaks_low[select]
		all_z_low = all_z_low[select]

		left_up = []
		right_up = []
		left_up_z = []
		right_up_z = []
		left_low = []
		right_low = []
		left_low_z = []
		right_low_z = []
		# for i_z,z in enumerate(z_fine):
		for i_z,z in z_array:
			i_z = int(i_z)
			if i_z in all_z_up:
				temp = all_peaks_up[all_z_up==i_z]
				if len(temp) == 1:
					right_up.append(temp[0])
					right_up_z.append(i_z)
				elif len(temp) == 2:
					# # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
					# if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
					if np.abs(z)>1.5 and np.sum(r_fine[temp]<0.8)==0:	# this identifies the condition when the outer separatrix curles up and it gets split in inner and outer
						right_up.append(temp[np.abs(temp - right_up[-1]).argmin()])
						right_up_z.append(i_z)
					else:
						left_up.append(temp.min())
						left_up_z.append(i_z)
						right_up.append(temp.max())
						right_up_z.append(i_z)
				elif len(temp) == 3:
					if np.abs(z)>1.5 and np.sum(r_fine[temp]<0.8)==0:	# this identifies the condition when the outer separatrix curles up and it gets split in inner and outer
						right_up.append(temp[np.abs(temp - right_up[-1]).argmin()])
						right_up_z.append(i_z)
					else:
						left_up.append(np.sort(temp)[1])
						left_up_z.append(i_z)
						right_up.append(temp.max())
						right_up_z.append(i_z)
				elif len(temp) == 4:
					left_up.append(np.sort(temp)[1])
					left_up_z.append(i_z)
					right_up.append(np.sort(temp)[2])
					right_up_z.append(i_z)
			if i_z in all_z_low:
				temp = all_peaks_low[all_z_low==i_z]
				if len(temp) == 1:
					right_low.append(temp[0])
					right_low_z.append(i_z)
				elif len(temp) == 2:
					# # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
					# if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
					if np.abs(z)>1.5 and np.sum(r_fine[temp]<0.8)==0:	# this identifies the condition when the outer separatrix curles up and it gets split in inner and outer
						right_low.append(temp[np.abs(temp - right_low[-1]).argmin()])
						right_low_z.append(i_z)
					else:
						left_low.append(temp.min())
						left_low_z.append(i_z)
						right_low.append(temp.max())
						right_low_z.append(i_z)
				elif len(temp) == 3:
					if np.abs(z)>1.5 and np.sum(r_fine[temp]<0.8)==0:	# this identifies the condition when the outer separatrix curles up and it gets split in inner and outer
						right_low.append(temp[np.abs(temp - right_low[-1]).argmin()])
						right_low_z.append(i_z)
					else:
						left_low.append(np.sort(temp)[1])
						left_low_z.append(i_z)
						right_low.append(temp.max())
						right_low_z.append(i_z)
				elif len(temp) == 4:
					left_low.append(np.sort(temp)[1])
					left_low_z.append(i_z)
					right_low.append(np.sort(temp)[2])
					right_low_z.append(i_z)
		# sep_r = [left_up,right_up,left_low,right_low]
		# sep_z = [left_up_z,right_up_z,left_low_z,right_low_z]
		left_up = np.array([y for _, y in sorted(zip(left_up_z, left_up))])
		left_up_z = np.sort(left_up_z)
		right_up = np.array([y for _, y in sorted(zip(right_up_z, right_up))])
		right_up_z = np.sort(right_up_z)
		left_low = np.array([y for _, y in sorted(zip(left_low_z, left_low))])
		left_low_z = np.sort(left_low_z)
		right_low = np.array([y for _, y in sorted(zip(right_low_z, right_low))])
		right_low_z = np.sort(right_low_z)
		# plt.figure()
		# plt.plot(z_fine[left_up_z],r_fine[left_up],'-b')
		# plt.plot(z_fine[right_up_z],r_fine[right_up],'-r')
		# plt.plot(z_fine[left_low_z],r_fine[left_low],'-b')
		# plt.plot(z_fine[right_low_z],r_fine[right_low],'-r')
		# plt.pause(0.01)
		all_time_sep_r.append([left_up,right_up,left_low,right_low])
		all_time_sep_z.append([left_up_z,right_up_z,left_low_z,right_low_z])
	return all_time_sep_r,all_time_sep_z,r_fine,z_fine


##########################################################################################################################################################################

def get_rotation_crop_parameters(testrot,foil_position_dict,laser_to_analyse,plasma_data_array,plasma_data_array_time,foilhorizw=0.09,foilvertw=0.07):
	# Created 2021/07/23 only to obtain the rotation / cropping parameters
	from scipy.ndimage.filters import generic_filter

	rotangle = foil_position_dict['angle'] #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter = foil_position_dict['foilcenter']
	foilhorizwpixel = foil_position_dict['foilhorizwpixel']
	# foilvertwpixel=int((foilhorizwpixel*foilvertw)//foilhorizw)
	foilvertwpixel=int(round((foilhorizwpixel*foilvertw)/foilhorizw))
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype('int')
	foilyint=(np.rint(foily)).astype('int')

	plt.figure(figsize=(20, 10))
	plt.title('Foil search in '+laser_to_analyse+'\nFoil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.ylabel('Horizontal axis [pixles]')
	plt.xlabel('Vertical axis [pixles]')
	temp = np.sort(testrot[testrot>0])
	plt.clim(vmin=np.nanmean(temp[:max(20,int(len(temp)/20))]), vmax=np.nanmean(temp[-max(20,int(len(temp)/20)):]))
	# plt.clim(vmax=27.1,vmin=26.8)
	# plt.clim(vmin=np.nanmin(testrot[testrot>0]), vmax=np.nanmax(testrot))
	plt.colorbar().set_label('counts [au]')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.savefig(laser_to_analyse[:-4]+'_foil_fit.eps', bbox_inches='tight')
	plt.close('all')

	plt.figure(figsize=(20, 10))
	temp = np.max(plasma_data_array[:,:,:200],axis=(-1,-2)).argmax()
	plt.title('Foil search in '+laser_to_analyse+'\nFoil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel\nhottest frame at %.4gsec' %(plasma_data_array_time[temp]))
	plt.imshow(plasma_data_array[temp],'rainbow',origin='lower',vmax=np.max(plasma_data_array[:,:,:200],axis=(-1,-2))[temp])
	plt.ylabel('Horizontal axis [pixles]')
	plt.xlabel('Vertical axis [pixles]')
	plt.clim(vmax=np.max(plasma_data_array[:,:,:200],axis=(-1,-2))[temp])
	plt.colorbar().set_label('count increase [au]')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.savefig(laser_to_analyse[:-4]+'_foil_fit_plasma.eps', bbox_inches='tight')
	plt.close('all')
	testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
	precisionincrease=25
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(np.rint(foily[0]*precisionincrease)),int(np.rint(foilx[0]*precisionincrease))]=3
	dummy[int(np.rint(foily[1]*precisionincrease)),int(np.rint(foilx[1]*precisionincrease))]=4
	dummy[int(np.rint(foily[2]*precisionincrease)),int(np.rint(foilx[2]*precisionincrease))]=5
	dummy[int(np.rint(foily[3]*precisionincrease)),int(np.rint(foilx[3]*precisionincrease))]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype('int')
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype('int')
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype('int')
	# plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
	# plt.plot(foilxrot,foilyrot,'r')
	# plt.title('Foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	# plt.colorbar().set_label('counts [au]')
	# plt.pause(0.01)

	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	out_of_ROI_mask = np.ones_like(testrotback)
	out_of_ROI_mask[testrotback<np.nanmin(testrot[testrot>0])]=np.nan
	out_of_ROI_mask[testrotback>np.nanmax(testrot[testrot>0])]=np.nan
	a = generic_filter((testrotback),np.std,size=[19,19])
	out_of_ROI_mask[a>np.mean(a)]=np.nan

	return foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx


def MASTU_pulse_process_FAST(laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,laser_framerate,laser_digitizer_ID,laser_int_time,seconds_for_reference_frame,start_time_of_pulse,laser_to_analyse,height,width,flag_use_of_first_frames_as_reference):
	# created 2021/07/23
	# Here I do a fast analysis of the unfiltered data so I can look at it during experiments
	max_ROI = [[0,255],[0,319]]
	# foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
	foil_position_dict = dict([('angle',0.7),('foilcenter',[157,136]),('foilhorizwpixel',240)])	# modified 2021/09/21 to match sensitivity matrix
	temp_ref_counts = []
	temp_counts_minus_background = []
	time_partial = []
	timesteps = np.inf
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		if external_clock_marker:
			time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers
		if flag_use_of_first_frames_as_reference:
			# temp_ref_counts.append(np.mean(laser_counts_corrected[i][time_of_experiment_digitizer_ID_seconds<0],axis=0))
			temp_ref_counts.append(np.mean(laser_counts_corrected[i][np.logical_and(time_of_experiment_digitizer_ID_seconds<0,time_of_experiment_digitizer_ID_seconds>-0.5)],axis=0))
		else:
			temp_ref_counts.append(np.mean(laser_counts_corrected[i][-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))
		select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds>=0,time_of_experiment_digitizer_ID_seconds<=1.5)
		temp_counts_minus_background.append(laser_counts_corrected[i][select_time]-temp_ref_counts[-1])
		time_partial.append(time_of_experiment_digitizer_ID_seconds[select_time])
		timesteps = min(timesteps,len(temp_counts_minus_background[-1]))

	for i in range(len(laser_digitizer_ID)):
		temp_counts_minus_background[i] = temp_counts_minus_background[i][:timesteps]
		time_partial[i] = time_partial[i][:timesteps]
	temp_counts_minus_background = np.nanmean(temp_counts_minus_background,axis=0)
	temp_ref_counts = np.nanmean(temp_ref_counts,axis=0)
	FAST_counts_minus_background_crop_time = np.nanmean(time_partial,axis=0)

	# I'm going to use the reference frames for foil position
	foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx = get_rotation_crop_parameters(temp_ref_counts,foil_position_dict,laser_to_analyse,temp_counts_minus_background,FAST_counts_minus_background_crop_time)

	# rotation and crop
	temp_counts_minus_background_rot=rotate(temp_counts_minus_background,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		temp_counts_minus_background_rot*=out_of_ROI_mask
		temp_counts_minus_background_rot[np.logical_and(temp_counts_minus_background_rot<np.nanmin(temp_counts_minus_background[i]),temp_counts_minus_background_rot>np.nanmax(temp_counts_minus_background[i]))]=0
	FAST_counts_minus_background_crop = temp_counts_minus_background_rot[:,foildw:foilup,foillx:foilrx]

	temp = FAST_counts_minus_background_crop[:,:,:int(np.shape(FAST_counts_minus_background_crop)[2]*0.75)]
	temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
	ani = movie_from_data(np.array([np.flip(np.transpose(FAST_counts_minus_background_crop,(0,2,1)),axis=2)]), laser_framerate/len(laser_digitizer_ID),integration=laser_int_time/1000,time_offset=FAST_counts_minus_background_crop_time[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Count increase [au]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
	ani.save(laser_to_analyse[:-4]+ '_FAST_count_increase.mp4', fps=5*laser_framerate/len(laser_digitizer_ID)/383, writer='ffmpeg',codec='mpeg4')
	plt.close('all')

	print('completed FAST rotating/cropping ' + laser_to_analyse)

	return foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop,FAST_counts_minus_background_crop_time

#######################################################################################################################################################################################################


def MASTU_pulse_process_FAST2(laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,laser_framerate,laser_digitizer_ID,laser_int_time,seconds_for_reference_frame,start_time_of_pulse,laser_to_analyse,height,width,flag_use_of_first_frames_as_reference,params,foil_position_dict):
	# created 2021/09/10
	# created modifying MASTU_pulse_process_FAST in order t have a quick power on the foil with a primitive filtering

	from scipy.ndimage.filters import generic_filter

	max_ROI = [[0,255],[0,319]]
	# foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
	temp_ref_counts = []
	temp_counts_minus_background = []
	time_partial = []
	timesteps = np.inf
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		if external_clock_marker:
			time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers

		# basic smoothing
		spectra_orig=np.fft.fft(np.mean(laser_counts_corrected[i],axis=(-1,-2)))
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		freq = np.fft.fftfreq(len(magnitude), d=np.mean(np.diff(time_of_experiment_digitizer_ID_seconds)))
		magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
		freq = np.sort(freq)
		magnitude_smooth = generic_filter(np.log(magnitude),np.median,size=[7])
		peak_oscillation = (magnitude-np.exp(magnitude_smooth))[np.logical_and(freq>10,freq<50)].argmax()
		peak_oscillation_freq = freq[np.logical_and(freq>10,freq<50)][peak_oscillation]
		frames_to_average = 1/peak_oscillation_freq/np.mean(np.diff(time_of_experiment_digitizer_ID_seconds))
		laser_counts_corrected_filtered = real_mean_filter_agent(laser_counts_corrected[i],frames_to_average)

		if flag_use_of_first_frames_as_reference:
			# temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[time_of_experiment_digitizer_ID_seconds<0],axis=0))
			temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<0,time_of_experiment_digitizer_ID_seconds>-0.5)],axis=0))
		else:
			temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))
		select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds>=0,time_of_experiment_digitizer_ID_seconds<=1.5)
		temp_counts_minus_background.append(laser_counts_corrected_filtered[select_time]-temp_ref_counts[-1])
		time_partial.append(time_of_experiment_digitizer_ID_seconds[select_time])
		timesteps = min(timesteps,len(temp_counts_minus_background[-1]))

	for i in range(len(laser_digitizer_ID)):
		temp_counts_minus_background[i] = temp_counts_minus_background[i][:timesteps]
		time_partial[i] = time_partial[i][:timesteps]
	temp_counts_minus_background = np.nanmean(temp_counts_minus_background,axis=0)
	temp_ref_counts = np.nanmean(temp_ref_counts,axis=0)
	FAST_counts_minus_background_crop_time = np.nanmean(time_partial,axis=0)

	# I'm going to use the reference frames for foil position
	foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx = get_rotation_crop_parameters(temp_ref_counts,foil_position_dict,laser_to_analyse,temp_counts_minus_background,FAST_counts_minus_background_crop_time)

	# rotation and crop
	temp_counts_minus_background_rot=rotate(temp_counts_minus_background,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		temp_counts_minus_background_rot*=out_of_ROI_mask
		temp_counts_minus_background_rot[np.logical_and(temp_counts_minus_background_rot<np.nanmin(temp_counts_minus_background),temp_counts_minus_background_rot>np.nanmax(temp_counts_minus_background))]=0
	FAST_counts_minus_background_crop = temp_counts_minus_background_rot[:,foildw:foilup,foillx:foilrx]

	temp = FAST_counts_minus_background_crop[:,:,:int(np.shape(FAST_counts_minus_background_crop)[2]*0.75)]
	temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
	ani = movie_from_data(np.array([np.flip(np.transpose(FAST_counts_minus_background_crop,(0,2,1)),axis=2)]), laser_framerate/len(laser_digitizer_ID),timesteps=FAST_counts_minus_background_crop_time,integration=laser_int_time/1000,time_offset=FAST_counts_minus_background_crop_time[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Count increase [au]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
	ani.save(laser_to_analyse[:-4]+ '_FAST_count_increase.mp4', fps=5*laser_framerate/len(laser_digitizer_ID)/383, writer='ffmpeg',codec='mpeg4')
	plt.close('all')

	print('completed FAST count rotating/cropping ' + laser_to_analyse)

	params = np.mean(params,axis=(0))
	temperature = params[:,:,-1] + params[:,:,-2] * (temp_counts_minus_background+temp_ref_counts) + params[:,:,-3] * ((temp_counts_minus_background+temp_ref_counts)**2)
	temperature_ref = params[:,:,-1] + params[:,:,-2] * temp_ref_counts + params[:,:,-3] * (temp_ref_counts**2)

	# rotation and crop
	temperature_rot=rotate(temperature,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		temperature_rot*=out_of_ROI_mask
		temperature_rot[np.logical_and(temperature_rot<np.nanmin(temperature),temperature_rot>np.nanmax(temperature))]=0
	temperature_crop = temperature_rot[:,foildw:foilup,foillx:foilrx]

	# rotation and crop
	temperature_ref_rot=rotate(temperature_ref,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		temperature_ref_rot*=out_of_ROI_mask
		temperature_ref_rot[np.logical_and(temperature_ref_rot<np.nanmin(temperature_ref),temperature_ref_rot>np.nanmax(temperature_ref))]=0
	temperature_ref_crop = temperature_ref_rot[foildw:foilup,foillx:foilrx]

	temperature_minus_background_crop = temperature_crop-temperature_ref_crop

	shrink_factor_t = int(round(frames_to_average))
	shrink_factor_x = 3	# with large time averaging this should be enough
	binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)

	FAST_counts_minus_background_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(FAST_counts_minus_background_crop,shrink_factor_t,shrink_factor_x)
	temperature_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(temperature_crop,shrink_factor_t,shrink_factor_x)
	temperature_minus_background_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(temperature_minus_background_crop,shrink_factor_t,shrink_factor_x)
	temperature_ref_crop_binned = proper_homo_binning_2D(temperature_ref_crop,shrink_factor_x)
	time_binned = proper_homo_binning_t(FAST_counts_minus_background_crop_time,shrink_factor_t)

	# reference foil properties
	# thickness = 1.4859095354482858e-06
	# emissivity = 0.9884061389741369
	# diffusivity = 1.045900223180454e-05
	# from 2021/09/17, Laser_data_analysis3_3.py
	thickness = 2.0531473351462095e-06
	emissivity = 0.9999999999999
	diffusivity = 1.0283685197530968e-05
	Ptthermalconductivity=71.6 #[W/(m·K)]
	zeroC=273.15 #K / C
	sigmaSB=5.6704e-08 #[W/(m2 K4)]

	foilemissivityscaled=emissivity*np.ones(np.array(temperature_ref_crop_binned.shape)-2)
	foilthicknessscaled=thickness*np.ones(np.array(temperature_ref_crop_binned.shape)-2)
	conductivityscaled=Ptthermalconductivity*np.ones(np.array(temperature_ref_crop_binned.shape)-2)
	reciprdiffusivityscaled=(1/diffusivity)*np.ones(np.array(temperature_ref_crop_binned.shape)-2)

	dt = time_binned[2:]-time_binned[:-2]
	dx=foil_position_dict['foilhorizw']/foil_position_dict['foilhorizwpixel']*shrink_factor_x
	dTdt=np.divide((temperature_crop_binned[2:,1:-1,1:-1]-temperature_crop_binned[:-2,1:-1,1:-1]).T,dt).T.astype(np.float32)
	d2Tdx2=np.divide(temperature_minus_background_crop_binned[1:-1,1:-1,2:]-np.multiply(2,temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])+temperature_minus_background_crop_binned[1:-1,1:-1,:-2],dx**2).astype(np.float32)
	d2Tdy2=np.divide(temperature_minus_background_crop_binned[1:-1,2:,1:-1]-np.multiply(2,temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])+temperature_minus_background_crop_binned[1:-1,:-2,1:-1],dx**2).astype(np.float32)
	d2Tdxy = np.ones_like(dTdt).astype(np.float32)*np.nan
	d2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]=np.add(d2Tdx2[:,nan_ROI_mask[1:-1,1:-1]],d2Tdy2[:,nan_ROI_mask[1:-1,1:-1]])
	del d2Tdx2,d2Tdy2
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	T4=(temperature_minus_background_crop_binned[1:-1,1:-1,1:-1]+np.nanmean(temperature_ref_crop_binned)+zeroC)**4
	T04=(np.nanmean(temperature_ref_crop_binned)+zeroC)**4 *np.ones_like(temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])
	T4_T04 = np.ones_like(dTdt).astype(np.float32)*np.nan
	T4_T04[:,nan_ROI_mask[1:-1,1:-1]] = (T4[:,nan_ROI_mask[1:-1,1:-1]]-T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)

	BBrad = np.ones_like(dTdt).astype(np.float32)*np.nan
	BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04[:,nan_ROI_mask[1:-1,1:-1]] * foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	diffusion = np.ones_like(dTdt).astype(np.float32)*np.nan
	diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	timevariation = np.ones_like(dTdt).astype(np.float32)*np.nan
	timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	powernoback = (diffusion + timevariation + BBrad).astype(np.float32)

	horizontal_coord = np.arange(np.shape(powernoback[0])[1])
	vertical_coord = np.arange(np.shape(powernoback[0])[0])
	horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
	horizontal_coord = (horizontal_coord+1+0.5)*dx	# +1 because in the process of calculating the power I eliminate the first and last pixel in spatial coordinates, +0.5 do be the centre of the pixel
	vertical_coord = (vertical_coord+1+0.5)*dx
	horizontal_coord -= foil_position_dict['foilhorizw']*0.5+0.0198
	vertical_coord -= foil_position_dict['foilvertw']*0.5-0.0198
	distance_from_vertical = (horizontal_coord**2+vertical_coord**2)**0.5
	pinhole_to_foil_vertical = 0.008 + 0.003 + 0.002 + 0.045	# pinhole holder, washer, foil holder, stand_off
	pinhole_to_pixel_distance = (pinhole_to_foil_vertical**2 + distance_from_vertical**2)**0.5

	etendue = np.ones_like(powernoback[0]) * (np.pi*(0.002**2)) / (pinhole_to_pixel_distance**2)	# I should include also the area of the pixel, but that is already in the w/m2 power
	etendue *= (pinhole_to_foil_vertical/pinhole_to_pixel_distance)**2	 # cos(a)*cos(b). for pixels not directly under the pinhole both pinhole and pixel are tilted respect to the vertical, with same angle.
	brightness = 4*np.pi*powernoback/etendue

	temp = brightness[:,:,:int(np.shape(brightness)[2]*0.75)]
	temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
	ani = movie_from_data(np.array([np.flip(np.transpose(brightness,(0,2,1)),axis=2)]), 1/np.median(np.diff(time_binned)),timesteps=time_binned[1:-1],integration=laser_int_time/1000,time_offset=time_binned[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
	ani.save('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+ '_FAST_brightness.mp4', fps=5*(1/np.mean(np.diff(time_binned)))/383, writer='ffmpeg',codec='mpeg4')
	plt.close('all')

	print('completed FAST power calculation ' + laser_to_analyse)

	return foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop_binned,time_binned,powernoback,brightness,binning_type

#######################################################################################################################################################################################################

def find_radiator_location(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,min_distance_between_peaks=0.5,radious_around_xpoint_for_radiator=0.2):
	from scipy.interpolate.fitpack2 import RectBivariateSpline
	# I want to filter points too close to each other
	# min_distance_between_peaks = 0.5	# m
	a,b = np.meshgrid(inversion_Z[1:-1],inversion_R[1:-1])
	dr = np.median(np.diff(inversion_R))
	dz = np.median(np.diff(inversion_Z))
	a_flat = a.flatten()
	b_flat = b.flatten()

	radiator_xpoint_distance_all = []
	radiator_above_xpoint_all = []
	radiator_magnetic_radious_all = []
	radiator_position_all = []
	radiator_baricentre_magnetic_radious_all = []
	radiator_baricentre_position_all = []
	radiator_baricentre_above_xpoint_all = []
	for i_t in range(len(time_full_binned_crop)):
		try:
			i_efit_time = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()

			central_voxel = inverted_data[i_t,1:-1,1:-1]
			central_voxel_flat = central_voxel.flatten()
			highest_surrounding_voxel = np.nanmax([inverted_data[i_t,1:-1,:-2],inverted_data[i_t,:-2,:-2],inverted_data[i_t,:-2,1:-1],inverted_data[i_t,:-2,2:],inverted_data[i_t,1:-1,2:],inverted_data[i_t,2:,2:],inverted_data[i_t,2:,1:-1],inverted_data[i_t,2:,:-2]],axis=0)
			peaks = np.logical_and(central_voxel>highest_surrounding_voxel,central_voxel>0)

			peaks_location = peaks.flatten()
			peaks_location = peaks_location*np.arange(len(peaks_location))
			peaks_location = peaks_location[peaks_location>0]
			new_peaks = np.zeros_like(peaks.flatten()).astype(int)
			for i_ in range(len(peaks_location)):
				distance = ((a_flat-a_flat[peaks_location[i_]])**2 + (b_flat-b_flat[peaks_location[i_]])**2)**0.5
				if np.sum(central_voxel_flat[np.logical_and(np.logical_and(distance < min_distance_between_peaks,distance >0),peaks.flatten())] > central_voxel_flat[peaks_location[i_]])==0:
					# print(peaks_location[i_])
					new_peaks[peaks_location[i_]] = peaks_location[i_]

			if False:
				plt.figure()
				plt.imshow(np.flip(np.transpose(central_voxel,(1,0)),axis=0),extent=[b_flat.min()-dr/2,b_flat.max()+dr/2,a_flat.min()-dz/2,a_flat.max()+dz/2])
				plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
				plt.plot(b_flat[new_peaks>0],a_flat[new_peaks>0],'+r')
				temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
				for i in range(len(all_time_sep_r[temp])):
					plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
				plt.pause(0.01)

			closer_peak = ((efit_reconstruction.lower_xpoint_z[i_efit_time] - a_flat[new_peaks>0])**2 +  + (efit_reconstruction.lower_xpoint_r[i_efit_time] - b_flat[new_peaks>0])**2).argmin()
			closer_peak = new_peaks[new_peaks>0][closer_peak]
		except:
			closer_peak = 0
		radiator_xpoint_distance = ((efit_reconstruction.lower_xpoint_z[i_efit_time] - a_flat[closer_peak])**2 + (efit_reconstruction.lower_xpoint_r[i_efit_time] - b_flat[closer_peak])**2)**0.5
		radiator_above_xpoint = a_flat[closer_peak]-efit_reconstruction.lower_xpoint_z[i_efit_time]
		interpolator = RectBivariateSpline(efit_reconstruction.R,efit_reconstruction.Z,efit_reconstruction.psidat[i_efit_time].T)
		radiator_magnetic_radious = (efit_reconstruction.psi_axis[i_efit_time] - interpolator(b_flat[closer_peak],a_flat[closer_peak])[0,0])/(efit_reconstruction.psi_axis[i_efit_time] - efit_reconstruction.psi_bnd[i_efit_time])
		radiator_xpoint_distance_all.append(radiator_xpoint_distance)
		radiator_above_xpoint_all.append(radiator_above_xpoint)
		radiator_magnetic_radious_all.append(radiator_magnetic_radious)
		radiator_position_all.append([b_flat[closer_peak],a_flat[closer_peak]])

		distance = ((a-efit_reconstruction.lower_xpoint_z[i_efit_time])**2 + (b-efit_reconstruction.lower_xpoint_r[i_efit_time])**2)**0.5
		select = distance < radious_around_xpoint_for_radiator
		z_baricentre = np.sum(a[select]*inverted_data[i_t,1:-1,1:-1][select])/np.sum(inverted_data[i_t,1:-1,1:-1][select])
		r_baricentre = np.sum(b[select]*inverted_data[i_t,1:-1,1:-1][select])/np.sum(inverted_data[i_t,1:-1,1:-1][select])
		radiator_baricentre_magnetic_radious = (efit_reconstruction.psi_axis[i_efit_time] - interpolator(r_baricentre,z_baricentre)[0,0])/(efit_reconstruction.psi_axis[i_efit_time] - efit_reconstruction.psi_bnd[i_efit_time])
		radiator_baricentre_magnetic_radious_all.append(radiator_baricentre_magnetic_radious)
		radiator_baricentre_position_all.append([r_baricentre,z_baricentre])
		radiator_baricentre_above_xpoint_all.append(z_baricentre-efit_reconstruction.lower_xpoint_z[i_efit_time])

	radiator_xpoint_distance_all = np.array(radiator_xpoint_distance_all)
	radiator_above_xpoint_all = np.array(radiator_above_xpoint_all)
	radiator_magnetic_radious_all = np.array(radiator_magnetic_radious_all)
	additional_points_dict = dict([])
	additional_points_dict['time'] = time_full_binned_crop
	additional_points_dict['0'] = np.array(radiator_position_all)
	additional_points_dict['1'] = np.array(radiator_baricentre_position_all)
	additional_points_dict['number_of_points'] = 2
	additional_points_dict['marker'] = ['xk','xb']
	return additional_points_dict,radiator_xpoint_distance_all,radiator_above_xpoint_all,radiator_magnetic_radious_all,radiator_baricentre_magnetic_radious_all,radiator_baricentre_above_xpoint_all

def track_outer_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,leg_resolution = 0.1):
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	from scipy.ndimage.filters import generic_filter

	all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
	all_time_strike_points_location = return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)

	a,b = np.meshgrid(inversion_Z,inversion_R)
	a_flat = a.flatten()
	b_flat = b.flatten()
	grid_resolution = np.median(np.concatenate([np.diff(inversion_Z),np.diff(inversion_R)]))
	# leg_resolution = 0.1	# m

	data_length = 0
	local_mean_emis_all = []
	local_power_all = []
	leg_length_all = []
	leg_length_interval_all = []
	for i_t in range(len(time_full_binned_crop)):
		print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
		try:
			i_efit_time = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
			if len(all_time_sep_r[i_efit_time][1])==0 and len(all_time_sep_r[i_efit_time][3])>0:
				i_closer_separatrix_to_x_point = 3
			elif len(all_time_sep_r[i_efit_time][3])==0 and len(all_time_sep_r[i_efit_time][1])>0:
				i_closer_separatrix_to_x_point = 1
			elif len(all_time_sep_r[i_efit_time][3])==0 and len(all_time_sep_r[i_efit_time][1])==0:
				local_mean_emis_all.append([])
				local_power_all.append([])
				leg_length_all.append(0)
				leg_length_interval_all.append([])
				continue
			elif np.abs((r_fine[all_time_sep_r[i_efit_time][1]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][1]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).min() < np.abs((r_fine[all_time_sep_r[i_efit_time][3]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][3]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).min():
				i_closer_separatrix_to_x_point = 1
			else:
				i_closer_separatrix_to_x_point = 3
			i_where_x_point_is = np.abs((r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).argmin()

			temp = np.array([((all_time_strike_points_location[i_efit_time][0][i] - r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2 + (all_time_strike_points_location[i_efit_time][1][i] - z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2).min() for i in range(len(all_time_strike_points_location[i_efit_time][1]))])
			temp[np.isnan(temp)] = np.inf
			i_which_strike_point_is = temp.argmin()
			i_where_strike_point_is = ((all_time_strike_points_location[i_efit_time][0][i_which_strike_point_is] - r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2 + (all_time_strike_points_location[i_efit_time][1][i_which_strike_point_is] - z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2).argmin()
			# plt.figure()
			# plt.plot(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is])
			# plt.pause(0.001)
			# r_coord_smooth = generic_filter(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],np.mean,size=11)
			# z_coord_smooth = generic_filter(z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],np.mean,size=11)
			r_coord_smooth = scipy.signal.savgol_filter(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],max(11,int(len(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is])/10//2*2+1)),2)
			z_coord_smooth = scipy.signal.savgol_filter(z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],max(11,int(len(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is])/10//2*2+1)),2)
			leg_length = np.sum((np.diff(r_coord_smooth)**2 + np.diff(z_coord_smooth)**2)**0.5)
			# I arbitrarily decide to cut the leg in 10cm pieces
			leg_length_interval = [0]
			target_length = 0 + leg_resolution
			i_ref_points = [0]
			ref_points = [[r_coord_smooth[0],z_coord_smooth[0]]]
			while target_length < leg_length + leg_resolution:
				# print(target_length)
				temp = np.abs(np.cumsum((np.diff(r_coord_smooth)**2 + np.diff(z_coord_smooth)**2)**0.5) - target_length).argmin()
				leg_length_interval.append(np.cumsum((np.diff(r_coord_smooth)**2 + np.diff(z_coord_smooth)**2)**0.5)[temp] - np.sum(leg_length_interval))
				i_ref_points.append(temp+1)
				ref_points.append([r_coord_smooth[temp+1],z_coord_smooth[temp+1]])
				target_length += leg_resolution
			ref_points = np.array(ref_points)
			leg_length_interval = leg_length_interval[1:]
			# I want to eliminate doubles
			i_ref_points = np.concatenate([[i_ref_points[0]],np.array(i_ref_points[1:])[np.abs(np.diff(i_ref_points))>0]])
			ref_points = np.concatenate([[ref_points[0]],ref_points[1:][np.abs(np.diff(ref_points[:,0]))+np.abs(np.diff(ref_points[:,1]))>0]])
			leg_length_interval = np.array(leg_length_interval)[np.array(leg_length_interval)>0].tolist()

			ref_points_1 = []
			ref_points_2 = []
			try:
				m = -1/((z_coord_smooth[i_ref_points[0]]-z_coord_smooth[i_ref_points[0]+1])/(r_coord_smooth[i_ref_points[0]]-r_coord_smooth[i_ref_points[0]+1]))
				ref_points_1.append([r_coord_smooth[i_ref_points[0]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[0]] - m*leg_resolution/((1+m**2)**0.5)])
				ref_points_2.append([r_coord_smooth[i_ref_points[0]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[0]] + m*leg_resolution/((1+m**2)**0.5)])
			except:
				pass
			for i_ref_point in range(1,len(i_ref_points)):
				try:
					m = -1/((z_coord_smooth[i_ref_points[i_ref_point]-1]-z_coord_smooth[i_ref_points[i_ref_point]+1])/(r_coord_smooth[i_ref_points[i_ref_point]-1]-r_coord_smooth[i_ref_points[i_ref_point]+1]))
					ref_points_1.append([r_coord_smooth[i_ref_points[i_ref_point]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[i_ref_point]] - m*leg_resolution/((1+m**2)**0.5)])
					ref_points_2.append([r_coord_smooth[i_ref_points[i_ref_point]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[i_ref_point]] + m*leg_resolution/((1+m**2)**0.5)])
				except:
					pass
			try:
				m = -1/((z_coord_smooth[i_ref_points[-1]-1]-z_coord_smooth[i_ref_points[-1]])/(r_coord_smooth[i_ref_points[-1]-1]-r_coord_smooth[i_ref_points[-1]]))
				ref_points_1.append([r_coord_smooth[i_ref_points[-1]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[-1]] - m*leg_resolution/((1+m**2)**0.5)])
				ref_points_2.append([r_coord_smooth[i_ref_points[-1]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[-1]] + m*leg_resolution/((1+m**2)**0.5)])
			except:
				pass
			ref_points_1 = np.array(ref_points_1)
			ref_points_2 = np.array(ref_points_2)

			# plt.figure()
			# plt.plot(r_coord_smooth,z_coord_smooth)
			# plt.plot(r_coord_smooth,z_coord_smooth,'+')
			# plt.plot(ref_points[:,0],ref_points[:,1],'+')
			# plt.plot(ref_points_1[:,0],ref_points_1[:,1],'o')
			# plt.plot(ref_points_2[:,0],ref_points_2[:,1],'+')
			# plt.pause(0.001)

			local_mean_emis = []
			local_power = []
			emissivity_flat = inverted_data[i_t].flatten()
			for i_ref_point in range(1,len(ref_points)):
				# print(i_ref_point)
				select = []
				polygon = Polygon([ref_points_1[i_ref_point-1], ref_points_1[i_ref_point], ref_points_2[i_ref_point], ref_points_2[i_ref_point-1]])
				for i_e in range(len(emissivity_flat)):
					point = Point((b_flat[i_e],a_flat[i_e]))
					select.append(polygon.contains(point))
				local_mean_emis.append(np.nanmean(emissivity_flat[select]))
				local_power.append(2*np.pi*np.nansum(emissivity_flat[select]*b_flat[select]*(grid_resolution**2)))
			# local_mean_emis = np.array(local_mean_emis)
			# local_power = np.array(local_power)
			# local_mean_emis = local_mean_emis[np.logical_not(np.isnan(local_mean_emis))].tolist()
			# local_power = local_power[np.logical_not(np.isnan(local_power))].tolist()
			local_mean_emis_all.append(local_mean_emis)
			local_power_all.append(local_power)
			data_length = max(data_length,len(local_power))
			leg_length_all.append(leg_length)
			leg_length_interval_all.append(leg_length_interval)
		except Exception as e:
			# logging.exception('with error: ' + str(e))
			local_mean_emis_all.append([])
			local_power_all.append([])
			leg_length_interval_all.append([])
			leg_length_all.append(0)

	for i_t in range(len(time_full_binned_crop)):
		if len(local_mean_emis_all[i_t])<data_length:
			local_mean_emis_all[i_t].extend([0]*(data_length-len(local_mean_emis_all[i_t])))
			local_power_all[i_t].extend([0]*(data_length-len(local_power_all[i_t])))
			leg_length_interval_all[i_t].extend([0]*(data_length-len(leg_length_interval_all[i_t])))

	return local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution

def track_inner_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,leg_resolution = 0.05):
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	from scipy.ndimage.filters import generic_filter

	all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
	all_time_strike_points_location = return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)

	a,b = np.meshgrid(inversion_Z,inversion_R)
	a_flat = a.flatten()
	b_flat = b.flatten()
	grid_resolution = np.median(np.concatenate([np.diff(inversion_Z),np.diff(inversion_R)]))
	# leg_resolution = 0.1	# m

	data_length = 0
	local_mean_emis_all = []
	local_power_all = []
	leg_length_all = []
	leg_length_interval_all = []
	for i_t in range(len(time_full_binned_crop)):
		print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
		try:
			i_efit_time = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
			if len(all_time_sep_r[i_efit_time][1])==0 and len(all_time_sep_r[i_efit_time][3])>0:
				i_closer_separatrix_to_x_point = 3
			elif len(all_time_sep_r[i_efit_time][3])==0 and len(all_time_sep_r[i_efit_time][1])>0:
				i_closer_separatrix_to_x_point = 1
			elif len(all_time_sep_r[i_efit_time][3])==0 and len(all_time_sep_r[i_efit_time][1])==0:
				local_mean_emis_all.append([])
				local_power_all.append([])
				leg_length_all.append(0)
				leg_length_interval_all.append([])
				continue
			elif np.abs((r_fine[all_time_sep_r[i_efit_time][0]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][0]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).min() < np.abs((r_fine[all_time_sep_r[i_efit_time][2]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][2]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).min():
				i_closer_separatrix_to_x_point = 0
			else:
				i_closer_separatrix_to_x_point = 2
			i_where_x_point_is = np.abs((r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).argmin()

			temp = np.array([((all_time_strike_points_location[i_efit_time][0][i] - r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2 + (all_time_strike_points_location[i_efit_time][1][i] - z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2).min() for i in range(len(all_time_strike_points_location[i_efit_time][1]))])
			temp[np.isnan(temp)] = np.inf
			i_which_strike_point_is = temp.argmin()
			i_where_strike_point_is = ((all_time_strike_points_location[i_efit_time][0][i_which_strike_point_is] - r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2 + (all_time_strike_points_location[i_efit_time][1][i_which_strike_point_is] - z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2).argmin()
			# plt.figure()
			# plt.plot(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is])
			# plt.pause(0.001)
			# r_coord_smooth = generic_filter(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],np.mean,size=11)
			# z_coord_smooth = generic_filter(z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],np.mean,size=11)
			r_coord_smooth = scipy.signal.savgol_filter(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],max(9,int(len(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is])/3//2*2+1)),2)
			z_coord_smooth = scipy.signal.savgol_filter(z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],max(9,int(len(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is])/3//2*2+1)),2)
			leg_length = np.sum((np.diff(r_coord_smooth)**2 + np.diff(z_coord_smooth)**2)**0.5)
			# I arbitrarily decide to cut the leg in 10cm pieces
			leg_length_interval = [0]
			target_length = 0 + leg_resolution
			i_ref_points = [0]
			ref_points = [[r_coord_smooth[0],z_coord_smooth[0]]]
			while target_length < leg_length + leg_resolution:
				# print(target_length)
				temp = np.abs(np.cumsum((np.diff(r_coord_smooth)**2 + np.diff(z_coord_smooth)**2)**0.5) - target_length).argmin()
				leg_length_interval.append(np.cumsum((np.diff(r_coord_smooth)**2 + np.diff(z_coord_smooth)**2)**0.5)[temp] - np.sum(leg_length_interval))
				i_ref_points.append(temp+1)
				ref_points.append([r_coord_smooth[temp+1],z_coord_smooth[temp+1]])
				target_length += leg_resolution
			ref_points = np.array(ref_points)
			leg_length_interval = leg_length_interval[1:]
			# I want to eliminate doubles
			i_ref_points = np.concatenate([[i_ref_points[0]],np.array(i_ref_points[1:])[np.abs(np.diff(i_ref_points))>0]])
			ref_points = np.concatenate([[ref_points[0]],ref_points[1:][np.abs(np.diff(ref_points[:,0]))+np.abs(np.diff(ref_points[:,1]))>0]])
			leg_length_interval = np.array(leg_length_interval)[np.array(leg_length_interval)>0].tolist()

			ref_points_1 = []
			ref_points_2 = []
			try:
				m = -1/((z_coord_smooth[i_ref_points[0]]-z_coord_smooth[i_ref_points[0]+1])/(r_coord_smooth[i_ref_points[0]]-r_coord_smooth[i_ref_points[0]+1]))
				ref_points_1.append([r_coord_smooth[i_ref_points[0]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[0]] - m*leg_resolution/((1+m**2)**0.5)])
				ref_points_2.append([r_coord_smooth[i_ref_points[0]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[0]] + m*leg_resolution/((1+m**2)**0.5)])
			except:
				pass
			for i_ref_point in range(1,len(i_ref_points)):
				try:
					m = -1/((z_coord_smooth[i_ref_points[i_ref_point]-1]-z_coord_smooth[i_ref_points[i_ref_point]+1])/(r_coord_smooth[i_ref_points[i_ref_point]-1]-r_coord_smooth[i_ref_points[i_ref_point]+1]))
					ref_points_1.append([r_coord_smooth[i_ref_points[i_ref_point]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[i_ref_point]] - m*leg_resolution/((1+m**2)**0.5)])
					ref_points_2.append([r_coord_smooth[i_ref_points[i_ref_point]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[i_ref_point]] + m*leg_resolution/((1+m**2)**0.5)])
				except:
					pass
			try:
				m = -1/((z_coord_smooth[i_ref_points[-1]-1]-z_coord_smooth[i_ref_points[-1]])/(r_coord_smooth[i_ref_points[-1]-1]-r_coord_smooth[i_ref_points[-1]]))
				ref_points_1.append([r_coord_smooth[i_ref_points[-1]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[-1]] - m*leg_resolution/((1+m**2)**0.5)])
				ref_points_2.append([r_coord_smooth[i_ref_points[-1]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[-1]] + m*leg_resolution/((1+m**2)**0.5)])
			except:
				pass
			ref_points_1 = np.array(ref_points_1)
			ref_points_2 = np.array(ref_points_2)

			# plt.figure()
			# plt.plot(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is])
			# plt.plot(r_coord_smooth,z_coord_smooth)
			# plt.plot(r_coord_smooth,z_coord_smooth,'+')
			# plt.plot(ref_points[:,0],ref_points[:,1],'x')
			# plt.plot(ref_points_1[:,0],ref_points_1[:,1],'o')
			# plt.plot(ref_points_2[:,0],ref_points_2[:,1],'+')
			# plt.pause(0.001)

			local_mean_emis = []
			local_power = []
			emissivity_flat = inverted_data[i_t].flatten()
			for i_ref_point in range(1,len(ref_points)):
				# print(i_ref_point)
				select = []
				polygon = Polygon([ref_points_1[i_ref_point-1], ref_points_1[i_ref_point], ref_points_2[i_ref_point], ref_points_2[i_ref_point-1]])
				# plt.plot([ref_points_1[i_ref_point-1],ref_points_1[i_ref_point-1],ref_points_1[i_ref_point],ref_points_1[i_ref_point]],[ref_points_2[i_ref_point],ref_points_2[i_ref_point-1],ref_points_2[i_ref_point-1],ref_points_2[i_ref_point]],'--')
				for i_e in range(len(emissivity_flat)):
					point = Point((b_flat[i_e],a_flat[i_e]))
					select.append(polygon.contains(point))
				local_mean_emis.append(np.nanmean(emissivity_flat[select]))
				local_power.append(2*np.pi*np.nansum(emissivity_flat[select]*b_flat[select]*(grid_resolution**2)))
			# local_mean_emis = np.array(local_mean_emis)
			# local_power = np.array(local_power)
			# local_mean_emis = local_mean_emis[np.logical_not(np.isnan(local_mean_emis))].tolist()
			# local_power = local_power[np.logical_not(np.isnan(local_power))].tolist()
			local_mean_emis_all.append(local_mean_emis)
			local_power_all.append(local_power)
			data_length = max(data_length,len(local_power))
			leg_length_all.append(leg_length)
			leg_length_interval_all.append(leg_length_interval)
		except Exception as e:
			# logging.exception('with error: ' + str(e))
			local_mean_emis_all.append([])
			local_power_all.append([])
			leg_length_interval_all.append([])
			leg_length_all.append(0)

	for i_t in range(len(time_full_binned_crop)):
		if len(local_mean_emis_all[i_t])<data_length:
			local_mean_emis_all[i_t].extend([0]*(data_length-len(local_mean_emis_all[i_t])))
			local_power_all[i_t].extend([0]*(data_length-len(local_power_all[i_t])))
			leg_length_interval_all[i_t].extend([0]*(data_length-len(leg_length_interval_all[i_t])))

	return local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution

def MASTU_pulse_process_FAST3(laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,laser_framerate,laser_digitizer_ID,laser_int_time,seconds_for_reference_frame,start_time_of_pulse,laser_to_analyse,height,width,flag_use_of_first_frames_as_reference,params,errparams,foil_position_dict):
	# created 2021/10/07
	# created modifying MASTU_pulse_process_FAST2 in order to include the bayesian inversion

	from scipy.ndimage.filters import generic_filter

	max_ROI = [[0,255],[0,319]]
	# foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
	temp_ref_counts = []
	temp_counts_minus_background = []
	time_partial = []
	timesteps = np.inf
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		if external_clock_marker:
			time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers

		# basic smoothing
		spectra_orig=np.fft.fft(np.mean(laser_counts_corrected[i],axis=(-1,-2)))
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		freq = np.fft.fftfreq(len(magnitude), d=np.mean(np.diff(time_of_experiment_digitizer_ID_seconds)))
		magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
		freq = np.sort(freq)
		magnitude_smooth = generic_filter(np.log(magnitude),np.median,size=[7])
		peak_oscillation = (magnitude-np.exp(magnitude_smooth))[np.logical_and(freq>10,freq<50)].argmax()
		peak_oscillation_freq = freq[np.logical_and(freq>10,freq<50)][peak_oscillation]
		frames_to_average = 1/peak_oscillation_freq/np.mean(np.diff(time_of_experiment_digitizer_ID_seconds))
		laser_counts_corrected_filtered = real_mean_filter_agent(laser_counts_corrected[i],frames_to_average)

		if flag_use_of_first_frames_as_reference:
			# temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[time_of_experiment_digitizer_ID_seconds<0],axis=0))
			temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<0,time_of_experiment_digitizer_ID_seconds>-0.5)],axis=0))
		else:
			temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))
		select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds>=0,time_of_experiment_digitizer_ID_seconds<=1.5)
		temp_counts_minus_background.append(laser_counts_corrected_filtered[select_time]-temp_ref_counts[-1])
		time_partial.append(time_of_experiment_digitizer_ID_seconds[select_time])
		timesteps = min(timesteps,len(temp_counts_minus_background[-1]))

	for i in range(len(laser_digitizer_ID)):
		temp_counts_minus_background[i] = temp_counts_minus_background[i][:timesteps]
		time_partial[i] = time_partial[i][:timesteps]
	temp_counts_minus_background = np.nanmean(temp_counts_minus_background,axis=0)
	temp_ref_counts = np.nanmean(temp_ref_counts,axis=0)
	FAST_counts_minus_background_crop_time = np.nanmean(time_partial,axis=0)

	# I'm going to use the reference frames for foil position
	foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx = get_rotation_crop_parameters(temp_ref_counts,foil_position_dict,laser_to_analyse,temp_counts_minus_background,FAST_counts_minus_background_crop_time)

	# rotation and crop
	temp_counts_minus_background_rot=rotate(temp_counts_minus_background,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		temp_counts_minus_background_rot*=out_of_ROI_mask
		temp_counts_minus_background_rot[np.logical_and(temp_counts_minus_background_rot<np.nanmin(temp_counts_minus_background),temp_counts_minus_background_rot>np.nanmax(temp_counts_minus_background))]=0
	FAST_counts_minus_background_crop = temp_counts_minus_background_rot[:,foildw:foilup,foillx:foilrx]

	# I drop this to save memory
	# temp = FAST_counts_minus_background_crop[:,:,:int(np.shape(FAST_counts_minus_background_crop)[2]*0.75)]
	# temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
	# ani,efit_reconstruction = movie_from_data(np.array([np.flip(np.transpose(FAST_counts_minus_background_crop,(0,2,1)),axis=2)]), laser_framerate/len(laser_digitizer_ID),timesteps=FAST_counts_minus_background_crop_time,integration=laser_int_time/1000,time_offset=FAST_counts_minus_background_crop_time[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Count increase [au]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,EFIT_output_requested=True)
	# ani.save(laser_to_analyse[:-4]+ '_FAST_count_increase.mp4', fps=5*laser_framerate/len(laser_digitizer_ID)/383, writer='ffmpeg',codec='mpeg4')
	# plt.close('all')

	print('completed FAST count rotating/cropping ' + laser_to_analyse)

	averaged_params = np.mean(params,axis=(0))
	averaged_errparams = np.mean(errparams,axis=(0))
	counts = temp_counts_minus_background+temp_ref_counts
	temperature = averaged_params[:,:,-1] + averaged_params[:,:,-2] * counts + averaged_params[:,:,-3] * (counts**2)
	counts_std = estimate_counts_std(counts,int_time=laser_int_time/1000)
	temperature_std = (averaged_errparams[:,:,2,2] + (counts_std**2)*(averaged_params[:,:,1]**2) + (counts**2+counts_std**2)*averaged_errparams[:,:,1,1] + (counts_std**2)*(4*counts**2+3*counts_std**2)*(averaged_params[:,:,0]**2) + (counts**4+6*(counts**2)*(counts_std**2)+3*counts_std**4)*averaged_errparams[:,:,0,0] + 2*counts*averaged_errparams[:,:,2,1] + 2*(counts**2+counts_std**2)*averaged_errparams[:,:,2,0] + 2*(counts**3+counts*(counts_std**2))*averaged_errparams[:,:,1,0])**0.5

	temperature_ref = averaged_params[:,:,-1] + averaged_params[:,:,-2] * temp_ref_counts + averaged_params[:,:,-3] * (temp_ref_counts**2)

	# rotation and crop
	temperature_rot=rotate(temperature,foilrotdeg,axes=(-1,-2))
	temperature_std_rot=rotate(temperature_std,foilrotdeg,axes=(-1,-2))
	counts_rot=rotate(counts,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		temperature_rot*=out_of_ROI_mask
		temperature_rot[np.logical_and(temperature_rot<np.nanmin(temperature),temperature_rot>np.nanmax(temperature))]=0
		temperature_std_rot*=out_of_ROI_mask
		temperature_std_rot[np.logical_and(temperature_std_rot<np.nanmin(temperature),temperature_std_rot>np.nanmax(temperature))]=0
		counts_rot*=out_of_ROI_mask
		counts_rot[np.logical_and(counts_rot<np.nanmin(temperature),counts_rot>np.nanmax(temperature))]=0
	temperature_crop = temperature_rot[:,foildw:foilup,foillx:foilrx]
	temperature_std_crop = temperature_std_rot[:,foildw:foilup,foillx:foilrx]
	counts_crop = counts_rot[:,foildw:foilup,foillx:foilrx]

	# rotation and crop
	temperature_ref_rot=rotate(temperature_ref,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		temperature_ref_rot*=out_of_ROI_mask
		temperature_ref_rot[np.logical_and(temperature_ref_rot<np.nanmin(temperature_ref),temperature_ref_rot>np.nanmax(temperature_ref))]=0
	temperature_ref_crop = temperature_ref_rot[foildw:foilup,foillx:foilrx]

	temperature_minus_background_crop = temperature_crop-temperature_ref_crop

	shrink_factor_t = int(round(frames_to_average))
	shrink_factor_x = 3	# with large time averaging this should be enough
	binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)

	FAST_counts_minus_background_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(FAST_counts_minus_background_crop,shrink_factor_t,shrink_factor_x)
	temperature_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(temperature_crop,shrink_factor_t,shrink_factor_x)
	temperature_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*(proper_homo_binning_t_2D(temperature_std_crop**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5)
	counts_crop_binned,trash = proper_homo_binning_t_2D(counts_crop,shrink_factor_t,shrink_factor_x)
	temperature_minus_background_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(temperature_minus_background_crop,shrink_factor_t,shrink_factor_x)
	temperature_ref_crop_binned = proper_homo_binning_2D(temperature_ref_crop,shrink_factor_x)
	time_binned = proper_homo_binning_t(FAST_counts_minus_background_crop_time,shrink_factor_t)

	averaged_params = np.mean(averaged_params,axis=(0,1))
	averaged_errparams = np.mean(averaged_errparams,axis=(0,1))

	# reference foil properties
	# thickness = 1.4859095354482858e-06
	# emissivity = 0.9884061389741369
	# diffusivity = 1.045900223180454e-05
	# from 2021/09/17, Laser_data_analysis3_3.py
	thickness = 2.0531473351462095e-06
	emissivity = 0.9999999999999
	diffusivity = 1.0283685197530968e-05
	Ptthermalconductivity=71.6 #[W/(m·K)]
	zeroC=273.15 #K / C
	sigmaSB=5.6704e-08 #[W/(m2 K4)]

	foilemissivityscaled=emissivity*np.ones(np.array(temperature_ref_crop_binned.shape)-2)
	foilthicknessscaled=thickness*np.ones(np.array(temperature_ref_crop_binned.shape)-2)
	conductivityscaled=Ptthermalconductivity*np.ones(np.array(temperature_ref_crop_binned.shape)-2)
	reciprdiffusivityscaled=(1/diffusivity)*np.ones(np.array(temperature_ref_crop_binned.shape)-2)

	dt = time_binned[2:]-time_binned[:-2]
	dx=foil_position_dict['foilhorizw']/foil_position_dict['foilhorizwpixel']*shrink_factor_x
	dTdt=np.divide((temperature_crop_binned[2:,1:-1,1:-1]-temperature_crop_binned[:-2,1:-1,1:-1]).T,dt).T.astype(np.float32)
	temp = averaged_errparams[2,2] + counts_crop_binned[2:,1:-1,1:-1]*counts_crop_binned[:-2,1:-1,1:-1]*averaged_errparams[1,1] + (counts_crop_binned[2:,1:-1,1:-1]**2)*(counts_crop_binned[:-2,1:-1,1:-1]**2)*averaged_errparams[0,0] + (estimate_counts_std(counts_crop_binned[2:,1:-1,1:-1])**2)*(estimate_counts_std(counts_crop_binned[:-2,1:-1,1:-1])**2)*(averaged_params[0]**2)/((shrink_factor_t*shrink_factor_x**2)**2) + (counts_crop_binned[2:,1:-1,1:-1]+counts_crop_binned[:-2,1:-1,1:-1])*averaged_errparams[1,2] + (counts_crop_binned[2:,1:-1,1:-1]**2 + counts_crop_binned[:-2,1:-1,1:-1]**2)*averaged_errparams[0,2] + (counts_crop_binned[2:,1:-1,1:-1]*(counts_crop_binned[:-2,1:-1,1:-1]**2)+counts_crop_binned[:-2,1:-1,1:-1]*(counts_crop_binned[2:,1:-1,1:-1]**2))*averaged_errparams[0,1]
	dTdt_std=np.divide((temperature_std_crop_binned[2:,1:-1,1:-1]**2 + temperature_std_crop_binned[:-2,1:-1,1:-1]**2 - 2*temp).T**0.5,dt).T.astype(np.float32)
	d2Tdx2=np.divide(temperature_minus_background_crop_binned[1:-1,1:-1,2:]-np.multiply(2,temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])+temperature_minus_background_crop_binned[1:-1,1:-1,:-2],dx**2).astype(np.float32)
	d2Tdy2=np.divide(temperature_minus_background_crop_binned[1:-1,2:,1:-1]-np.multiply(2,temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])+temperature_minus_background_crop_binned[1:-1,:-2,1:-1],dx**2).astype(np.float32)
	d2Tdxy = np.ones_like(dTdt).astype(np.float32)*np.nan
	d2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]=np.add(d2Tdx2[:,nan_ROI_mask[1:-1,1:-1]],d2Tdy2[:,nan_ROI_mask[1:-1,1:-1]])
	del d2Tdx2,d2Tdy2
	d2Tdx2_std=np.divide((temperature_std_crop_binned[1:-1,1:-1,2:]**2+np.multiply(2**2,temperature_std_crop_binned[1:-1,1:-1,1:-1])**2+temperature_std_crop_binned[1:-1,1:-1,:-2]**2)**0.5,dx**2).astype(np.float32)
	d2Tdy2_std=np.divide((temperature_std_crop_binned[1:-1,2:,1:-1]**2+np.multiply(2**2,temperature_std_crop_binned[1:-1,1:-1,1:-1])**2+temperature_std_crop_binned[1:-1,:-2,1:-1]**2)**0.5,dx**2).astype(np.float32)
	d2Tdxy_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	temp1 = averaged_errparams[2,2] + counts_crop_binned[1:-1,1:-1,2:]*counts_crop_binned[1:-1,1:-1,:-2]*averaged_errparams[1,1] + (counts_crop_binned[1:-1,1:-1,2:]**2)*(counts_crop_binned[1:-1,1:-1,:-2]**2)*averaged_errparams[0,0] + (estimate_counts_std(counts_crop_binned[1:-1,1:-1,2:],int_time=laser_int_time/1000)**2)*(estimate_counts_std(counts_crop_binned[1:-1,1:-1,:-2],int_time=laser_int_time/1000)**2)*(averaged_params[0]**2)/((shrink_factor_t*shrink_factor_x**2)**2) + (counts_crop_binned[1:-1,1:-1,2:]+counts_crop_binned[1:-1,1:-1,:-2])*averaged_errparams[1,2] + (counts_crop_binned[1:-1,1:-1,2:]**2+counts_crop_binned[1:-1,1:-1,:-2]**2)*averaged_errparams[0,2] + (counts_crop_binned[1:-1,1:-1,2:]*(counts_crop_binned[1:-1,1:-1,:-2]**2)+counts_crop_binned[1:-1,1:-1,:-2]*(counts_crop_binned[1:-1,1:-1,2:]**2))*averaged_errparams[0,1]
	temp2 = averaged_errparams[2,2] + counts_crop_binned[1:-1,1:-1,2:]*counts_crop_binned[1:-1,1:-1,1:-1]*averaged_errparams[1,1] + (counts_crop_binned[1:-1,1:-1,2:]**2)*(counts_crop_binned[1:-1,1:-1,1:-1]**2)*averaged_errparams[0,0] + (estimate_counts_std(counts_crop_binned[1:-1,1:-1,2:],int_time=laser_int_time/1000)**2)*(estimate_counts_std(counts_crop_binned[1:-1,1:-1,1:-1],int_time=laser_int_time/1000)**2)*(averaged_params[0]**2)/((shrink_factor_t*shrink_factor_x**2)**2) + (counts_crop_binned[1:-1,1:-1,2:]+counts_crop_binned[1:-1,1:-1,1:-1])*averaged_errparams[1,2] + (counts_crop_binned[1:-1,1:-1,2:]**2+counts_crop_binned[1:-1,1:-1,1:-1]**2)*averaged_errparams[0,2] + (counts_crop_binned[1:-1,1:-1,2:]*(counts_crop_binned[1:-1,1:-1,1:-1]**2)+counts_crop_binned[1:-1,1:-1,1:-1]*(counts_crop_binned[1:-1,1:-1,2:]**2))*averaged_errparams[0,1]
	temp3 = averaged_errparams[2,2] + counts_crop_binned[1:-1,1:-1,1:-1]*counts_crop_binned[1:-1,1:-1,:-2]*averaged_errparams[1,1] + (counts_crop_binned[1:-1,1:-1,1:-1]**2)*(counts_crop_binned[1:-1,1:-1,:-2]**2)*averaged_errparams[0,0] + (estimate_counts_std(counts_crop_binned[1:-1,1:-1,1:-1],int_time=laser_int_time/1000)**2)*(estimate_counts_std(counts_crop_binned[1:-1,1:-1,:-2],int_time=laser_int_time/1000)**2)*(averaged_params[0]**2)/((shrink_factor_t*shrink_factor_x**2)**2) + (counts_crop_binned[1:-1,1:-1,1:-1]+counts_crop_binned[1:-1,1:-1,:-2])*averaged_errparams[1,2] + (counts_crop_binned[1:-1,1:-1,1:-1]**2+counts_crop_binned[1:-1,1:-1,:-2]**2)*averaged_errparams[0,2] + (counts_crop_binned[1:-1,1:-1,1:-1]*(counts_crop_binned[1:-1,1:-1,:-2]**2)+counts_crop_binned[1:-1,1:-1,:-2]*(counts_crop_binned[1:-1,1:-1,1:-1]**2))*averaged_errparams[0,1]
	temp = 2*temp1-4*temp2-4*temp3
	temp1 = averaged_errparams[2,2] + counts_crop_binned[1:-1,2:,1:-1]*counts_crop_binned[1:-1,:-2,1:-1]*averaged_errparams[1,1] + (counts_crop_binned[1:-1,2:,1:-1]**2)*(counts_crop_binned[1:-1,:-2,1:-1]**2)*averaged_errparams[0,0] + (estimate_counts_std(counts_crop_binned[1:-1,2:,1:-1],int_time=laser_int_time/1000)**2)*(estimate_counts_std(counts_crop_binned[1:-1,:-2,1:-1],int_time=laser_int_time/1000)**2)*(averaged_params[0]**2)/((shrink_factor_t*shrink_factor_x**2)**2) + (counts_crop_binned[1:-1,2:,1:-1]+counts_crop_binned[1:-1,:-2,1:-1])*averaged_errparams[1,2] + (counts_crop_binned[1:-1,2:,1:-1]**2+counts_crop_binned[1:-1,:-2,1:-1]**2)*averaged_errparams[0,2] + (counts_crop_binned[1:-1,2:,1:-1]*(counts_crop_binned[1:-1,:-2,1:-1]**2)+counts_crop_binned[1:-1,:-2,1:-1]*(counts_crop_binned[1:-1,2:,1:-1]**2))*averaged_errparams[0,1]
	temp2 = averaged_errparams[2,2] + counts_crop_binned[1:-1,2:,1:-1]*counts_crop_binned[1:-1,1:-1,1:-1]*averaged_errparams[1,1] + (counts_crop_binned[1:-1,2:,1:-1]**2)*(counts_crop_binned[1:-1,1:-1,1:-1]**2)*averaged_errparams[0,0] + (estimate_counts_std(counts_crop_binned[1:-1,2:,1:-1],int_time=laser_int_time/1000)**2)*(estimate_counts_std(counts_crop_binned[1:-1,1:-1,1:-1],int_time=laser_int_time/1000)**2)*(averaged_params[0]**2)/((shrink_factor_t*shrink_factor_x**2)**2) + (counts_crop_binned[1:-1,2:,1:-1]+counts_crop_binned[1:-1,1:-1,1:-1])*averaged_errparams[1,2] + (counts_crop_binned[1:-1,2:,1:-1]**2+counts_crop_binned[1:-1,1:-1,1:-1]**2)*averaged_errparams[0,2] + (counts_crop_binned[1:-1,2:,1:-1]*(counts_crop_binned[1:-1,1:-1,1:-1]**2)+counts_crop_binned[1:-1,1:-1,1:-1]*(counts_crop_binned[1:-1,2:,1:-1]**2))*averaged_errparams[0,1]
	temp3 = averaged_errparams[2,2] + counts_crop_binned[1:-1,1:-1,1:-1]*counts_crop_binned[1:-1,:-2,1:-1]*averaged_errparams[1,1] + (counts_crop_binned[1:-1,1:-1,1:-1]**2)*(counts_crop_binned[1:-1,:-2,1:-1]**2)*averaged_errparams[0,0] + (estimate_counts_std(counts_crop_binned[1:-1,1:-1,1:-1],int_time=laser_int_time/1000)**2)*(estimate_counts_std(counts_crop_binned[1:-1,:-2,1:-1],int_time=laser_int_time/1000)**2)*(averaged_params[0]**2)/((shrink_factor_t*shrink_factor_x**2)**2) + (counts_crop_binned[1:-1,1:-1,1:-1]+counts_crop_binned[1:-1,:-2,1:-1])*averaged_errparams[1,2] + (counts_crop_binned[1:-1,1:-1,1:-1]**2+counts_crop_binned[1:-1,:-2,1:-1]**2)*averaged_errparams[0,2] + (counts_crop_binned[1:-1,1:-1,1:-1]*(counts_crop_binned[1:-1,:-2,1:-1]**2)+counts_crop_binned[1:-1,:-2,1:-1]*(counts_crop_binned[1:-1,1:-1,1:-1]**2))*averaged_errparams[0,1]
	temp += 2*temp1-4*temp2-4*temp3
	d2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]=np.add(temp[:,nan_ROI_mask[1:-1,1:-1]]/(dx**4),np.add(d2Tdx2_std[:,nan_ROI_mask[1:-1,1:-1]]**2,d2Tdy2_std[:,nan_ROI_mask[1:-1,1:-1]]**2))**0.5
	del d2Tdx2_std,d2Tdy2_std
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	negd2Tdxy_std=d2Tdxy_std
	T4=(temperature_minus_background_crop_binned[1:-1,1:-1,1:-1]+np.nanmean(temperature_ref_crop_binned)+zeroC)**4
	T04=(np.nanmean(temperature_ref_crop_binned)+zeroC)**4 *np.ones_like(temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])
	T4_T04 = np.ones_like(dTdt).astype(np.float32)*np.nan
	T4_T04[:,nan_ROI_mask[1:-1,1:-1]] = (T4[:,nan_ROI_mask[1:-1,1:-1]]-T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	T4_std=T4**(3/4) *4 *temperature_std_crop_binned[1:-1,1:-1,1:-1]	# the error resulting from doing the average on the whole ROI is completely negligible
	T04_std=0
	T4_T04_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]] = ((T4_std[:,nan_ROI_mask[1:-1,1:-1]]**2+T04_std**2)**0.5).astype(np.float32)

	BBrad = np.ones_like(dTdt).astype(np.float32)*np.nan
	BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04[:,nan_ROI_mask[1:-1,1:-1]] * foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	diffusion = np.ones_like(dTdt).astype(np.float32)*np.nan
	diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	timevariation = np.ones_like(dTdt).astype(np.float32)*np.nan
	timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	powernoback = (diffusion + timevariation + BBrad).astype(np.float32)
	BBrad_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]]*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	diffusion_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	timevariation_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	powernoback_std = np.ones_like(powernoback)*np.nan
	powernoback_std[:,nan_ROI_mask[1:-1,1:-1]] = ((diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5).astype(np.float32)

	horizontal_coord = np.arange(np.shape(powernoback[0])[1])
	vertical_coord = np.arange(np.shape(powernoback[0])[0])
	horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
	horizontal_coord = (horizontal_coord+1+0.5)*dx	# +1 because in the process of calculating the power I eliminate the first and last pixel in spatial coordinates, +0.5 do be the centre of the pixel
	vertical_coord = (vertical_coord+1+0.5)*dx
	horizontal_coord -= foil_position_dict['foilhorizw']*0.5+0.0198
	vertical_coord -= foil_position_dict['foilvertw']*0.5-0.0198
	distance_from_vertical = (horizontal_coord**2+vertical_coord**2)**0.5
	pinhole_to_foil_vertical = 0.008 + 0.003 + 0.002 + 0.045	# pinhole holder, washer, foil holder, standoff
	pinhole_to_pixel_distance = (pinhole_to_foil_vertical**2 + distance_from_vertical**2)**0.5

	etendue = np.ones_like(powernoback[0]) * (np.pi*(0.002**2)) / (pinhole_to_pixel_distance**2)	# I should include also the area of the pixel, but that is already in the w/m2 power
	etendue *= (pinhole_to_foil_vertical/pinhole_to_pixel_distance)**2	 # cos(a)*cos(b). for pixels not directly under the pinhole both pinhole and pixel are tilted respect to the vertical, with same angle.
	brightness = 4*np.pi*powernoback/etendue

	temp = brightness[:,:,:int(np.shape(brightness)[2]*0.75)]
	temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
	ani,efit_reconstruction = movie_from_data(np.array([np.flip(np.transpose(brightness,(0,2,1)),axis=2)]), 1/np.median(np.diff(time_binned)),timesteps=time_binned[1:-1],integration=laser_int_time/1000,time_offset=time_binned[0],extvmin=0,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,EFIT_output_requested=True)
	ani.save('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+ '_FAST_brightness.mp4', fps=5*(1/np.mean(np.diff(time_binned)))/383, writer='ffmpeg',codec='mpeg4')
	plt.close('all')
	powernoback_output = cp.deepcopy(powernoback)

	print('completed FAST power calculation ' + laser_to_analyse)

	inverted_dict = dict([])
	from scipy.ndimage import geometric_transform
	import time as tm
	import pickle
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	shot_number = int(laser_to_analyse[-9:-4])
	for grid_resolution in [4, 2]:
	# for grid_resolution in [4]:
		inverted_dict[str(grid_resolution)] = dict([])
		# grid_resolution = 8  # in cm
		foil_resolution = '187'

		foil_res = '_foil_pixel_h_' + str(foil_resolution)

		grid_type = 'core_res_' + str(grid_resolution) + 'cm'
		path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res + '_power'
		try:
			sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity.npz')).todense())
		except:
			sensitivities = np.load(path_sensitivity + '/sensitivity.npy')

		filenames = all_file_names(path_sensitivity, '.csv')[0]
		with open(os.path.join(path_sensitivity, filenames)) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if row[0] == 'foil vertical pixels ':
					pixel_v = int(row[1])
				if row[0] == 'foil horizontal pixels ':
					pixel_h = int(row[1])
				if row[0] == 'pipeline type ':
					pipeline = row[1]
				if row[0] == 'type of volume grid ':
					grid_type = row[1]
			# print(row)

		directory = '/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction'
		grid_file = os.path.join(directory,'{}_rectilinear_grid.pickle'.format(grid_type))
		with open(grid_file, 'rb') as f:
			grid_data_all = pickle.load(f)
		grid_laplacian = grid_data_all['laplacian']
		grid_mask = grid_data_all['mask']
		grid_data = grid_data_all['voxels']
		grid_index_2D_to_1D_map = grid_data_all['index_2D_to_1D_map']
		grid_index_1D_to_2D_map = grid_data_all['index_1D_to_2D_map']

		sensitivities_reshaped = sensitivities.reshape((pixel_v,pixel_h,len(grid_laplacian)))
		sensitivities_reshaped = np.transpose(sensitivities_reshaped , (1,0,2))

		if grid_resolution==8:
			# temp=1e-3
			temp=1e-7
		elif grid_resolution==2:
			temp=1e-4
		elif grid_resolution==4:
			temp=0
		sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,std_treshold = temp,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False)

		# this step is to adapt the matrix to the size of the foil I measure, that can be slightly different
		binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
		shape = list(FAST_counts_minus_background_crop.shape[1:])
		if shape!=list(sensitivities_reshaped_masked.shape[:-1]):
			shape.extend([len(grid_laplacian_masked)])
			def mapping(output_coords):
				return(output_coords[0]/shape[0]*pixel_h,output_coords[1]/shape[1]*pixel_v,output_coords[2])
			sensitivities_reshaped_masked2 = geometric_transform(sensitivities_reshaped_masked,mapping,output_shape=shape)
		else:
			sensitivities_reshaped_masked2 = cp.deepcopy(sensitivities_reshaped_masked)

		sensitivities_binned = proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
		sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
		sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

		# additional cropping of the foil to exlude regions without plasma LOS, the frame of the foil and gas puff
		# ROI = np.array([[0.2,0.85],[0.1,0.9]])
		# ROI = np.array([[0.05,0.95],[0.05,0.95]])
		# ROI = np.array([[0.2,0.95],[0.1,1]])
		ROI1 = np.array([[0.03,0.80],[0.03,0.85]])
		ROI2 = np.array([[0.03,0.7],[0.03,0.91]])
		ROI_beams = np.array([[0.,0.3],[0.5,1]])
		sensitivities_binned_crop,selected_ROI = cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_binned,ROI1,ROI2,ROI_beams,laser_to_analyse)

		select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-3)
		selected_ROI_no_plasma = np.logical_and(selected_ROI,np.logical_not(select_foil_region_with_plasma))
		select_foil_region_with_plasma = select_foil_region_with_plasma.flatten()

		if grid_resolution==8:
			# temp=1e-3
			temp=1e-7
		elif grid_resolution==2:
			temp=1e-4
		elif grid_resolution==4:
			temp=0
		sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked,std_treshold = temp)

		selected_super_x_cells = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>0.85,np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.65)

		x1 = [1.55,0.25]	# r,z
		x2 = [1.1,-0.15]
		interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
		select = np.mean(grid_data_masked_crop,axis=1)[:,1]>interp(np.mean(grid_data_masked_crop,axis=1)[:,0])
		selected_central_border_cells = np.logical_and(select,np.logical_and(np.max(grid_Z_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,1]>-0.5))
		selected_central_border_cells = np.dot(grid_laplacian_masked_crop,selected_central_border_cells*np.random.random(selected_central_border_cells.shape))!=0

		selected_central_column_border_cells = np.logical_and(np.logical_and(np.max(grid_R_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)
		selected_central_column_border_cells = np.logical_and(np.logical_and(np.dot(grid_laplacian_masked_crop,selected_central_column_border_cells*np.random.random(selected_central_column_border_cells.shape))!=0,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)

		selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=6,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1)
		selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=6,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.5))

		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells*np.random.random(selected_edge_cells.shape))!=0)
		if grid_resolution<8:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
		if grid_resolution<4:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)

		sensitivities_binned_crop_shape = sensitivities_binned_crop.shape
		sensitivities_binned_crop = sensitivities_binned_crop.reshape((sensitivities_binned_crop.shape[0]*sensitivities_binned_crop.shape[1],sensitivities_binned_crop.shape[2]))

		if shrink_factor_x > 1:
			foil_resolution = str(shrink_factor_x) + 'x' + str(shrink_factor_x)
		else:
			foil_resolution = str(shape[0])

		foil_res = '_foil_pixel_h_' + str(foil_resolution)
		path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
		path_sensitivity_original = cp.deepcopy(path_sensitivity)

		binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
		print('starting '+binning_type)
		# powernoback_full = saved_file_dict_short[binning_type].all()['powernoback_full']
		# powernoback_std_full = saved_file_dict_short[binning_type].all()['powernoback_std_full']

		# from here I make the new method.
		# I consider the nominal properties as central value, with:
		# emissivity -10% (from Japanese properties i have std of ~5%, but my nominal value is ~1 and emissivity cannot be >1 so I double the interval down)
		# thickness +/-15% (from Japanese properties i have std of ~15%)
		# diffusivity -10% (this is missing from the Japanese data, so I guess std ~10%)

		emissivity_steps = 5
		thickness_steps = 9
		rec_diffusivity_steps = 9
		sigma_emissivity = 0.1
		sigma_thickness = 0.15
		sigma_rec_diffusivity = 0.1
		emissivity_array = np.linspace(1-sigma_emissivity*3,1,num=emissivity_steps)
		emissivity_log_prob =  -(0.5*(((1-emissivity_array)/sigma_emissivity)**2))**1	# super gaussian order 1, probability assigned linearly
		emissivity_log_prob = emissivity_log_prob -np.log(np.trapz(np.exp(emissivity_log_prob),x=emissivity_array))	# normalisation for logarithmic probabilities
		thickness_array = np.linspace(1-sigma_thickness*3,1+sigma_thickness*3,num=thickness_steps)
		thickness_log_prob =  -(0.5*(((1-thickness_array)/sigma_thickness)**2))**1	# super gaussian order 1, probability assigned linearly
		thickness_log_prob = thickness_log_prob -np.log(np.trapz(np.exp(thickness_log_prob),x=thickness_array))	# normalisation for logarithmic probabilities
		rec_diffusivity = 1/diffusivity
		rec_diffusivity_array = np.linspace(1-sigma_rec_diffusivity*3,1+sigma_rec_diffusivity*3,num=rec_diffusivity_steps)
		rec_diffusivity_log_prob =  -(0.5*(((1-rec_diffusivity_array)/sigma_rec_diffusivity)**2))**1	# super gaussian order 1, probability assigned linearly
		rec_diffusivity_log_prob = rec_diffusivity_log_prob -np.log(np.trapz(np.exp(rec_diffusivity_log_prob),x=rec_diffusivity_array))	# normalisation for logarithmic probabilities

		tend = get_tend(laser_to_analyse[-9:-4])+0.01	 # I add 10ms just for safety and to catch disruptions

		time_full_binned = time_binned[1:-1]
		BBrad_full_crop = BBrad[time_full_binned<tend]
		BBrad_full_crop[:,np.logical_not(selected_ROI)] = 0
		BBrad_std_full_crop = BBrad_std[time_full_binned<tend]
		BBrad_std_full_crop[:,np.logical_not(selected_ROI)] = 0
		diffusion_full_crop = diffusion[time_full_binned<tend]
		diffusion_full_crop[:,np.logical_not(selected_ROI)] = 0
		diffusion_std_full_crop = diffusion_std[time_full_binned<tend]
		diffusion_std_full_crop[:,np.logical_not(selected_ROI)] = 0
		timevariation_full_crop = timevariation[time_full_binned<tend]
		timevariation_full_crop[:,np.logical_not(selected_ROI)] = 0
		timevariation_std_full_crop = timevariation_std[time_full_binned<tend]
		timevariation_std_full_crop[:,np.logical_not(selected_ROI)] = 0
		time_full_binned_crop = time_full_binned[time_full_binned<tend]

		powernoback_full = (np.array([[[BBrad_full_crop[0].tolist()]*rec_diffusivity_steps]*thickness_steps]*emissivity_steps).T*emissivity_array).T	# emissivity, thickness, rec_diffusivity
		powernoback_full += (np.array([(np.array([[diffusion_full_crop[0].tolist()]*rec_diffusivity_steps]*thickness_steps).T*thickness_array).T.tolist()]*emissivity_steps))
		powernoback_full += (np.array([(np.array([(np.array([timevariation_full_crop[0].tolist()]*rec_diffusivity_steps).T*rec_diffusivity_array).T.tolist()]*thickness_steps).T*thickness_array).T.tolist()]*emissivity_steps))

		alpha = 1e-4
		A_ = np.dot(sensitivities_binned_crop.T, sensitivities_binned_crop) + (alpha**2) * np.dot(grid_laplacian_masked_crop.T, grid_laplacian_masked_crop)
		d=powernoback_full.reshape(emissivity_steps*thickness_steps*rec_diffusivity_steps,powernoback_full.shape[-2]*powernoback_full.shape[-1])
		b_ = np.dot(sensitivities_binned_crop.T,d.T)

		U, s, Vh = np.linalg.svd(A_)
		sigma = np.diag(s)
		inv_sigma = np.diag(1 / s)
		a1 = np.dot(U, np.dot(sigma, Vh))
		a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
		m = np.dot(a1_inv, b_).T
		neg_m_penalty = np.zeros_like(m)
		neg_m_penalty[m<0] = m[m<0]
		# neg_d_penalty = np.dot(sensitivities_binned_crop,neg_m_penalty.T).T
		neg_m_penalty = -20*neg_m_penalty/np.median(np.flip(np.sort(m[m<0]),axis=0)[-np.sum(m<0)//10:])
		neg_m_penalty = neg_m_penalty.reshape((*powernoback_full.shape[:-2],neg_m_penalty.shape[-1]))

		edge_penalty = np.zeros_like(m)
		edge_penalty[:,selected_edge_cells] = np.max(m[:,selected_edge_cells],0)
		edge_penalty = -50*edge_penalty/np.median(np.sort(edge_penalty[edge_penalty>0])[-np.sum(edge_penalty>0)//10:])
		edge_penalty = edge_penalty.reshape((*powernoback_full.shape[:-2],edge_penalty.shape[-1]))

		if True:	# if I want to bypass this penalty
			neg_powernoback_full_penalty = np.zeros_like(powernoback_full)	# emissivity, thickness, rec_diffusivity
			neg_powernoback_full_penalty[powernoback_full<0] = powernoback_full[powernoback_full<0]
			neg_powernoback_full_penalty = neg_powernoback_full_penalty.reshape((emissivity_steps*thickness_steps*rec_diffusivity_steps,neg_powernoback_full_penalty.shape[-2]*neg_powernoback_full_penalty.shape[-1]))
			neg_powernoback_full_penalty = np.dot(a1_inv, np.dot(sensitivities_binned_crop.T,neg_powernoback_full_penalty.T)).T
			neg_powernoback_full_penalty -= np.max(neg_powernoback_full_penalty)
			# neg_powernoback_full_penalty[neg_powernoback_full_penalty<0] = -10*neg_powernoback_full_penalty[neg_powernoback_full_penalty<0]/np.min(neg_powernoback_full_penalty[neg_powernoback_full_penalty<0])
			if neg_powernoback_full_penalty.min()<0:
				neg_powernoback_full_penalty = -20*neg_powernoback_full_penalty/np.median(np.flip(np.sort(neg_powernoback_full_penalty[neg_powernoback_full_penalty<0]),axis=0)[-np.sum(neg_powernoback_full_penalty<0)//10:])
			neg_powernoback_full_penalty = neg_powernoback_full_penalty.reshape(neg_m_penalty.shape)
		else:
			neg_powernoback_full_penalty = np.zeros_like(neg_m_penalty)
		# neg_powernoback_full_penalty = np.zeros_like(powernoback_full[:,:,:,0])	# emissivity, thickness, rec_diffusivity,coord_1,coord_2
		likelihood = np.transpose(np.transpose(neg_powernoback_full_penalty, (1,2,3,0)) + emissivity_log_prob, (3,0,1,2))
		likelihood = np.transpose(np.transpose(likelihood, (0,2,3,1)) + thickness_log_prob, (0,3,1,2))
		likelihood = np.transpose(np.transpose(likelihood, (0,1,3,2)) + rec_diffusivity_log_prob, (0,1,3,2))
		likelihood += neg_m_penalty
		likelihood += edge_penalty
		# likelihood = np.sum(likelihood, axis=-3)
		likelihood = likelihood -np.log(np.trapz(np.trapz(np.trapz(np.exp(likelihood),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0))	# normalisation for logarithmic probabilities
		total_volume = np.trapz(np.trapz(np.trapz(np.ones((emissivity_steps,thickness_steps,rec_diffusivity_steps)),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0)
		final_emissivity = np.trapz(np.trapz(np.trapz(np.exp(likelihood)*(m.reshape(neg_m_penalty.shape)),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0)

		powernoback_full_orig = diffusion_full_crop + timevariation_full_crop + BBrad_full_crop
		sigma_powernoback_full = ( (diffusion_full_crop**2)*((diffusion_std_full_crop/diffusion_full_crop)**2+sigma_thickness**2) + (timevariation_full_crop**2)*((timevariation_std_full_crop/timevariation_full_crop)**2+sigma_thickness**2+sigma_rec_diffusivity**2) + (BBrad_full_crop**2)*((BBrad_std_full_crop/BBrad_full_crop)**2+sigma_emissivity**2) )**0.5

		grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
		grid_Z_derivate_masked_crop_scaled = grid_Z_derivate_masked_crop/((1e-2*grid_resolution)**1)
		grid_R_derivate_masked_crop_scaled = grid_R_derivate_masked_crop/((1e-2*grid_resolution)**1)
		reference_sigma_powernoback = np.nanmedian(sigma_powernoback_full)
		regolarisation_coeff = 1e-3	# ok for np.median(sigma_powernoback_full)=78.18681
		if grid_resolution==4:
			regolarisation_coeff = 3e-4
		elif grid_resolution==2:
			regolarisation_coeff = 1e-4
		# regolarisation_coeff = 1e-5 / ((reference_sigma_powernoback/78.18681)**0.5)
		sigma_emissivity = 2e3	# this is completely arbitrary
		# sigma_emissivity = 1e4 * ((np.median(sigma_powernoback_full)/78)**0.5)	# I think it must go hand in hand with the uncertanty in the pixels
		regolarisation_coeff_edge = 1e1	# I raiset it artificially from 1e-3 to engage regolarisation_coeff_central_border_Z_derivate and regolarisation_coeff_central_column_border_R_derivate
		regolarisation_coeff_central_border_Z_derivate = 1e-20
		regolarisation_coeff_central_column_border_R_derivate = 1e-20
		regolarisation_coeff_divertor = regolarisation_coeff/1.5
		sigma_emissivity_2 = sigma_emissivity**2

		sigma_powernoback_full[np.isnan(sigma_powernoback_full)] = 1
		selected_ROI_internal = selected_ROI.flatten()
		inverted_data = []
		inverted_data_likelihood = []
		inverted_data_info = []
		inverted_data_plasma_region_offset = []
		inverted_data_homogeneous_offset = []
		fitted_foil_power = []
		foil_power = []
		foil_power_residuals = []
		fit_error = []
		chi_square_all = []
		regolarisation_coeff_all = []
		time_per_iteration = []
		for i_t in range(len(time_full_binned_crop)):
			time_start = tm.time()

			print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
			# plt.figure()
			# plt.imshow(powernoback_full_orig[i_t])
			# plt.colorbar()
			# plt.pause(0.01)
			#
			# plt.figure()
			# plt.imshow(sigma_powernoback_full[i_t])
			# plt.colorbar()
			# plt.pause(0.01)

			powernoback = powernoback_full_orig[i_t].flatten()
			sigma_powernoback = sigma_powernoback_full[i_t].flatten()
			# sigma_powernoback = np.ones_like(powernoback)*10
			sigma_powernoback_2 = sigma_powernoback**2
			homogeneous_scaling=1e-4

			if time_full_binned_crop[i_t]<0.2:
				if False:	# I can start from a random sample no problem
					guess = final_emissivity+(np.random.random(final_emissivity.shape)-0.5)*1e3	# the 1e3 is to make things harder for the solver, but also to go away a bit from the best fit of other functions
					guess[selected_edge_cells] = guess[selected_edge_cells]*1e-4
					guess = np.concatenate((guess,[0.1/homogeneous_scaling,0.1/homogeneous_scaling]))
				else:
					guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
			else:
				guess = cp.deepcopy(x_optimal)
				guess[:-2] += (np.random.random(x_optimal.shape[0]-2)-0.5)*1e3	# I still add a bit of scramble to give it the freedom to find the best configuration
			# target_chi_square = 1020	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
			target_chi_square = sensitivities_binned_crop.shape[1]	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
			target_chi_square_sigma = 200	# this should be tight, because for such a high number of degrees of freedom things should average very well

			def prob_and_gradient(emissivity_plus,*powernoback):
				homogeneous_offset = emissivity_plus[-1]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
				homogeneous_offset_plasma = emissivity_plus[-2]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
				# print(homogeneous_offset,homogeneous_offset_plasma)
				emissivity = emissivity_plus[:-2]
				emissivity[emissivity==0] = 1e-10
				foil_power_guess = np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma
				foil_power_error = powernoback - foil_power_guess
				emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
				Z_derivate = np.dot(grid_Z_derivate_masked_crop_scaled,emissivity)
				R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)

				likelihood_power_fit = np.sum((foil_power_error/sigma_powernoback)**2)
				likelihood_chi_square =0# ((likelihood_power_fit-target_chi_square)/target_chi_square_sigma)**2
				likelihood_emissivity_pos = np.sum((np.minimum(0.,emissivity)/sigma_emissivity)**2)
				likelihood_emissivity_laplacian = (regolarisation_coeff**2)* np.sum(((emissivity_laplacian*np.logical_not(selected_super_x_cells) /sigma_emissivity)**2))
				likelihood_emissivity_laplacian_superx = (regolarisation_coeff_divertor**2)* np.sum(((emissivity_laplacian*selected_super_x_cells /sigma_emissivity)**2))
				likelihood_emissivity_edge_laplacian = 0#(regolarisation_coeff_edge**2)* np.sum(((emissivity_laplacian*selected_edge_cells_for_laplacian /sigma_emissivity)**2))
				likelihood_emissivity_edge = (regolarisation_coeff_edge**2)*np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
				likelihood_emissivity_central_border_Z_derivate = (regolarisation_coeff_central_border_Z_derivate**2)* np.sum((Z_derivate*selected_central_border_cells/sigma_emissivity)**2)
				likelihood_emissivity_central_column_border_R_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)* np.sum((R_derivate*selected_central_column_border_cells/sigma_emissivity)**2)
				likelihood = likelihood_power_fit + likelihood_emissivity_pos + likelihood_emissivity_laplacian + likelihood_emissivity_edge_laplacian + likelihood_emissivity_edge + likelihood_emissivity_central_border_Z_derivate + likelihood_emissivity_central_column_border_R_derivate + likelihood_emissivity_laplacian_superx
				likelihood_homogeneous_offset = 0#(homogeneous_offset/reference_sigma_powernoback)**2
				likelihood_homogeneous_offset_plasma = (homogeneous_offset_plasma/reference_sigma_powernoback)**2
				likelihood = likelihood + likelihood_homogeneous_offset + likelihood_homogeneous_offset_plasma + likelihood_chi_square

				likelihood_power_fit_derivate = np.concatenate((-2*np.dot((foil_power_error/sigma_powernoback_2),sensitivities_binned_crop),[-2*np.sum(foil_power_error*select_foil_region_with_plasma/sigma_powernoback_2)*homogeneous_scaling,-2*np.sum(foil_power_error*selected_ROI_internal/sigma_powernoback_2)*homogeneous_scaling]))
				likelihood_chi_square_derivate =0# likelihood_power_fit_derivate * 2 *(likelihood_power_fit-target_chi_square)/(target_chi_square_sigma**2)
				likelihood_emissivity_pos_derivate = 2*(np.minimum(0.,emissivity)**2)/emissivity/sigma_emissivity_2
				likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian*np.logical_not(selected_super_x_cells) , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian*selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				likelihood_emissivity_edge_laplacian_derivate = 0#2*(regolarisation_coeff_edge**2) * np.dot(emissivity_laplacian*selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				likelihood_emissivity_edge_derivate = 2*(regolarisation_coeff_edge**2)*emissivity*selected_edge_cells/sigma_emissivity_2
				likelihood_emissivity_central_border_Z_derivate_derivate = 2*(regolarisation_coeff_central_border_Z_derivate**2)*np.dot(Z_derivate*selected_central_border_cells,grid_Z_derivate_masked_crop_scaled)/sigma_emissivity_2
				likelihood_emissivity_central_column_border_R_derivate_derivate = 2*(regolarisation_coeff_central_column_border_R_derivate**2)*np.dot(R_derivate*selected_central_column_border_cells,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
				likelihood_derivate = likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate + likelihood_emissivity_edge_laplacian_derivate + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_border_Z_derivate_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate + likelihood_emissivity_laplacian_derivate_superx
				likelihood_homogeneous_offset_derivate = 0#2*homogeneous_offset*homogeneous_scaling/(reference_sigma_powernoback**2)
				likelihood_homogeneous_offset_plasma_derivate = 2*homogeneous_offset_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
				likelihood_derivate = np.concatenate((likelihood_derivate,[likelihood_homogeneous_offset_plasma_derivate,likelihood_homogeneous_offset_derivate])) + likelihood_power_fit_derivate + likelihood_chi_square_derivate
				# likelihood_derivate = likelihood_emissivity_central_border_derivate
				# print([likelihood,likelihood_derivate.max(),likelihood_derivate.min()])
				return likelihood,likelihood_derivate

			if time_full_binned_crop[i_t]<0.1:
				x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (powernoback), iprint=0, factr=1e0, pgtol=1e-6, maxiter=5000)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
			else:
				x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (powernoback), iprint=0, factr=1e0, pgtol=1e-7, maxiter=5000)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
			# if opt_info['warnflag']>0:
			# 	print('incomplete fit so restarted')
			# 	x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=x_optimal, args = (powernoback), iprint=0, factr=1e0, pgtol=1e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
			# x_optimal[-2:] *= homogeneous_scaling
			x_optimal[-2:] *= np.array([homogeneous_scaling,homogeneous_scaling])

			foil_power_guess = np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal
			foil_power_error = powernoback - foil_power_guess
			chi_square = np.sum((foil_power_error/sigma_powernoback)**2)
			print('chi_square '+str(chi_square))

			voxels_centre = np.mean(grid_data_masked_crop,axis=1)
			dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
			dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
			dist_mean = (dz**2 + dr**2)/2
			recompose_voxel_emissivity = np.zeros((len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))*np.nan
			for i_r,r in enumerate(np.unique(voxels_centre[:,0])):
				for i_z,z in enumerate(np.unique(voxels_centre[:,1])):
					dist = (voxels_centre[:,0]-r)**2 + (voxels_centre[:,1]-z)**2
					if dist.min()<dist_mean/2:
						index = np.abs(dist).argmin()
						# recompose_voxel_emissivity[i_r,i_z] = guess[index]
						# recompose_voxel_emissivity[i_r,i_z] = (x_optimal-guess)[index]
						# recompose_voxel_emissivity[i_r,i_z] = (x_optimal2-x_optimal3)[index]
						recompose_voxel_emissivity[i_r,i_z] = x_optimal[index]
						# recompose_voxel_emissivity[i_r,i_z] = likelihood_emissivity_laplacian[index]

			recompose_voxel_emissivity *= 4*np.pi	# this exist because the sensitivity matrix is built with 1W/str/m^3/ x nm emitters while I use 1W as reference, so I need to multiply the results by 4pi

			if False:	# only for visualisation
				plt.figure(figsize=(12,13))
				# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
				plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
				plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
				temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
				for i in range(len(all_time_sep_r[temp])):
					plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
				plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
				plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
				plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
				plt.colorbar().set_label('emissivity [W/m3]')
				plt.ylim(top=0.5)
				# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example19.eps')
				plt.pause(0.01)

			inverted_data.append(recompose_voxel_emissivity)
			inverted_data_likelihood.append(y_opt)
			inverted_data_plasma_region_offset.append(x_optimal[-2])
			inverted_data_homogeneous_offset.append(x_optimal[-1])
			inverted_data_info.append(opt_info)
			fitted_foil_power.append((np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape))
			foil_power.append(powernoback_full_orig[i_t])
			foil_power_residuals.append(powernoback_full_orig[i_t]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape))
			fit_error.append(np.sum(((powernoback_full_orig[i_t][selected_ROI]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape)[[selected_ROI]]))**2)**0.5/np.sum(selected_ROI))
			chi_square_all.append(chi_square)
			regolarisation_coeff_all.append(regolarisation_coeff)
			time_per_iteration.append(tm.time()-time_start)

		inverted_data = np.array(inverted_data)
		inverted_data_likelihood = -np.array(inverted_data_likelihood)
		inverted_data_plasma_region_offset = np.array(inverted_data_plasma_region_offset)
		inverted_data_homogeneous_offset = np.array(inverted_data_homogeneous_offset)
		fit_error = np.array(fit_error)
		chi_square_all = np.array(chi_square_all)
		regolarisation_coeff_all = np.array(regolarisation_coeff_all)
		time_per_iteration = np.array(time_per_iteration)
		fitted_foil_power = np.array(fitted_foil_power)
		foil_power = np.array(foil_power)
		foil_power_residuals = np.array(foil_power_residuals)

		timeout = 20*60	# 20 minutes
		while efit_reconstruction==None and timeout>0:
			try:
				EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
				efit_reconstruction = mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])

			except:
				print('EFIT missing, waiting 20 seconds')
			tm.sleep(20)
			timeout -= 20

		if efit_reconstruction!=None:

			temp = brightness[:,:,:int(np.shape(brightness)[2]*0.75)]
			temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
			ani,efit_reconstruction = movie_from_data(np.array([np.flip(np.transpose(brightness,(0,2,1)),axis=2)]), 1/np.median(np.diff(time_binned)),timesteps=time_binned[1:-1],integration=laser_int_time/1000,time_offset=time_binned[0],extvmin=0,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction)
			ani.save('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+ '_FAST_brightness.mp4', fps=5*(1/np.mean(np.diff(time_binned)))/383, writer='ffmpeg',codec='mpeg4')
			plt.close('all')

			all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
			all_time_strike_points_location = return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			outer_leg_tot_rad_power_all = []
			sxd_tot_rad_power_all = []
			inner_leg_tot_rad_power_all = []
			core_tot_rad_power_all = []
			x_point_tot_rad_power_all = []
			for i_t in range(len(time_full_binned_crop)):
				temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
				xpoint_r = efit_reconstruction.lower_xpoint_r[temp]
				xpoint_z = efit_reconstruction.lower_xpoint_z[temp]
				z_,r_ = np.meshgrid(np.unique(voxels_centre[:,1]),np.unique(voxels_centre[:,0]))
				temp = cp.deepcopy(inverted_data[i_t])
				temp[z_>xpoint_z] = 0
				temp[r_<xpoint_r] = 0
				outer_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				temp = cp.deepcopy(inverted_data[i_t])
				temp[z_>xpoint_z] = 0
				temp[r_>xpoint_r] = 0
				inner_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				temp = cp.deepcopy(inverted_data[i_t])
				temp[z_<xpoint_z] = 0
				temp[z_>0] = 0
				core_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				temp = cp.deepcopy(inverted_data[i_t])
				temp[z_<-1.5] = 0
				temp[r_>0.8] = 0
				sxd_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				temp = cp.deepcopy(inverted_data[i_t])
				temp[((z_-xpoint_z)**2+(r_-xpoint_r)**2)**0.5>0.10] = 0
				x_point_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				outer_leg_tot_rad_power_all.append(outer_leg_tot_rad_power)
				inner_leg_tot_rad_power_all.append(inner_leg_tot_rad_power)
				core_tot_rad_power_all.append(core_tot_rad_power)
				sxd_tot_rad_power_all.append(sxd_tot_rad_power)
				x_point_tot_rad_power_all.append(x_point_tot_rad_power)
			outer_leg_tot_rad_power_all = np.array(outer_leg_tot_rad_power_all)
			inner_leg_tot_rad_power_all = np.array(inner_leg_tot_rad_power_all)
			core_tot_rad_power_all = np.array(core_tot_rad_power_all)
			sxd_tot_rad_power_all = np.array(sxd_tot_rad_power_all)
			x_point_tot_rad_power_all = np.array(x_point_tot_rad_power_all)

			plt.figure(figsize=(20, 10))
			plt.plot(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3,label='outer_leg')
			plt.plot(time_full_binned_crop,sxd_tot_rad_power_all/1e3,label='sxd')
			plt.plot(time_full_binned_crop,inner_leg_tot_rad_power_all/1e3,label='inner_leg')
			plt.plot(time_full_binned_crop,core_tot_rad_power_all/1e3,label='core')
			plt.plot(time_full_binned_crop,x_point_tot_rad_power_all/1e3,label='x_point')
			plt.plot(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3+inner_leg_tot_rad_power_all/1e3+core_tot_rad_power_all/1e3,label='tot')
			plt.title('radiated power in the lower half of the machine')
			plt.legend(loc='best', fontsize='x-small')
			plt.xlabel('time [s]')
			plt.ylabel('power [kW]')
			plt.grid()
			plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_tot_rad_power.eps')
			plt.close()

			inverted_dict[str(grid_resolution)]['outer_leg_tot_rad_power_all'] = outer_leg_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['inner_leg_tot_rad_power_all'] = inner_leg_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['core_tot_rad_power_all'] = core_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['sxd_tot_rad_power_all'] = sxd_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_all'] = x_point_tot_rad_power_all


		path_power_output = os.path.split(laser_to_analyse)[0] + '/' + str(shot_number)
		if not os.path.exists(path_power_output):
			os.makedirs(path_power_output)
		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,time_per_iteration)
		# plt.semilogy()
		plt.title('time spent per iteration')
		plt.xlabel('time [s]')
		plt.ylabel('time [s]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_time_trace.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,inverted_data_likelihood)
		# plt.semilogy()
		plt.title('Fit log likelihood')
		plt.xlabel('time [s]')
		plt.ylabel('log likelihoog [au]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_likelihood.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,chi_square_all)
		plt.plot(time_full_binned_crop,np.ones_like(time_full_binned_crop)*target_chi_square,'--k')
		# plt.semilogy()
		if False:
			plt.title('chi square obtained vs requested\nfixed regularisation of '+str(regolarisation_coeff))
		else:
			plt.title('chi square obtained vs requested\nflexible regolarisation coefficient')
		plt.xlabel('time [s]')
		plt.ylabel('chi square [au]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_chi_square.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,regolarisation_coeff_all)
		# plt.semilogy()
		plt.title('regolarisation coefficient obtained')
		plt.semilogy()
		plt.xlabel('time [s]')
		plt.ylabel('regolarisation coefficient [au]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_regolarisation_coeff.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,fit_error)
		# plt.semilogy()
		plt.title('Fit error ( sum((image-fit)^2)^0.5/num pixels )')
		plt.xlabel('time [s]')
		plt.ylabel('average fit error [W/m2]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_fit_error.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,inverted_data_plasma_region_offset,label='plasma region')
		plt.plot(time_full_binned_crop,inverted_data_homogeneous_offset,label='whole foil')
		plt.title('Offsets to match foil power')
		plt.legend(loc='best', fontsize='x-small')
		plt.xlabel('time [s]')
		plt.ylabel('power density [W/m2]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_offsets.eps')
		plt.close()

		if efit_reconstruction!=None:

			additional_points_dict,radiator_xpoint_distance_all,radiator_above_xpoint_all,radiator_magnetic_radious_all,radiator_baricentre_magnetic_radious_all,radiator_baricentre_above_xpoint_all = find_radiator_location(inverted_data,np.unique(voxels_centre[:,0]),np.unique(voxels_centre[:,1]),time_full_binned_crop,efit_reconstruction)

			inverted_dict[str(grid_resolution)]['radiator_location_all'] = additional_points_dict['0']
			inverted_dict[str(grid_resolution)]['radiator_xpoint_distance_all'] = radiator_xpoint_distance_all
			inverted_dict[str(grid_resolution)]['radiator_above_xpoint_all'] = radiator_above_xpoint_all
			inverted_dict[str(grid_resolution)]['radiator_magnetic_radious_all'] = radiator_magnetic_radious_all

			fig, ax = plt.subplots( 2,1,figsize=(8, 12), squeeze=False,sharex=True)
			ax[0,0].plot(time_full_binned_crop,radiator_magnetic_radious_all)
			ax[0,0].plot(time_full_binned_crop,radiator_baricentre_magnetic_radious_all,'--')
			ax[0,0].set_ylim(top=min(np.nanmax(radiator_magnetic_radious_all),1.1),bottom=max(np.nanmin(radiator_magnetic_radious_all),0.9))
			ax[1,0].plot(time_full_binned_crop,radiator_above_xpoint_all)
			ax[1,0].plot(time_full_binned_crop,radiator_baricentre_above_xpoint_all,'--')
			fig.suptitle('Location of the x-point radiator\n"--"=baricentre r=20cm around x-point')
			ax[0,0].set_ylabel('normalised psi [au]')
			ax[0,0].grid()
			ax[1,0].set_xlabel('time [s]')
			ax[1,0].set_ylabel('position above x-point [m]')
			ax[1,0].grid()
			plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_x_point_location.eps')
			plt.close()

			extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			ani,trash = movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nregolarisation_coeff_divertor %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,regolarisation_coeff_divertor,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,additional_points_dict=additional_points_dict)#,extvmin=0,extvmax=4e4)

		else:

			# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(recompose_voxel_emissivity,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))),integration=laser_int_time/1000,barlabel='Emissivity [W/m3]')#,extvmin=0,extvmax=4e4)
			extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			ani,trash = movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nregolarisation_coeff_divertor %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,regolarisation_coeff_divertor,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)#,extvmin=0,extvmax=4e4)
		ani.save('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_reconstruct_emissivity_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
		plt.close()


		inverted_dict[str(grid_resolution)]['binning_type'] = binning_type
		inverted_dict[str(grid_resolution)]['inverted_data'] = inverted_data
		inverted_dict[str(grid_resolution)]['inverted_data_likelihood'] = inverted_data_likelihood
		inverted_dict[str(grid_resolution)]['inverted_data_info'] = inverted_data_info
		inverted_dict[str(grid_resolution)]['select_foil_region_with_plasma'] = select_foil_region_with_plasma
		inverted_dict[str(grid_resolution)]['inverted_data_plasma_region_offset'] = inverted_data_plasma_region_offset
		inverted_dict[str(grid_resolution)]['inverted_data_homogeneous_offset'] = inverted_data_homogeneous_offset
		inverted_dict[str(grid_resolution)]['time_full_binned_crop'] = time_full_binned_crop
		inverted_dict[str(grid_resolution)]['fitted_foil_power'] = fitted_foil_power
		inverted_dict[str(grid_resolution)]['foil_power'] = foil_power
		inverted_dict[str(grid_resolution)]['foil_power_residuals'] = foil_power_residuals
		inverted_dict[str(grid_resolution)]['fit_error'] = fit_error
		inverted_dict[str(grid_resolution)]['chi_square_all'] = chi_square_all
		inverted_dict[str(grid_resolution)]['geometry'] = dict([])
		inverted_dict[str(grid_resolution)]['geometry']['R'] = np.unique(voxels_centre[:,0])
		inverted_dict[str(grid_resolution)]['geometry']['Z'] = np.unique(voxels_centre[:,1])

		if efit_reconstruction!=None:

			inversion_R = np.unique(voxels_centre[:,0])
			inversion_Z = np.unique(voxels_centre[:,1])
			local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution = track_outer_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)

			try:
				fig, ax = plt.subplots( 1,2,figsize=(10, 20), squeeze=False,sharey=True)
				temp = np.array(local_power_all)
				temp[np.isnan(temp)] = 0
				im1 = ax[0,0].imshow(temp,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(temp[:-4]),vmax=np.max(temp[:-4]))
				ax[0,0].plot(leg_length_all,time_full_binned_crop,'--k')
				temp = np.array(local_mean_emis_all)
				temp[np.isnan(temp)] = 0
				im2 = ax[0,1].imshow(temp,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(temp[:-4]),vmax=np.max(temp[:-4]))
				ax[0,1].plot(leg_length_all,time_full_binned_crop,'--k')
				fig.suptitle('tracking radiation on the outer leg')
				ax[0,0].set_xlabel('distance from the strike point [m]')
				ax[0,0].grid()
				ax[0,0].set_ylabel('time [s]')
				plt.colorbar(im1,ax=ax[0,0]).set_label('Integrated power [W]')
				# ax[0,0].colorbar().set_label('Integrated power [W]')
				ax[0,1].set_xlabel('distance from the strike point [m]')
				ax[0,1].grid()
				# ax[0,1].colorbar().set_label('Emissivity [W/m3]')
				plt.colorbar(im2,ax=ax[0,1]).set_label('Emissivity [W/m3]')
				plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking.eps')
				plt.close()


				number_of_curves_to_plot = 12
				alpha = 0.9
				# colors_smooth = np.array([np.linspace(0,1,num=number_of_curves_to_plot),np.linspace(1,0,num=number_of_curves_to_plot),[0]*number_of_curves_to_plot]).T
				colors_smooth = np.linspace(0,0.9,num=number_of_curves_to_plot).astype(str)
				select = np.unique(np.round(np.linspace(0,len(local_mean_emis_all)-1,num=number_of_curves_to_plot))).astype(int)
				fig, ax = plt.subplots( 2,1,figsize=(10, 20), squeeze=False,sharex=True)
				fig.suptitle('tracking radiation on the outer leg')
				# plt.figure()
				for i_i_t,i_t in enumerate(select):
					# ax[0,0].plot(np.cumsum(leg_length_interval_all[i_t]),local_mean_emis_all[i_t],color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3)
					# ax[1,0].plot(np.cumsum(leg_length_interval_all[i_t]),local_power_all[i_t],color=colors_smooth[i_i_t],linewidth=3)
					to_plot_x = np.array(leg_length_interval_all[i_t])
					to_plot_y1 = np.array(local_mean_emis_all[i_t])
					to_plot_y1 = to_plot_y1[to_plot_x>0]
					to_plot_y2 = np.array(local_power_all[i_t])
					to_plot_y2 = to_plot_y2[to_plot_x>0]
					to_plot_x = to_plot_x[to_plot_x>0]
					to_plot_x = np.flip(np.sum(to_plot_x)-(np.cumsum(to_plot_x)-np.array(to_plot_x)/2),axis=0)
					to_plot_y1 = np.flip(to_plot_y1,axis=0)
					to_plot_y2 = np.flip(to_plot_y2,axis=0)
					if i_i_t%3==0:
						ax[0,0].plot(to_plot_x,to_plot_y1,'-',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'-',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
					elif i_i_t%3==1:
						ax[0,0].plot(to_plot_x,to_plot_y1,'-.',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'-.',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
					elif i_i_t%3==2:
						ax[0,0].plot(to_plot_x,to_plot_y1,'--',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'--',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
				ax[0,0].legend(loc='best', fontsize='x-small')
				ax[0,0].set_ylabel('average emissivity [W/m3]')
				ax[1,0].set_ylabel('local radiated power [W]')
				# ax[1,0].set_xlabel('distance from target [m]')
				ax[1,0].set_xlabel('distance from x-point [m]')
				ax[0,0].grid()
				ax[1,0].grid()
				plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking_2.png')
				plt.close()

			except Exception as e:
				logging.exception('with error: ' + str(e))
				print('failed to print\n'+'/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking.eps')

			inverted_dict[str(grid_resolution)]['local_outer_leg_power'] = local_power_all
			inverted_dict[str(grid_resolution)]['local_outer_leg_mean_emissivity'] = local_mean_emis_all
			inverted_dict[str(grid_resolution)]['leg_length_all'] = leg_length_all
			inverted_dict[str(grid_resolution)]['leg_length_interval_all'] = leg_length_interval_all

	return foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop_binned,time_binned,powernoback_output,brightness,binning_type,inverted_dict


def MASTU_pulse_process_FAST3_BB(laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,laser_framerate,laser_digitizer_ID,laser_int_time,seconds_for_reference_frame,start_time_of_pulse,laser_to_analyse,height,width,flag_use_of_first_frames_as_reference,params,errparams,params_BB,errparams_BB,photon_flux_over_temperature_interpolator,BB_proportional,BB_proportional_std,foil_position_dict,pass_number = 0):
	# created 2022/01/11
	# created modifying MASTU_pulse_process_FAST3 to go from polynomial temperature calibration fo black body

	from scipy.ndimage.filters import generic_filter

	max_ROI = [[0,255],[0,319]]
	# foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
	temp_ref_counts = []
	temp_ref_counts_std = []
	temp_counts_minus_background = []
	counts_std = []
	time_partial = []
	timesteps = np.inf
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		if external_clock_marker:
			time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers

		if laser_framerate>60:	# this is to try to see something for the shots in which the framerate is 50Hz
			# basic smoothing
			spectra_orig=np.fft.fft(np.mean(laser_counts_corrected[i],axis=(-1,-2)))
			magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
			freq = np.fft.fftfreq(len(magnitude), d=np.mean(np.diff(time_of_experiment_digitizer_ID_seconds)))
			magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
			freq = np.sort(freq)
			magnitude_smooth = generic_filter(np.log(magnitude),np.median,size=[7])
			peak_oscillation = (magnitude-np.exp(magnitude_smooth))[np.logical_and(freq>20,freq<50)].argmax()
			peak_oscillation_freq = freq[np.logical_and(freq>20,freq<50)][peak_oscillation]
		else:
			# peak_oscillation_freq = freq[np.logical_and(freq>1,freq<2)][peak_oscillation]	# I should do this, but it removes too much signal
			peak_oscillation_freq=laser_framerate/2
		frames_to_average = 1/peak_oscillation_freq/np.mean(np.diff(time_of_experiment_digitizer_ID_seconds))
		laser_counts_corrected_filtered = real_mean_filter_agent(laser_counts_corrected[i],frames_to_average)

		if flag_use_of_first_frames_as_reference:
			# temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[time_of_experiment_digitizer_ID_seconds<0],axis=0))
			temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<-0.5,time_of_experiment_digitizer_ID_seconds>time_of_experiment_digitizer_ID_seconds.min()+0.5)],axis=0))
			temp_ref_counts_std.append(np.std(laser_counts_corrected_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<-0.5,time_of_experiment_digitizer_ID_seconds>time_of_experiment_digitizer_ID_seconds.min()+0.5)],axis=0))
		else:
			temp_ref_counts.append(np.mean(laser_counts_corrected_filtered[-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID))+np.abs(time_of_experiment_digitizer_ID_seconds-40).argmin():+np.abs(time_of_experiment_digitizer_ID_seconds-40).argmin()],axis=0))	# for framerate=50 I avoid it considers too late times
			temp_ref_counts_std.append(np.std(laser_counts_corrected_filtered[-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID))+np.abs(time_of_experiment_digitizer_ID_seconds-40).argmin():+np.abs(time_of_experiment_digitizer_ID_seconds-40).argmin()],axis=0))
		select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds>=-0.1,time_of_experiment_digitizer_ID_seconds<=1.5)
		temp_counts_minus_background.append(laser_counts_corrected_filtered[select_time]-temp_ref_counts[-1])
		counts_std.append(estimate_counts_std(laser_counts_corrected_filtered[select_time],int_time=laser_int_time/1000))
		time_partial.append(time_of_experiment_digitizer_ID_seconds[select_time])
		timesteps = min(timesteps,len(temp_counts_minus_background[-1]))

	for i in range(len(laser_digitizer_ID)):
		temp_counts_minus_background[i] = temp_counts_minus_background[i][:timesteps]
		counts_std[i] = counts_std[i][:timesteps]
		time_partial[i] = time_partial[i][:timesteps]
	temp_counts_minus_background = np.nanmean(temp_counts_minus_background,axis=0)
	counts_std = (1/len(laser_digitizer_ID))*np.nansum(np.array(counts_std)**2,axis=0)**0.5
	temp_ref_counts = np.nanmean(temp_ref_counts,axis=0)
	temp_ref_counts_std = (1/len(laser_digitizer_ID))*np.nansum(np.array(temp_ref_counts_std)**2,axis=0)**0.5
	FAST_counts_minus_background_crop_time = np.nanmean(time_partial,axis=0)

	# I'm going to use the reference frames for foil position
	foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx = get_rotation_crop_parameters(temp_ref_counts,foil_position_dict,laser_to_analyse,temp_counts_minus_background,FAST_counts_minus_background_crop_time)

	# rotation and crop
	# temp_counts_minus_background_rot=rotate(temp_counts_minus_background,foilrotdeg,axes=(-1,-2))
	# if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
	# 	temp_counts_minus_background_rot*=out_of_ROI_mask
	# 	temp_counts_minus_background_rot[np.logical_and(temp_counts_minus_background_rot<np.nanmin(temp_counts_minus_background),temp_counts_minus_background_rot>np.nanmax(temp_counts_minus_background))]=0
	# FAST_counts_minus_background_crop = temp_counts_minus_background_rot[:,foildw:foilup,foillx:foilrx]
	FAST_counts_minus_background_crop = rotate_and_crop_3D(temp_counts_minus_background,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)

	# I drop this to save memory
	# temp = FAST_counts_minus_background_crop[:,:,:int(np.shape(FAST_counts_minus_background_crop)[2]*0.75)]
	# temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
	# ani,efit_reconstruction = movie_from_data(np.array([np.flip(np.transpose(FAST_counts_minus_background_crop,(0,2,1)),axis=2)]), laser_framerate/len(laser_digitizer_ID),timesteps=FAST_counts_minus_background_crop_time,integration=laser_int_time/1000,time_offset=FAST_counts_minus_background_crop_time[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Count increase [au]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,EFIT_output_requested=True)
	# ani.save(laser_to_analyse[:-4]+ '_FAST_count_increase.mp4', fps=5*laser_framerate/len(laser_digitizer_ID)/383, writer='ffmpeg',codec='mpeg4')
	# plt.close('all')

	print('completed FAST count rotating/cropping ' + laser_to_analyse)

	averaged_params = np.mean(params,axis=(0))
	averaged_errparams = np.mean(errparams,axis=(0))
	averaged_params_BB = np.mean(params_BB,axis=(0))
	averaged_errparams_BB = np.mean(errparams_BB,axis=(0))
	averaged_BB_proportional = np.mean(BB_proportional,axis=(0))
	averaged_BB_proportional_std = np.mean(BB_proportional_std,axis=(0))
	counts = temp_counts_minus_background+temp_ref_counts
	# counts_std = estimate_counts_std(counts,laser_framerate)
	temperature_ref = averaged_params[:,:,-1] + averaged_params[:,:,-2] * temp_ref_counts + averaged_params[:,:,-3] * (temp_ref_counts**2)
	temperature_ref_std = (averaged_errparams[:,:,2,2] + (temp_ref_counts_std**2)*(averaged_params[:,:,1]**2) + (temp_ref_counts**2+temp_ref_counts_std**2)*averaged_errparams[:,:,1,1] + (temp_ref_counts_std**2)*(4*temp_ref_counts**2+3*temp_ref_counts_std**2)*(averaged_params[:,:,0]**2) + (temp_ref_counts**4+6*(counts**2)*(temp_ref_counts_std**2)+3*temp_ref_counts_std**4)*averaged_errparams[:,:,0,0] + 2*temp_ref_counts*averaged_errparams[:,:,2,1] + 2*(temp_ref_counts**2+temp_ref_counts_std**2)*averaged_errparams[:,:,2,0] + 2*(temp_ref_counts**3+temp_ref_counts*(temp_ref_counts_std**2))*averaged_errparams[:,:,1,0])**0.5
	try:
		ref_temperature = retrive_vessel_average_temp_archve(int(laser_to_analyse[-9:-4]))
		ref_temperature_std = 0.25	# coming from the fact that there is no noise during transition, so the std must be quite below 1K
	except:
		print('reading of vessel temperature failed')
		ref_temperature = np.mean(temperature_ref)
		ref_temperature_std = (np.sum(np.array(temperature_ref_std)**2)**0.5 / len(np.array(temperature_ref_std).flatten()))

	temperature,temperature_std = count_to_temp_BB_multi_digitizer(np.array([counts]),np.array([averaged_params_BB]),np.array([averaged_errparams_BB]),[0],reference_background=np.array([temp_ref_counts]),reference_background_std=np.array([temp_ref_counts_std]),ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5,wavelength_bottom=2.5,inttime=laser_int_time/1000)
	temperature = temperature[0]
	temperature_std = temperature_std[0]

	temperature_minus_background = temperature - ref_temperature
	temperature_minus_background_std = (temperature_std**2 + ref_temperature_std**2)**0.5

	# rotation and crop
	# temperature_rot=rotate(temperature,foilrotdeg,axes=(-1,-2))
	# temperature_std_rot=rotate(temperature_std,foilrotdeg,axes=(-1,-2))
	# temperature_minus_background_rot=rotate(temperature_minus_background,foilrotdeg,axes=(-1,-2))
	# temperature_minus_background_std_rot=rotate(temperature_minus_background_std,foilrotdeg,axes=(-1,-2))
	# counts_rot=rotate(counts,foilrotdeg,axes=(-1,-2))
	# counts_std_rot=rotate(counts_std,foilrotdeg,axes=(-1,-2))
	# if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
	# 	temperature_rot*=out_of_ROI_mask
	# 	temperature_rot[np.logical_and(temperature_rot<np.nanmin(temperature),temperature_rot>np.nanmax(temperature))]=0
	# 	temperature_std_rot*=out_of_ROI_mask
	# 	temperature_std_rot[np.logical_and(temperature_std_rot<np.nanmin(temperature_std),temperature_std_rot>np.nanmax(temperature_std))]=0
	# 	temperature_minus_background_rot*=out_of_ROI_mask
	# 	temperature_minus_background_rot[np.logical_and(temperature_minus_background_rot<np.nanmin(temperature_minus_background),temperature_minus_background_rot>np.nanmax(temperature_minus_background))]=0
	# 	temperature_minus_background_std_rot*=out_of_ROI_mask
	# 	temperature_minus_background_std_rot[np.logical_and(temperature_minus_background_std_rot<np.nanmin(temperature_minus_background_std),temperature_minus_background_std_rot>np.nanmax(temperature_minus_background_std))]=0
	# 	counts_rot*=out_of_ROI_mask
	# 	counts_rot[np.logical_and(counts_rot<np.nanmin(counts),counts_rot>np.nanmax(counts))]=0
	# 	counts_std_rot*=out_of_ROI_mask
	# 	counts_std_rot[np.logical_and(counts_std_rot<np.nanmin(counts_std),counts_std_rot>np.nanmax(counts_std))]=0
	# temperature_crop = temperature_rot[:,foildw:foilup,foillx:foilrx]
	# temperature_std_crop = temperature_std_rot[:,foildw:foilup,foillx:foilrx]
	# temperature_minus_background_crop = temperature_minus_background_rot[:,foildw:foilup,foillx:foilrx]
	# temperature_minus_background_std_crop = temperature_minus_background_std_rot[:,foildw:foilup,foillx:foilrx]
	# counts_crop = counts_rot[:,foildw:foilup,foillx:foilrx]
	# counts_std_crop = counts_std_rot[:,foildw:foilup,foillx:foilrx]
	temperature_crop = rotate_and_crop_3D(temperature,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)
	temperature_std_crop = rotate_and_crop_3D(temperature_std,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)
	temperature_minus_background_crop = rotate_and_crop_3D(temperature_minus_background,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)
	temperature_minus_background_std_crop = rotate_and_crop_3D(temperature_minus_background_std,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)
	counts_crop = rotate_and_crop_3D(counts,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)
	counts_std_crop = rotate_and_crop_3D(counts_std,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)


	# rotation and crop
	# temp_ref_counts_rot=rotate(temp_ref_counts,foilrotdeg,axes=(-1,-2))
	# temp_ref_counts_std_rot=rotate(temp_ref_counts_std,foilrotdeg,axes=(-1,-2))
	# averaged_BB_proportional_rot=rotate(averaged_BB_proportional,foilrotdeg,axes=(-1,-2))
	# averaged_BB_proportional_std_rot=rotate(averaged_BB_proportional_std,foilrotdeg,axes=(-1,-2))
	# if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
	# 	temp_ref_counts_rot*=out_of_ROI_mask
	# 	temp_ref_counts_rot[np.logical_and(temp_ref_counts_rot<np.nanmin(temp_ref_counts),temp_ref_counts_rot>np.nanmax(temp_ref_counts))]=0
	# 	temp_ref_counts_std_rot*=out_of_ROI_mask
	# 	temp_ref_counts_std_rot[np.logical_and(temp_ref_counts_std_rot<np.nanmin(temp_ref_counts_std),temp_ref_counts_std_rot>np.nanmax(temp_ref_counts_std))]=0
	# 	averaged_BB_proportional_rot*=out_of_ROI_mask
	# 	averaged_BB_proportional_rot[np.logical_and(averaged_BB_proportional_rot<np.nanmin(averaged_BB_proportional),averaged_BB_proportional_rot>np.nanmax(averaged_BB_proportional))]=0
	# 	averaged_BB_proportional_std_rot*=out_of_ROI_mask
	# 	averaged_BB_proportional_std_rot[np.logical_and(averaged_BB_proportional_std_rot<np.nanmin(averaged_BB_proportional_std),averaged_BB_proportional_std_rot>np.nanmax(averaged_BB_proportional_std))]=0
	# temp_ref_counts_crop = temp_ref_counts_rot[foildw:foilup,foillx:foilrx]
	# temp_ref_counts_std_crop = temp_ref_counts_std_rot[foildw:foilup,foillx:foilrx]
	# averaged_BB_proportional_crop = averaged_BB_proportional_rot[foildw:foilup,foillx:foilrx]
	# averaged_BB_proportional_std_crop = averaged_BB_proportional_std_rot[foildw:foilup,foillx:foilrx]
	temp_ref_counts_crop = rotate_and_crop_2D(temp_ref_counts,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)
	temp_ref_counts_std_crop = rotate_and_crop_2D(temp_ref_counts_std,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)
	averaged_BB_proportional_crop = rotate_and_crop_2D(averaged_BB_proportional,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)
	averaged_BB_proportional_std_crop = rotate_and_crop_2D(averaged_BB_proportional_std,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx)

	shrink_factor_t = int(round(frames_to_average))
	shrink_factor_x = 3	# with large time averaging this should be enough
	if laser_framerate<60:
		shrink_factor_x = 5	# increase spatial averaging if I have very low framerate
	binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)

	FAST_counts_minus_background_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(FAST_counts_minus_background_crop,shrink_factor_t,shrink_factor_x)
	temperature_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(temperature_crop,shrink_factor_t,shrink_factor_x)
	temperature_std_add_crop_binned = proper_homo_binning_t_2D(temperature_crop,shrink_factor_t,shrink_factor_x,type='np.nanstd')[0]
	temperature_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*(proper_homo_binning_t_2D(temperature_std_crop**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5)
	temperature_std_crop_binned = (temperature_std_crop_binned**2 + temperature_std_add_crop_binned**2)**0.5
	counts_crop_binned,trash = proper_homo_binning_t_2D(counts_crop,shrink_factor_t,shrink_factor_x)
	counts_std_add_crop_binned = proper_homo_binning_t_2D(counts_crop,shrink_factor_t,shrink_factor_x,type='np.nanstd')[0]
	counts_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*(proper_homo_binning_t_2D(counts_std_crop**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5)
	counts_std_crop_binned = (counts_std_crop_binned**2 + counts_std_add_crop_binned**2)**0.5
	temperature_minus_background_crop_binned,nan_ROI_mask = proper_homo_binning_t_2D(temperature_minus_background_crop,shrink_factor_t,shrink_factor_x)
	temperature_minus_background_std_add_crop_binned = proper_homo_binning_t_2D(temperature_minus_background_crop,shrink_factor_t,shrink_factor_x,type='np.nanstd')[0]
	temperature_minus_background_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*(proper_homo_binning_t_2D(temperature_minus_background_std_crop**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5)
	temperature_minus_background_std_crop_binned = (temperature_minus_background_std_crop_binned**2 + temperature_minus_background_std_add_crop_binned**2)**0.5
	# temperature_ref_crop_binned = proper_homo_binning_2D(temperature_ref_crop,shrink_factor_x)
	temp_ref_counts_std_add_crop_binned = proper_homo_binning_2D(temp_ref_counts_crop,shrink_factor_x,type='np.nanstd')
	temp_ref_counts_std_crop_binned = 1/((shrink_factor_t**0.5)*shrink_factor_x**2)*(proper_homo_binning_2D(temp_ref_counts_std_crop**2,shrink_factor_x,type='np.nansum')**0.5)
	temp_ref_counts_std_crop_binned = (temp_ref_counts_std_crop_binned**2  + temp_ref_counts_std_add_crop_binned**2)**0.5
	averaged_BB_proportional_crop_binned = proper_homo_binning_2D(averaged_BB_proportional_crop,shrink_factor_x)
	averaged_BB_proportional_std_add_crop_binned = proper_homo_binning_2D(averaged_BB_proportional_crop,shrink_factor_x,type='np.nanstd')[0]
	averaged_BB_proportional_std_crop_binned = 1/(shrink_factor_x**2)*(proper_homo_binning_2D(averaged_BB_proportional_std_crop**2,shrink_factor_x,type='np.nansum')**0.5)
	averaged_BB_proportional_std_crop_binned = (averaged_BB_proportional_std_crop_binned**2 + averaged_BB_proportional_std_add_crop_binned**2)**0.5
	time_binned = proper_homo_binning_t(FAST_counts_minus_background_crop_time,shrink_factor_t)


	# reference foil properties
	# thickness = 1.4859095354482858e-06
	# emissivity = 0.9884061389741369
	# diffusivity = 1.045900223180454e-05
	# from 2021/09/17, Laser_data_analysis3_3.py
	thickness = 2.0531473351462095e-06
	emissivity = 0.9999999999999
	diffusivity = 1.0283685197530968e-05
	Ptthermalconductivity=71.6 #[W/(m·K)]
	zeroC=273.15 #K / C
	sigmaSB=5.6704e-08 #[W/(m2 K4)]

	foilemissivityscaled=emissivity*np.ones(np.array(averaged_BB_proportional_crop_binned.shape)-2)
	foilthicknessscaled=thickness*np.ones(np.array(averaged_BB_proportional_crop_binned.shape)-2)
	conductivityscaled=Ptthermalconductivity*np.ones(np.array(averaged_BB_proportional_crop_binned.shape)-2)
	reciprdiffusivityscaled=(1/diffusivity)*np.ones(np.array(averaged_BB_proportional_crop_binned.shape)-2)

	dx=foil_position_dict['foilhorizw']/foil_position_dict['foilhorizwpixel']*shrink_factor_x

	dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std = calc_temp_to_power_BB_1(photon_flux_over_temperature_interpolator,temperature_minus_background_crop_binned,ref_temperature,time_binned,dx,counts_std_crop_binned,averaged_BB_proportional_crop_binned,averaged_BB_proportional_std_crop_binned,temp_ref_counts_std_crop_binned,temperature_std_crop_binned,nan_ROI_mask,ref_temperature_std=ref_temperature_std)
	BBrad,diffusion,timevariation,powernoback,BBrad_std,diffusion_std,timevariation_std,powernoback_std = calc_temp_to_power_BB_2(dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std,nan_ROI_mask,foilemissivityscaled,foilthicknessscaled,reciprdiffusivityscaled,Ptthermalconductivity)

	horizontal_coord = np.arange(np.shape(powernoback[0])[1])
	vertical_coord = np.arange(np.shape(powernoback[0])[0])
	horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
	horizontal_coord = (horizontal_coord+1+0.5)*dx	# +1 because in the process of calculating the power I eliminate the first and last pixel in spatial coordinates, +0.5 do be the centre of the pixel
	vertical_coord = (vertical_coord+1+0.5)*dx
	horizontal_coord -= foil_position_dict['foilhorizw']*0.5+0.0198
	vertical_coord -= foil_position_dict['foilvertw']*0.5-0.0198
	distance_from_vertical = (horizontal_coord**2+vertical_coord**2)**0.5
	pinhole_to_foil_vertical = 0.008 + 0.003 + 0.002 + 0.045	# pinhole holder, washer, foil holder, standoff
	pinhole_to_pixel_distance = (pinhole_to_foil_vertical**2 + distance_from_vertical**2)**0.5

	etendue = np.ones_like(powernoback[0]) * (np.pi*(0.002**2)) / (pinhole_to_pixel_distance**2)	# I should include also the area of the pixel, but that is already in the w/m2 power
	etendue *= (pinhole_to_foil_vertical/pinhole_to_pixel_distance)**2	 # cos(a)*cos(b). for pixels not directly under the pinhole both pinhole and pixel are tilted respect to the vertical, with same angle.
	brightness = 4*np.pi*powernoback/etendue

	# temp = brightness[:,:,:int(np.shape(brightness)[2]*0.75)]
	# temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
	# ani,efit_reconstruction = movie_from_data(np.array([np.flip(np.transpose(brightness,(0,2,1)),axis=2)]), 1/np.median(np.diff(time_binned)),timesteps=time_binned[1:-1],integration=laser_int_time/1000,time_offset=time_binned[0],extvmin=0,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,EFIT_output_requested=True)
	# ani.save('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+ '_FAST_brightness.mp4', fps=5*(1/np.mean(np.diff(time_binned)))/383, writer='ffmpeg',codec='mpeg4')
	# plt.close('all')
	powernoback_output = cp.deepcopy(powernoback)

	print('completed FAST power calculation ' + laser_to_analyse)

	inverted_dict = dict([])
	from scipy.ndimage import geometric_transform
	import time as tm
	import pickle
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	shot_number = int(laser_to_analyse[-9:-4])
	# for grid_resolution in [4, 2]:
	# for grid_resolution in [2, 4]:
	for grid_resolution in [2]:
		inverted_dict[str(grid_resolution)] = dict([])
		# grid_resolution = 8  # in cm
		foil_resolution = '187'

		foil_res = '_foil_pixel_h_' + str(foil_resolution)

		grid_type = 'core_res_' + str(grid_resolution) + 'cm'
		path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res + '_power'
		try:
			sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity.npz')).todense())
		except:
			sensitivities = np.load(path_sensitivity + '/sensitivity.npy')

		filenames = all_file_names(path_sensitivity, '.csv')[0]
		with open(os.path.join(path_sensitivity, filenames)) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if row[0] == 'foil vertical pixels ':
					pixel_v = int(row[1])
				if row[0] == 'foil horizontal pixels ':
					pixel_h = int(row[1])
				if row[0] == 'pipeline type ':
					pipeline = row[1]
				if row[0] == 'type of volume grid ':
					grid_type = row[1]
			# print(row)

		directory = '/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction'
		grid_file = os.path.join(directory,'{}_rectilinear_grid.pickle'.format(grid_type))
		with open(grid_file, 'rb') as f:
			grid_data_all = pickle.load(f)
		grid_laplacian = grid_data_all['laplacian']
		grid_mask = grid_data_all['mask']
		grid_data = grid_data_all['voxels']
		grid_index_2D_to_1D_map = grid_data_all['index_2D_to_1D_map']
		grid_index_1D_to_2D_map = grid_data_all['index_1D_to_2D_map']

		sensitivities_reshaped = sensitivities.reshape((pixel_v,pixel_h,len(grid_laplacian)))
		sensitivities_reshaped = np.transpose(sensitivities_reshaped , (1,0,2))

		if grid_resolution==8:
			# temp=1e-3
			temp=1e-7
			temp2=0
		elif grid_resolution==2:
			temp=1e-4
			temp2=1e-4
		elif grid_resolution==4:
			temp=0
			temp2=0

		sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,sum_treshold=temp2,std_treshold = temp,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False)

		# this step is to adapt the matrix to the size of the foil I measure, that can be slightly different
		binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
		shape = list(FAST_counts_minus_background_crop.shape[1:])
		if shape!=list(sensitivities_reshaped_masked.shape[:-1]):
			shape.extend([len(grid_laplacian_masked)])
			def mapping(output_coords):
				return(output_coords[0]/shape[0]*pixel_h,output_coords[1]/shape[1]*pixel_v,output_coords[2])
			sensitivities_reshaped_masked2 = geometric_transform(sensitivities_reshaped_masked,mapping,output_shape=shape)
		else:
			sensitivities_reshaped_masked2 = cp.deepcopy(sensitivities_reshaped_masked)

		sensitivities_binned = proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
		sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
		sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

		# additional cropping of the foil to exlude regions without plasma LOS, the frame of the foil and gas puff
		# ROI = np.array([[0.2,0.85],[0.1,0.9]])
		# ROI = np.array([[0.05,0.95],[0.05,0.95]])
		# ROI = np.array([[0.2,0.95],[0.1,1]])
		ROI1 = np.array([[0.03,0.80],[0.03,0.85]])	# horizontal, vertical
		ROI2 = np.array([[0.03,0.7],[0.03,0.91]])
		ROI_beams = np.array([[0.,0.32],[0.4,1]])
		sensitivities_binned_crop,selected_ROI,ROI1,ROI2,ROI_beams = cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_binned,ROI1,ROI2,ROI_beams,laser_to_analyse,additional_output=True)

		additional_polygons_dict = dict([])
		additional_polygons_dict['time'] = np.array([0])	# in this case I plot the same polygon for the whole movie
		additional_polygons_dict['0'] = np.array([[[ROI1[0,0],ROI1[0,1],ROI1[0,1],ROI1[0,0],ROI1[0,0]],[ROI1[1,0],ROI1[1,0],ROI1[1,1],ROI1[1,1],ROI1[1,0]]]])
		additional_polygons_dict['1'] = np.array([[[ROI2[0,0],ROI2[0,1],ROI2[0,1],ROI2[0,0],ROI2[0,0]],[ROI2[1,0],ROI2[1,0],ROI2[1,1],ROI2[1,1],ROI2[1,0]]]])
		additional_polygons_dict['2'] = np.array([[[ROI_beams[0,0],ROI_beams[0,1],ROI_beams[0,1],ROI_beams[0,0],ROI_beams[0,0]],[ROI_beams[1,0],ROI_beams[1,0],ROI_beams[1,1],ROI_beams[1,1],ROI_beams[1,0]]]])
		additional_polygons_dict['number_of_polygons'] = 3
		additional_polygons_dict['marker'] = ['--k','--k','--k']

		temp = brightness[:,:,:int(np.shape(brightness)[2]*0.75)]
		temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
		ani,efit_reconstruction = movie_from_data(np.array([np.flip(np.transpose(brightness,(0,2,1)),axis=2)]), 1/np.median(np.diff(time_binned)),timesteps=time_binned[1:-1],integration=laser_int_time/1000,time_offset=time_binned[0],extvmin=0,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,EFIT_output_requested=True,additional_polygons_dict=additional_polygons_dict)
		ani.save('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+ '_FAST_brightness.mp4', fps=5*(1/np.mean(np.diff(time_binned)))/383, writer='ffmpeg',codec='mpeg4')
		plt.close('all')

		if grid_resolution==8:
			# temp=1e-3
			temp=1e-7
			temp2=0
		elif grid_resolution==2:
			temp=1e-4
			temp2=np.sum(sensitivities_binned_crop,axis=(0,1)).max()*1e-3
		elif grid_resolution==4:
			temp=0
			temp2=0
		sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked,sum_treshold=temp2,std_treshold = temp)

		selected_super_x_cells = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>0.85,np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.65)
		select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-3)
		selected_ROI_no_plasma = np.logical_and(selected_ROI,np.logical_not(select_foil_region_with_plasma))
		select_foil_region_with_plasma = select_foil_region_with_plasma.flatten()


		x1 = [1.55,0.25]	# r,z
		x2 = [1.1,-0.15]
		interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
		select = np.mean(grid_data_masked_crop,axis=1)[:,1]>interp(np.mean(grid_data_masked_crop,axis=1)[:,0])
		selected_central_border_cells = np.logical_and(select,np.logical_and(np.max(grid_Z_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,1]>-0.5))
		selected_central_border_cells = np.dot(grid_laplacian_masked_crop,selected_central_border_cells*np.random.random(selected_central_border_cells.shape))!=0

		selected_central_column_border_cells = np.logical_and(np.logical_and(np.max(grid_R_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)
		selected_central_column_border_cells = np.logical_and(np.logical_and(np.dot(grid_laplacian_masked_crop,selected_central_column_border_cells*np.random.random(selected_central_column_border_cells.shape))!=0,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)

		selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1,np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6))
		selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6))

		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells*np.random.random(selected_edge_cells.shape))!=0)
		if grid_resolution<8:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
		if grid_resolution<4:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
			# selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)

		sensitivities_binned_crop_shape = sensitivities_binned_crop.shape
		sensitivities_binned_crop = sensitivities_binned_crop.reshape((sensitivities_binned_crop.shape[0]*sensitivities_binned_crop.shape[1],sensitivities_binned_crop.shape[2]))

		if shrink_factor_x > 1:
			foil_resolution = str(shrink_factor_x) + 'x' + str(shrink_factor_x)
		else:
			foil_resolution = str(shape[0])

		foil_res = '_foil_pixel_h_' + str(foil_resolution)
		path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
		path_sensitivity_original = cp.deepcopy(path_sensitivity)

		binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
		print('starting '+binning_type)
		# powernoback_full = saved_file_dict_short[binning_type].all()['powernoback_full']
		# powernoback_std_full = saved_file_dict_short[binning_type].all()['powernoback_std_full']

		# from here I make the new method.
		# I consider the nominal properties as central value, with:
		# emissivity -10% (from Japanese properties i have std of ~5%, but my nominal value is ~1 and emissivity cannot be >1 so I double the interval down)
		# thickness +/-15% (from Japanese properties i have std of ~15%)
		# diffusivity -10% (this is missing from the Japanese data, so I guess std ~10%)

		emissivity_steps = 5
		thickness_steps = 9
		rec_diffusivity_steps = 9
		# sigma_emissivity = 0.1
		# sigma_thickness = 0.15
		# sigma_rec_diffusivity = 0.1
		sigma_emissivity = 0.04999 # np.std(foilemissivity)/np.mean(foilemissivity)	# I use the varition on the japanese data as reference
		sigma_thickness = 0.1325 # np.std(foilthickness)/np.mean(foilthickness)
		sigma_rec_diffusivity = 0	# this was not consodered as variable

		tend = get_tend(laser_to_analyse[-9:-4])+0.05	 # I add 50ms just for safety and to catch disruptions
		time_full_binned = time_binned[1:-1]
		BBrad_full_crop = BBrad[time_full_binned<tend]
		BBrad_full_crop[:,np.logical_not(selected_ROI)] = 0
		BBrad_std_full_crop = BBrad_std[time_full_binned<tend]
		BBrad_std_full_crop[:,np.logical_not(selected_ROI)] = 0
		diffusion_full_crop = diffusion[time_full_binned<tend]
		diffusion_full_crop[:,np.logical_not(selected_ROI)] = 0
		diffusion_std_full_crop = diffusion_std[time_full_binned<tend]
		diffusion_std_full_crop[:,np.logical_not(selected_ROI)] = 0
		timevariation_full_crop = timevariation[time_full_binned<tend]
		timevariation_full_crop[:,np.logical_not(selected_ROI)] = 0
		timevariation_std_full_crop = timevariation_std[time_full_binned<tend]
		timevariation_std_full_crop[:,np.logical_not(selected_ROI)] = 0
		time_full_binned_crop = time_full_binned[time_full_binned<tend]

		powernoback_full_orig = diffusion_full_crop + timevariation_full_crop + BBrad_full_crop
		sigma_powernoback_full = ( (diffusion_full_crop**2)*((diffusion_std_full_crop/diffusion_full_crop)**2+sigma_thickness**2) + (timevariation_full_crop**2)*((timevariation_std_full_crop/timevariation_full_crop)**2+sigma_thickness**2+sigma_rec_diffusivity**2) + (BBrad_full_crop**2)*((BBrad_std_full_crop/BBrad_full_crop)**2+sigma_emissivity**2) )**0.5

		grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
		grid_Z_derivate_masked_crop_scaled = grid_Z_derivate_masked_crop/((1e-2*grid_resolution)**1)
		grid_R_derivate_masked_crop_scaled = grid_R_derivate_masked_crop/((1e-2*grid_resolution)**1)
		reference_sigma_powernoback_all = np.nanmedian(sigma_powernoback_full[:,selected_ROI],axis=1)
		number_cells_ROI = np.sum(selected_ROI)
		number_cells_plasma = np.sum(select_foil_region_with_plasma)
		regolarisation_coeff = 1e-3	# ok for np.median(sigma_powernoback_full)=78.18681
		if grid_resolution==4:
			regolarisation_coeff = 3e-4
		elif grid_resolution==2:
			regolarisation_coeff = 1e-4
		sigma_emissivity = 1e6	# 2e3	# this is completely arbitrary
		sigma_emissivity_2 = sigma_emissivity**2
		r_int = np.mean(grid_data_masked_crop,axis=1)[:,0]
		r_int_2 = r_int**2

		sigma_powernoback_full[np.isnan(sigma_powernoback_full)] = 1e10
		# inverted_dict[str(grid_resolution)]['foil_power'] = powernoback_full_orig
		# inverted_dict[str(grid_resolution)]['foil_power_std'] = sigma_powernoback_full
		# inverted_dict[str(grid_resolution)]['time_full_binned_crop'] = time_full_binned_crop
		# full_saved_file_dict_FAST = dict([])
		# full_saved_file_dict_FAST['inverted_dict'] = inverted_dict
		# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
		selected_ROI_internal = selected_ROI.flatten()
		not_selected_super_x_cells = np.logical_not(selected_super_x_cells)
		inverted_data = []
		inverted_data_sigma = []
		inverted_data_likelihood = []
		inverted_data_info = []
		inverted_data_plasma_region_offset = []
		inverted_data_homogeneous_offset = []
		fitted_foil_power = []
		foil_power = []
		foil_power_std = []
		foil_power_residuals = []
		fit_error = []
		chi_square_all = []
		regolarisation_coeff_all = []
		time_per_iteration = []
		score_x_all = []
		score_y_all = []
		regolarisation_coeff_range_all = []
		Lcurve_curvature_all = []
		x_optimal_ext = []

		plt.figure(10,figsize=(20, 10))
		plt.title('L-curve evolution\nlight=early, dark=late')
		plt.figure(11,figsize=(20, 10))
		plt.title('L-curve curvature evolution\nlight=early, dark=late')
		first_guess = []
		x_optimal_all_guess = []
		# regolarisation_coeff_upper_limit = 10**-0.2
		regolarisation_coeff_upper_limit = 0.4
		regolarisation_coeff_lower_limit = 3e-4
		if laser_framerate<60:
			regolarisation_coeff_lower_limit = 8e-4
		for i_t in range(len(time_full_binned_crop)):
			time_start = tm.time()

			print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
			# plt.figure()
			# plt.imshow(powernoback_full_orig[i_t])
			# plt.colorbar()
			# plt.pause(0.01)
			#
			# plt.figure()
			# plt.imshow(sigma_powernoback_full[i_t])
			# plt.colorbar()
			# plt.pause(0.01)

			powernoback = powernoback_full_orig[i_t].flatten()
			sigma_powernoback = sigma_powernoback_full[i_t].flatten()
			reference_sigma_powernoback = reference_sigma_powernoback_all[i_t]
			# sigma_powernoback = np.ones_like(powernoback)*10
			sigma_powernoback_2 = sigma_powernoback**2
			homogeneous_scaling=1e-4

			guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
			if len(first_guess) != 0:
				guess = cp.deepcopy(first_guess)

			target_chi_square = sensitivities_binned_crop.shape[1]	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
			target_chi_square_sigma = 200	# this should be tight, because for such a high number of degrees of freedom things should average very well

			# regolarisation_coeff_edge = 10
			# regolarisation_coeff_edge_multiplier = 100
			regolarisation_coeff_central_border_Z_derivate_multiplier = 0
			regolarisation_coeff_central_column_border_R_derivate_multiplier = 0
			# regolarisation_coeff_edge_laplacian_multiplier = 2	# 1e1
			regolarisation_coeff_divertor_multiplier = 1
			regolarisation_coeff_non_negativity_multiplier = 40
			regolarisation_coeff_offsets_multiplier = 1e-10
			regolarisation_coeff_edge_laplacian = 0.01
			regolarisation_coeff_edge = 100

			def prob_and_gradient(emissivity_plus,*args):
				# time_start = tm.time()
				# emissivity_plus = emissivity_plus
				powernoback = args[0]
				sigma_powernoback = args[1]
				sigma_emissivity = args[2]
				regolarisation_coeff = args[3]
				sigma_powernoback_2 = args[4]
				sigma_emissivity_2 = args[5]
				homogeneous_offset = emissivity_plus[-1]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
				homogeneous_offset_plasma = emissivity_plus[-2]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
				regolarisation_coeff_divertor = regolarisation_coeff*regolarisation_coeff_divertor_multiplier
				regolarisation_coeff_central_column_border_R_derivate = regolarisation_coeff*regolarisation_coeff_central_column_border_R_derivate_multiplier
				# regolarisation_coeff_edge_laplacian = regolarisation_coeff*regolarisation_coeff_edge_laplacian_multiplier
				# regolarisation_coeff_edge = regolarisation_coeff*regolarisation_coeff_edge_multiplier
				# print(homogeneous_offset,homogeneous_offset_plasma)
				emissivity = emissivity_plus[:-2]
				# emissivity[emissivity==0] = 1e-10
				# foil_power_guess = np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma
				foil_power_error = powernoback - (np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma)
				emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
				emissivity_laplacian_not_selected_super_x_cells = emissivity_laplacian*not_selected_super_x_cells
				emissivity_laplacian_selected_super_x_cells = emissivity_laplacian*selected_super_x_cells
				emissivity_laplacian_selected_edge_cells_for_laplacian = emissivity_laplacian*selected_edge_cells_for_laplacian
				if regolarisation_coeff_central_column_border_R_derivate!=0:
					R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)
					R_derivate_selected_central_column_border_cells = R_derivate*selected_central_column_border_cells
				# print(tm.time()-time_start)
				# time_start = tm.time()

				likelihood_power_fit = np.sum((foil_power_error/sigma_powernoback)**2)
				likelihood_emissivity_pos = (regolarisation_coeff_non_negativity_multiplier**2)*np.sum((np.minimum(0.,emissivity*np.logical_not(selected_edge_cells))*r_int/sigma_emissivity*1)**2)	# I added a weight on the redious, becaus the power increase with radious and a negative voxel at high r is more important that one at low r
				likelihood_emissivity_laplacian = (regolarisation_coeff**2)* np.sum(((emissivity_laplacian_not_selected_super_x_cells /sigma_emissivity)**2))
				likelihood_emissivity_laplacian_superx = (regolarisation_coeff_divertor**2)* np.sum(((emissivity_laplacian_selected_super_x_cells /sigma_emissivity)**2))
				likelihood_emissivity_edge_laplacian = (regolarisation_coeff_edge_laplacian**2)* np.sum(((emissivity_laplacian_selected_edge_cells_for_laplacian /sigma_emissivity)**2))
				likelihood_emissivity_edge = (regolarisation_coeff_edge**2)*np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
				if regolarisation_coeff_central_column_border_R_derivate==0:
					likelihood_emissivity_central_column_border_R_derivate = 0
				else:
					likelihood_emissivity_central_column_border_R_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)* np.sum((R_derivate_selected_central_column_border_cells/sigma_emissivity)**2)
				likelihood = likelihood_power_fit + likelihood_emissivity_pos + likelihood_emissivity_laplacian + likelihood_emissivity_edge + likelihood_emissivity_laplacian_superx + likelihood_emissivity_central_column_border_R_derivate + likelihood_emissivity_edge_laplacian
				likelihood_homogeneous_offset = regolarisation_coeff_offsets_multiplier*number_cells_ROI*(homogeneous_offset/reference_sigma_powernoback)**2
				likelihood_homogeneous_offset_plasma = regolarisation_coeff_offsets_multiplier*number_cells_plasma*(homogeneous_offset_plasma/reference_sigma_powernoback)**2
				likelihood = likelihood + likelihood_homogeneous_offset + likelihood_homogeneous_offset_plasma
				# print(tm.time()-time_start)
				# time_start = tm.time()

				temp = foil_power_error/sigma_powernoback_2
				likelihood_power_fit_derivate = np.concatenate((-2*np.dot(temp,sensitivities_binned_crop),[-2*np.sum(temp*select_foil_region_with_plasma)*homogeneous_scaling,-2*np.sum(temp*selected_ROI_internal)*homogeneous_scaling]))
				likelihood_emissivity_pos_derivate = 2*(regolarisation_coeff_non_negativity_multiplier**2)*np.minimum(0.,emissivity*np.logical_not(selected_edge_cells))*r_int_2/sigma_emissivity_2*1

				# likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian_not_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				# likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				# likelihood_emissivity_edge_laplacian_derivate = 2*(regolarisation_coeff_edge_laplacian**2) * np.dot(emissivity_laplacian_selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				likelihood_emissivity_laplacian_derivate_all = 2* np.dot( (regolarisation_coeff**2)*emissivity_laplacian_not_selected_super_x_cells + (regolarisation_coeff_edge_laplacian**2)*emissivity_laplacian_selected_edge_cells_for_laplacian + (regolarisation_coeff_divertor**2)*emissivity_laplacian_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)

				likelihood_emissivity_edge_derivate = 2*(regolarisation_coeff_edge**2)*emissivity*selected_edge_cells/sigma_emissivity_2
				if regolarisation_coeff_central_column_border_R_derivate==0:
					likelihood_emissivity_central_column_border_R_derivate_derivate = 0
				else:
					likelihood_emissivity_central_column_border_R_derivate_derivate = 2*(regolarisation_coeff_central_column_border_R_derivate**2)*np.dot(R_derivate_selected_central_column_border_cells,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
				likelihood_derivate = likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate_all + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate
				likelihood_homogeneous_offset_derivate = 2*regolarisation_coeff_offsets_multiplier*number_cells_ROI*homogeneous_offset*homogeneous_scaling/(reference_sigma_powernoback**2)
				likelihood_homogeneous_offset_plasma_derivate = 2*regolarisation_coeff_offsets_multiplier*number_cells_plasma*homogeneous_offset_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
				likelihood_derivate = np.concatenate((likelihood_derivate,[likelihood_homogeneous_offset_plasma_derivate,likelihood_homogeneous_offset_derivate])) + likelihood_power_fit_derivate
				# print(tm.time()-time_start)
				# time_start = tm.time()
				return likelihood,likelihood_derivate

			def calc_hessian(emissivity_plus,*args):
				# time_start = tm.time()
				# emissivity_plus = emissivity_plus
				powernoback = args[0]
				sigma_powernoback = args[1]
				sigma_emissivity = args[2]
				regolarisation_coeff = args[3]
				sigma_powernoback_2 = args[4]
				sigma_emissivity_2 = args[5]
				homogeneous_offset = emissivity_plus[-1]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
				homogeneous_offset_plasma = emissivity_plus[-2]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
				regolarisation_coeff_divertor = regolarisation_coeff*regolarisation_coeff_divertor_multiplier
				regolarisation_coeff_central_column_border_R_derivate = regolarisation_coeff*regolarisation_coeff_central_column_border_R_derivate_multiplier
				# regolarisation_coeff_edge_laplacian = regolarisation_coeff*regolarisation_coeff_edge_laplacian_multiplier
				# regolarisation_coeff_edge = regolarisation_coeff*regolarisation_coeff_edge_multiplier
				# print(homogeneous_offset,homogeneous_offset_plasma)
				emissivity = emissivity_plus[:-2]
				# emissivity[emissivity==0] = 1e-10
				# foil_power_guess = np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma
				foil_power_error = powernoback - (np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma)
				emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
				emissivity_laplacian_not_selected_super_x_cells = emissivity_laplacian*not_selected_super_x_cells
				emissivity_laplacian_selected_super_x_cells = emissivity_laplacian*selected_super_x_cells
				emissivity_laplacian_selected_edge_cells_for_laplacian = emissivity_laplacian*selected_edge_cells_for_laplacian
				if regolarisation_coeff_central_column_border_R_derivate!=0:
					R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)
					R_derivate_selected_central_column_border_cells = R_derivate*selected_central_column_border_cells
				# print(tm.time()-time_start)
				# time_start = tm.time()

				# based on https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DAVIES1/rd_bhatt_cvonline/node9.html#SECTION00041000000000000000
				likelihood_power_fit_derivate = np.dot(sensitivities_binned_crop.T*sigma_powernoback_2,sensitivities_binned_crop)
				temp = np.zeros((np.shape(sensitivities_binned_crop)[1]+2,np.shape(sensitivities_binned_crop)[1]+2))
				temp[:-2,:-2] = likelihood_power_fit_derivate
				temp[-2:,:-2] = np.array([np.sum(-(sensitivities_binned_crop.T/sigma_powernoback_2*select_foil_region_with_plasma).T,axis=0)*homogeneous_scaling,np.sum(-(sensitivities_binned_crop.T/sigma_powernoback_2*selected_ROI_internal).T,axis=0)*homogeneous_scaling])
				temp[:-2,-2:] = np.array([np.sum(-(sensitivities_binned_crop.T/sigma_powernoback_2*select_foil_region_with_plasma).T,axis=0)*homogeneous_scaling,np.sum(-(sensitivities_binned_crop.T/sigma_powernoback_2*selected_ROI_internal).T,axis=0)*homogeneous_scaling]).T
				temp[-2,-2] = -np.sum(select_foil_region_with_plasma/sigma_powernoback_2*select_foil_region_with_plasma)*homogeneous_scaling
				temp[-1,-1] = -np.sum(selected_ROI_internal/sigma_powernoback_2*selected_ROI_internal)*homogeneous_scaling
				temp[-1,-2] = -np.sum(selected_ROI_internal/sigma_powernoback_2*select_foil_region_with_plasma)*homogeneous_scaling
				temp[-2,-1] = -np.sum(selected_ROI_internal/sigma_powernoback_2*select_foil_region_with_plasma)*homogeneous_scaling
				likelihood_power_fit_derivate = cp.deepcopy(temp)

				likelihood_emissivity_pos_derivate = (regolarisation_coeff_non_negativity_multiplier**2)*np.diag((emissivity<0)*np.logical_not(selected_edge_cells)*r_int_2/sigma_emissivity_2*1)

				# likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian_not_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				# likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				# likelihood_emissivity_edge_laplacian_derivate = 2*(regolarisation_coeff_edge_laplacian**2) * np.dot(emissivity_laplacian_selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
				likelihood_emissivity_laplacian_derivate_all = np.dot(grid_laplacian_masked_crop_scaled*( (regolarisation_coeff**2)*not_selected_super_x_cells + (regolarisation_coeff_edge_laplacian**2)*selected_edge_cells_for_laplacian + (regolarisation_coeff_divertor**2)*selected_super_x_cells) , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)

				likelihood_emissivity_edge_derivate = (regolarisation_coeff_edge**2)*np.diag(selected_edge_cells*r_int_2/sigma_emissivity_2*1)
				if regolarisation_coeff_central_column_border_R_derivate==0:
					likelihood_emissivity_central_column_border_R_derivate_derivate = 0
				else:
					likelihood_emissivity_central_column_border_R_derivate_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)*np.dot( grid_R_derivate_masked_crop_scaled*selected_central_column_border_cells ,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
				likelihood_derivate = likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate_all + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate
				likelihood_homogeneous_offset_derivate = regolarisation_coeff_offsets_multiplier*number_cells_ROI*homogeneous_scaling/(reference_sigma_powernoback**2)
				likelihood_homogeneous_offset_plasma_derivate = regolarisation_coeff_offsets_multiplier*number_cells_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
				likelihood_power_fit_derivate[:-2,:-2]+=likelihood_derivate
				likelihood_power_fit_derivate[-1,-1] += likelihood_homogeneous_offset_derivate
				likelihood_power_fit_derivate[-2,-2] += likelihood_homogeneous_offset_plasma_derivate
				return likelihood_power_fit_derivate

			if pass_number==0:
				regolarisation_coeff_range = np.array([1e-2])
			else:
				regolarisation_coeff_range = 10**np.linspace(1,-6,num=120)
				# regolarisation_coeff_range = 10**np.linspace(1,-5,num=102)
				# regolarisation_coeff_range = 10**np.linspace(0.5,-4.5,num=85)
				# if laser_framerate<60:
				# 	regolarisation_coeff_range = 10**np.linspace(0.5,-4,num=76)

			x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,voxels_centre = loop_fit_over_regularisation(prob_and_gradient,regolarisation_coeff_range,guess,grid_data_masked_crop,powernoback,sigma_powernoback,sigma_emissivity,x_optimal_all_guess=x_optimal_all_guess,factr=1e10)
			# if first_guess == []:
			x_optimal_all_guess = cp.deepcopy(x_optimal_all)
			first_guess = x_optimal_all[0]

			regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
			x_optimal_all = np.flip(x_optimal_all,axis=0)
			recompose_voxel_emissivity_all = np.flip(recompose_voxel_emissivity_all,axis=0)
			y_opt_all = np.flip(y_opt_all,axis=0)
			opt_info_all = np.flip(opt_info_all,axis=0)

			score_x = np.sum(((np.dot(sensitivities_binned_crop,np.array(x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback) ** 2) / (sigma_powernoback**2),axis=1)
			score_y = np.sum(((np.dot(grid_laplacian_masked_crop_scaled,np.array(x_optimal_all)[:,:-2].T).T) ** 2) / (sigma_emissivity**2),axis=1)
			score_x_all.append(score_x)
			score_y_all.append(score_y)
			regolarisation_coeff_range_all.append(regolarisation_coeff_range)

			if pass_number==0:
				recompose_voxel_emissivity,x_optimal,regolarisation_coeff,y_opt,opt_info = recompose_voxel_emissivity_all[0],x_optimal_all[0],regolarisation_coeff_range[0],y_opt_all[0],opt_info_all[0]
			else:
				score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,recompose_voxel_emissivity,x_optimal,points_removed,regolarisation_coeff,regolarisation_coeff_range,y_opt,opt_info,curvature_range_left_all,curvature_range_right_all,peaks,best_index = find_optimal_regularisation(score_x,score_y,regolarisation_coeff_range,x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,regolarisation_coeff_upper_limit=regolarisation_coeff_upper_limit,regolarisation_coeff_lower_limit=regolarisation_coeff_lower_limit)

				plt.figure(10)
				plt.plot(score_x,score_y,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(score_x,score_y,'+',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(score_x[best_index],score_y[best_index],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(score_x[peaks],score_y[peaks],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)),fillstyle='none',markersize=10)
				plt.plot(score_x[np.abs(regolarisation_coeff_range-regolarisation_coeff_upper_limit).argmin()],score_y[np.abs(regolarisation_coeff_range-regolarisation_coeff_upper_limit).argmin()],'s',color='r')
				plt.plot(score_x[np.abs(regolarisation_coeff_range-regolarisation_coeff_lower_limit).argmin()],score_y[np.abs(regolarisation_coeff_range-regolarisation_coeff_lower_limit).argmin()],'s',color='r')
				plt.xlabel('log ||Gm-d||2')
				plt.ylabel('log ||Laplacian(m)||2')
				plt.title('L-curve evolution\nlight=early, dark=late\ncurvature_range = '+str(curvature_range)+'\ntime_per_iteration [s] '+str(np.round(time_per_iteration).astype(int)))
				plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_evolution.eps')
				plt.figure(11)
				plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,'+',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(regolarisation_coeff_range[best_index],Lcurve_curvature[best_index-curvature_range],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(regolarisation_coeff_range[peaks],Lcurve_curvature[peaks-curvature_range],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)),fillstyle='none',markersize=10)
				plt.axvline(x=regolarisation_coeff_upper_limit,color='r')
				plt.axvline(x=regolarisation_coeff_lower_limit,color='r')
				plt.semilogx()
				plt.xlabel('regularisation coeff')
				plt.ylabel('L-curve turvature')
				plt.title('L-curve curvature evolution\nlight=early, dark=late\ncurvature_range = '+str(curvature_range)+'\ncurvature_range_left_all = '+str(curvature_range_left_all)+'\ncurvature_range_right_all = '+str(curvature_range_right_all))
				plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_curvature_evolution.eps')


			foil_power_guess = np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal[-1]*selected_ROI_internal*homogeneous_scaling
			foil_power_error = powernoback - foil_power_guess
			chi_square = np.sum((foil_power_error/sigma_powernoback)**2)
			print('chi_square '+str(chi_square))

			# regolarisation_coeff_edge = regolarisation_coeff*regolarisation_coeff_edge_multiplier
			regolarisation_coeff_central_border_Z_derivate = regolarisation_coeff*regolarisation_coeff_central_border_Z_derivate_multiplier
			regolarisation_coeff_central_column_border_R_derivate = regolarisation_coeff*regolarisation_coeff_central_column_border_R_derivate_multiplier
			# regolarisation_coeff_edge_laplacian = regolarisation_coeff*regolarisation_coeff_edge_laplacian_multiplier
			regolarisation_coeff_divertor = regolarisation_coeff*regolarisation_coeff_divertor_multiplier

			if False:	# only for visualisation
				plt.figure(figsize=(12,13))
				# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
				plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
				plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
				temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
				for i in range(len(all_time_sep_r[temp])):
					plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
				plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
				plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
				plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
				plt.colorbar().set_label('emissivity [W/m3]')
				plt.ylim(top=0.5)
				# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example19.eps')
				plt.pause(0.01)

			inverted_data.append(recompose_voxel_emissivity)
			inverted_data_plasma_region_offset.append(x_optimal[-2]*homogeneous_scaling)
			inverted_data_homogeneous_offset.append(x_optimal[-1]*homogeneous_scaling)
			x_optimal_ext.append(x_optimal)
			inverted_data_likelihood.append(y_opt)
			inverted_data_info.append(opt_info)
			fitted_foil_power.append((np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal[-1]*selected_ROI_internal*homogeneous_scaling).reshape(powernoback_full_orig[i_t].shape))
			foil_power.append(powernoback_full_orig[i_t])
			foil_power_std.append(sigma_powernoback_full[i_t])
			foil_power_residuals.append(powernoback_full_orig[i_t]-fitted_foil_power[-1])
			fit_error.append(np.sum(((powernoback_full_orig[i_t][selected_ROI]-fitted_foil_power[-1][[selected_ROI]]))**2)**0.5/np.sum(selected_ROI))
			chi_square_all.append(chi_square)
			regolarisation_coeff_all.append(regolarisation_coeff)
			if pass_number!=0:
				for value in points_removed:
					Lcurve_curvature = np.concatenate([Lcurve_curvature[:value],[np.nan],Lcurve_curvature[value:]])
				Lcurve_curvature_all.append(Lcurve_curvature)

			args = [powernoback,sigma_powernoback,sigma_emissivity,regolarisation_coeff,sigma_powernoback**2,sigma_emissivity**2]
			hessian=calc_hessian(x_optimal,*args)
			covariance = np.linalg.inv(hessian)
			trash,recompose_voxel_sigma = translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.diag(covariance)**0.5,np.mean(grid_data_masked_crop,axis=1))
			inverted_data_sigma.append(recompose_voxel_sigma)

			# inverted_dict[str(grid_resolution)]['score_x_all'] = score_x_all
			# inverted_dict[str(grid_resolution)]['score_y_all'] = score_y_all
			# inverted_dict[str(grid_resolution)]['regolarisation_coeff_range_all'] = regolarisation_coeff_range_all
			# inverted_dict[str(grid_resolution)]['Lcurve_curvature_all'] = Lcurve_curvature_all
			# inverted_dict[str(grid_resolution)]['foil_power_residuals'] = foil_power_residuals
			# inverted_dict[str(grid_resolution)]['fitted_foil_power'] = fitted_foil_power
			# full_saved_file_dict_FAST = dict([])
			# full_saved_file_dict_FAST['inverted_dict'] = inverted_dict
			# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
			time_per_iteration.append(tm.time()-time_start)

		inverted_data = np.array(inverted_data)
		inverted_data_sigma = np.array(inverted_data_sigma)
		inverted_data_likelihood = -np.array(inverted_data_likelihood)
		inverted_data_plasma_region_offset = np.array(inverted_data_plasma_region_offset)
		inverted_data_homogeneous_offset = np.array(inverted_data_homogeneous_offset)
		x_optimal_ext = np.array(x_optimal_ext)
		fit_error = np.array(fit_error)
		chi_square_all = np.array(chi_square_all)
		regolarisation_coeff_all = np.array(regolarisation_coeff_all)
		time_per_iteration = np.array(time_per_iteration)
		fitted_foil_power = np.array(fitted_foil_power)
		foil_power = np.array(foil_power)
		foil_power_std = np.array(foil_power_std)
		foil_power_residuals = np.array(foil_power_residuals)
		fitted_brightness = 4*np.pi*fitted_foil_power/etendue

		timeout = 20*60	# 20 minutes
		while efit_reconstruction==None and timeout>0:
			try:
				EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
				efit_reconstruction = mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])

			except:
				print('EFIT missing, waiting 1 min')
			tm.sleep(60)
			timeout -= 60

		if efit_reconstruction!=None:

			temp = brightness[:,:,:int(np.shape(brightness)[2]*0.75)]
			temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
			ani,efit_reconstruction = movie_from_data(np.array([np.flip(np.transpose(brightness,(0,2,1)),axis=2)]), 1/np.median(np.diff(time_binned)),timesteps=time_binned[1:-1],integration=laser_int_time/1000,time_offset=time_binned[0],extvmin=0,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,additional_polygons_dict=additional_polygons_dict)
			ani.save('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+ '_FAST_brightness.mp4', fps=5*(1/np.mean(np.diff(time_binned)))/383, writer='ffmpeg',codec='mpeg4')
			plt.close('all')

			all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
			all_time_strike_points_location = return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			outer_leg_tot_rad_power_all = []
			inner_leg_tot_rad_power_all = []
			core_tot_rad_power_all = []
			sxd_tot_rad_power_all = []
			x_point_tot_rad_power_all = []
			outer_leg_tot_rad_power_sigma_all = []
			inner_leg_tot_rad_power_sigma_all = []
			core_tot_rad_power_sigma_all = []
			sxd_tot_rad_power_sigma_all = []
			x_point_tot_rad_power_sigma_all = []
			for i_t in range(len(time_full_binned_crop)):
				temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
				xpoint_r = efit_reconstruction.lower_xpoint_r[temp]
				xpoint_z = efit_reconstruction.lower_xpoint_z[temp]
				z_,r_ = np.meshgrid(np.unique(voxels_centre[:,1]),np.unique(voxels_centre[:,0]))
				temp = cp.deepcopy(inverted_data[i_t])
				temp[z_>xpoint_z] = 0
				temp[r_<xpoint_r] = 0
				temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
				temp_sigma[z_>xpoint_z] = 0
				temp_sigma[r_<xpoint_r] = 0
				outer_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				outer_leg_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
				temp = cp.deepcopy(inverted_data[i_t])
				temp[z_>xpoint_z] = 0
				temp[r_>xpoint_r] = 0
				temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
				temp_sigma[z_>xpoint_z] = 0
				temp_sigma[r_>xpoint_r] = 0
				inner_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				inner_leg_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
				temp = cp.deepcopy(inverted_data[i_t])
				temp[z_<xpoint_z] = 0
				temp[z_>0] = 0
				temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
				temp_sigma[z_<xpoint_z] = 0
				temp_sigma[z_>0] = 0
				core_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				core_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
				temp = cp.deepcopy(inverted_data[i_t])
				temp[z_>-1.5] = 0
				temp[r_<0.8] = 0
				temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
				temp_sigma[z_>-1.5] = 0
				temp_sigma[r_<0.8] = 0
				sxd_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				sxd_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
				temp = cp.deepcopy(inverted_data[i_t])
				temp[((z_-xpoint_z)**2+(r_-xpoint_r)**2)**0.5>0.20] = 0
				temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
				temp_sigma[((z_-xpoint_z)**2+(r_-xpoint_r)**2)**0.5>0.20] = 0
				x_point_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
				x_point_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
				outer_leg_tot_rad_power_all.append(outer_leg_tot_rad_power)
				inner_leg_tot_rad_power_all.append(inner_leg_tot_rad_power)
				core_tot_rad_power_all.append(core_tot_rad_power)
				sxd_tot_rad_power_all.append(sxd_tot_rad_power)
				x_point_tot_rad_power_all.append(x_point_tot_rad_power)
				outer_leg_tot_rad_power_sigma_all.append(outer_leg_tot_rad_power_sigma)
				inner_leg_tot_rad_power_sigma_all.append(inner_leg_tot_rad_power_sigma)
				core_tot_rad_power_sigma_all.append(core_tot_rad_power_sigma)
				sxd_tot_rad_power_sigma_all.append(sxd_tot_rad_power_sigma)
				x_point_tot_rad_power_sigma_all.append(x_point_tot_rad_power_sigma)
			outer_leg_tot_rad_power_all = np.array(outer_leg_tot_rad_power_all)
			inner_leg_tot_rad_power_all = np.array(inner_leg_tot_rad_power_all)
			core_tot_rad_power_all = np.array(core_tot_rad_power_all)
			sxd_tot_rad_power_all = np.array(sxd_tot_rad_power_all)
			x_point_tot_rad_power_all = np.array(x_point_tot_rad_power_all)
			outer_leg_tot_rad_power_sigma_all = np.array(outer_leg_tot_rad_power_sigma_all)
			inner_leg_tot_rad_power_sigma_all = np.array(inner_leg_tot_rad_power_sigma_all)
			core_tot_rad_power_sigma_all = np.array(core_tot_rad_power_sigma_all)
			sxd_tot_rad_power_sigma_all = np.array(sxd_tot_rad_power_sigma_all)
			x_point_tot_rad_power_sigma_all = np.array(x_point_tot_rad_power_sigma_all)

			plt.figure(figsize=(20, 15))
			plt.errorbar(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3,yerr=outer_leg_tot_rad_power_sigma_all/1e3,label='outer_leg',capsize=5)
			plt.errorbar(time_full_binned_crop,sxd_tot_rad_power_all/1e3,yerr=sxd_tot_rad_power_sigma_all/1e3,label='sxd',capsize=5)
			plt.errorbar(time_full_binned_crop,inner_leg_tot_rad_power_all/1e3,yerr=inner_leg_tot_rad_power_sigma_all/1e3,label='inner_leg',capsize=5)
			plt.errorbar(time_full_binned_crop,core_tot_rad_power_all/1e3,yerr=core_tot_rad_power_sigma_all/1e3,label='core',capsize=5)
			plt.errorbar(time_full_binned_crop,x_point_tot_rad_power_all/1e3,yerr=x_point_tot_rad_power_sigma_all/1e3,label='x_point (dist<20cm)',capsize=5)
			plt.errorbar(time_full_binned_crop,(outer_leg_tot_rad_power_all+inner_leg_tot_rad_power_all+core_tot_rad_power_all)/1e3,yerr=((outer_leg_tot_rad_power_sigma_all**2+inner_leg_tot_rad_power_sigma_all**2+core_tot_rad_power_sigma_all**2)**0.5)/1e3,label='tot',capsize=5)
			plt.title('radiated power in the lower half of the machine')
			plt.legend(loc='best', fontsize='x-small')
			plt.xlabel('time [s]')
			plt.ylabel('power [kW]')
			plt.grid()
			plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_tot_rad_power.eps')
			plt.close()

			inverted_dict[str(grid_resolution)]['outer_leg_tot_rad_power_all'] = outer_leg_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['inner_leg_tot_rad_power_all'] = inner_leg_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['core_tot_rad_power_all'] = core_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['sxd_tot_rad_power_all'] = sxd_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_all'] = x_point_tot_rad_power_all
			inverted_dict[str(grid_resolution)]['outer_leg_tot_rad_power_sigma_all'] = outer_leg_tot_rad_power_sigma_all
			inverted_dict[str(grid_resolution)]['inner_leg_tot_rad_power_sigma_all'] = inner_leg_tot_rad_power_sigma_all
			inverted_dict[str(grid_resolution)]['core_tot_rad_power_sigma_all'] = core_tot_rad_power_sigma_all
			inverted_dict[str(grid_resolution)]['sxd_tot_rad_power_sigma_all'] = sxd_tot_rad_power_sigma_all
			inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_sigma_all'] = x_point_tot_rad_power_sigma_all


		path_power_output = os.path.split(laser_to_analyse)[0] + '/' + str(shot_number)
		if not os.path.exists(path_power_output):
			os.makedirs(path_power_output)
		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,time_per_iteration)
		# plt.semilogy()
		plt.title('time spent per iteration')
		plt.xlabel('time [s]')
		plt.ylabel('time [s]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_time_trace.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,inverted_data_likelihood)
		# plt.semilogy()
		plt.title('Fit log likelihood')
		plt.xlabel('time [s]')
		plt.ylabel('log likelihoog [au]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_likelihood.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,chi_square_all)
		plt.plot(time_full_binned_crop,np.ones_like(time_full_binned_crop)*target_chi_square,'--k')
		# plt.semilogy()
		if False:
			plt.title('chi square obtained vs requested\nfixed regularisation of '+str(regolarisation_coeff))
		else:
			plt.title('chi square obtained vs requested\nflexible regolarisation coefficient')
		plt.xlabel('time [s]')
		plt.ylabel('chi square [au]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_chi_square.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,regolarisation_coeff_all)
		# plt.semilogy()
		plt.title('regolarisation coefficient obtained')
		plt.semilogy()
		plt.xlabel('time [s]')
		plt.ylabel('regolarisation coefficient [au]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_regolarisation_coeff.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,fit_error)
		# plt.semilogy()
		plt.title('Fit error ( sum((image-fit)^2)^0.5/num pixels )')
		plt.xlabel('time [s]')
		plt.ylabel('average fit error [W/m2]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_fit_error.eps')
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(time_full_binned_crop,inverted_data_plasma_region_offset,label='plasma region')
		plt.plot(time_full_binned_crop,inverted_data_homogeneous_offset,label='whole foil')
		plt.title('Offsets to match foil power')
		plt.legend(loc='best', fontsize='x-small')
		plt.xlabel('time [s]')
		plt.ylabel('power density [W/m2]')
		plt.grid()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_offsets.eps')
		plt.close()

		if efit_reconstruction!=None:

			additional_points_dict,radiator_xpoint_distance_all,radiator_above_xpoint_all,radiator_magnetic_radious_all,radiator_baricentre_magnetic_radious_all,radiator_baricentre_above_xpoint_all = find_radiator_location(inverted_data,np.unique(voxels_centre[:,0]),np.unique(voxels_centre[:,1]),time_full_binned_crop,efit_reconstruction)

			inverted_dict[str(grid_resolution)]['radiator_location_all'] = additional_points_dict['0']
			inverted_dict[str(grid_resolution)]['radiator_baricentre_location_all'] = additional_points_dict['1']
			inverted_dict[str(grid_resolution)]['radiator_xpoint_distance_all'] = radiator_xpoint_distance_all
			inverted_dict[str(grid_resolution)]['radiator_above_xpoint_all'] = radiator_above_xpoint_all
			inverted_dict[str(grid_resolution)]['radiator_magnetic_radious_all'] = radiator_magnetic_radious_all

			fig, ax = plt.subplots( 2,1,figsize=(8, 12), squeeze=False,sharex=True)
			ax[0,0].plot(time_full_binned_crop,radiator_magnetic_radious_all)
			ax[0,0].plot(time_full_binned_crop,radiator_baricentre_magnetic_radious_all,'--')
			ax[0,0].set_ylim(top=min(np.nanmax(radiator_magnetic_radious_all),1.1),bottom=max(np.nanmin(radiator_magnetic_radious_all),0.9))
			ax[1,0].plot(time_full_binned_crop,radiator_above_xpoint_all)
			ax[1,0].plot(time_full_binned_crop,radiator_baricentre_above_xpoint_all,'--')
			fig.suptitle('Location of the x-point radiator\n"--"=baricentre r=20cm around x-point')
			ax[0,0].set_ylabel('normalised psi [au]')
			ax[0,0].grid()
			ax[1,0].set_xlabel('time [s]')
			ax[1,0].set_ylabel('position above x-point [m]')
			ax[1,0].grid()
			plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_x_point_location.eps')
			plt.close()

			extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			additional_each_frame_label_description = ['reg coeff=']*len(inverted_data)
			additional_each_frame_label_number = np.array(regolarisation_coeff_all)
			ani,trash = movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate_multiplier %.3g\nregolarisation_coeff_central_column_border_R_derivate_multiplier %.3g\nregolarisation_coeff_edge_laplacian %.3g\nregolarisation_coeff_divertor_multiplier %.3g\nregolarisation_coeff_non_negativity_multiplier %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate_multiplier,regolarisation_coeff_central_column_border_R_derivate_multiplier,regolarisation_coeff_edge_laplacian,regolarisation_coeff_divertor_multiplier,regolarisation_coeff_non_negativity_multiplier,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,additional_points_dict=additional_points_dict,additional_each_frame_label_description=additional_each_frame_label_description,additional_each_frame_label_number=additional_each_frame_label_number)#,extvmin=0,extvmax=4e4)

		else:

			# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(recompose_voxel_emissivity,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))),integration=laser_int_time/1000,barlabel='Emissivity [W/m3]')#,extvmin=0,extvmax=4e4)
			extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			additional_each_frame_label_description = ['reg coeff=']*len(inverted_data)
			additional_each_frame_label_number = np.array(regolarisation_coeff_all)
			ani,trash = movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate_multiplier %.3g\nregolarisation_coeff_central_column_border_R_derivate_multiplier %.3g\nregolarisation_coeff_edge_laplacian %.3g\nregolarisation_coeff_divertor_multiplier %.3g\nregolarisation_coeff_non_negativity_multiplier %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate_multiplier,regolarisation_coeff_central_column_border_R_derivate_multiplier,regolarisation_coeff_edge_laplacian,regolarisation_coeff_divertor_multiplier,regolarisation_coeff_non_negativity_multiplier,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,additional_each_frame_label_description=additional_each_frame_label_description,additional_each_frame_label_number=additional_each_frame_label_number)#,extvmin=0,extvmax=4e4)
		ani.save('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_FAST_reconstruct_emissivity_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
		plt.close()


		inverted_dict[str(grid_resolution)]['binning_type'] = binning_type
		inverted_dict[str(grid_resolution)]['inverted_data'] = inverted_data
		inverted_dict[str(grid_resolution)]['inverted_data_sigma'] = inverted_data_sigma
		inverted_dict[str(grid_resolution)]['inverted_data_likelihood'] = inverted_data_likelihood
		inverted_dict[str(grid_resolution)]['inverted_data_info'] = inverted_data_info
		inverted_dict[str(grid_resolution)]['select_foil_region_with_plasma'] = select_foil_region_with_plasma
		inverted_dict[str(grid_resolution)]['inverted_data_plasma_region_offset'] = inverted_data_plasma_region_offset
		inverted_dict[str(grid_resolution)]['inverted_data_homogeneous_offset'] = inverted_data_homogeneous_offset
		inverted_dict[str(grid_resolution)]['x_optimal_ext'] = x_optimal_ext
		inverted_dict[str(grid_resolution)]['time_full_binned_crop'] = time_full_binned_crop
		inverted_dict[str(grid_resolution)]['fitted_foil_power'] = fitted_foil_power
		inverted_dict[str(grid_resolution)]['fitted_brightness'] = fitted_brightness
		inverted_dict[str(grid_resolution)]['foil_power'] = foil_power
		inverted_dict[str(grid_resolution)]['foil_power_std'] = foil_power_std
		inverted_dict[str(grid_resolution)]['foil_power_residuals'] = foil_power_residuals
		inverted_dict[str(grid_resolution)]['fit_error'] = fit_error
		inverted_dict[str(grid_resolution)]['chi_square_all'] = chi_square_all
		inverted_dict[str(grid_resolution)]['geometry'] = dict([])
		inverted_dict[str(grid_resolution)]['geometry']['R'] = np.unique(voxels_centre[:,0])
		inverted_dict[str(grid_resolution)]['geometry']['Z'] = np.unique(voxels_centre[:,1])
		inverted_dict[str(grid_resolution)]['score_x_all'] = score_x_all
		inverted_dict[str(grid_resolution)]['score_y_all'] = score_y_all
		inverted_dict[str(grid_resolution)]['regolarisation_coeff_range_all'] = regolarisation_coeff_range_all
		inverted_dict[str(grid_resolution)]['Lcurve_curvature_all'] = Lcurve_curvature_all
		# inverted_dict[str(grid_resolution)]['sensitivities_binned_crop'] = sensitivities_binned_crop
		inverted_dict[str(grid_resolution)]['regolarisation_coeff_all'] = regolarisation_coeff_all

		if efit_reconstruction!=None:

			inversion_R = np.unique(voxels_centre[:,0])
			inversion_Z = np.unique(voxels_centre[:,1])
			local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution = track_outer_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)

			try:
				fig, ax = plt.subplots( 1,2,figsize=(15, 10), squeeze=False,sharey=True)
				temp = np.array(local_power_all)
				temp[np.isnan(temp)] = 0
				im1 = ax[0,0].imshow(temp,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(temp[:-4]),vmax=np.max(temp[:-4]))
				ax[0,0].plot(leg_length_all,time_full_binned_crop,'--k')
				ax[0,0].set_aspect(3)
				temp = np.array(local_mean_emis_all)
				temp[np.isnan(temp)] = 0
				im2 = ax[0,1].imshow(temp,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(temp[:-4]),vmax=np.max(temp[:-4]))
				ax[0,1].plot(leg_length_all,time_full_binned_crop,'--k')
				ax[0,1].set_aspect(3)
				fig.suptitle('tracking radiation on the outer leg\naveraged/summed %.3gcm above and below the separatrix'%(leg_resolution*100))
				ax[0,0].set_xlabel('distance from the strike point [m]')
				ax[0,0].grid()
				ax[0,0].set_ylabel('time [s]')
				plt.colorbar(im1,ax=ax[0,0]).set_label('Integrated power [W]')
				# ax[0,0].colorbar().set_label('Integrated power [W]')
				ax[0,1].set_xlabel('distance from the strike point [m]')
				ax[0,1].grid()
				# ax[0,1].colorbar().set_label('Emissivity [W/m3]')
				plt.colorbar(im2,ax=ax[0,1]).set_label('Emissivity [W/m3]')
				plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking.eps')
				plt.close()


				number_of_curves_to_plot = 12
				alpha = 0.9
				# colors_smooth = np.array([np.linspace(0,1,num=number_of_curves_to_plot),np.linspace(1,0,num=number_of_curves_to_plot),[0]*number_of_curves_to_plot]).T
				colors_smooth = np.linspace(0,0.9,num=number_of_curves_to_plot).astype(str)
				select = np.unique(np.round(np.linspace(0,len(local_mean_emis_all)-1,num=number_of_curves_to_plot))).astype(int)
				fig, ax = plt.subplots( 2,1,figsize=(10, 20), squeeze=False,sharex=True)
				fig.suptitle('tracking radiation on the outer leg')
				# plt.figure()
				for i_i_t,i_t in enumerate(select):
					# ax[0,0].plot(np.cumsum(leg_length_interval_all[i_t]),local_mean_emis_all[i_t],color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3)
					# ax[1,0].plot(np.cumsum(leg_length_interval_all[i_t]),local_power_all[i_t],color=colors_smooth[i_i_t],linewidth=3)
					to_plot_x = np.array(leg_length_interval_all[i_t])
					to_plot_y1 = np.array(local_mean_emis_all[i_t])
					to_plot_y1 = to_plot_y1[to_plot_x>0]
					to_plot_y2 = np.array(local_power_all[i_t])
					to_plot_y2 = to_plot_y2[to_plot_x>0]
					to_plot_x = to_plot_x[to_plot_x>0]
					to_plot_x = np.flip(np.sum(to_plot_x)-(np.cumsum(to_plot_x)-np.array(to_plot_x)/2),axis=0)
					to_plot_y1 = np.flip(to_plot_y1,axis=0)
					to_plot_y2 = np.flip(to_plot_y2,axis=0)
					if i_i_t%3==0:
						ax[0,0].plot(to_plot_x,to_plot_y1,'-',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'-',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
					elif i_i_t%3==1:
						ax[0,0].plot(to_plot_x,to_plot_y1,'-.',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'-.',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
					elif i_i_t%3==2:
						ax[0,0].plot(to_plot_x,to_plot_y1,'--',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'--',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
				ax[0,0].legend(loc='best', fontsize='x-small')
				ax[0,0].set_ylabel('average emissivity [W/m3]')
				ax[1,0].set_ylabel('local radiated power [W]')
				# ax[1,0].set_xlabel('distance from target [m]')
				ax[1,0].set_xlabel('distance from x-point [m]')
				ax[0,0].grid()
				ax[1,0].grid()
				plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking_2.png')
				plt.close()

			except Exception as e:
				logging.exception('with error: ' + str(e))
				print('failed to print\n'+'/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking.eps')

			inverted_dict[str(grid_resolution)]['local_outer_leg_power'] = local_power_all
			inverted_dict[str(grid_resolution)]['local_outer_leg_mean_emissivity'] = local_mean_emis_all
			inverted_dict[str(grid_resolution)]['leg_outer_length_all'] = leg_length_all
			inverted_dict[str(grid_resolution)]['leg_outer_length_interval_all'] = leg_length_interval_all

			local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution = track_inner_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)

			try:
				fig, ax = plt.subplots( 1,2,figsize=(15, 10), squeeze=False,sharey=True)
				temp = np.array(local_power_all)
				temp[np.isnan(temp)] = 0
				im1 = ax[0,0].imshow(temp,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(temp[:-4]),vmax=np.max(temp[:-4]))
				ax[0,0].plot(leg_length_all,time_full_binned_crop,'--k')
				ax[0,0].set_aspect(3)
				temp = np.array(local_mean_emis_all)
				temp[np.isnan(temp)] = 0
				im2 = ax[0,1].imshow(temp,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(temp[:-4]),vmax=np.max(temp[:-4]))
				ax[0,1].plot(leg_length_all,time_full_binned_crop,'--k')
				ax[0,1].set_aspect(3)
				fig.suptitle('tracking radiation on the inner leg\naveraged/summed %.3gcm above and below the separatrix'%(leg_resolution*100))
				ax[0,0].set_xlabel('distance from the strike point [m]')
				ax[0,0].grid()
				ax[0,0].set_ylabel('time [s]')
				plt.colorbar(im1,ax=ax[0,0]).set_label('Integrated power [W]')
				# ax[0,0].colorbar().set_label('Integrated power [W]')
				ax[0,1].set_xlabel('distance from the strike point [m]')
				ax[0,1].grid()
				# ax[0,1].colorbar().set_label('Emissivity [W/m3]')
				plt.colorbar(im2,ax=ax[0,1]).set_label('Emissivity [W/m3]')
				plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_inner_leg_radiation_tracking.eps')
				plt.close()


				number_of_curves_to_plot = 12
				alpha = 0.9
				# colors_smooth = np.array([np.linspace(0,1,num=number_of_curves_to_plot),np.linspace(1,0,num=number_of_curves_to_plot),[0]*number_of_curves_to_plot]).T
				colors_smooth = np.linspace(0,0.9,num=number_of_curves_to_plot).astype(str)
				select = np.unique(np.round(np.linspace(0,len(local_mean_emis_all)-1,num=number_of_curves_to_plot))).astype(int)
				fig, ax = plt.subplots( 2,1,figsize=(10, 20), squeeze=False,sharex=True)
				fig.suptitle('tracking radiation on the inner leg')
				# plt.figure()
				for i_i_t,i_t in enumerate(select):
					# ax[0,0].plot(np.cumsum(leg_length_interval_all[i_t]),local_mean_emis_all[i_t],color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3)
					# ax[1,0].plot(np.cumsum(leg_length_interval_all[i_t]),local_power_all[i_t],color=colors_smooth[i_i_t],linewidth=3)
					to_plot_x = np.array(leg_length_interval_all[i_t])
					to_plot_y1 = np.array(local_mean_emis_all[i_t])
					to_plot_y1 = to_plot_y1[to_plot_x>0]
					to_plot_y2 = np.array(local_power_all[i_t])
					to_plot_y2 = to_plot_y2[to_plot_x>0]
					to_plot_x = to_plot_x[to_plot_x>0]
					to_plot_x = np.flip(np.sum(to_plot_x)-(np.cumsum(to_plot_x)-np.array(to_plot_x)/2),axis=0)
					to_plot_y1 = np.flip(to_plot_y1,axis=0)
					to_plot_y2 = np.flip(to_plot_y2,axis=0)
					if i_i_t%3==0:
						ax[0,0].plot(to_plot_x,to_plot_y1,'-',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'-',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
					elif i_i_t%3==1:
						ax[0,0].plot(to_plot_x,to_plot_y1,'-.',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'-.',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
					elif i_i_t%3==2:
						ax[0,0].plot(to_plot_x,to_plot_y1,'--',color=colors_smooth[i_i_t],label = '%.3gs' %(time_full_binned_crop[i_t]),linewidth=3,alpha=alpha)
						ax[1,0].plot(to_plot_x,to_plot_y2,'--',color=colors_smooth[i_i_t],linewidth=3,alpha=alpha)
				ax[0,0].legend(loc='best', fontsize='x-small')
				ax[0,0].set_ylabel('average emissivity [W/m3]')
				ax[1,0].set_ylabel('local radiated power [W]')
				# ax[1,0].set_xlabel('distance from target [m]')
				ax[1,0].set_xlabel('distance from x-point [m]')
				ax[0,0].grid()
				ax[1,0].grid()
				plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_inner_leg_radiation_tracking_2.png')
				plt.close()

			except Exception as e:
				logging.exception('with error: ' + str(e))
				print('failed to print\n'+'/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_inner_leg_radiation_tracking.eps')

			inverted_dict[str(grid_resolution)]['local_inner_leg_power'] = local_power_all
			inverted_dict[str(grid_resolution)]['local_inner_leg_mean_emissivity'] = local_mean_emis_all
			inverted_dict[str(grid_resolution)]['leg_inner_length_all'] = leg_length_all
			inverted_dict[str(grid_resolution)]['leg_inner_length_interval_all'] = leg_length_interval_all

	return foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop_binned,time_binned,powernoback_output,brightness,binning_type,inverted_dict


def loop_fit_over_regularisation(prob_and_gradient,regolarisation_coeff_range,guess,grid_data_masked_crop,powernoback,sigma_powernoback,sigma_emissivity,factr=1e9,pgtol=5e-7,x_optimal_all_guess=[],iprint=0):
	import scipy
	import time as tm

	voxels_centre = np.mean(grid_data_masked_crop,axis=1)
	dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
	dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
	dist_mean = (dz**2 + dr**2)/2

	x_optimal_all = []
	recompose_voxel_emissivity_all = []
	y_opt_all = []
	opt_info_all = []
	start = tm.time()
	for i_regolarisation_coeff,regolarisation_coeff in enumerate(regolarisation_coeff_range):
		print('regolarisation_coeff = '+str(regolarisation_coeff))
		args = [powernoback,sigma_powernoback,sigma_emissivity,regolarisation_coeff,sigma_powernoback**2,sigma_emissivity**2]
		if len(regolarisation_coeff_range)==len(x_optimal_all_guess):
			temp = np.mean([x_optimal_all_guess[i_regolarisation_coeff],guess],axis=0)
			x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=temp, args = (args), iprint=iprint, factr=factr, pgtol=pgtol)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
		else:
			x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (args), iprint=iprint, factr=factr, pgtol=pgtol)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
		x_optimal_all.append(x_optimal)
		y_opt_all.append(y_opt)
		opt_info_all.append(opt_info)
		guess = x_optimal

		recompose_voxel_emissivity = np.zeros((len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))*np.nan
		for i_r,r in enumerate(np.unique(voxels_centre[:,0])):
			for i_z,z in enumerate(np.unique(voxels_centre[:,1])):
				dist = (voxels_centre[:,0]-r)**2 + (voxels_centre[:,1]-z)**2
				if dist.min()<dist_mean/2:
					index = np.abs(dist).argmin()
					recompose_voxel_emissivity[i_r,i_z] = x_optimal[index]
		recompose_voxel_emissivity *= 4*np.pi	# this exist because the sensitivity matrix is built with 1W/str/m^3/ x nm emitters while I use 1W as reference, so I need to multiply the results by 4pi
		recompose_voxel_emissivity_all.append(recompose_voxel_emissivity)
		print('done in %.3g' %(tm.time()-start))
		start = tm.time()
	return x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,voxels_centre

def find_optimal_regularisation(score_x,score_y,regolarisation_coeff_range,x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,curvature_fit_regularisation_interval = 0.13,fraction_of_L_curve_for_fit = 0.12,regolarisation_coeff_upper_limit = 10**-0.2,regolarisation_coeff_lower_limit = 1e-4,forward_model_residuals=False):
	import collections
	from scipy.signal import find_peaks, peak_prominences as get_proms

	# I need to remove the cases where points are overlapped
	points_removed = []
	counter_score_x = collections.Counter(score_x)
	counter_score_y = collections.Counter(score_y)
	test = np.logical_and( [value in np.array(list(counter_score_x.items()))[:,0][np.array(list(counter_score_x.items()))[:,1]>1] for value in score_x] , [value in np.array(list(counter_score_y.items()))[:,0][np.array(list(counter_score_y.items()))[:,1]>1] for value in score_y] )
	while np.sum(test)>0:	# removing points one on top of each other
		i__ = test.argmax()
		# print(i__)
		regolarisation_coeff_range = np.concatenate([regolarisation_coeff_range[:i__],regolarisation_coeff_range[i__+1:]])
		x_optimal_all = np.concatenate([x_optimal_all[:i__],x_optimal_all[i__+1:]])
		recompose_voxel_emissivity_all = np.concatenate([recompose_voxel_emissivity_all[:i__],recompose_voxel_emissivity_all[i__+1:]])
		y_opt_all = np.concatenate([y_opt_all[:i__],y_opt_all[i__+1:]])
		opt_info_all = np.concatenate([opt_info_all[:i__],opt_info_all[i__+1:]])
		score_x = np.concatenate([score_x[:i__],score_x[i__+1:]])
		score_y = np.concatenate([score_y[:i__],score_y[i__+1:]])
		counter_score_x = collections.Counter(score_x)
		counter_score_y = collections.Counter(score_y)
		test = np.logical_and( [value in np.array(list(counter_score_x.items()))[:,0][np.array(list(counter_score_x.items()))[:,1]>1] for value in score_x] , [value in np.array(list(counter_score_y.items()))[:,0][np.array(list(counter_score_y.items()))[:,1]>1] for value in score_y] )
		points_removed.append(i__)
	if forward_model_residuals == False:
		test = np.diff(score_x)
		while np.sum(test<0)>0:	# removing points for which quality of the fit derease for decreasing regularisation. it shouldn't happen, but it actually can for forwards modelled data
			i__ = test.argmin()+1
			if i__<5:
				i__ = test.argmin()
			# print(i__)
			regolarisation_coeff_range = np.concatenate([regolarisation_coeff_range[:i__],regolarisation_coeff_range[i__+1:]])
			x_optimal_all = np.concatenate([x_optimal_all[:i__],x_optimal_all[i__+1:]])
			recompose_voxel_emissivity_all = np.concatenate([recompose_voxel_emissivity_all[:i__],recompose_voxel_emissivity_all[i__+1:]])
			y_opt_all = np.concatenate([y_opt_all[:i__],y_opt_all[i__+1:]])
			opt_info_all = np.concatenate([opt_info_all[:i__],opt_info_all[i__+1:]])
			score_x = np.concatenate([score_x[:i__],score_x[i__+1:]])
			score_y = np.concatenate([score_y[:i__],score_y[i__+1:]])
			test = np.diff(score_x)
			points_removed.append(i__)
	test = np.diff(score_y)
	while np.sum(test>0)>0:	# removing points for which the laplacian reduces for increasing regularisation (it shouldn't happen)
		i__ = test.argmax()+1
		# print(i__)
		regolarisation_coeff_range = np.concatenate([regolarisation_coeff_range[:i__],regolarisation_coeff_range[i__+1:]])
		x_optimal_all = np.concatenate([x_optimal_all[:i__],x_optimal_all[i__+1:]])
		recompose_voxel_emissivity_all = np.concatenate([recompose_voxel_emissivity_all[:i__],recompose_voxel_emissivity_all[i__+1:]])
		y_opt_all = np.concatenate([y_opt_all[:i__],y_opt_all[i__+1:]])
		opt_info_all = np.concatenate([opt_info_all[:i__],opt_info_all[i__+1:]])
		score_x = np.concatenate([score_x[:i__],score_x[i__+1:]])
		score_y = np.concatenate([score_y[:i__],score_y[i__+1:]])
		test = np.diff(score_y)
		points_removed.append(i__)
	length_of_line = (np.diff(np.log10(score_y))**2 + np.diff(np.log10(score_x))**2)**0.5
	test = length_of_line<np.median(length_of_line)/100
	while np.sum(test)>0:	# removing points that are too close each other
		i__ = test.argmax()
		# print(i__)
		regolarisation_coeff_range = np.concatenate([regolarisation_coeff_range[:i__],regolarisation_coeff_range[i__+1:]])
		x_optimal_all = np.concatenate([x_optimal_all[:i__],x_optimal_all[i__+1:]])
		recompose_voxel_emissivity_all = np.concatenate([recompose_voxel_emissivity_all[:i__],recompose_voxel_emissivity_all[i__+1:]])
		y_opt_all = np.concatenate([y_opt_all[:i__],y_opt_all[i__+1:]])
		opt_info_all = np.concatenate([opt_info_all[:i__],opt_info_all[i__+1:]])
		score_x = np.concatenate([score_x[:i__],score_x[i__+1:]])
		score_y = np.concatenate([score_y[:i__],score_y[i__+1:]])
		length_of_line = (np.diff(np.log10(score_y))**2 + np.diff(np.log10(score_x))**2)**0.5
		test = length_of_line<np.median(length_of_line)/100
		points_removed.append(i__)
	points_removed = np.flip(points_removed,axis=0)
	# plt.figure()
	# plt.plot(score_x,score_y)
	# plt.plot(score_x,score_y,'+')
	# plt.xlabel('||Gm-d||2')
	# plt.ylabel('||Laplacian(m)||2')
	# plt.semilogx()
	# plt.semilogy()
	# plt.grid()
	# plt.pause(0.01)

	score_y = np.log(score_y)
	score_x = np.log(score_x)

	score_y_record_rel = (score_y-score_y.min())/(score_y.max()-score_y.min())
	score_x_record_rel = (score_x-score_x.min())/(score_x.max()-score_x.min())

	length_of_line = np.sum((np.diff(score_y_record_rel)**2 + np.diff(score_x_record_rel)**2)**0.5)
	fraction_of_line_as_range = length_of_line*fraction_of_L_curve_for_fit/2*(3+6)/(np.log10(regolarisation_coeff_range).max()-np.log10(regolarisation_coeff_range).min())	# arbitrary
	length_of_line = (np.diff(score_y_record_rel)**2 + np.diff(score_x_record_rel)**2)**0.5
	# if I use a fine regularisation range I need to fit the curvature over more points. this takes care of that.
	# curvature_range was originally = 2
	curvature_range_int = max(1,int(np.ceil(np.abs(-curvature_fit_regularisation_interval*(np.max(np.log10(regolarisation_coeff_range))-np.min(np.log10(regolarisation_coeff_range)))/np.median(np.diff(np.log10(regolarisation_coeff_range)))-1)/2)))
	print('curvature_range_int = '+str(curvature_range_int))


	# plt.figure()
	# plt.plot(regolarisation_coeff_range,score_x_record_rel,label='fit error')
	# plt.plot(regolarisation_coeff_range,score_y_record_rel,label='laplacian error')
	# plt.legend()
	# plt.semilogx()
	# plt.pause(0.01)
	#
	# plt.figure()
	# plt.plot(score_x_record_rel,score_y_record_rel)
	# plt.plot(score_x_record_rel,score_y_record_rel,'+')
	# plt.xlabel('log ||Gm-d||2')
	# plt.ylabel('log ||Laplacian(m)||2')
	# plt.grid()
	# plt.pause(0.01)


	def distance_spread(coord):
		def int(trash,px,py):
			x = coord[0]
			y = coord[1]
			dist = ((x-px)**2 + (y-py)**2)**0.5
			spread = np.sum((dist-np.mean(dist))**2)
			# print(spread)
			return [spread]*5
		return int

	def distance_spread_and_gradient(coord):
		def int(arg):
			x = coord[0]
			y = coord[1]
			px = arg[0]
			py = arg[1]
			dist = ((x-px)**2 + (y-py)**2)**0.5
			spread = np.sum((dist-np.mean(dist))**2)
			# print(str([spread,dist]))
			temp = (((x-px)**2 + (y-py)**2)**-0.5)
			derivate = np.array([np.sum(2*(dist-np.mean(dist))*( -0.5*temp*2*(x-px) - np.mean(-0.5*temp*2*(x-px)) )) , np.sum(2*(dist-np.mean(dist))*( -0.5*temp*2*(y-py) - np.mean(-0.5*temp*2*(y-py)) ))])
			return spread,derivate
		return int

	# plt.figure()
	# plt.plot(score_x_record_rel,score_y_record_rel)
	curvature_range = 1
	curvature_radious = []
	curvature_range_left_all = []
	curvature_range_right_all = []
	for ii in range(curvature_range,len(score_y_record_rel)-curvature_range):

		curvature_range_left = 1
		while np.sum(length_of_line[max(0,ii-curvature_range_left):ii])<fraction_of_line_as_range and curvature_range_left<curvature_range_int*2:
			curvature_range_left += 1
		curvature_range_right = 1
		while np.sum(length_of_line[ii+1:ii+curvature_range_right+1])<fraction_of_line_as_range and curvature_range_right<curvature_range_int*2:
			curvature_range_right += 1

		curvature_range_left_all.append(curvature_range_left)
		curvature_range_right_all.append(curvature_range_right)
		# print(ii)
		# # print(curvature_range)
		# print(curvature_range_left)
		# print(curvature_range_right)
		# # try:
		# # 	guess = centre[0]
		# # except:
		# print(ii)
		try:
			# bds = [[np.min(score_y_record_rel[ii-2:ii+2+1]),np.min(score_x_record_rel[ii-2:ii+2+1])],[np.inf,np.inf]]
			# centre = curve_fit(distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]]),[0]*5,[0]*5,p0=guess,bounds = bds,maxfev=1e5,gtol=1e-12,verbose=1)
			if forward_model_residuals:
				guess = [np.max(score_x_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1])*1,np.mean(score_y_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1])*1]
				bds = [[np.min(score_x_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]),np.inf],[-np.inf,np.inf]]
			else:
				guess = np.max([score_x_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]*1,score_y_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]*1],axis=1)
				bds = [[np.min(score_x_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]),np.inf],[np.min(score_y_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]),np.inf]]
			# bds = [[score_y_record_rel[ii],np.inf],[score_x_record_rel[ii],np.inf]]
			centre, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(distance_spread_and_gradient([score_x_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1],score_y_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]]), x0=guess, bounds = bds, iprint=0, factr=1e3, pgtol=1e-8)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
			centre = [centre]

			dist = ((score_x_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]-centre[0][0])**2 + (score_y_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]-centre[0][1])**2)**0.5
			radious = np.mean(dist)
			# plt.figure()
			# plt.plot(score_x_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1],score_y_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1],'+')
			# plt.plot(score_x_record_rel[ii],score_y_record_rel[ii],'o')
			# plt.plot(centre[0][0],centre[0][1],'o')
			# plt.plot(np.linspace(centre[0][0]-radious,centre[0][0]+radious),centre[0][1]+(radious**2-np.linspace(-radious,+radious)**2)**0.5)
			# plt.plot(np.linspace(centre[0][0]-radious,centre[0][0]+radious),centre[0][1]-(radious**2-np.linspace(-radious,+radious)**2)**0.5)
			# plt.axhline(y=np.min(score_y_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]),linestyle='--')
			# plt.axvline(x=np.min(score_x_record_rel[max(0,ii-curvature_range_left):ii+curvature_range_right+1]),linestyle='--')
			# plt.pause(0.01)
		except:
			radious = np.inf
		curvature_radious.append(radious)
	# curvature_radious = [np.max(curvature_radious)]+curvature_radious+[np.max(curvature_radious)]
	Lcurve_curvature = 1/np.array(curvature_radious)

	peaks_all = find_peaks(Lcurve_curvature)[0]
	peaks = cp.deepcopy(peaks_all)
	proms = get_proms(Lcurve_curvature,peaks)[0]
	if False:	 # a free search doesn't really work, so I force sensible results
		proms = proms[Lcurve_curvature[peaks]>np.max(Lcurve_curvature[peaks])/4]
		peaks = peaks[Lcurve_curvature[peaks]>np.max(Lcurve_curvature[peaks])/4]	# for the case there is mainly only one peak and the rest is noise
		# peaks = np.array([y for _, y in sorted(zip(proms, peaks))])
		# proms = np.sort(proms)
		peaks = np.sort(peaks)
		peaks += curvature_range
		# best_index = peaks[max(-len(peaks),-2)]
		best_index = peaks[0]
	else:
		if len(peaks[regolarisation_coeff_range[peaks+curvature_range]<regolarisation_coeff_upper_limit])>0:
			peaks = peaks[regolarisation_coeff_range[peaks+curvature_range]<regolarisation_coeff_upper_limit]
		if len(peaks[regolarisation_coeff_range[peaks+curvature_range]>regolarisation_coeff_lower_limit])>0:
			peaks = peaks[regolarisation_coeff_range[peaks+curvature_range]>regolarisation_coeff_lower_limit]
		best_index = peaks[np.array(Lcurve_curvature[peaks]).argmax()] + curvature_range
		peaks_all += curvature_range

	# recompose_voxel_emissivity = recompose_voxel_emissivity_all[Lcurve_curvature.argmax()+curvature_range]
	# regolarisation_coeff = regolarisation_coeff_range[Lcurve_curvature.argmax()+curvature_range]
	# x_optimal = x_optimal_all[Lcurve_curvature.argmax()+curvature_range]
	# y_opt = y_opt_all[Lcurve_curvature.argmax()+curvature_range]
	# opt_info = opt_info_all[Lcurve_curvature.argmax()+curvature_range]
	recompose_voxel_emissivity = recompose_voxel_emissivity_all[best_index]
	regolarisation_coeff = regolarisation_coeff_range[best_index]
	x_optimal = x_optimal_all[best_index]
	y_opt = y_opt_all[best_index]
	opt_info = opt_info_all[best_index]

	return score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,recompose_voxel_emissivity,x_optimal,points_removed,regolarisation_coeff,regolarisation_coeff_range,y_opt,opt_info,curvature_range_left_all,curvature_range_right_all,peaks_all,best_index

def cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_binned,ROI1,ROI2,ROI_beams,laser_to_analyse,additional_output=False):
	# additional cropping of the foil to exlude regions without plasma LOS, the frame of the foil and gas puff
	foil_shape = np.shape(sensitivities_binned)[:-1]
	ROI1 = np.round((ROI1.T*foil_shape).T).astype(int)
	ROI2 = np.round((ROI2.T*foil_shape).T).astype(int)
	ROI_beams = np.round((ROI_beams.T*foil_shape).T).astype(int)
	a,b = np.meshgrid(np.arange(foil_shape[1]),np.arange(foil_shape[0]))
	selected_ROI = np.logical_and(np.logical_and(a>=ROI1[1,0],a<ROI1[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI1[0,1],b<sensitivities_binned.shape[0]-ROI1[0,0]))
	selected_ROI = np.logical_or(selected_ROI,np.logical_and(np.logical_and(a>=ROI2[1,0],a<ROI2[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI2[0,1],b<sensitivities_binned.shape[0]-ROI2[0,0])))
	if check_beams_on(laser_to_analyse[-9:-4]):
		selected_ROI = np.logical_and(selected_ROI,np.logical_not(np.logical_and(np.logical_and(a>=ROI_beams[1,0],a<ROI_beams[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI_beams[0,1],b<sensitivities_binned.shape[0]-ROI_beams[0,0]))))
	else:
		ROI_beams = np.array([[0,0],[0,0]])

	if True:	# setting zero to the sensitivities I want to exclude
		sensitivities_binned_crop = cp.deepcopy(sensitivities_binned)
		sensitivities_binned_crop[np.logical_not(selected_ROI),:] = 0
	else:	# cutting sensitivity out of ROI
		sensitivities_binned_crop = sensitivities_binned[sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]
	if additional_output:
		return sensitivities_binned_crop,selected_ROI,ROI1,ROI2,ROI_beams
	else:
		return sensitivities_binned_crop,selected_ROI

def calc_temp_to_power_BB_1(photon_flux_over_temperature_interpolator,temperature_minus_background_crop_binned,ref_temperature,time_binned,dx,counts_std_crop_binned,BB_proportional_crop_binned,BB_proportional_std_crop_binned,reference_background_std_crop_binned,temperature_std_crop_binned,nan_ROI_mask,grid_laplacian=[],ref_temperature_std=0):
	zeroC=273.15 #K / C

	dt = time_binned[2:]-time_binned[:-2]
	photon_flux_over_temperature = photon_flux_over_temperature_interpolator(temperature_minus_background_crop_binned+ref_temperature)
	# basetemp=np.nanmean(datatempcrop[0,frame-7:frame+7,1:-1,1:-1],axis=0)
	dT = (temperature_minus_background_crop_binned[2:,1:-1,1:-1]-temperature_minus_background_crop_binned[:-2,1:-1,1:-1]).astype(np.float32)
	dTdt = np.gradient(temperature_minus_background_crop_binned,time_binned,axis=0)[1:-1,1:-1,1:-1].astype(np.float32)	# this is still a central difference but it doesn't rely on hand made code
	dTdt_std=np.divide(( ((counts_std_crop_binned[2:,1:-1,1:-1]/(photon_flux_over_temperature[2:,1:-1,1:-1]*BB_proportional_crop_binned[1:-1,1:-1]))**2 + (counts_std_crop_binned[:-2,1:-1,1:-1] /(photon_flux_over_temperature[:-2,1:-1,1:-1]*BB_proportional_crop_binned[1:-1,1:-1]))**2 + (dT*BB_proportional_std_crop_binned[1:-1,1:-1]/BB_proportional_crop_binned[1:-1,1:-1])**2)**0.5 ).T,dt).T.astype(np.float32)

	if len(np.shape(grid_laplacian))!=2:
		horizontal_coord = np.arange(np.shape(temperature_minus_background_crop_binned)[2])
		vertical_coord = np.arange(np.shape(temperature_minus_background_crop_binned)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		grid = np.array([[horizontal_coord.flatten()]*4,[vertical_coord.flatten()]*4]).T
		grid_laplacian = -build_laplacian(grid,diagonal_factor=0.5) / (dx**2) / 2	# the /2 comes from the fact that including the diagonals amounts to double counting, so i do a mean by summing half of it
	d2Tdxy = np.dot(temperature_minus_background_crop_binned.reshape((len(temperature_minus_background_crop_binned),len(grid_laplacian))),grid_laplacian).reshape(np.shape(temperature_minus_background_crop_binned))[1:-1,1:-1,1:-1]
	d2Tdxy_std = (np.dot( ((counts_std_crop_binned/(BB_proportional_crop_binned*photon_flux_over_temperature))**2+(reference_background_std_crop_binned/(BB_proportional_crop_binned*photon_flux_over_temperature_interpolator(ref_temperature)))**2 + (temperature_minus_background_crop_binned*BB_proportional_std_crop_binned/BB_proportional_crop_binned)**2 ).reshape((len(temperature_minus_background_crop_binned),len(grid_laplacian))),grid_laplacian**2)**0.5).reshape(np.shape(temperature_minus_background_crop_binned))[1:-1,1:-1,1:-1]

	# temp = np.ones_like(dTdt).astype(np.float32)*np.nan
	# temp[:,nan_ROI_mask[1:-1,1:-1]]=d2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]
	# d2Tdxy = cp.copy(temp)
	# temp = np.ones_like(dTdt).astype(np.float32)*np.nan
	# temp[:,nan_ROI_mask[1:-1,1:-1]]=d2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]
	# d2Tdxy_std = cp.copy(temp)
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	negd2Tdxy_std=d2Tdxy_std

	T4=(temperature_minus_background_crop_binned[1:-1,1:-1,1:-1]+ref_temperature+zeroC)**4
	T4_std=T4**(3/4) *4 *temperature_std_crop_binned[1:-1,1:-1,1:-1]
	T04=(ref_temperature+zeroC)**4 *np.ones_like(temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])
	T04_std=T04**(3/4) *4 *ref_temperature_std
	T4_T04 = (T4-T04).astype(np.float32)
	T4_T04_std = ((T4_std**2+T04_std**2)**0.5).astype(np.float32)
	# T4_T04 = np.ones_like(dTdt).astype(np.float32)*np.nan
	# T4_T04[:,nan_ROI_mask[1:-1,1:-1]] = (T4[:,nan_ROI_mask[1:-1,1:-1]]-T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	# T4_T04_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	# T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]] = ((T4_std[:,nan_ROI_mask[1:-1,1:-1]]**2+T04_std**2)**0.5).astype(np.float32)

	# temp = np.ones_like(dTdt).astype(np.float32)*np.nan
	# temp[:,nan_ROI_mask[1:-1,1:-1]]=dTdt[:,nan_ROI_mask[1:-1,1:-1]]
	# dTdt = cp.copy(temp)
	# temp = np.ones_like(dTdt).astype(np.float32)*np.nan
	# temp[:,nan_ROI_mask[1:-1,1:-1]]=dTdt_std[:,nan_ROI_mask[1:-1,1:-1]]
	# dTdt_std = cp.copy(temp)

	return dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std

def calc_temp_to_power_BB_2(dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std,nan_ROI_mask,foilemissivityscaled,foilthicknessscaled,reciprdiffusivityscaled,Ptthermalconductivity):
	sigmaSB=5.6704e-08 #[W/(m2 K4)]

	BBrad = np.ones_like(dTdt).astype(np.float32)*np.nan
	BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*((T4_T04 * foilemissivityscaled)[:,nan_ROI_mask[1:-1,1:-1]])).astype(np.float32)
	diffusion = np.ones_like(dTdt).astype(np.float32)*np.nan
	diffusion[:,nan_ROI_mask[1:-1,1:-1]] = ((Ptthermalconductivity*negd2Tdxy*foilthicknessscaled)[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	timevariation = np.ones_like(dTdt).astype(np.float32)*np.nan
	timevariation[:,nan_ROI_mask[1:-1,1:-1]] = ((Ptthermalconductivity*dTdt*foilthicknessscaled*reciprdiffusivityscaled)[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	powernoback = (diffusion + timevariation + BBrad).astype(np.float32)
	BBrad_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*((T4_T04_std*foilemissivityscaled)[:,nan_ROI_mask[1:-1,1:-1]])).astype(np.float32)
	diffusion_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (((Ptthermalconductivity*negd2Tdxy_std*foilthicknessscaled)[:,nan_ROI_mask[1:-1,1:-1]])).astype(np.float32)
	timevariation_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = ((Ptthermalconductivity*dTdt_std*foilthicknessscaled*reciprdiffusivityscaled)[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	powernoback_std = np.ones_like(powernoback)*np.nan
	powernoback_std[:,nan_ROI_mask[1:-1,1:-1]] = ((diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5).astype(np.float32)
	return BBrad,diffusion,timevariation,powernoback,BBrad_std,diffusion_std,timevariation_std,powernoback_std


def rotate_and_crop_2D(data,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx):
	data_rot=rotate(data,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		data_rot*=out_of_ROI_mask
		data_rot[np.logical_and(data_rot<np.nanmin(data[i]),data_rot>np.nanmax(data[i]))]=0
	return data_rot[foildw:foilup,foillx:foilrx]

def rotate_and_crop_3D(data,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx):
	data_rot=rotate(data,foilrotdeg,axes=(-1,-2))
	if not (height==max_ROI[0][1]+1 and width==max_ROI[1][1]+1):
		data_rot*=out_of_ROI_mask
		data_rot[np.logical_and(data_rot<np.nanmin(data[i]),data_rot>np.nanmax(data[i]))]=0
	return data_rot[:,foildw:foilup,foillx:foilrx]

def rotate_and_crop_multi_digitizer(data,dimensions,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx):
	if dimensions==2:
		out = [rotate_and_crop_2D(data_int,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx) for data_int in data]
	elif dimensions==3:
		out = [rotate_and_crop_3D(data_int,foilrotdeg,max_ROI,height,width,out_of_ROI_mask,foildw,foilup,foillx,foilrx) for data_int in data]
	else:
		print('ERROR, dimensions must be 2 or 3, not '+str(dimensions))
	return out


def translate_emissivity_profile_with_homo_temp(original_voxels_centre,original_x_optimal,output_voxels_centre):
	output_x_optimal = np.zeros((len(output_voxels_centre)+2))
	output_x_optimal[-2:] = original_x_optimal[-2:]

	output_recompose_voxel_emissivity = np.zeros((len(np.unique(output_voxels_centre[:,0])),len(np.unique(output_voxels_centre[:,1]))))*np.nan
	dr = np.median(np.diff(np.unique(output_voxels_centre[:,0])))
	dz = np.median(np.diff(np.unique(output_voxels_centre[:,1])))
	dist_mean = (dz**2 + dr**2)/2
	for i_r,r in enumerate(np.unique(output_voxels_centre[:,0])):
		for i_z,z in enumerate(np.unique(output_voxels_centre[:,1])):
			dist = (output_voxels_centre[:,0]-r)**2 + (output_voxels_centre[:,1]-z)**2
			if dist.min()<dist_mean/2:
				index = np.abs(dist).argmin()
				original_dist = (original_voxels_centre[:,0]-r)**2 + (original_voxels_centre[:,1]-z)**2
				if original_dist.min()<dist_mean/2:
					original_index = np.abs(original_dist).argmin()
					output_x_optimal[index] = original_x_optimal[original_index]
					output_recompose_voxel_emissivity[i_r,i_z] = output_x_optimal[index]
	output_recompose_voxel_emissivity *= 4*np.pi	# this exist because the sensitivity matrix is built with 1W/str/m^3/ x nm emitters while I use 1W as reference, so I need to multiply the results by 4pi
	return output_x_optimal,output_recompose_voxel_emissivity

def find_temperature_from_power_residuals(dt,grid_laplacian,input_foil_power,temperature_full_resolution_initial,ref_temperature=25,thickness=2.0531473351462095e-06,emissivity=0.9999999999999,diffusivity=1.0283685197530968e-05,Ptthermalconductivity=71.6):
	zeroC=273.15 #K / C
	sigmaSB=5.6704e-08 #[W/(m2 K4)]
	# # from 2021/09/17, Laser_data_analysis3_3.py
	# thickness = 2.0531473351462095e-06
	# emissivity = 0.9999999999999
	# diffusivity = 1.0283685197530968e-05
	T04=(ref_temperature+zeroC)**4
	temperature_full_resolution = np.zeros(np.array(np.shape(input_foil_power))+2)
	def residual(temp):
		temperature_full_resolution[1:-1,1:-1] = temp.reshape(temperature_full_resolution[1:-1,1:-1].shape)
		dTdt = (temperature_full_resolution-temperature_full_resolution_initial)/dt	# the full central difference gradient requires a global fit that takes forever, so I do this as mu
		dTdt_derivate = np.ones_like(temperature_full_resolution)/dt
		d2Tdxy = np.dot(temperature_full_resolution.flatten(),grid_laplacian).reshape(np.shape(temperature_full_resolution))
		# negd2Tdxy=np.multiply(-1,d2Tdxy)
		# negd2Tdxy_derivate=np.ones_like(temperature_full_resolution)#np.multiply(-1,np.dot(np.ones_like(temperature_full_resolution).flatten(),grid_laplacian).reshape(np.shape(temperature_full_resolution)))
		T4=(temperature_full_resolution+ref_temperature+zeroC)**4
		T4_T04 = (T4-T04)
		T4_T04_derivate = 4*(temperature_full_resolution+ref_temperature+zeroC)**3

		BBrad = (2*sigmaSB*T4_T04 * emissivity)
		BBrad_derivate = (2*sigmaSB*T4_T04_derivate * emissivity)
		diffusion = -(Ptthermalconductivity*d2Tdxy*thickness)
		# diffusion_derivate = (Ptthermalconductivity*negd2Tdxy_derivate*thickn/ess)
		timevariation = (Ptthermalconductivity*dTdt*thickness/diffusivity)
		timevariation_derivate = (Ptthermalconductivity*dTdt_derivate*thickness/diffusivity)
		powernoback = (diffusion + timevariation + BBrad)
		# powernoback_derivate = (diffusion_derivate + timevariation_derivate + BBrad_derivate)
		powernoback[1:-1,1:-1]-=input_foil_power
		powernoback[0]=0
		powernoback[-1]=0
		powernoback[:,0]=0
		powernoback[:,-1]=0
		gradient = 2*powernoback * (BBrad_derivate + timevariation_derivate) - 2*Ptthermalconductivity*thickness*np.dot(powernoback.flatten(),grid_laplacian).reshape(np.shape(temperature_full_resolution))

		return np.sum(powernoback[1:-1,1:-1]**2),gradient[1:-1,1:-1].flatten()
	return residual


#######################################################################################################################################################################################################################################

def movie_from_data_radial_profile(data,framerate,integration=1,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',form_factor_size=15,extent = [], image_extent = [],timesteps='auto',extvmin='auto',extvmax='auto',time_offset=0,prelude='',vline=None,hline=None,EFIT_path=EFIT_path_default,include_EFIT=False,efit_reconstruction=None,EFIT_output_requested = False,pulse_ID=None,overlay_x_point=False,overlay_mag_axis=False,overlay_structure=False,overlay_strike_points=False,overlay_separatrix=False,structure_alpha=0.5,foil_size=foil_size,additional_points_dict = dict([]), additional_each_frame_label_description=[], additional_each_frame_label_number=[],x_markersize=30,x_linewidth=3):
	import matplotlib.animation as animation
	import numpy as np

	if len(extent) == 0 or len(image_extent) == 0:
		print('ERROR. for coleval.movie_from_data_radial_profile you must supply an extent and image_extent of shape=1')
		# exit()

	form_factor = (image_extent[1]-image_extent[0])/(image_extent[3]-image_extent[2])
	# form_factor = (np.shape(data[0][0])[1])/(np.shape(data[0][0])[0])
	fig = plt.figure(figsize=(form_factor_size*form_factor, form_factor_size))
	ax = fig.add_subplot(111)

	data_rotated = np.array([np.flip(data[0],axis=2)])

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#	 return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	if len(image_extent)==4:
		ver = np.linspace(extent[2],extent[3],num=np.shape(data_rotated[0][0])[0]+1)
		up = np.abs(image_extent[3]-ver).argmin()-1
		down = np.abs(image_extent[2]-ver).argmin()
		hor = np.linspace(extent[0],extent[1],num=np.shape(data_rotated[0][0])[1]+1)
		left = np.abs(image_extent[0]-hor).argmin()
		right = np.abs(image_extent[1]-hor).argmin()-1
		data_rotated[0][:,:,:left] = np.nan
		data_rotated[0][:,:,right+1:] = np.nan
		data_rotated[0][:,:down] = np.nan
		data_rotated[0][:,up+1:] = np.nan

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data_rotated[0])
	frames[0]=data_rotated[0,0]

	for i in range(len(data_rotated[0])):
		# x	   += 1
		# curVals  = f(x, y)
		frames[i]=(data_rotated[0,i])

	cv0 = frames[0]
	im = ax.imshow(cv0,cmap, origin='lower', interpolation='none', extent = extent)# [0,np.shape(data)[0]-1,0,np.shape(data)[1]-1]) # Here make an AxesImage rather than contour
	ax.set_ylim(top=image_extent[3],bottom=image_extent[2])
	ax.set_xlim(right=image_extent[1],left=image_extent[0])

	if include_EFIT:
		try:
			if efit_reconstruction == None:
				print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc')
				efit_reconstruction = mclass(EFIT_path+'/epm0'+str(pulse_ID)+'.nc',pulse_ID=pulse_ID)
			else:
				print('EFIT reconstruction externally supplied')
			EFIT_dt = np.median(np.diff(efit_reconstruction.time))
		except Exception as e:
			print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc failed')
			logging.exception('with error: ' + str(e))
			include_EFIT=False
			overlay_x_point=False
			overlay_mag_axis=False
			overlay_separatrix=False
			overlay_strike_points=False
			overlay_separatrix=False
			efit_reconstruction = None
		if overlay_x_point:
			all_time_x_point_location = np.array([efit_reconstruction.lower_xpoint_r,efit_reconstruction.lower_xpoint_z]).T
			plot1 = ax.plot(0,0,'+r', alpha=1)[0]
		if overlay_mag_axis:
			all_time_mag_axis_location = np.array([efit_reconstruction.mag_axis_r,efit_reconstruction.mag_axis_z]).T
			plot2 = ax.plot(0,0,'+r', alpha=1)[0]
		if overlay_separatrix or overlay_strike_points:
			all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
		if overlay_strike_points:
			all_time_strike_points_location = return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			plot3 = ax.plot(0,0,'xr',markersize=x_markersize,linewidth=x_linewidth, alpha=1)[0]
		if overlay_separatrix:
			all_time_separatrix = return_all_time_separatrix_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			plot5 = []
			for __i in range(len(all_time_separatrix[0])):
				plot5.append(ax.plot(0,0,'--b', alpha=1)[0])
	if overlay_structure:
		for i in range(len(structure_radial_profile)):
			ax.plot(structure_radial_profile[i][:,0],structure_radial_profile[i][:,1],'--k',alpha=structure_alpha)
	if overlay_structure:
		for i in range(len(res_bolo_radial_profile)):
			ax.plot(res_bolo_radial_profile[i][:,0],res_bolo_radial_profile[i][:,1],'--r',alpha=structure_alpha)
	if len(list(additional_points_dict.keys()))!=0:
		plot6 = []
		for __i in range(additional_points_dict['number_of_points']):
			plot6.append(ax.plot(0,0,additional_points_dict['marker'][__i], alpha=1,markersize=x_markersize,linewidth=x_linewidth)[0])

	# if len(np.shape(mask)) == 2:
	# im = ax.imshow(mask,'gray',interpolation='none',alpha=1)
	if np.sum(vline == None)==0:
		if np.shape(vline)==():
			vline = max(0,min(vline,np.shape(cv0)[1]-1))
			axvline = ax.axvline(x=vline,linestyle='--',color='k')
		else:
			for i in range(len(vline)):
				vline[i] = max(0,min(vline[i],np.shape(cv0)[1]-1))
				axvline = ax.axvline(x=vline[i],linestyle='--',color='k')
	if np.sum(hline == None)==0:
		if np.shape(hline)==():
			hline = max(0,min(hline,np.shape(cv0)[0]-1))
			axhline = ax.axhline(y=hline,linestyle='--',color='k')
		else:
			for i in range(len(hline)):
				hline[i] = max(0,min(hline[i],np.shape(cv0)[0]-1))
				axhline = ax.axhline(y=hline[i],linestyle='--',color='k')

	cb = fig.colorbar(im).set_label(barlabel)
	cb = ax.set_xlabel(xlabel)
	cb = ax.set_ylabel(ylabel)
	tx = ax.set_title('Frame 0')


	# if timesteps=='auto':
	# 	timesteps_int = time_offset+np.arange(len(data[0])+1)/framerate
	# else:
	# 	timesteps_int = cp.deepcopy(timesteps)
	def animate(i):
		arr = frames[i]
		if extvmax=='auto':
			vmax = np.nanmax(arr)
		elif extvmax=='allmax':
			vmax = np.nanmax(data_rotated)
		else:
			vmax = extvmax

		if extvmin=='auto':
			vmin = np.nanmin(arr)
		elif extvmin=='allmin':
			vmin = np.nanmin(data_rotated)
		else:
			vmin = extvmin
		im.set_data(arr)
		im.set_clim(vmin, vmax)
		if timesteps=='auto':
			time_int = time_offset+i/framerate
			each_frame_label = prelude + 'Frame {0}'.format(i)+', FR %.3gHz, t %.3gs, int %.3gms' %(framerate,time_int,integration)
		else:
			time_int = timesteps[i]
			each_frame_label = prelude + 'Frame {0}'.format(i)+', t %.3gs, int %.3gms' %(time_int,integration)
		if len(additional_each_frame_label_description) != 0:
			additional_each_frame_label = '\n' + additional_each_frame_label_description[i]
		else:
			additional_each_frame_label = '\n'
		if len(additional_each_frame_label_number) != 0:
			additional_each_frame_label = additional_each_frame_label + ' %.5g' %(additional_each_frame_label_number[i])
		each_frame_label = each_frame_label + additional_each_frame_label
		tx.set_text(each_frame_label)
		if include_EFIT:
			if np.min(np.abs(time_int-efit_reconstruction.time))>EFIT_dt:	# means that the reconstruction is not available for that time
				if overlay_x_point:
					plot1.set_data(([],[]))
				if overlay_mag_axis:
					plot2.set_data(([],[]))
				if overlay_strike_points:
					plot3.set_data(([],[]))
				if overlay_separatrix:
					for __i in range(len(plot5)):
						plot5[__i].set_data(([],[]))
			else:
				i_time = np.abs(time_int-efit_reconstruction.time).argmin()
				if overlay_x_point:
					if np.sum(np.isnan(all_time_x_point_location[i_time]))>=len(all_time_x_point_location[i_time]):	# means that all the points calculated are outside the foil
						plot1.set_data(([],[]))
					else:
						plot1.set_data((all_time_x_point_location[i_time][0],all_time_x_point_location[1]))
				if overlay_mag_axis:
					# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
					# 	plot2.set_data(([],[]))
					# else:
					plot2.set_data((all_time_mag_axis_location[i_time][0],all_time_mag_axis_location[i_time][1]))
				if overlay_strike_points:
					# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
					# 	plot3.set_data(([],[]))
					# 	for __i in range(len(plot4)):
					# 		plot4[__i].set_data(([],[]))
					# else:
					plot3.set_data((all_time_strike_points_location[i_time][0],all_time_strike_points_location[i_time][1]))
				if overlay_separatrix:
					# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
					for __i in range(len(plot5)):
						plot5[__i].set_data((all_time_separatrix[i_time][__i][0],all_time_separatrix[i_time][__i][1]))
		if len(list(additional_points_dict.keys()))!=0:
			i_time2 = np.abs(time_int-additional_points_dict['time']).argmin()
			# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
			for __i in range(additional_points_dict['number_of_points']):
				plot6[__i].set_data((additional_points_dict[str(__i)][i_time2][0],additional_points_dict[str(__i)][i_time2][1]))
			# 	masked = np.ma.masked_where(mask == 0, mask)
			# 	ax.imshow(masked, 'gray', interpolation='none', alpha=0.2,origin='lower',extent = [0,np.shape(data)[0]-1,0,np.shape(data)[1]-1])

			# In this version you don't have to do anything to the colorbar,
			# it updates itself when the mappable it watches (im) changes
	# else:
	# 	def animate(i):
	# 		arr = frames[i]
	# 		if extvmax=='auto':
	# 			vmax = np.max(arr)
	# 		elif extvmax=='allmax':
	# 			vmax = np.max(data)
	# 		else:
	# 			vmax = extvmax
	#
	# 		if extvmin=='auto':
	# 			vmin = np.min(arr)
	# 		elif extvmin=='allmin':
	# 			vmin = np.min(data)
	# 		else:
	# 			vmin = extvmin
	# 		im.set_data(arr)
	# 		im.set_clim(vmin, vmax)
	# 		tx.set_text(prelude + 'Frame {0}'.format(i)+', t %.3gs, int %.3gms' %(timesteps[i],integration))
	# 		# if len(np.shape(mask)) == 2:
	# 		# 	# 	im.imshow(mask,'binary',interpolation='none',alpha=0.3)
	# 		# 	masked = np.ma.masked_where(mask == 0, mask)
	# 		# 	ax.imshow(masked, 'gray', interpolation='none', alpha=0.2,origin='lower',extent = [0,np.shape(data)[0]-1,0,np.shape(data)[1]-1])

	ani = animation.FuncAnimation(fig, animate, frames=len(data[0]))

	if EFIT_output_requested == False:
		return ani
	else:
		return ani,efit_reconstruction

def image_from_data_radial_profile(data,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',form_factor_size=15,extent = [], image_extent = [],ref_time=None,extvmin='auto',extvmax='auto',prelude='',vline=None,hline=None,EFIT_path=EFIT_path_default,include_EFIT=False,efit_reconstruction=None,EFIT_output_requested = False,pulse_ID=None,overlay_x_point=False,overlay_mag_axis=False,overlay_structure=False,overlay_strike_points=False,overlay_separatrix=False,overlay_res_bolo=False,structure_alpha=0.5,foil_size=foil_size,additional_points_dict = dict([])):
	import numpy as np
	from matplotlib import cm	# to print nan as white

	if len(extent) == 0 or len(image_extent) == 0:
		print('ERROR. for coleval.image_from_data_radial_profile you must supply an extent and image_extent of shape=1')
		# exit()

	form_factor = (image_extent[1]-image_extent[0])/(image_extent[3]-image_extent[2])
	fig = plt.figure(figsize=(form_factor_size*form_factor, form_factor_size))
	ax = fig.add_subplot(111)

	data_rotated = np.array([np.flip(data[0],axis=2)])

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#	 return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	if len(image_extent)==4:
		ver = np.linspace(extent[2],extent[3],num=np.shape(data_rotated[0][0])[0]+1)
		up = np.abs(image_extent[3]-ver).argmin()-1
		down = np.abs(image_extent[2]-ver).argmin()
		hor = np.linspace(extent[0],extent[1],num=np.shape(data_rotated[0][0])[1]+1)
		left = np.abs(image_extent[0]-hor).argmin()
		right = np.abs(image_extent[1]-hor).argmin()-1
		data_rotated[0][:,:,:left] = np.nan
		data_rotated[0][:,:,right+1:] = np.nan
		data_rotated[0][:,:down] = np.nan
		data_rotated[0][:,up+1:] = np.nan

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data_rotated[0])
	frames[0]=data_rotated[0,0]

	for i in range(len(data_rotated[0])):
		# x	   += 1
		# curVals  = f(x, y)
		frames[i]=(data_rotated[0,i])

	cv0 = frames[0]
	masked_array = np.ma.array (cv0, mask=np.isnan(cv0))
	# exec('cmap=cm.' + cmap)
	cmap=cm.rainbow
	cmap.set_bad('white',1.)
	im = ax.imshow(masked_array,cmap=cmap, origin='lower', interpolation='none', extent = extent)# [0,np.shape(data)[0]-1,0,np.shape(data)[1]-1]) # Here make an AxesImage rather than contour
	ax.set_ylim(top=image_extent[3],bottom=image_extent[2])
	ax.set_xlim(right=image_extent[1],left=image_extent[0])

	if ref_time==None:
		include_EFIT = False

	if include_EFIT:
		try:
			if efit_reconstruction == None:
				print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc')
				efit_reconstruction = mclass(EFIT_path+'/epm0'+str(pulse_ID)+'.nc',pulse_ID=pulse_ID)
			else:
				print('EFIT reconstruction externally supplied')
			EFIT_dt = np.median(np.diff(efit_reconstruction.time))
		except Exception as e:
			print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc failed')
			logging.exception('with error: ' + str(e))
			include_EFIT=False
			overlay_x_point=False
			overlay_mag_axis=False
			overlay_separatrix=False
			overlay_strike_points=False
			overlay_separatrix=False
			efit_reconstruction = None
		if overlay_x_point:
			all_time_x_point_location = np.array([efit_reconstruction.lower_xpoint_r,efit_reconstruction.lower_xpoint_z]).T
			plot1 = ax.plot(0,0,'+r', alpha=1)[0]
		if overlay_mag_axis:
			all_time_mag_axis_location = np.array([efit_reconstruction.mag_axis_r,efit_reconstruction.mag_axis_z]).T
			plot2 = ax.plot(0,0,'+r', alpha=1)[0]
		if overlay_separatrix or overlay_strike_points:
			all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
		if overlay_strike_points:
			all_time_strike_points_location = return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			plot3 = ax.plot(0,0,'xr',markersize=20, alpha=1)[0]
		if overlay_separatrix:
			all_time_separatrix = return_all_time_separatrix_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
			plot5 = []
			for __i in range(len(all_time_separatrix[0])):
				plot5.append(ax.plot(0,0,'--b', alpha=1)[0])
	if overlay_structure:
		for i in range(len(structure_radial_profile)):
			ax.plot(structure_radial_profile[i][:,0],structure_radial_profile[i][:,1],'--k',alpha=structure_alpha)
	if overlay_res_bolo:
		for i in range(len(res_bolo_radial_profile)):
			ax.plot(res_bolo_radial_profile[i][:,0],res_bolo_radial_profile[i][:,1],'--r',alpha=structure_alpha)
	if len(list(additional_points_dict.keys()))!=0:
		plot6 = []
		for __i in range(additional_points_dict['number_of_points']):
			plot6.append(ax.plot(0,0,additional_points_dict['marker'][__i], alpha=1,markersize=20)[0])

	# if len(np.shape(mask)) == 2:
	# im = ax.imshow(mask,'gray',interpolation='none',alpha=1)
	if np.sum(vline == None)==0:
		if np.shape(vline)==():
			vline = max(0,min(vline,np.shape(cv0)[1]-1))
			axvline = ax.axvline(x=vline,linestyle='--',color='k')
		else:
			for i in range(len(vline)):
				vline[i] = max(0,min(vline[i],np.shape(cv0)[1]-1))
				axvline = ax.axvline(x=vline[i],linestyle='--',color='k')
	if np.sum(hline == None)==0:
		if np.shape(hline)==():
			hline = max(0,min(hline,np.shape(cv0)[0]-1))
			axhline = ax.axhline(y=hline,linestyle='--',color='k')
		else:
			for i in range(len(hline)):
				hline[i] = max(0,min(hline[i],np.shape(cv0)[0]-1))
				axhline = ax.axhline(y=hline[i],linestyle='--',color='k')

	cb = fig.colorbar(im).set_label(barlabel)
	cb = ax.set_xlabel(xlabel)
	cb = ax.set_ylabel(ylabel)
	tx = ax.set_title('Frame 0')


	arr = frames[0]
	if extvmax=='auto':
		vmax = np.nanmax(arr)
	elif extvmax=='allmax':
		vmax = np.nanmax(data_rotated)
	else:
		vmax = extvmax

	if extvmin=='auto':
		vmin = np.nanmin(arr)
	elif extvmin=='allmin':
		vmin = np.nanmin(data_rotated)
	else:
		vmin = extvmin
	masked_array = np.ma.array (arr, mask=np.isnan(arr))
	im.set_data(masked_array)
	im.set_clim(vmin, vmax)
	tx.set_text(prelude)
	if include_EFIT:
		if np.min(np.abs(ref_time-efit_reconstruction.time))>EFIT_dt:	# means that the reconstruction is not available for that time
			if overlay_x_point:
				plot1.set_data(([],[]))
			if overlay_mag_axis:
				plot2.set_data(([],[]))
			if overlay_strike_points:
				plot3.set_data(([],[]))
			if overlay_separatrix:
				for __i in range(len(plot5)):
					plot5[__i].set_data(([],[]))
		else:
			i_time = np.abs(ref_time-efit_reconstruction.time).argmin()
			if overlay_x_point:
				if np.sum(np.isnan(all_time_x_point_location[i_time]))>=len(all_time_x_point_location[i_time]):	# means that all the points calculated are outside the foil
					plot1.set_data(([],[]))
				else:
					plot1.set_data((all_time_x_point_location[i_time][0],all_time_x_point_location[1]))
			if overlay_mag_axis:
				# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
				# 	plot2.set_data(([],[]))
				# else:
				plot2.set_data((all_time_mag_axis_location[i_time][0],all_time_mag_axis_location[i_time][1]))
			if overlay_strike_points:
				# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
				# 	plot3.set_data(([],[]))
				# 	for __i in range(len(plot4)):
				# 		plot4[__i].set_data(([],[]))
				# else:
				plot3.set_data((all_time_strike_points_location[i_time][0],all_time_strike_points_location[i_time][1]))
			if overlay_separatrix:
				# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
				for __i in range(len(plot5)):
					plot5[__i].set_data((all_time_separatrix[i_time][__i][0],all_time_separatrix[i_time][__i][1]))
			if len(list(additional_points_dict.keys()))!=0:
				i_time2 = np.abs(ref_time-additional_points_dict['time']).argmin()
				# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
				for __i in range(additional_points_dict['number_of_points']):
					plot6[__i].set_data((additional_points_dict[str(__i)][i_time2][0],additional_points_dict[str(__i)][i_time2][1]))

	if EFIT_output_requested == False:
		return fig
	else:
		return fig,efit_reconstruction

##############################################################################################################################################################################################################

def estimate_counts_std(counts,int_time=2):#framerate=383):
	# if framerate<994+3:
	if np.abs(int_time-2)<0.001:
		# counts_std_fit = np.array([6.96431538e-20, -3.26767625e-15,  5.84235577e-11, -4.97550821e-07, 2.18103951e-03])	# found 2021-10-06 by looking at 383Hz 1ms and 2ms data. it is likely not correct for 0.5ms integration time
		counts_std_fit = np.array([-5.32486228e-31,  3.53268894e-26, -9.85692764e-22,  1.50423830e-17,-1.36726859e-13,  7.54763409e-10, -2.46270434e-06,  4.48742195e-03])	# found 2021-10-06 with generate_count_std_VS_count_coeff.py
	# else:
	elif np.abs(int_time-1)<0.001:
		counts_std_fit = np.array([-5.32486228e-31,  3.53268894e-26, -9.85692764e-22,  1.50423830e-17,-1.36726859e-13,  7.54763409e-10, -2.46270434e-06,  4.48742195e-03])/2	# only copied from the 2ms case and dividewd by 2
	else:
		print(str(framerate)+'Hz framerate does not have std coefficients assotiated with it, this should be checked')
		a=b	# a and b are not defined so this will cause an error to occour
	counts_temp_std = np.polyval(counts_std_fit,counts)*counts
	return counts_temp_std

##############################################################################################################################################################################################################

def get_tend(shot_id):
	from pycpf import pycpf
	# import pyuda as uda
	client_int = pyuda.Client()

	try:
		tend = pycpf.query(['tend'], filters=['exp_number = '+shot_id])['tend'][0]
	except:
		try:
			signal_name = '/AMC/Plasma_current'
			temp = client_int.get(signal_name,shot_id)
			Plasma_current_time = temp.time.data
			Plasma_current = temp.data
			tend = Plasma_current_time[np.logical_and(Plasma_current_time>0.1,Plasma_current<20).argmax()]	# the 20 (kA) treshold comes from Jimmy Measures
		except:
			tend = 1	# arbitrary time
	reset_connection(client_int)
	del client_int
	return tend

def check_beams_on(shot_id):
	# import pyuda as uda
	client_int = pyuda.Client()

	try:
		try:
			signal_name = '/XNB/SS/BEAMPOWER'
			temp = client_int.get(signal_name,shot_id)
			beams_on_flag = True
		except:
			signal_name = '/XNB/SW/BEAMPOWER'
			temp = client_int.get(signal_name,shot_id)
			beams_on_flag = True
	except:
		beams_on_flag = False
	reset_connection(client_int)
	del client_int
	return beams_on_flag

################################################################################################################################################################################################################

# sensitivity matrix manipulation

def build_laplacian(grid,diagonal_factor=0.5):
	# Try making grid laplacian matrix for spatial regularisation
	num_cells = len(grid)
	grid_laplacian = np.zeros((num_cells, num_cells))
	unique_x = np.unique(np.mean(grid,axis=1)[:,0])
	unique_y = np.unique(np.mean(grid,axis=1)[:,1])
	if diagonal_factor==0:
		account_diagonals = False
	else:
		account_diagonals = True
	# 	# change to take into account that the distance is different on the diagonals.
	# 	# https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
	# 	# here they mention too a laplacian operator with a different coefficient on the diagonals
	# 	diagonal_factor = 0.5
	# else:
	# 	diagonal_factor = 1

	for ith_cell in range(num_cells):

		# get the 2D mesh coordinates of this cell
		ix = np.abs(unique_x - np.mean(grid,axis=1)[ith_cell,0]).argmin()	# radious
		iy = np.abs(unique_y - np.mean(grid,axis=1)[ith_cell,1]).argmin()	# Z

		neighbours = 0

		if ix>0:
			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix-1],np.mean(grid,axis=1)[:,1]==unique_y[iy])  # neighbour 1 left
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = -1
					neighbours += 1
			except KeyError:
				pass

		try:
			select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix],np.mean(grid,axis=1)[:,1]==unique_y[iy+1])  # neighbour 3 top
			if np.sum(select)>0:
				n1 = select.argmax()
				grid_laplacian[ith_cell, n1] = -1
				neighbours += 1
		except:
			pass

		try:
			select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix+1],np.mean(grid,axis=1)[:,1]==unique_y[iy])  # neighbour 5 right
			if np.sum(select)>0:
				n1 = select.argmax()
				grid_laplacian[ith_cell, n1] = -1
				neighbours += 1
		except:
			pass

		if iy>0:
			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix],np.mean(grid,axis=1)[:,1]==unique_y[iy-1])  # neighbour 7 down
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = -1
					neighbours += 1
			except:
				pass

		if account_diagonals:
			if ix>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix-1],np.mean(grid,axis=1)[:,1]==unique_y[iy+1])  # neighbour 2 top left
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -diagonal_factor
						neighbours += diagonal_factor
				except:
					pass

			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix+1],np.mean(grid,axis=1)[:,1]==unique_y[iy+1])  # neighbour 4 top right
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = -diagonal_factor
					neighbours += diagonal_factor
			except:
				pass

			if iy>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix+1],np.mean(grid,axis=1)[:,1]==unique_y[iy-1])  # neighbour 6 down right
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -diagonal_factor
						neighbours += diagonal_factor
				except:
					pass

			if ix>0 and iy>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix-1],np.mean(grid,axis=1)[:,1]==unique_y[iy-1])  # neighbour 8 down left
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -diagonal_factor
						neighbours += diagonal_factor
				except:
					pass

		grid_laplacian[ith_cell, ith_cell] = neighbours
	return grid_laplacian

def build_Z_derivate(grid):
	# Try making grid Z direction derivate matrix for spatial regularisation
	num_cells = len(grid)
	grid_laplacian = np.zeros((num_cells, num_cells))
	unique_x = np.unique(np.mean(grid,axis=1)[:,0])
	unique_y = np.unique(np.mean(grid,axis=1)[:,1])

	for ith_cell in range(num_cells):

		# get the 2D mesh coordinates of this cell
		ix = np.abs(unique_x - np.mean(grid,axis=1)[ith_cell,0]).argmin()	# radious
		iy = np.abs(unique_y - np.mean(grid,axis=1)[ith_cell,1]).argmin()	# Z

		neighbours = 0

		try:
			select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix],np.mean(grid,axis=1)[:,1]==unique_y[iy+1])  # neighbour 3 top
			if np.sum(select)>0:
				n1 = select.argmax()
				grid_laplacian[ith_cell, n1] = 1
				neighbours += 1
		except:
			pass

		if iy>0:
			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix],np.mean(grid,axis=1)[:,1]==unique_y[iy-1])  # neighbour 7 down
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = -1
					neighbours += 1
			except:
				pass

		if neighbours==2:
			grid_laplacian[ith_cell, ith_cell] = 0
			grid_laplacian[ith_cell] /=2
		elif neighbours==1:
			grid_laplacian[ith_cell, ith_cell] = 1
		else:
			grid_laplacian[ith_cell, ith_cell] = 0

	return grid_laplacian

def build_R_derivate(grid):
	# Try making grid R direction derivate matrix for spatial regularisation
	num_cells = len(grid)
	grid_laplacian = np.zeros((num_cells, num_cells))
	unique_x = np.unique(np.mean(grid,axis=1)[:,0])
	unique_y = np.unique(np.mean(grid,axis=1)[:,1])

	for ith_cell in range(num_cells):

		# get the 2D mesh coordinates of this cell
		ix = np.abs(unique_x - np.mean(grid,axis=1)[ith_cell,0]).argmin()	# radious
		iy = np.abs(unique_y - np.mean(grid,axis=1)[ith_cell,1]).argmin()	# Z

		neighbours = 0

		if ix>0:
			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix-1],np.mean(grid,axis=1)[:,1]==unique_y[iy])  # neighbour 1 left
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = -1
					neighbours += 1
			except KeyError:
				pass

		try:
			select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix+1],np.mean(grid,axis=1)[:,1]==unique_y[iy])  # neighbour 5 right
			if np.sum(select)>0:
				n1 = select.argmax()
				grid_laplacian[ith_cell, n1] = 1
				neighbours += 1
		except:
			pass

		if neighbours==2:
			grid_laplacian[ith_cell, ith_cell] = 0
			grid_laplacian[ith_cell] /=2
		elif neighbours==1:
			grid_laplacian[ith_cell, ith_cell] = 1
		else:
			grid_laplacian[ith_cell, ith_cell] = 0
	return grid_laplacian

def reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,std_treshold = 1e-4, sum_treshold = 0.000, core_radious_treshold = 1.9, divertor_radious_treshold = 1.9,chop_top_corner = False, extra_chop_top_corner = False , chop_corner_close_to_baffle = False, restrict_polygon = FULL_MASTU_CORE_GRID_POLYGON):
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	# masking the voxels whose emission does not reach the foil
	# select = np.sum(sensitivities_reshaped,axis=(0,1))>0.05
	select = np.logical_and(np.std(sensitivities_reshaped,axis=(0,1))>std_treshold*(np.std(sensitivities_reshaped,axis=(0,1)).max()),np.sum(sensitivities_reshaped,axis=(0,1))>sum_treshold)
	grid_data_masked = grid_data[select]
	sensitivities_reshaped_masked = sensitivities_reshaped[:,:,select]

	# this is not enough because the voxels close to the pinhole have a too large influence on it and the inversion is weird
	# select = np.median(sensitivities_reshaped_masked,axis=(0,1))<5*np.mean(np.median(sensitivities_reshaped_masked,axis=(0,1)))
	select = np.logical_or(np.mean(grid_data_masked,axis=1)[:,0]<core_radious_treshold,np.mean(grid_data_masked,axis=1)[:,1]<-1.3)
	if chop_top_corner:
		x1 = [1.1,0.6]	# r,z
		x2 = [1.6,0.0]
		interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
		select = np.logical_and(select,np.mean(grid_data_masked,axis=1)[:,1]<interp(np.mean(grid_data_masked,axis=1)[:,0]))
	if extra_chop_top_corner:
		x1 = [1.1,0.3]	# r,z
		x2 = [1.4,-0.05]
		interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
		select = np.logical_and(select,np.mean(grid_data_masked,axis=1)[:,1]<interp(np.mean(grid_data_masked,axis=1)[:,0]))
	if chop_corner_close_to_baffle:
		x1 = [1.33,-0.9]	# r,z
		x2 = [1.5,-1.03]	# r,z
		x3 = [1.2,-1.07]	# r,z
		interp1 = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
		interp2 = interp1d([x1[0],x3[0]],[x1[1],x3[1]],fill_value="extrapolate",bounds_error=False)
		# select2 = np.logical_or(np.logical_or(np.mean(grid_data_masked,axis=1)[:,1]<-1.1,np.mean(grid_data_masked,axis=1)[:,1]>interp1(np.mean(grid_data_masked,axis=1)[:,0])),np.mean(grid_data_masked,axis=1)[:,1]>interp2(np.mean(grid_data_masked,axis=1)[:,0]))
		x4 = [1.4,-0.55]	# r,z
		interp3 = interp1d([x4[0],x3[0]],[x4[1],x3[1]],fill_value="extrapolate",bounds_error=False)
		select2 = np.logical_or(np.mean(grid_data_masked,axis=1)[:,1]<-1.1,np.mean(grid_data_masked,axis=1)[:,1]>interp3(np.mean(grid_data_masked,axis=1)[:,0]))
		select = np.logical_and(select,select2)
	if len(restrict_polygon)>0:
		polygon = Polygon(restrict_polygon)
		for i_e in range(len(grid_data_masked)):
			if np.sum([polygon.contains(Point((grid_data_masked[i_e][i__e,0],grid_data_masked[i_e][i__e,1]))) for i__e in range(4)])==0:
				select[i_e] = False
	grid_data_masked = grid_data_masked[select]
	sensitivities_reshaped_masked = sensitivities_reshaped_masked[:,:,select]
	select = np.logical_or(np.mean(grid_data_masked,axis=1)[:,0]<divertor_radious_treshold,np.mean(grid_data_masked,axis=1)[:,1]>-1.3)
	grid_data_masked = grid_data_masked[select]
	sensitivities_reshaped_masked = sensitivities_reshaped_masked[:,:,select]
	grid_laplacian_masked = build_laplacian(grid_data_masked)
	grid_Z_derivate_masked = build_Z_derivate(grid_data_masked)
	grid_R_derivate_masked = build_R_derivate(grid_data_masked)


	return sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked



def retrive_vessel_average_temp_archve(shot):
	import csv
	from datetime import datetime
	# import pyuda
	client_int=pyuda.Client()

	filename = '/home/ffederic/work/irvb/MAST-U/MU01_VesselTemp_Raw_2021.csv'
	csvfile=open(filename,'r')
	reader = csv.reader(csvfile)
	read = np.array(list(reader))

	date = client_int.get_shot_date_time(shot)[0]+', '+client_int.get_shot_date_time(shot)[1][:8]
	date_format = datetime.strptime(date,"%Y-%m-%d, %H:%M:%S")
	date_unix = datetime.timestamp(date_format)

	index = np.abs(read[1:].astype(int)[:,0]-date_unix).argmin()
	temperature = int(read[1:][index,1])
	reset_connection(client_int)
	del client_int

	return temperature
