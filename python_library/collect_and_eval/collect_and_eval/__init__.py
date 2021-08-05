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
	#     return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data[0])
	frames[0]=data[0,0]

	for i in range(len(data[0])):
	    # x       += 1
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
Rf=1.54967	# m
plane_equation = np.array([1,-1,0,2**0.5 * Rf])
pinhole_location = np.array([-1.04087,1.068856,-0.7198])
centre_of_foil = np.array([-1.095782166, 1.095782166, -0.7])
foil_size = [0.07,0.09]
R_centre_column = 0.261	# m

def point_toroidal_to_cartesian(coords):	# r,z,teta deg	to	x,y,z
	out = np.zeros_like(coords).astype(float)
	out.T[0]=coords.T[0] * np.cos(coords.T[2]*2*np.pi/360)
	out.T[1]=coords.T[0] * np.sin(coords.T[2]*2*np.pi/360)
	out.T[2]=coords.T[1]
	return out

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

def return_structure_point_location_on_foil():
	return structure_point_location_on_foil

fueling_point_location_on_foil = []
for time in range(len(fueling_r)):
	point_location = np.array([fueling_r[time],fueling_z[time],fueling_t[time]]).T
	point_location = point_toroidal_to_cartesian(point_location)
	point_location = find_location_on_foil(point_location)
	fueling_point_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location))

def return_fueling_point_location_on_foil():
	return fueling_point_location_on_foil

def return_all_time_x_point_location(efit_reconstruction,resolution = 1000):
	all_time_x_point_location = []
	for time in range(len(efit_reconstruction.time)):
		x_point_location = np.array([[efit_reconstruction.lower_xpoint_r[time]]*resolution,[efit_reconstruction.lower_xpoint_z[time]]*resolution,np.linspace(0,360,resolution)]).T
		x_point_location = point_toroidal_to_cartesian(x_point_location)
		x_point_location = find_location_on_foil(x_point_location)
		all_time_x_point_location.append(absolute_position_on_foil_to_foil_coord(x_point_location))
	all_time_x_point_location = np.array(all_time_x_point_location)
	return all_time_x_point_location

def return_all_time_mag_axis_location(efit_reconstruction,resolution = 1000):
	all_time_mag_axis_location = []
	for time in range(len(efit_reconstruction.time)):
		mag_axis_location = np.array([[efit_reconstruction.mag_axis_r[time]]*resolution,[efit_reconstruction.mag_axis_z[time]]*resolution,np.linspace(0,360,resolution)]).T
		mag_axis_location = point_toroidal_to_cartesian(mag_axis_location)
		mag_axis_location = find_location_on_foil(mag_axis_location)
		all_time_mag_axis_location.append(absolute_position_on_foil_to_foil_coord(mag_axis_location))
	all_time_mag_axis_location = np.array(all_time_mag_axis_location)
	return all_time_mag_axis_location

def return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,resolution = 1000):
	from scipy.signal import find_peaks, peak_prominences as get_proms
	temp_R = np.ones((len(efit_reconstruction.time),20))*np.nan
	temp_Z = np.ones((len(efit_reconstruction.time),20))*np.nan
	for time in range(len(efit_reconstruction.time)):
		temp = np.array([efit_reconstruction.strikepointR[time],efit_reconstruction.strikepointZ[time]]).T
		if temp.max()<1e-1:
			a=np.concatenate([r_fine[all_time_sep_r[time][0]],r_fine[all_time_sep_r[time][2]]])
			b=np.concatenate([z_fine[all_time_sep_z[time][0]],z_fine[all_time_sep_z[time][2]]])
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
		strike_point_location = find_location_on_foil(strike_point_location)
		all_time_strike_points_location.append(absolute_position_on_foil_to_foil_coord(strike_point_location))
	all_time_strike_points_location_rot = []
	for time in range(len(efit_reconstruction.time)):
		temp = []
		for __i in range(len(temp_R[time])):
			# strike_point_location = np.array([[efit_reconstruction.strikepointR[time][__i]]*resolution,[-np.abs(efit_reconstruction.strikepointZ[time][__i])]*resolution,np.linspace(0,360,resolution)]).T
			strike_point_location = np.array([[temp_R[time][__i]]*resolution,[temp_Z[time][__i]]*resolution,np.linspace(0,360,resolution)]).T
			strike_point_location = point_toroidal_to_cartesian(strike_point_location)
			strike_point_location = find_location_on_foil(strike_point_location)
			temp.append(absolute_position_on_foil_to_foil_coord(strike_point_location))
		all_time_strike_points_location_rot.append(temp)
	return all_time_strike_points_location,all_time_strike_points_location_rot

def return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,ref_angle=60):
	all_time_separatrix = []
	for time in range(len(efit_reconstruction.time)):
		separatrix = []
		for i in range(len(all_time_sep_r[time])):
			point_location = np.array([r_fine[all_time_sep_r[time][i]],z_fine[all_time_sep_z[time][i]],[ref_angle]*len(all_time_sep_z[time][i])]).T
			point_location = point_toroidal_to_cartesian(point_location)
			point_location = find_location_on_foil(point_location)
			separatrix.append(absolute_position_on_foil_to_foil_coord(point_location))
		all_time_separatrix.append(separatrix)
	return all_time_separatrix

def return_core_tangential_location_on_foil(resolution = 10000):
	core_tangential_location_on_foil = []
	for i in range(len(core_tangential_arrival)):
		point_location = np.array([np.linspace(core_tangential_arrival[i][0],core_tangential_common_point[0],resolution),np.linspace(core_tangential_arrival[i][1],core_tangential_common_point[1],resolution),[core_tangential_arrival[i][2]]*resolution]).T
		point_location = find_location_on_foil(point_location)
		core_tangential_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location))
	return core_tangential_location_on_foil

def return_core_poloidal_location_on_foil(angle=60,resolution = 10000):
	core_poloidal_location_on_foil = []
	for i in range(len(core_poloidal_arrival)):
		point_location = np.array([np.linspace(core_poloidal_arrival[i][0],core_poloidal_common_point[0],resolution),np.linspace(core_poloidal_arrival[i][1],core_poloidal_common_point[1],resolution),[angle]*resolution]).T
		point_location = point_toroidal_to_cartesian(point_location)
		point_location = find_location_on_foil(point_location)
		core_poloidal_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location))
	return core_poloidal_location_on_foil

def return_divertor_poloidal_location_on_foil(angle=60,resolution = 10000):
	divertor_poloidal_location_on_foil = []
	for i in range(len(divertor_poloidal_arrival)):
		point_location = np.array([np.linspace(divertor_poloidal_arrival[i][0],divertor_poloidal_common_point[0],resolution),np.linspace(divertor_poloidal_arrival[i][1],divertor_poloidal_common_point[1],resolution),[angle]*resolution]).T
		point_location = point_toroidal_to_cartesian(point_location)
		point_location = find_location_on_foil(point_location)
		divertor_poloidal_location_on_foil.append(absolute_position_on_foil_to_foil_coord(point_location))
	return divertor_poloidal_location_on_foil

def movie_from_data(data,framerate,integration=1,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',timesteps='auto',extvmin='auto',extvmax='auto',mask=[0],mask_alpha=0.2,time_offset=0,prelude='',vline=None,hline=None,EFIT_path=EFIT_path_default,include_EFIT=False,pulse_ID=None,overlay_x_point=False,overlay_mag_axis=False,overlay_structure=False,overlay_strike_points=False,overlay_separatrix=False,structure_alpha=0.5,foil_size=foil_size):
	import matplotlib.animation as animation
	import numpy as np

	form_factor = (np.shape(data[0][0])[1])/(np.shape(data[0][0])[0])
	fig = plt.figure(figsize=(12*form_factor, 12))
	ax = fig.add_subplot(111)

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#     return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data[0])
	frames[0]=data[0,0]

	for i in range(len(data[0])):
	    # x       += 1
	    # curVals  = f(x, y)
	    frames[i]=(data[0,i])

	cv0 = frames[0]
	im = ax.imshow(cv0,cmap, origin='lower', interpolation='none') # Here make an AxesImage rather than contour

	if len(np.shape(mask))==2:
		if np.shape(mask)!=np.shape(data[0][0]):
			print('The shape of the mask '+str(np.shape(mask))+' does not match with the shape of the record '+str(np.shape(data[0][0])))
		masked = np.ma.masked_where(mask == 0, mask)
		im2 = ax.imshow(masked, 'gray', interpolation='none', alpha=mask_alpha,origin='lower',extent = [0,np.shape(cv0)[1]-1,0,np.shape(cv0)[0]-1])

	if include_EFIT:
		try:
			print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc')
			efit_reconstruction = mclass(EFIT_path+'/epm0'+str(pulse_ID)+'.nc')
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


	# if timesteps=='auto':
	# 	timesteps_int = time_offset+np.arange(len(data[0])+1)/framerate
	# else:
	# 	timesteps_int = cp.deepcopy(timesteps)
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

	return ani


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
		if filenames[index][-3:]=='npy':
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


def find_dead_pixels(data, start_interval='auto',end_interval='auto', framerate='auto',from_data=True,treshold_for_bad_low_std=0,treshold_for_bad_std=10,treshold_for_bad_difference=13):

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
						if (mean[i, j] > max(temp2) + treshold_for_bad_difference or mean[i, j] < min(
								temp2) - treshold_for_bad_difference):
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


	counter = collections.Counter(flatten_full(flag_check))
	print('Number of pixels that trepass '+str(treshold_for_bad_difference)+' counts difference with neighbours: '+str(counter.get(3)))
	print('Number of pixels with standard deviation > '+str(treshold_for_bad_std)+' counts: '+str(counter.get(6)))
	print('Number of pixels that trepass both limits: '+str(counter.get(9)))

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
		print('There is something wrong, the shape of the data and the dead pixels map you want to use are not the same')
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

def ptw_to_dict(full_path):
	os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
	import pyradi.ryptw as ryptw
	import datetime

	header = ryptw.readPTWHeader(full_path)
	width = header.h_Cols
	height = header.h_Rows
	camera_SN = header.h_CameraSerialNumber
	NUCpresetUsed = header.h_NucTable
	FrameRate = 1/header.h_CEDIPAquisitionPeriod # Hz
	IntegrationTime = round(header.h_CEDIPIntegrationTime*1e6,0) # microseconds
	ExternalTrigger = None	# I couldn't find this signal in the header

	digitizer_ID = []
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
		data.append(np.flip(frame.flatten().reshape(new_shape).T,axis=0))
		frame_header = ryptw.getPTWFrames(header, [i])[1][0]
		digitizer_ID.append(frame_header.h_framepointer%2)	# I couldn't find this so as a proxy, given the digitisers are always alternated, I only use if the frame is even or odd
		# yyyy = frame_header.h_FileSaveYear
		# mm = frame_header.h_FileSaveMonth
		# dd = frame_header.h_FileSaveDay
		hh = frame_header.h_frameHour
		minutes = frame_header.h_frameMinute
		ss_sss = frame_header.h_frameSecond
		time_of_measurement.append((datetime.datetime(frame_header.h_FileSaveYear-30,frame_header.h_FileSaveMonth,frame_header.h_FileSaveDay).timestamp() + hh*60*60 + minutes*60 + ss_sss)*1e6)	# I leave if as a true timestamp, t=0 is 1970
		frame_counter.append(None)	# I couldn't fine this in the header
		DetectorTemp.append(frame_header.h_detectorTemp)
		SensorTemp_0.append(frame_header.h_detectorTemp)	# this data is missing so I use the more similar again
		SensorTemp_3.append(frame_header.h_sensorTemp4)
		AtmosphereTemp.append(frame_header.h_AtmosphereTemp)	# additional data not present in the .ats format
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


def build_poly_coeff_multi_digitizer(temperature,files,int,path,n):
	# modified 2018-10-08 to build the coefficient only for 1 degree of polinomial
	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)
	meancounttot=[]
	meancountstdtot=[]
	all_SensorTemp_0 = []
	all_DetectorTemp = []
	all_frame_counter = []
	all_time_of_measurement = []
	for i_file,file in enumerate(files):
		full_saved_file_dict=np.load(file+'.npz')
		data_per_digitizer,uniques_digitizer_ID = separate_data_with_digitizer(full_saved_file_dict)
		if i_file==0:
			digitizer_ID = np.array(uniques_digitizer_ID)
		if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
			print('ERROR: problem with the ID of the digitizer in \n' + file)
			exit()
		meancounttot.append([np.mean(x,axis=0) for x in data_per_digitizer])
		meancountstdtot.append([np.std(x,axis=0) for x in data_per_digitizer])
		all_SensorTemp_0.append(np.mean(full_saved_file_dict['SensorTemp_0']))
		all_DetectorTemp.append(np.mean(full_saved_file_dict['DetectorTemp']))
		all_time_of_measurement.append(np.mean(full_saved_file_dict['time_of_measurement']))
		all_frame_counter.append(np.mean(full_saved_file_dict['frame_counter']))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)
	shapex=np.shape(meancounttot)[-2]
	shapey=np.shape(meancounttot)[-1]
	score=np.zeros((len(digitizer_ID),shapex,shapey))

	# WARNING; THIS CREATE COEFFICIENTS INCOMPATIBLE WITH PREVIOUS build_poly_coeff FUNCTION
	coeff=np.zeros((len(digitizer_ID),shapex,shapey,n))
	errcoeff=np.zeros((len(digitizer_ID),shapex,shapey,n,n))

	for j in range(shapex):
		for k in range(shapey):
			for i_z,z in enumerate(digitizer_ID):
				x=np.array(meancounttot[:,z==digitizer_ID,j,k]).flatten()
				xerr=np.array(meancountstdtot[:,z==digitizer_ID,j,k]).flatten()
				temp1,temp2=np.polyfit(temperature,x,n-1,cov='unscaled')
				yerr=(np.polyval(temp1,x+xerr)-np.polyval(temp1,x-xerr))/2
				temp1,temp2=np.polyfit(x,temperature,n-1,w=1/yerr,cov='unscaled')
				# plt.figure()
				# plt.errorbar(x,temperature,xerr=xerr)
				# plt.plot(x,np.polyval(temp1,x),'--')
				# plt.pause(0.01)
				coeff[i_z,j,k,:]=temp1
				errcoeff[i_z,j,k,:]=temp2
				score[i_z,j,k]=rsquared(temperature,np.polyval(temp1,x))
	np.savez_compressed(os.path.join(path,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int)+'ms'),**dict([('coeff',coeff),('errcoeff',errcoeff),('score',score)]))
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
				# this is approximate because it does not account properly for the error matrix but only the diagonal, but it's massively faster
				temp2 = (errparams[i][:,:,2,2] + (counts_temp**2)*errparams[i][:,:,1,1] + (counts_temp**4)*errparams[i][:,:,0,0])**0.5
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

import netCDF4
from . import efitData

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from scipy import interpolate

class mclass:


	def __init__(self,path):

		f = netCDF4.Dataset(path)

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
					if len(zcr) < 2:	# made to prevent the error when there is only one zero in (mp_p_arr-self.psi_bnd[i])
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
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def proper_homo_binning_t_2D(data,shrink_factor_t,shrink_factor_x,type='mean'):
	old_shape = np.array(np.shape(data))
	new_shape=np.array([int(np.ceil(old_shape[0]/shrink_factor_t)),int(np.ceil(old_shape[1]/shrink_factor_x)),int(np.ceil(old_shape[2]/shrink_factor_x))]).astype(int)
	to_pad=np.array([(shrink_factor_t-old_shape[0]%shrink_factor_t)*(old_shape[0]%shrink_factor_t>0),(shrink_factor_x-old_shape[1]%shrink_factor_x)*(old_shape[1]%shrink_factor_x>0),(shrink_factor_x-old_shape[2]%shrink_factor_x)*(old_shape[2]%shrink_factor_x>0)]).astype(int)
	to_pad_right = to_pad//2
	to_pad_left = to_pad - to_pad_right
	to_pad = np.array([to_pad_left,to_pad_right]).T
	data_binned = np.pad(data,to_pad,mode='mean',stat_length=((max(1,shrink_factor_t//2),max(1,shrink_factor_t//2)),(max(1,shrink_factor_x//2),max(1,shrink_factor_x//2)),(max(1,shrink_factor_x//2),max(1,shrink_factor_x//2))))
	data_binned = bin_ndarray(data_binned, new_shape=new_shape, operation=type)
	nan_ROI_mask = np.isfinite(np.nanmedian(data_binned[:10],axis=0))
	return data_binned,nan_ROI_mask

def proper_homo_binning_2D(data,shrink_factor_x,type='mean'):
	old_shape = np.array(np.shape(data))
	new_shape=np.array([int(np.ceil(old_shape[0]/shrink_factor_x)),int(np.ceil(old_shape[1]/shrink_factor_x))]).astype(int)
	to_pad=np.array([(shrink_factor_x-old_shape[0]%shrink_factor_x)*(old_shape[0]%shrink_factor_x>0),(shrink_factor_x-old_shape[1]%shrink_factor_x)*(old_shape[1]%shrink_factor_x>0)]).astype(int)
	to_pad_right = to_pad//2
	to_pad_left = to_pad - to_pad_right
	to_pad = np.array([to_pad_left,to_pad_right]).T
	data_binned = np.pad(data,to_pad,mode='mean',stat_length=((max(1,shrink_factor_x//2),max(1,shrink_factor_x//2)),(max(1,shrink_factor_x//2),max(1,shrink_factor_x//2))))
	data_binned = bin_ndarray(data_binned, new_shape=new_shape, operation=type)
	return data_binned

def proper_homo_binning_t(time,shrink_factor_t,type='mean'):
	old_shape = np.array(np.shape(time))
	new_shape=np.array([int(np.ceil(old_shape[0]/shrink_factor_t))]).astype(int)
	to_pad=np.array([(shrink_factor_t-old_shape[0]%shrink_factor_t)*(old_shape[0]%shrink_factor_t>0)]).astype(int)
	to_pad_right = to_pad//2
	to_pad_left = to_pad - to_pad_right
	to_pad = np.array([to_pad_left,to_pad_right]).T
	time_binned = np.pad(time,to_pad,mode='mean',stat_length=((max(1,shrink_factor_t//2),max(1,shrink_factor_t//2))))
	time_binned = bin_ndarray(time_binned, new_shape=new_shape, operation=type)
	time_binned[0] = time_binned[1] - np.median(np.diff(time_binned[1:-1]))
	time_binned[-1] = time_binned[-2] + np.median(np.diff(time_binned[1:-1]))
	return time_binned

def efit_reconstruction_to_separatrix_on_foil(efit_reconstruction,refinement=1000):
	from scipy.signal import find_peaks, peak_prominences as get_proms
	from scipy.interpolate import interp2d
	all_time_sep_r = []
	all_time_sep_z = []
	r_fine = np.unique(np.linspace(efit_reconstruction.R.min(),efit_reconstruction.R.max(),refinement).tolist() + np.linspace(R_centre_column-0.01,R_centre_column+0.08,refinement).tolist())
	z_fine = np.linspace(efit_reconstruction.Z.min(),efit_reconstruction.Z.max(),refinement)
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
		for i_z,z in enumerate(z_fine):
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
		# plt.plot(all_z[found_points<(gna.max()-gna.min())/500],all_peaks[found_points<(gna.max()-gna.min())/500],'+b')
		all_peaks_up = all_peaks_up[found_psi_up<(gna.max()-gna.min())/500]
		all_z_up = all_z_up[found_psi_up<(gna.max()-gna.min())/500]
		all_peaks_low = all_peaks_low[found_psi_low<(gna.max()-gna.min())/500]
		all_z_low = all_z_low[found_psi_low<(gna.max()-gna.min())/500]

		left_up = []
		right_up = []
		left_up_z = []
		right_up_z = []
		left_low = []
		right_low = []
		left_low_z = []
		right_low_z = []
		for i_z,z in enumerate(z_fine):
			if i_z in all_z_up:
				temp = all_peaks_up[all_z_up==i_z]
				if len(temp) == 1:
					right_up.append(temp[0])
					right_up_z.append(i_z)
				elif len(temp) == 2:
					# # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
					# if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
					left_up.append(temp.min())
					left_up_z.append(i_z)
					right_up.append(temp.max())
					right_up_z.append(i_z)
				elif len(temp) == 3:
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
					left_low.append(temp.min())
					left_low_z.append(i_z)
					right_low.append(temp.max())
					right_low_z.append(i_z)
				elif len(temp) == 3:
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
	foilvertwpixel=int((foilhorizwpixel*foilvertw)//foilhorizw)
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
	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
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
	foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
	temp_ref_counts = []
	temp_counts_minus_background = []
	time_partial = []
	timesteps = np.inf
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		if external_clock_marker:
			time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers
		if flag_use_of_first_frames_as_reference:
			temp_ref_counts.append(np.mean(laser_counts_corrected[i][time_of_experiment_digitizer_ID_seconds<0],axis=0))
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
	ani = movie_from_data(np.array([np.flip(np.transpose(FAST_counts_minus_background_crop,(0,2,1)),axis=2)]), laser_framerate/len(laser_digitizer_ID),integration=laser_int_time/1000,time_offset=FAST_counts_minus_background_crop_time[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
	ani.save(laser_to_analyse[:-4]+ '_FAST_count_increase.mp4', fps=5*laser_framerate/len(laser_digitizer_ID)/383, writer='ffmpeg',codec='mpeg4')
	plt.close('all')

	print('completed FAST rotating/cropping ' + laser_to_analyse)

	return foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop,FAST_counts_minus_background_crop_time
