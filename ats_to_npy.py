# Created 03/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())


# Header for every frame in a .ats file
# separator in HEX = 49 49
# unknown int = 3	# int(8)
# unknown int = 3	# int(8)
# time in ns from camera startup = 0	# int(8)
# unknown int = 0	# int(8)
# Frame taken from the camera startup = 1	# int(8)
# unknown int = 0	# int(8)
# unknown flag = 1	# bool (2)
# unknown flag = 0	# bool (2)
# unknown flag = 1	# bool (2)
# unknown flag = 0	# bool (2)
# unknown flag = 1	# bool (2)
# unknown flag = 0	# bool (2)
# unknown flag = 1	# bool (2)
# unknown flag = 0	# bool (2)
# DetectorTemp = 68.3792	# float(8)
# SensorTemp[0] = 302.212	# float(8)
# SensorTemp[1] = 0	# float(8)
# SensorTemp[2] = 0	# float(8)
# SensorTemp[3] = 68.395	# float(8)
# MasterClock = 1e+07	# float(8)
# IntegrationTime = 1000	# float(8)
# FrameRate = 50	# float(8)
# unknown flag	# bool (2)
# ExternalTrigger = False	# bool (2)
# unknown flag	# bool (2)
# PageIndex = 0	# int(8)



# path = '/home/ffederic/work/irvb/FLIR 17-11-2020'
# fileats = 'irvb_50hz_external-000012.ats'

def hex8_to_int(hex):
	temp = hex[-2:]+hex[4:6]+hex[2:4]+hex[0:2]
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

# this script returns all the properties even if only one is requested
# def FLIR_frame_header_decomposition(header):
# 	# trash = header[:8]
# 	# trash = header[8:16]
# 	time = hex8_to_int(header[16:24])	# ns
# 	# trash = header[24:32]
# 	frame_counter = hex8_to_int(header[32:40])
# 	# trash = header[40:48]
# 	# trash = header[48:50]
# 	# trash = header[50:52]
# 	# trash = header[52:54]
# 	# trash = header[54:56]
# 	# trash = header[56:58]
# 	# trash = header[58:60]
# 	# trash = header[60:62]
# 	# trash = header[62:64]
# 	DetectorTemp = hex8_to_float(header[64:72])	# K
# 	SensorTemp_0 = hex8_to_float(header[72:80])	# K
# 	SensorTemp_1 = hex8_to_float(header[80:88])	# K
# 	SensorTemp_2 = hex8_to_float(header[88:96])	# K
# 	SensorTemp_3 = hex8_to_float(header[96:104])	# K
# 	MasterClock = hex8_to_float(header[104:112])	# unknown
# 	IntegrationTime = hex8_to_float(header[112:120])	# ns
# 	FrameRate = hex8_to_float(header[120:128])	# Hz
# 	# trash = header[128:130]
# 	ExternalTrigger = int(header[130:132])
# 	# trash = header[132:134]
# 	PageIndex = hex8_to_int(header[134:142])	# index of the digitiser used
# 	return dict([('time',time),('frame_counter',frame_counter),('DetectorTemp',DetectorTemp),('SensorTemp_0',SensorTemp_0),('SensorTemp_1',SensorTemp_1),('SensorTemp_2',SensorTemp_2),('SensorTemp_3',SensorTemp_3),('MasterClock',MasterClock),('IntegrationTime',IntegrationTime),('FrameRate',FrameRate),('ExternalTrigger',ExternalTrigger),('PageIndex',PageIndex)])

def FLIR_frame_header_decomposition(header):
	def return_requested_output(request):
		if request=='time':
			return hex8_to_int(header[16:24])	# ns
		elif request=='frame_counter':
			return hex8_to_int(header[32:40])
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
			return hex8_to_int(header[134:142])	# index of the digitiser used
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


def ats_to_dict(path,fileats,digital_level_bytes=4,header_marker = '4949'):
	data = open(os.path.join(path,fileats),'rb').read()
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
	last = 0
	import time as tm
	# value = data.find(string_for_digitizer)
	value = hexdata.find(header_marker)
	# last+=value+len(hex_for_digitizer)	# the first one is part of the neader of the whole file
	# value = hexdata[last:].find(hex_for_digitizer)
	while len(hexdata)-last>header_length:
		# start_time = tm.time()
		# print(len(time_of_measurement))
		header = hexdata[last+value+length_header_marker:last+value+header_length+length_header_marker]
		header = FLIR_frame_header_decomposition(header)
		digitizer_ID.append(header('PageIndex'))
		time_of_measurement.append(header('time'))
		frame_counter.append(header('frame_counter'))
		DetectorTemp.append(header('DetectorTemp'))
		SensorTemp_0.append(header('SensorTemp_0'))
		# time_lapsed = tm.time()-start_time
		# print(time_lapsed)
		raw_digital_level = hexdata[last+value-data_length:last+value]
		# time_lapsed = tm.time()-start_time-time_lapsed
		# print(time_lapsed)
		data.append(raw_to_image(raw_digital_level,width,height,digital_level_bytes))
		# time_lapsed = tm.time()-start_time-time_lapsed
		# print(time_lapsed)
		last+=value+header_length+data_length
		if len(time_of_measurement)<=1:	# the spacing between separators seems constant, and take very long, so I do it once
			value = hexdata[last:].find(header_marker)
			IntegrationTime = header('IntegrationTime')
			FrameRate = header('FrameRate')
			ExternalTrigger = header('ExternalTrigger')
		# print(value)
	data = np.array(data)
	digitizer_ID = np.array(digitizer_ID)
	time_of_measurement = np.array(time_of_measurement)
	frame_counter = np.array(frame_counter)
	DetectorTemp = np.array(DetectorTemp)
	SensorTemp_0 = np.array(SensorTemp_0)
	# return data,digitizer_ID,time_of_measurement,IntegrationTime,FrameRate,ExternalTrigger,SensorTemp_0,DetectorTemp,width,height,camera_SN,frame_counter
	return dict([('data',data),('digitizer_ID',digitizer_ID),('time_of_measurement',time_of_measurement),('IntegrationTime',IntegrationTime),('FrameRate',FrameRate),('ExternalTrigger',ExternalTrigger),('SensorTemp_0',SensorTemp_0),('DetectorTemp',DetectorTemp),('width',width),('height',height),('camera_SN',camera_SN),('frame_counter',frame_counter)])

# plt.figure(),plt.imshow(data[1]),plt.pause(0.01)

path = '/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018'
fileats = 'irvb_sample-000001.ats'
out = ats_to_dict(path,fileats)

data = out['data']
digitizer_ID = out['digitizer_ID']
time_of_measurement = out['time_of_measurement']

data_digitizer_0 = data[digitizer_ID==0]
time_of_measurement_digitizer_0 = time_of_measurement[digitizer_ID==0]
data_digitizer_1 = data[digitizer_ID==1]
time_of_measurement_digitizer_1 = time_of_measurement[digitizer_ID==1]


spectra = np.fft.fft(np.array(data_digitizer_0),axis=0)
magnitude = 2 * np.abs(spectra) / len(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1e-6*np.median(np.diff(time_of_measurement_digitizer_0)))
plt.figure(),plt.plot(freq,magnitude[:,100,100]),plt.semilogy(),plt.pause(0.01)

spectra = np.fft.fft(np.array(data_digitizer_1),axis=0)
magnitude = 2 * np.abs(spectra) / len(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1e-6*np.median(np.diff(time_of_measurement_digitizer_1)))
plt.plot(freq,magnitude[:,100,100]),plt.semilogy(),plt.pause(0.01)

spectra = np.fft.fft(np.array(data),axis=0)
magnitude = 2 * np.abs(spectra) / len(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1e-6*np.median(np.diff(time_of_measurement)))
plt.plot(freq,magnitude[:,100,100],'--'),plt.semilogy(),plt.pause(0.01)

plt.figure(),plt.imshow(np.mean(data_digitizer_0,axis=0),'rainbow'),plt.colorbar(),plt.pause(0.01)
plt.figure(),plt.imshow(np.mean(data_digitizer_1,axis=0),'rainbow'),plt.colorbar(),plt.pause(0.01)
temp = np.mean(data_digitizer_0,axis=0)-np.mean(data_digitizer_1,axis=0)
flag = np.abs(temp)>10
temp[np.not(flag)]=np.nan
plt.figure(),plt.imshow(temp,'rainbow',origin='lower'),plt.colorbar(),plt.pause(0.01)

ani = coleval.movie_from_data(np.array([data]), 1/(1e-6*np.median(np.diff(time_of_measurement))), 1000, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]')#,extvmin=90,extvmax=np.min([np.max(np.array(data_all)[:30]),np.max(np.array(data_all)[-70:])]))
plt.show()

# data2 = np.array(data,dtype=float)
# data2[:,flag]=np.mean(data2)
data2 = coleval.replace_dead_pixels([data],flag)[0]
ani = coleval.movie_from_data(np.array([data2]), 1/(1e-6*np.median(np.diff(time_of_measurement))), 1000, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]')#,extvmin=90,extvmax=np.min([np.max(np.array(data_all)[:30]),np.max(np.array(data_all)[-70:])]))
plt.show()


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


from astropy.io import fits

datafit=fits.open(os.path.join(path,filefits[4]))
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

plt.figure(),plt.imshow(data[0,0]),plt.pause(0.01)



# The two images are the same, so I can use directly the .ats file now
