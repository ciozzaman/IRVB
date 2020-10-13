# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())










# 2018-10-10 I want to explore better the oscillation in the counts






'''




pathfiles = vacuum2[23]
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '.npy'
filenames = coleval.all_file_names(pathfiles, type)[0]

data = np.load(os.path.join(pathfiles, filenames))[0]

# real=np.zeros(np.shape(data))
# imag=np.zeros(np.shape(data))
# phase=np.zeros(np.shape(data))
# for i in range(np.shape(data)[1]):
# 	for j in range(np.shape(data)[2]):
# 		temp=np.fft.fft(data[:,i,j])
# 		real[:,i,j]=np.real(temp)
# 		imag[:,i,j]=np.imag(temp)
# 		phase[:,i,j]=np.angle(temp)
spectra = np.fft.fft(data, axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra) / len(spectra)
phase = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)


samplefreq = 243
plt.title('Normalised amplitude of ' + str(
	freq[samplefreq]) + 'Hz oscillation from fast Fourier transform in counts in \n ' + pathfiles + ' framerate ' + str(
	framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.imshow(np.divide(magnitude[samplefreq], np.shape(magnitude)[0]), 'rainbow', origin='lower')
plt.colorbar().set_label('Amplitude [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
# plt.show()
plt.figure()
plt.title('Phase of ' + str(
	freq[samplefreq]) + 'Hz oscillation from fast Fourier transform in counts in \n ' + pathfiles + ' framerate ' + str(
	framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.imshow(phase[samplefreq], 'rainbow', origin='lower')
plt.colorbar().set_label('Phase [rad]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.show()

posx = 250
posy = 240
plt.plot(freq, magnitude[:, posy, posx])
# plt.title(
plt.semilogy()
plt.show()

posx = 10
posy = 10
plt.plot(freq, phase[:, posy, posx])
posx = 50
posy = 50
plt.plot(freq, phase[:, posy, posx])
posx = 250
posy = 240
plt.plot(freq, phase[:, posy, posx])
# plt.title(
plt.show()

average = np.mean(magnitude, axis=(-1, -2))
plt.plot(freq, average)
plt.plot(freq, average, '+')
# plt.title(
plt.semilogy()
plt.title(
	'Amplitued from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.grid()
# plt.show()

plt.figure()
average = np.mean(phase, axis=(-1, -2))
plt.plot(freq, average)
plt.plot(freq, average, '+')
# plt.title(
# plt.semilogy()
plt.title(
	'Phase from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [rad]')
plt.grid()
plt.show()

# Here I want to see if the amplitude remains constant varying the length of time subsection of data I look at
sections = 10
record_magnitude = []
record_phase = []
record_datasection_start = []
for i in range(sections):
	datasection = data[0:(i + 1) * (len(data) // sections)]
	# record_datasection_start.append(i*sections)
	spectra = np.fft.fft(datasection, axis=0)
	# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
	magnitude = 2 * np.abs(spectra) / len(spectra)
	record_magnitude.append(magnitude)
	phase = np.angle(spectra)
	record_phase.append(phase)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	y = np.mean(magnitude, axis=(-1, -2))
	y = np.array([y for _, y in sorted(zip(freq, y))])
	x = np.sort(freq)
	plt.plot(x, y, label='size of the analysed window ' + str((i + 1) * (len(data) // sections) / framerate))
# plt.title()

plt.title(
	'Amplitued from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.grid()
plt.semilogy()
plt.legend()
plt.show()

record_magnitude = np.array(record_magnitude)
record_phase = np.array(record_phase)
record_datasection_start = np.array(record_datasection_start)
freq = np.fft.fftfreq(len(record_magnitude[0]), d=1 / framerate)

# Here I want to see if the amplitude remains constant for same length of the interval but different starting time
sectionlength = 3 * framerate
sections = len(data) // sectionlength
record_magnitude = []
record_phase = []
record_datasection_start = []
for i in range(sections):
	datasection = data[i * sectionlength:(i + 1) * sectionlength]
	record_datasection_start.append(i * sectionlength / framerate)
	spectra = np.fft.fft(datasection, axis=0)
	# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
	magnitude = 2 * np.abs(spectra) / len(spectra)
	record_magnitude.append(magnitude)
	phase = np.angle(spectra)
	record_phase.append(phase)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	y = np.mean(magnitude, axis=(-1, -2))
	y = np.array([y for _, y in sorted(zip(freq, y))])
	x = np.sort(freq)
	plt.plot(x, y, label='window of ' + str(sectionlength / framerate) + 's starting at ' + str(
		i * sectionlength / framerate) + 's')
# plt.title()

plt.title(
	'Amplitued from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.grid()
plt.semilogy()
plt.legend()
plt.show()

# Here I want to see if the amplitude remains constant for same length of the interval and different locations
poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
sectionlength = 7 * framerate
datasection = data[sectionlength:2 * sectionlength]
spectra = np.fft.fft(datasection, axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra) / len(spectra)
phase = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
window = 2
for i in range(len(poscentred)):
	pos = poscentred[i]
	y = np.mean(magnitude[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=(-1, -2))
	# y=magnitude[:,pos[0],pos[1]]
	y = np.array([y for _, y in sorted(zip(freq, y))])
	x = np.sort(freq)
	plt.plot(x, y, label='window of ' + str(sectionlength / framerate) + 's and the point ' + str(pos))
# plt.title()

plt.title('Amplitued from fast Fourier transform for different groups of ' + str(window * 2) + 'x' + str(
	window * 2) + ' pixels in counts in \n ' + pathfiles + ' framerate ' + str(framerate) + 'Hz, int. time ' + str(
	inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.grid()
plt.semilogy()
plt.legend()
plt.show()

samplefreq = 88
plt.title('Normalised amplitude of ' + str(freq[samplefreq]) + 'Hz oscillation from fast Fourier transform for a ' + str(
		sectionlength / framerate) + 's window in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.imshow(np.divide(magnitude[samplefreq], np.shape(magnitude)[0]), 'rainbow', origin='lower')
plt.colorbar().set_label('Amplitude [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
# plt.show()
plt.figure()
plt.title('Phase of ' + str(freq[samplefreq]) + 'Hz oscillation from fast Fourier transform for a ' + str(
	sectionlength / framerate) + 's window in counts in \n ' + pathfiles + ' framerate ' + str(
	framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.imshow(phase[samplefreq], 'rainbow', origin='lower')
plt.colorbar().set_label('Phase [rad]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.show()

# TEST  -  I will detect amplitude and phase in 3s, substarct it to all the record, compare the fourier analisys before and AFTER
# first I just sample the initial status
poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
datasection = data
spectra_orig = np.fft.fft(datasection, axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
phase = np.angle(spectra_orig)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
window = 10
color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
for i in range(len(poscentred)):
	pos = poscentred[i]
	y = np.mean(magnitude[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=(-1, -2))
	# y=magnitude[:,pos[0],pos[1]]
	y = np.array([y for _, y in sorted(zip(freq, y))])
	x = np.sort(freq)
	plt.plot(x, y, color[i], label='original data at the point ' + str(pos))
# plt.title()

plt.title('Amplitued from fast Fourier transform for different groups of ' + str(window * 2) + 'x' + str(
	window * 2) + ' pixels in counts in \n ' + pathfiles + ' framerate ' + str(framerate) + 'Hz, int. time ' + str(
	inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.grid()
plt.semilogy()
plt.legend()
# plt.show()

# # here I individuate the amplitude and phase of a 3 seconds section
# sectionlength = 3 * framerate
# datasection = data[0:sectionlength]
# spectra_orig = np.fft.fft(datasection, axis=0)
# # magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
# magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
# phase = np.angle(spectra_orig)
# freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
#
# plt.figure(1)
# average = np.mean(magnitude, axis=(-1, -2))
# plt.plot(freq, average)
# plt.plot(freq, average, '+')
# # plt.title(
# plt.semilogy()
# plt.title(
# 	'Amplitued from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
# 		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Amplitude [au]')
# plt.grid()
# # plt.show()
#
# plt.figure(2)
# average = np.mean(phase, axis=(-1, -2))
# plt.plot(freq, average)
# plt.plot(freq, average, '+')
# # plt.title(
# # plt.semilogy()
# plt.title(
# 	'Phase from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
# 		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Phase [rad]')
# plt.grid()

# I need to find the peak

sections = 23	#number found with practice, no specific mathematical reasons
max_time = 5	# seconds of record that I can use to filter the signal. I assume to start from zero
poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]
record_magnitude = []
record_phase = []
record_freq = []
peak_freq_record = []
peak_value_record = []
section_frames_record = []
if (len(data) / framerate) <= 1:
	min_start = max(1,int(0.2 * framerate / (len(data) / sections)))
	max_start = int(sections // 2)
else:
	min_start = max(1,int(2 * framerate / (len(data) / sections)))		# with too little intervals it can interpret noise for signal
	max_start = min(int(1 + sections / 2),int(max_time * framerate / (len(data) / sections)))		# I can use only a part of the record to filter the signal
for i in range(min_start, max_start):
	section_frames = (i) * (len(data) // sections)
	section_frames_record.append(section_frames)
	datasection = data[0:section_frames, poscentre[0] - window:poscentre[0] + window, poscentre[1] - window:poscentre[1] +window]
	spectra = np.fft.fft(datasection, axis=0)
	# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
	magnitude = 2 * np.abs(spectra) / len(spectra)
	record_magnitude.append(magnitude[0:len(magnitude) // 2])
	phase = np.angle(spectra)
	record_phase.append(phase[0:len(magnitude) // 2])
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	record_freq.append(freq[0:len(magnitude) // 2])
	magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	# plt.plot(x, y, label='size of the analysed window ' + str((i + 1) * (len(data) // sections) / framerate))
	# plt.title()
	index_24 = int(coleval.find_nearest_index(x, 24))	# I restric the window over which I do the peak search
	index_34 = int(coleval.find_nearest_index(x, 34))
	index_7 = int(coleval.find_nearest_index(x, 7))
	index_n7 = int(coleval.find_nearest_index(x, -7))
	index_n24 = int(coleval.find_nearest_index(x, -24))
	index_n34 = int(coleval.find_nearest_index(x, -34))
	index_0 = int(coleval.find_nearest_index(x, 0))
	noise = np.mean(np.array(y[3:index_n34].tolist()+y[index_n24:index_n7].tolist()+y[index_7:index_24].tolist()+y[index_34:-3].tolist()), axis=(-1))
	temp = peakutils.indexes(y[index_24:index_34], thres=noise + np.abs(magnitude.min()),min_dist=(index_34 - index_24) // 2)
	if len(temp) == 1:
		peak_index = index_24 + int(temp)
		peak_freq_record.append(x[peak_index])
		peak_value = float(y[peak_index])
		peak_value_record.append(peak_value)
record_magnitude = np.array(record_magnitude)
record_phase = np.array(record_phase)
record_freq = np.array(record_freq)
peak_freq_record = np.array(peak_freq_record)
peak_value_record = np.array(peak_value_record)
section_frames_record = np.array(section_frames_record)
# plt.title('Amplitued from fast Fourier transform averaged in a wondow of ' + str(window) + 'pixels around ' + str(
# 	poscentre) + ' in counts in \n ' + pathfiles + ' framerate ' + str(framerate) + 'Hz, int. time ' + str(
# 	inttime) + 'ms')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Amplitude [au]')
# plt.grid()
# plt.semilogy()
# plt.legend()
# plt.show()

# I find the highest peak and that will be the one I use
index = int(coleval.find_nearest_index(peak_value_record, 100000))
section_frames = section_frames_record[index]
datasection = data[0:section_frames]
spectra = np.fft.fft(datasection, axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra) / len(spectra)
phase = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
freq_to_erase = peak_freq_record[index]
freq_to_erase_index=int(coleval.find_nearest_index(freq, freq_to_erase))
framenumber = np.linspace(0, len(data) - 1, len(data))
data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))

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

plt.title('Amplitued from fast Fourier transform for different groups of ' + str(window * 2) + 'x' + str(
	window * 2) + ' pixels in counts in \n ' + pathfiles + ' framerate ' + str(framerate) + 'Hz, int. time ' + str(
	inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.grid()
plt.semilogy()
plt.legend()
plt.show()

# Manual search
sectionlength = 3 * framerate
datasection2 = data2[sectionlength:2 * sectionlength]
spectra = np.fft.fft(datasection2, axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude2 = 2 * np.abs(spectra) / len(spectra)
phase2 = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)

plt.figure(1)
average = np.mean(magnitude2, axis=(-1, -2))
plt.plot(freq, average)
plt.plot(freq, average, '+')
# plt.title(
plt.semilogy()
plt.title(
	'Amplitued from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.grid()
# plt.show()

plt.figure(2)
average = np.mean(phase2, axis=(-1, -2))
plt.plot(freq, average)
plt.plot(freq, average, '+')
# plt.title(
# plt.semilogy()
plt.title(
	'Phase from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [rad]')
plt.grid()
plt.show()

samplefreq = 88
plt.title(
	'Normalised amplitude of ' + str(freq[samplefreq]) + 'Hz oscillation from fast Fourier transform for a ' + str(
		sectionlength / framerate) + 's window in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.imshow(np.divide(magnitude[samplefreq], np.shape(magnitude)[0]), 'rainbow', origin='lower')
plt.colorbar().set_label('Amplitude [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
# plt.show()
plt.figure()
plt.title('Phase of ' + str(freq[samplefreq]) + 'Hz oscillation from fast Fourier transform for a ' + str(
	sectionlength / framerate) + 's window in counts in \n ' + pathfiles + ' framerate ' + str(
	framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.imshow(phase[samplefreq], 'rainbow', origin='lower')
plt.colorbar().set_label('Phase [rad]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.show()

'''





# I want to take real laser data, substract the oscillation ad see the difference

# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = laser10[0]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = 'npy'
# type='csv'

fullpathparams = os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy')
params = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))

frame = np.shape(data)[1] // 2
base_frame_range = np.shape(data)[1] // 2
testorig = data[0, frame]
testrot = testorig
rotangle = -1.5  # in degrees
foilrot = rotangle * 2 * np.pi / 360
foilrotdeg = rotangle
foilcenter = [160, 133]
foilhorizw = 0.09
foilvertw = 0.07
foilhorizwpixel = 240
foilvertwpixel = np.int((foilhorizwpixel * foilvertw) // foilhorizw)
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

precisionincrease = 10
dummy = np.ones(np.multiply(np.shape(testrot), precisionincrease))
dummy[foilcenter[1] * precisionincrease, foilcenter[0] * precisionincrease] = 2
dummy[int(foily[0] * precisionincrease), int(foilx[0] * precisionincrease)] = 3
dummy[int(foily[1] * precisionincrease), int(foilx[1] * precisionincrease)] = 4
dummy[int(foily[2] * precisionincrease), int(foilx[2] * precisionincrease)] = 5
dummy[int(foily[3] * precisionincrease), int(foilx[3] * precisionincrease)] = 6
dummy2 = rotate(dummy, foilrotdeg, axes=(-1, -2), order=0)
foilcenterrot = (
	np.rint([np.where(dummy2 == 2)[1][0] / precisionincrease, np.where(dummy2 == 2)[0][0] / precisionincrease])).astype(
	int)
foilxrot = (np.rint([np.where(dummy2 == 3)[1][0] / precisionincrease, np.where(dummy2 == 4)[1][0] / precisionincrease,
					 np.where(dummy2 == 5)[1][0] / precisionincrease, np.where(dummy2 == 6)[1][0] / precisionincrease,
					 np.where(dummy2 == 3)[1][0] / precisionincrease])).astype(int)
foilyrot = (np.rint([np.where(dummy2 == 3)[0][0] / precisionincrease, np.where(dummy2 == 4)[0][0] / precisionincrease,
					 np.where(dummy2 == 5)[0][0] / precisionincrease, np.where(dummy2 == 6)[0][0] / precisionincrease,
					 np.where(dummy2 == 3)[0][0] / precisionincrease])).astype(int)

foillx = min(foilxrot)
foilrx = max(foilxrot)
foilhorizwpixel = foilrx - foillx
foildw = min(foilyrot)
foilup = max(foilyrot)
foilvertwpixel = foilup - foildw

# FOIL PROPERTY ADJUSTMENT

# foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
# foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
# conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
# reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

foilemissivityscaled = 1 * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
foilthicknessscaled = (2.5 / 1000000) * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
conductivityscaled = Ptthermalconductivity * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
reciprdiffusivityscaled = (1 / Ptthermaldiffusivity) * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))

basecounts = data

maxpower = []
pathfiles = laser20[0]

filenames = coleval.all_file_names(pathfiles, type)[0]
if (pathfiles in laser12):
	framerate = 994
	datashort = np.load(os.path.join(pathfiles, filenames))
	data = np.multiply(6000, np.ones((1, np.shape(datashort)[1], 256, 320)))
	data[:, :, 64:96, :] = datashort
	type_of_experiment = 'low duty cycle partially defocused'
	poscentred = [[15, 80], [40, 75], [80, 85]]
elif (pathfiles in [vacuum1[1], vacuum1[3]]):
	framerate = 994
	datashort = np.load(os.path.join(pathfiles, filenames))
	data = np.multiply(6000, np.ones((1, np.shape(datashort)[1], 256, 320)))
	data[:, :, 64:96, :] = datashort
	type_of_experiment = 'low duty cycle partially defocused'
	poscentred = [[15, 80], [40, 75], [80, 85]]
elif (pathfiles in laser11):
	framerate = 383
	data = np.load(os.path.join(pathfiles, filenames))
	type_of_experiment = 'partially defocused'
	poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
else:
	framerate = 383
	data = np.load(os.path.join(pathfiles, filenames))
	type_of_experiment = 'focused'
	poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]

# Added 01/08/2018 to account for variations in the background counts over time
data_mean_difference = []
data_mean_difference_std = []
# poscentred=[[70,70],[160,133],[250,200]]
# poscentred=[[60,12],[170,12],[290,12]]
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
base_counts_correction, temp2 = curve_fit(costant, np.linspace(1, len(data_mean_difference), len(data_mean_difference)),
										  data_mean_difference, sigma=data_mean_difference_std, p0=guess,
										  maxfev=100000000)
print('background correction = ' + str(int(base_counts_correction * 1000) / 1000))

datatemp, errdatatemp = coleval.count_to_temp_poly2(
	basecounts[:, frame - base_frame_range // 2:frame + base_frame_range // 2] + base_counts_correction, params,
	errparams)
datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
# errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
# errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]
basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)

datatemp, errdatatemp = coleval.count_to_temp_poly2(data, params, errparams)

# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
# plt.close()
# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=5900,extvmax=6020)
# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
# plt.close()

datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
errdatatemprot = rotate(errdatatemp, foilrotdeg, axes=(-1, -2))
errdatatempcrop = errdatatemprot[:, :, foildw:foilup, foillx:foilrx]

# PRE - PROCESSING 2 12/08/2018 CREATED TO DEAL WITH THE EXTRAPOLATION OF THE BACKGROUND FROM THE TIMESTAMP
# OF THE MEASURE INSTEAD OF THE CHANGE OF BACKGROUND IN DIFFERENT POSITIONS.

# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = vacuum2[20]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '_stat.npy'
# type='csv'

fullpathparams = os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy')
params = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))[0]

# frame=np.shape(data)[1]//2
# base_frame_range=np.shape(data)[1]//2
testrot = data
rotangle = -1.5  # in degrees
foilrot = rotangle * 2 * np.pi / 360
foilrotdeg = rotangle
foilcenter = [160, 133]
foilhorizw = 0.09
foilvertw = 0.07
foilhorizwpixel = 240
foilvertwpixel = np.int((foilhorizwpixel * foilvertw) // foilhorizw)
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

precisionincrease = 10
dummy = np.ones(np.multiply(np.shape(testrot), precisionincrease))
dummy[foilcenter[1] * precisionincrease, foilcenter[0] * precisionincrease] = 2
dummy[int(foily[0] * precisionincrease), int(foilx[0] * precisionincrease)] = 3
dummy[int(foily[1] * precisionincrease), int(foilx[1] * precisionincrease)] = 4
dummy[int(foily[2] * precisionincrease), int(foilx[2] * precisionincrease)] = 5
dummy[int(foily[3] * precisionincrease), int(foilx[3] * precisionincrease)] = 6
dummy2 = rotate(dummy, foilrotdeg, axes=(-1, -2), order=0)
foilcenterrot = (
	np.rint([np.where(dummy2 == 2)[1][0] / precisionincrease, np.where(dummy2 == 2)[0][0] / precisionincrease])).astype(
	int)
foilxrot = (np.rint([np.where(dummy2 == 3)[1][0] / precisionincrease, np.where(dummy2 == 4)[1][0] / precisionincrease,
					 np.where(dummy2 == 5)[1][0] / precisionincrease, np.where(dummy2 == 6)[1][0] / precisionincrease,
					 np.where(dummy2 == 3)[1][0] / precisionincrease])).astype(int)
foilyrot = (np.rint([np.where(dummy2 == 3)[0][0] / precisionincrease, np.where(dummy2 == 4)[0][0] / precisionincrease,
					 np.where(dummy2 == 5)[0][0] / precisionincrease, np.where(dummy2 == 6)[0][0] / precisionincrease,
					 np.where(dummy2 == 3)[0][0] / precisionincrease])).astype(int)

foillx = min(foilxrot)
foilrx = max(foilxrot)
foilhorizwpixel = foilrx - foillx
foildw = min(foilyrot)
foilup = max(foilyrot)
foilvertwpixel = foilup - foildw

# FOIL PROPERTY ADJUSTMENT

# foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
# foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
# conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
# reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

# foilemissivityscaled=1*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
# foilthicknessscaled=(2.5/1000000)*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
# conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
# reciprdiffusivityscaled=(1/Ptthermaldiffusivity)*np.ones((foilvertwpixel-2,foilhorizwpixel-2))

basecounts = data

files = laser20[0]

type = '.npy'
filenames = coleval.all_file_names(pathfiles, type)[0]
if (pathfiles in laser12):
	framerate = 994
	datashort = np.load(os.path.join(pathfiles, filenames))
	data = np.multiply(6000, np.ones((1, np.shape(datashort)[1], 256, 320)))
	data[:, :, 64:96, :] = datashort
	type_of_experiment = 'low duty cycle partially defocused'
	poscentred = [[15, 80], [40, 75], [80, 85]]
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	inttime = 1
elif (pathfiles in coleval.flatten_full([laser15, laser16, laser21, laser23])):
	framerate = 994
	datashort = np.load(os.path.join(pathfiles, filenames))
	data = np.multiply(6000, np.ones((1, np.shape(datashort)[1], 256, 320)))
	data[:, :, 64:128, :] = datashort
	type_of_experiment = 'low duty cycle partially defocused'
	poscentred = [[15, 80], [40, 75], [80, 85]]
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	inttime = 1
elif (pathfiles in coleval.flatten_full([laser24, laser26, laser27, laser28, laser29, laser31])):
	framerate = 1974
	datashort = np.load(os.path.join(pathfiles, filenames))
	data = np.multiply(3000, np.ones((1, np.shape(datashort)[1], 256, 320)))
	data[:, :, 64:128, 128:] = datashort
	type_of_experiment = 'low duty cycle partially defocused'
	poscentred = [[15, 80], [40, 75], [80, 85]]
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/0.5ms383Hz/average'
	inttime = 0.5
elif (pathfiles in [vacuum1[1], vacuum1[3]]):
	framerate = 994
	datashort = np.load(os.path.join(pathfiles, filenames))
	data = np.multiply(6000, np.ones((1, np.shape(datashort)[1], 256, 320)))
	data[:, :, 64:96, :] = datashort
	type_of_experiment = 'low duty cycle partially defocused'
	poscentred = [[15, 80], [40, 75], [80, 85]]
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	inttime = 1
elif (pathfiles in laser11):
	framerate = 383
	data = np.load(os.path.join(pathfiles, filenames))
	type_of_experiment = 'partially defocused'
	poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	inttime = 1
elif (pathfiles in coleval.flatten_full([laser25, laser30])):
	framerate = 383
	data = np.load(os.path.join(pathfiles, filenames))
	type_of_experiment = 'focused'
	poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/2ms383Hz/average'
	inttime = 2
else:
	framerate = 383
	data = np.load(os.path.join(pathfiles, filenames))
	type_of_experiment = 'focused'
	poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	inttime = 1

	params = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
	errparams = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))

if (pathfiles in coleval.flatten_full([laser12])):
	# Added 01/08/2018 to account for variations in the background counts over time
	data_mean_difference = []
	data_mean_difference_std = []
	# poscentred=[[70,70],[160,133],[250,200]]
	# poscentred=[[60,12],[170,12],[290,12]]
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

	datatemp, errdatatemp = coleval.count_to_temp_poly2(
		basecounts[:, frame - base_frame_range // 2:frame + base_frame_range // 2] + base_counts_correction, params,
		errparams)
	datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
	datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
	basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)
elif (pathfiles in coleval.flatten_full([laser10[8:]])):
	# Added 19/08/2018 to use the best data available for building the reference
	type = '_reference.npy'
	filenames_basecounts = coleval.all_file_names(laser10[7], type)[0]
	basecounts = np.load(os.path.join(laser10[7], filenames_basecounts))
	filenames = filenames[:-4] + '_reference' + filenames[-4:]
	data_mean_difference = []
	data_mean_difference_std = []
	# poscentred=[[70,70],[160,133],[250,200]]
	# poscentred=[[60,12],[170,12],[290,12]]
	for pos in poscentred:
		for a in [5, 10]:
			temp1 = np.mean(data[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2, -3))
			temp1std = np.std(np.mean(data[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2)))
			temp2 = np.mean(basecounts[pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2))
			temp2std = 0
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

	datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]] + base_counts_correction, params, errparams)
	datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
	datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
	basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)
elif (pathfiles in coleval.flatten_full([laser11[8:]])):
	# Added 19/08/2018 to use the best data available for building the reference
	type = '_reference.npy'
	filenames_basecounts = coleval.all_file_names(laser11[7], type)[0]
	basecounts = np.load(os.path.join(laser11[7], filenames_basecounts))
	filenames = filenames[:-4] + '_reference' + filenames[-4:]
	data_mean_difference = []
	data_mean_difference_std = []
	# poscentred=[[70,70],[160,133],[250,200]]
	# poscentred=[[60,12],[170,12],[290,12]]
	for pos in poscentred:
		for a in [5, 10]:
			temp1 = np.mean(data[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2, -3))
			temp1std = np.std(np.mean(data[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2)))
			temp2 = np.mean(basecounts[pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2))
			temp2std = 0
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

	datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]] + base_counts_correction, params, errparams)
	datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
	datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
	basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)

elif (pathfiles in coleval.flatten_full([laser15, laser16, laser21, laser23])):

	datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
	datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
	datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
	basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)
elif (pathfiles in coleval.flatten_full([laser24, laser26, laser27, laser28, laser29, laser31])):

	datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
	datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
	datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
	basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)
else:

	datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
	datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
	datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
	basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)

datatemp, errdatatemp = coleval.count_to_temp_poly2(data, params, errparams)

# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
# plt.close()
# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=5900,extvmax=6020)
# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
# plt.close()

datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
errdatatemprot = rotate(errdatatemp, foilrotdeg, axes=(-1, -2))
errdatatempcrop = errdatatemprot[:, :, foildw:foilup, foillx:foilrx]

foilemissivityscaled_orig = 1 * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
conductivityscaled = Ptthermalconductivity * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
reciprdiffusivityscaled_orig = (1 / Ptthermaldiffusivity) * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
flat_properties = True
diffusivity_mult_range = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
emissivity_mult_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
thickness_mult_range = [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]

# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = laser10[13]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '_foil_temperature_temperror_basetemp.npz'
# type='csv'

maxpower = []
files = [laser20[0]]
control_frequency = [freqlaser20[0]]
control_dutycycle = [dutylaser20[0]]

files = coleval.flatten_full(files)
control_frequency_flat = coleval.flatten_full(control_frequency)
control_ducycycle_flat = coleval.flatten_full(control_dutycycle)

index = 0
pathfiles = laser20[0]

if (pathfiles in laser12):
	framerate = 994
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'low duty cycle partially defocused'
# poscentred=[[15,80],[40,75],[80,85]]
elif (pathfiles in [vacuum1[1], vacuum1[3]]):
	framerate = 994
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'low duty cycle partially defocused'
# poscentred=[[15,80],[40,75],[80,85]]
elif (pathfiles in laser11):
	framerate = 383
	# data=np.load(os.path.join(pathfiles,filenames))
	type_of_experiment = 'partially defocused'
# poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
elif (pathfiles in coleval.flatten_full([laser15, laser16])):
	framerate = 994
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'low duty cycle partially defocused'
# poscentred=[[15,80],[40,75],[80,85]]
elif (pathfiles in coleval.flatten_full([laser21, laser23])):
	framerate = 994
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'fully defocused'
# poscentred=[[15,80],[40,75],[80,85]]
elif (pathfiles in coleval.flatten_full([laser24, laser26, laser27, laser28, laser29, laser31])):
	framerate = 1976
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'fully defocused'
# poscentred=[[15,80],[40,75],[80,85]]
else:
	framerate = 383
	# data=np.load(os.path.join(pathfiles,filenames))
	type_of_experiment = 'focused'
# poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]


dt = 1 / framerate
dx = foilhorizw / (foilhorizwpixel - 1)
dTdt = np.divide(datatempcrop[:, 2:, 1:-1, 1:-1] - datatempcrop[:, :-2, 1:-1, 1:-1], 2 * dt)
d2Tdx2 = np.divide(
	datatempcrop[:, 1:-1, 1:-1, 2:] - np.multiply(2, datatempcrop[:, 1:-1, 1:-1, 1:-1]) + datatempcrop[:, 1:-1, 1:-1,
																						  :-2], dx ** 2)
d2Tdy2 = np.divide(
	datatempcrop[:, 1:-1, 2:, 1:-1] - np.multiply(2, datatempcrop[:, 1:-1, 1:-1, 1:-1]) + datatempcrop[:, 1:-1, :-2,
																						  1:-1], dx ** 2)
d2Tdxy = np.add(d2Tdx2, d2Tdy2)
negd2Tdxy = np.multiply(-1, d2Tdxy)
T4 = np.power(np.add(zeroC, datatempcrop[:, 1:-1, 1:-1, 1:-1]), 4)
T04 = np.power(np.add(zeroC, basetemp), 4)

diffusivity_mult = 1.4
emissivity_mult = 1.2
thickness_mult = 0.8

reciprdiffusivityscaled = np.multiply(1 / diffusivity_mult, reciprdiffusivityscaled_orig)
foilemissivityscaled = np.multiply(emissivity_mult, foilemissivityscaled_orig)
foilthicknessscaled = np.multiply(thickness_mult, foilthicknessscaled_orig)

BBrad = []
diffusion = []
timevariation = []
ktf = np.multiply(conductivityscaled, foilthicknessscaled)
for i in range(len(datatempcrop[:, 0, 0, 0])):
	BBrad.append([])
	diffusion.append([])
	timevariation.append([])
	for j in range(len(datatempcrop[0, 1:-1, 0, 0])):
		BBradtemp = np.multiply(np.multiply(2 * sigmaSB, foilemissivityscaled), np.add(T4[i, j], np.negative(T04)))
		BBrad[i].append(BBradtemp)
		diffusiontemp = np.multiply(ktf, negd2Tdxy[i, j])
		diffusion[i].append(diffusiontemp)
		timevariationtemp = np.multiply(ktf, np.multiply(reciprdiffusivityscaled, dTdt[i, j]))
		timevariation[i].append(timevariationtemp)
BBrad = np.array(BBrad)
diffusion = np.array(diffusion)
timevariation = np.array(timevariation)

BBradnoback = np.add(BBrad, 0)
diffusionnoback = np.add(diffusion, 0)
timevariationnoback = np.add(timevariation, 0)

powernoback = np.add(np.add(diffusionnoback, timevariationnoback), BBradnoback)

totalpower = np.multiply(np.sum(powernoback[0], axis=(-1, -2)), dx ** 2)
totalBBrad = np.multiply(np.sum(BBradnoback[0], axis=(-1, -2)), dx ** 2)
totaldiffusion = np.multiply(np.sum(diffusionnoback[0], axis=(-1, -2)), dx ** 2)
totaltimevariation = np.multiply(np.sum(timevariationnoback[0], axis=(-1, -2)), dx ** 2)
#

addition = 0
if (pathfiles in laser12):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_temperature'),powernoback=powernoback)
	powerzoom = 5 + addition
	laserspot = [41, 172]
elif (pathfiles in laser13):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [26, 196]
elif (pathfiles in laser14):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [27, 156]
elif (pathfiles in laser15):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 2 + addition
	laserspot = [43, 188]
elif (pathfiles in laser16):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [41, 186]
elif (pathfiles in laser17):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 2 + addition
	laserspot = [21, 216]
elif (pathfiles in laser18):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 1 + addition
	laserspot = [22, 159]
elif (pathfiles in laser19):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 1 + addition
	laserspot = [54, 187]
elif (pathfiles in laser20):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 6 + addition
	laserspot = [42, 180]
elif (pathfiles in laser21):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 5 + addition
	laserspot = [42, 180]
elif (pathfiles in coleval.flatten_full([laser22, laser23, laser24, laser25, lase26, laser27])):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 1 + addition
	laserspot = [42, 189]
elif (pathfiles in coleval.flatten_full([laser28, laser29])):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [43, 188]
elif (pathfiles in coleval.flatten_full([laser30, laser31])):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 6 + addition
	laserspot = [42, 190]
elif (pathfiles in coleval.flatten_full([laser32])):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 2 + addition
	laserspot = [49, 214]
else:
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total-BB-diff-time'),powernoback=powernoback,BBradnoback=BBradnoback,diffusionnoback=diffusionnoback,timevariationnoback=timevariationnoback)
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [41, 172]

# powerzoom=2
# laserspot=[41,172]
maxdw = laserspot[0] - powerzoom
maxup = laserspot[0] + powerzoom + 1
maxlx = laserspot[1] - powerzoom
maxdx = laserspot[1] + powerzoom + 1

powernobackcrop = powernoback[:, :, maxdw:maxup, maxlx:maxdx]
BBradnobackcrop = BBradnoback[:, :, maxdw:maxup, maxlx:maxdx]
diffusionnobackcrop = diffusionnoback[:, :, maxdw:maxup, maxlx:maxdx]
timevariationnobackcrop = timevariationnoback[:, :, maxdw:maxup, maxlx:maxdx]

totalpowercrop = np.multiply(np.sum(powernobackcrop[0], axis=(-1, -2)), dx ** 2)
totalBBradcrop = np.multiply(np.sum(BBradnobackcrop[0], axis=(-1, -2)), dx ** 2)
totaldiffusioncrop = np.multiply(np.sum(diffusionnobackcrop[0], axis=(-1, -2)), dx ** 2)
totaltimevariationcrop = np.multiply(np.sum(timevariationnobackcrop[0], axis=(-1, -2)), dx ** 2)

# 2018/08/16 modified because i can make aebtter use of the mobile average, knowing the pulse frequency and duty cycle
flat_time_high = control_ducycycle_flat[index] / control_frequency_flat[index]
flat_frames_high = int(flat_time_high * framerate)
flat_time_low = (1 - control_ducycycle_flat[index]) / control_frequency_flat[index]
flat_frames_low = int(flat_time_low * framerate)

if flat_frames_high > 2:
	flat_frames_high = flat_frames_high - 1
	moving_average_high = np.convolve(totalpowercrop, np.ones((flat_frames_high,)) / flat_frames_high, mode='full')
	sample_high = []
	for i in range(1, len(moving_average_high) - 2):
		border_left = np.array([i + 1 - int(flat_frames_high // 1.1), 0])
		border_left = (border_left[border_left >= 0]).max()
		border_right = np.array([i + 1 + int(flat_frames_low // 1.1), len(moving_average_high) - 1])
		border_right = (border_right[border_right <= len(moving_average_high)]).min()
		if ((moving_average_high[i + 1] > (moving_average_high[border_left:i + 1]).max()) & (
				moving_average_high[i + 1] >= (moving_average_high[i + 1:border_right]).max())):
			sample_high.append(moving_average_high[i + 1])
	sample_high = np.array(sample_high)

	moving_average_low = np.convolve(totalpowercrop, np.ones((flat_frames_low,)) / flat_frames_low, mode='full')
	sample_low = moving_average_low[1:-1][((moving_average_low[1:-1] - moving_average_low[0:-2] < 0) & (
				moving_average_low[1:-1] - moving_average_low[2:] < 0))]

	sample_high_max = sample_high.max()
	sample_low_min = sample_low.min()
	sample_high = sample_high[(sample_high > (sample_high_max - sample_low_min) / 2)]
	sample_low = sample_low[(sample_low < (sample_high_max - sample_low_min) / 2)]

	if len(sample_high) > 3:
		test4 = np.mean(sample_high)
		test3 = np.std(sample_high)
		sample = sample_high
	else:
		max_power_index = (np.ndarray.tolist(moving_average_high)).index(sample_high.max())
		test4 = moving_average_high[max_power_index]
		test3 = np.std(totalpowercrop[
					   max(0, max_power_index - flat_frames_high):min(max_power_index + 1, len(totalpowercrop) - 1)])
		sample = np.array([test4 - test3, test4 + test3])
else:
	# powermean=np.mean(totalpowercrop)
	sample = totalpowercrop[1:-1][
		((totalpowercrop[1:-1] - totalpowercrop[0:-2] > 0) & (totalpowercrop[1:-1] - totalpowercrop[2:] > 0))]
	sample = sample[(sample > sample.max() / 2)]
	test4 = np.mean(sample)
	test3 = np.std(sample)
# guess=[sample.max()]
# 	# test4,test3=curve_fit(costant,np.linspace(1,len(sample),len(sample)),sample, p0=guess, maxfev=100000000)


x_axis = (1 / framerate) * np.linspace(1, len(totalpowercrop), len(totalpowercrop))
plt.figure(figsize=(20, 10))
plt.plot(x_axis, totalpowercrop, label='totalpowercrop_unfiltered')
plt.plot(x_axis, totalBBradcrop, label='totalBBradcrop_unfiltered')
plt.plot(x_axis, totaldiffusioncrop, label='totaldiffusioncrop_unfiltered')
plt.plot(x_axis, totaltimevariationcrop, label='totaltimevariationcrop_unfiltered')
plt.plot(x_axis, costant(np.linspace(1, len(totalpowercrop), len(totalpowercrop)), *[test4]), label='sampled power')
plt.plot(x_axis, costant(np.linspace(1, len(totalpowercrop), len(totalpowercrop)), *[sample.min()]), 'k--',
		 linewidth=0.5)
plt.plot(x_axis, costant(np.linspace(1, len(totalpowercrop), len(totalpowercrop)), *[sample.max()]), 'k--',
		 label='used for sampling', linewidth=0.5)
plt.title('Sum of the power over the area \n in ' + str(laserspot) + ' +/- ' + str(powerzoom) + ' max=' + str(
	int(test4 * 100000) / 100) + 'mW, std=' + str(int(test3 * 10000000) / 10000) + 'mW')
plt.xlabel('Time [s]')
plt.ylabel('Power sum [W]')
plt.legend()
plt.legend(loc='best')
plt.grid()

data = coleval.clear_oscillation_central2(data, framerate,plot_conparison=False)

datatemp, errdatatemp = coleval.count_to_temp_poly2(data, params, errparams)

# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
# plt.close()
# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=5900,extvmax=6020)
# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
# plt.close()

datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
errdatatemprot = rotate(errdatatemp, foilrotdeg, axes=(-1, -2))
errdatatempcrop = errdatatemprot[:, :, foildw:foilup, foillx:foilrx]

foilemissivityscaled_orig = 1 * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
conductivityscaled = Ptthermalconductivity * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
reciprdiffusivityscaled_orig = (1 / Ptthermaldiffusivity) * np.ones((foilvertwpixel - 2, foilhorizwpixel - 2))
flat_properties = True
diffusivity_mult_range = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
emissivity_mult_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
thickness_mult_range = [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]

# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = laser10[13]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '_foil_temperature_temperror_basetemp.npz'
# type='csv'

maxpower = []
files = [laser20[0]]
control_frequency = [freqlaser20[0]]
control_dutycycle = [dutylaser20[0]]

files = coleval.flatten_full(files)
control_frequency_flat = coleval.flatten_full(control_frequency)
control_ducycycle_flat = coleval.flatten_full(control_dutycycle)

index = 0
pathfiles = laser20[0]

if (pathfiles in laser12):
	framerate = 994
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'low duty cycle partially defocused'
# poscentred=[[15,80],[40,75],[80,85]]
elif (pathfiles in [vacuum1[1], vacuum1[3]]):
	framerate = 994
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'low duty cycle partially defocused'
# poscentred=[[15,80],[40,75],[80,85]]
elif (pathfiles in laser11):
	framerate = 383
	# data=np.load(os.path.join(pathfiles,filenames))
	type_of_experiment = 'partially defocused'
# poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
elif (pathfiles in coleval.flatten_full([laser15, laser16])):
	framerate = 994
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'low duty cycle partially defocused'
# poscentred=[[15,80],[40,75],[80,85]]
elif (pathfiles in coleval.flatten_full([laser21, laser23])):
	framerate = 994
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'fully defocused'
# poscentred=[[15,80],[40,75],[80,85]]
elif (pathfiles in coleval.flatten_full([laser24, laser26, laser27, laser28, laser29, laser31])):
	framerate = 1976
	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:96,:]=datashort
	type_of_experiment = 'fully defocused'
# poscentred=[[15,80],[40,75],[80,85]]
else:
	framerate = 383
	# data=np.load(os.path.join(pathfiles,filenames))
	type_of_experiment = 'focused'
# poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]


dt = 1 / framerate
dx = foilhorizw / (foilhorizwpixel - 1)
dTdt = np.divide(datatempcrop[:, 2:, 1:-1, 1:-1] - datatempcrop[:, :-2, 1:-1, 1:-1], 2 * dt)
d2Tdx2 = np.divide(
	datatempcrop[:, 1:-1, 1:-1, 2:] - np.multiply(2, datatempcrop[:, 1:-1, 1:-1, 1:-1]) + datatempcrop[:, 1:-1, 1:-1,
																						  :-2], dx ** 2)
d2Tdy2 = np.divide(
	datatempcrop[:, 1:-1, 2:, 1:-1] - np.multiply(2, datatempcrop[:, 1:-1, 1:-1, 1:-1]) + datatempcrop[:, 1:-1, :-2,
																						  1:-1], dx ** 2)
d2Tdxy = np.add(d2Tdx2, d2Tdy2)
negd2Tdxy = np.multiply(-1, d2Tdxy)
T4 = np.power(np.add(zeroC, datatempcrop[:, 1:-1, 1:-1, 1:-1]), 4)
T04 = np.power(np.add(zeroC, basetemp), 4)

diffusivity_mult = 1.4
emissivity_mult = 1.2
thickness_mult = 0.8

reciprdiffusivityscaled = np.multiply(1 / diffusivity_mult, reciprdiffusivityscaled_orig)
foilemissivityscaled = np.multiply(emissivity_mult, foilemissivityscaled_orig)
foilthicknessscaled = np.multiply(thickness_mult, foilthicknessscaled_orig)

BBrad = []
diffusion = []
timevariation = []
ktf = np.multiply(conductivityscaled, foilthicknessscaled)
for i in range(len(datatempcrop[:, 0, 0, 0])):
	BBrad.append([])
	diffusion.append([])
	timevariation.append([])
	for j in range(len(datatempcrop[0, 1:-1, 0, 0])):
		BBradtemp = np.multiply(np.multiply(2 * sigmaSB, foilemissivityscaled), np.add(T4[i, j], np.negative(T04)))
		BBrad[i].append(BBradtemp)
		diffusiontemp = np.multiply(ktf, negd2Tdxy[i, j])
		diffusion[i].append(diffusiontemp)
		timevariationtemp = np.multiply(ktf, np.multiply(reciprdiffusivityscaled, dTdt[i, j]))
		timevariation[i].append(timevariationtemp)
BBrad = np.array(BBrad)
diffusion = np.array(diffusion)
timevariation = np.array(timevariation)

BBradnoback = np.add(BBrad, 0)
diffusionnoback = np.add(diffusion, 0)
timevariationnoback = np.add(timevariation, 0)

powernoback = np.add(np.add(diffusionnoback, timevariationnoback), BBradnoback)

totalpower = np.multiply(np.sum(powernoback[0], axis=(-1, -2)), dx ** 2)
totalBBrad = np.multiply(np.sum(BBradnoback[0], axis=(-1, -2)), dx ** 2)
totaldiffusion = np.multiply(np.sum(diffusionnoback[0], axis=(-1, -2)), dx ** 2)
totaltimevariation = np.multiply(np.sum(timevariationnoback[0], axis=(-1, -2)), dx ** 2)
#

addition = 0
if (pathfiles in laser12):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_temperature'),powernoback=powernoback)
	powerzoom = 5 + addition
	laserspot = [41, 172]
elif (pathfiles in laser13):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [26, 196]
elif (pathfiles in laser14):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [27, 156]
elif (pathfiles in laser15):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 2 + addition
	laserspot = [43, 188]
elif (pathfiles in laser16):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [41, 186]
elif (pathfiles in laser17):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 2 + addition
	laserspot = [21, 216]
elif (pathfiles in laser18):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 1 + addition
	laserspot = [22, 159]
elif (pathfiles in laser19):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 1 + addition
	laserspot = [54, 187]
elif (pathfiles in laser20):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 6 + addition
	laserspot = [42, 180]
elif (pathfiles in laser21):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 5 + addition
	laserspot = [42, 180]
elif (pathfiles in coleval.flatten_full([laser22, laser23, laser24, laser25, lase26, laser27])):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 1 + addition
	laserspot = [42, 189]
elif (pathfiles in coleval.flatten_full([laser28, laser29])):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [43, 188]
elif (pathfiles in coleval.flatten_full([laser30, laser31])):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 6 + addition
	laserspot = [42, 190]
elif (pathfiles in coleval.flatten_full([laser32])):
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 2 + addition
	laserspot = [49, 214]
else:
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total-BB-diff-time'),powernoback=powernoback,BBradnoback=BBradnoback,diffusionnoback=diffusionnoback,timevariationnoback=timevariationnoback)
	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
	powerzoom = 3 + addition
	laserspot = [41, 172]

# powerzoom=2
# laserspot=[41,172]
maxdw = laserspot[0] - powerzoom
maxup = laserspot[0] + powerzoom + 1
maxlx = laserspot[1] - powerzoom
maxdx = laserspot[1] + powerzoom + 1

powernobackcrop = powernoback[:, :, maxdw:maxup, maxlx:maxdx]
BBradnobackcrop = BBradnoback[:, :, maxdw:maxup, maxlx:maxdx]
diffusionnobackcrop = diffusionnoback[:, :, maxdw:maxup, maxlx:maxdx]
timevariationnobackcrop = timevariationnoback[:, :, maxdw:maxup, maxlx:maxdx]

totalpowercrop = np.multiply(np.sum(powernobackcrop[0], axis=(-1, -2)), dx ** 2)
totalBBradcrop = np.multiply(np.sum(BBradnobackcrop[0], axis=(-1, -2)), dx ** 2)
totaldiffusioncrop = np.multiply(np.sum(diffusionnobackcrop[0], axis=(-1, -2)), dx ** 2)
totaltimevariationcrop = np.multiply(np.sum(timevariationnobackcrop[0], axis=(-1, -2)), dx ** 2)

# 2018/08/16 modified because i can make aebtter use of the mobile average, knowing the pulse frequency and duty cycle
flat_time_high = control_ducycycle_flat[index] / control_frequency_flat[index]
flat_frames_high = int(flat_time_high * framerate)
flat_time_low = (1 - control_ducycycle_flat[index]) / control_frequency_flat[index]
flat_frames_low = int(flat_time_low * framerate)

if flat_frames_high > 2:
	flat_frames_high = flat_frames_high - 1
	moving_average_high = np.convolve(totalpowercrop, np.ones((flat_frames_high,)) / flat_frames_high, mode='full')
	sample_high = []
	for i in range(1, len(moving_average_high) - 2):
		border_left = np.array([i + 1 - int(flat_frames_high // 1.1), 0])
		border_left = (border_left[border_left >= 0]).max()
		border_right = np.array([i + 1 + int(flat_frames_low // 1.1), len(moving_average_high) - 1])
		border_right = (border_right[border_right <= len(moving_average_high)]).min()
		if ((moving_average_high[i + 1] > (moving_average_high[border_left:i + 1]).max()) & (
				moving_average_high[i + 1] >= (moving_average_high[i + 1:border_right]).max())):
			sample_high.append(moving_average_high[i + 1])
	sample_high = np.array(sample_high)

	moving_average_low = np.convolve(totalpowercrop, np.ones((flat_frames_low,)) / flat_frames_low, mode='full')
	sample_low = moving_average_low[1:-1][((moving_average_low[1:-1] - moving_average_low[0:-2] < 0) & (
				moving_average_low[1:-1] - moving_average_low[2:] < 0))]

	sample_high_max = sample_high.max()
	sample_low_min = sample_low.min()
	sample_high = sample_high[(sample_high > (sample_high_max - sample_low_min) / 2)]
	sample_low = sample_low[(sample_low < (sample_high_max - sample_low_min) / 2)]

	if len(sample_high) > 3:
		test4 = np.mean(sample_high)
		test3 = np.std(sample_high)
		sample = sample_high
	else:
		max_power_index = (np.ndarray.tolist(moving_average_high)).index(sample_high.max())
		test4 = moving_average_high[max_power_index]
		test3 = np.std(totalpowercrop[
					   max(0, max_power_index - flat_frames_high):min(max_power_index + 1, len(totalpowercrop) - 1)])
		sample = np.array([test4 - test3, test4 + test3])
else:
	# powermean=np.mean(totalpowercrop)
	sample = totalpowercrop[1:-1][
		((totalpowercrop[1:-1] - totalpowercrop[0:-2] > 0) & (totalpowercrop[1:-1] - totalpowercrop[2:] > 0))]
	sample = sample[(sample > sample.max() / 2)]
	test4 = np.mean(sample)
	test3 = np.std(sample)
# guess=[sample.max()]
# 	# test4,test3=curve_fit(costant,np.linspace(1,len(sample),len(sample)),sample, p0=guess, maxfev=100000000)


x_axis = (1 / framerate) * np.linspace(1, len(totalpowercrop), len(totalpowercrop))
plt.figure(figsize=(20, 10))
plt.plot(x_axis, totalpowercrop, label='totalpowercrop_filtered')
plt.plot(x_axis, totalBBradcrop, label='totalBBradcrop_filtered')
plt.plot(x_axis, totaldiffusioncrop, label='totaldiffusioncrop_filtered')
plt.plot(x_axis, totaltimevariationcrop, label='totaltimevariationcrop_filtered')
plt.plot(x_axis, costant(np.linspace(1, len(totalpowercrop), len(totalpowercrop)), *[test4]), label='sampled power')
plt.plot(x_axis, costant(np.linspace(1, len(totalpowercrop), len(totalpowercrop)), *[sample.min()]), 'k--',
		 linewidth=0.5)
plt.plot(x_axis, costant(np.linspace(1, len(totalpowercrop), len(totalpowercrop)), *[sample.max()]), 'k--',
		 label='used for sampling', linewidth=0.5)
plt.title('Sum of the power over the area \n in ' + str(laserspot) + ' +/- ' + str(powerzoom) + ' max=' + str(
	int(test4 * 100000) / 100) + 'mW, std=' + str(int(test3 * 10000000) / 10000) + 'mW')
plt.xlabel('Time [s]')
plt.ylabel('Power sum [W]')
plt.legend()
plt.legend(loc='best')
plt.grid()

plt.show()


