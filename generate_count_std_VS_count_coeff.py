# Created 06/10/2021
# Fabio Federici
# created because I realise that the error of the counts if not the normally assumet (count)**0.5 but uch less. I need to find out how much it is


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

# from multiprocessing import Pool,cpu_count
# number_cpu_available = 8	#cpu_count()
# print('Number of cores available: '+str(number_cpu_available))
# import mkl
# mkl.set_num_threads(number_cpu_available)

fileshot=np.array([files24,files28])
temperaturehot=np.array([temperature24,temperature28])
filescold=np.array([files26,files30])
temperaturecold=np.array([temperature26,temperature30])
inttime=0.5	# ms
framerate=383	# Hz
n=3
pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)

temperature = temperaturehot.tolist()+temperaturecold.tolist()
files = fileshot.tolist()+filescold.tolist()
while np.shape(temperature[0])!=():
	temperature=np.concatenate(temperature)
	files=np.concatenate(files)

sin_fun = lambda x,A,f,p : A*np.sin(x*f*2*np.pi+p)
meancounttot=[]
meancountstdtot=[]
for i_file,file in enumerate(files):
	full_saved_file_dict=np.load(file+'.npz')
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	if i_file==0:
		digitizer_ID = np.array(uniques_digitizer_ID)
	if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
		print('ERROR: problem with the ID of the digitizer in \n' + file)
		exit()
	# this data contains the pesky constant oscillation, that will is removed when the data is processed and I need to remofe from here too
	a = [np.mean(x,axis=(-1,-2)) for x in data_per_digitizer]
	b = []
	for i in digitizer_ID:
		time_axis = np.arange(len(a[i]))*2*1 / framerate
		lin_fit = np.polyfit(time_axis,a[i],1)
		baseline = np.polyval(lin_fit,time_axis)
		bds = [[0,20,-4*np.pi],[np.inf,40,4*np.pi]]
		guess = [1,29,max(-4*np.pi,min(4*np.pi,-np.pi*np.trapz((a[i]-baseline)[time_axis<1/29/2])*2/np.trapz(np.abs(a[i]-baseline)[time_axis<1/29])))]
		fit = curve_fit(sin_fun, time_axis,a[i]-baseline, p0=guess, bounds = bds, maxfev=100000000)
		# plt.figure()
		# plt.plot(time_axis,a[i]-baseline)
		# plt.plot(time_axis,sin_fun(time_axis,*fit[0]))
		# plt.plot(time_axis,sin_fun(time_axis,*guess),'--')
		# plt.pause(0.001)
		b.append((data_per_digitizer[i].T-baseline-sin_fun(time_axis,*fit[0])).T)
	meancounttot.append([np.mean(x) for x in data_per_digitizer])
	meancountstdtot.append([np.mean(np.std(x,axis=0)) for x in b])


fileshot=np.array([files16[2:],files14[5:]])
temperaturehot=np.array([temperature16[2:],temperature14[5:]])
filescold=np.array([files18[4:],files20[2:]])
temperaturecold=np.array([temperature18[4:],temperature20[2:]])
inttime=1.0	# ms
framerate=383	# Hz
n=3
pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)

temperature = temperaturehot.tolist()+temperaturecold.tolist()
files = fileshot.tolist()+filescold.tolist()
while np.shape(temperature[0])!=():
	temperature=np.concatenate(temperature)
	files=np.concatenate(files)

for i_file,file in enumerate(files):
	full_saved_file_dict=np.load(file+'.npz')
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	if i_file==0:
		digitizer_ID = np.array(uniques_digitizer_ID)
	if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
		print('ERROR: problem with the ID of the digitizer in \n' + file)
		exit()
	# this data contains the pesky constant oscillation, that will is removed when the data is processed and I need to remofe from here too
	a = [np.mean(x,axis=(-1,-2)) for x in data_per_digitizer]
	b = []
	for i in digitizer_ID:
		time_axis = np.arange(len(a[i]))*2*1 / framerate
		lin_fit = np.polyfit(time_axis,a[i],1)
		baseline = np.polyval(lin_fit,time_axis)
		bds = [[0,20,-4*np.pi],[np.inf,40,4*np.pi]]
		guess = [1,29,max(-4*np.pi,min(4*np.pi,-np.pi*np.trapz((a[i]-baseline)[time_axis<1/29/2])*2/np.trapz(np.abs(a[i]-baseline)[time_axis<1/29])))]
		fit = curve_fit(sin_fun, time_axis,a[i]-baseline, p0=guess, bounds = bds, maxfev=100000000)
		# plt.figure()
		# plt.plot(time_axis,a[i]-baseline)
		# plt.plot(time_axis,sin_fun(time_axis,*fit[0]))
		# plt.plot(time_axis,sin_fun(time_axis,*guess),'--')
		# plt.pause(0.001)
		b.append((data_per_digitizer[i].T-baseline-sin_fun(time_axis,*fit[0])).T)
	meancounttot.append([np.mean(x) for x in data_per_digitizer])
	meancountstdtot.append([np.mean(np.std(x,axis=0)) for x in b])

fileshot=np.array([files15,files17])
temperaturehot=np.array([temperature15,temperature17])
filescold=np.array([files19,files21])
temperaturecold=np.array([temperature19,temperature21])
inttime=2.0	# ms
framerate=383	# Hz
n=3
pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)

temperature = temperaturehot.tolist()+temperaturecold.tolist()
files = fileshot.tolist()+filescold.tolist()
while np.shape(temperature[0])!=():
	temperature=np.concatenate(temperature)
	files=np.concatenate(files)

for i_file,file in enumerate(files):
	full_saved_file_dict=np.load(file+'.npz')
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	if i_file==0:
		digitizer_ID = np.array(uniques_digitizer_ID)
	if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
		print('ERROR: problem with the ID of the digitizer in \n' + file)
		exit()
	# this data contains the pesky constant oscillation, that will is removed when the data is processed and I need to remofe from here too
	a = [np.mean(x,axis=(-1,-2)) for x in data_per_digitizer]
	b = []
	for i in digitizer_ID:
		time_axis = np.arange(len(a[i]))*2*1 / framerate
		lin_fit = np.polyfit(time_axis,a[i],1)
		baseline = np.polyval(lin_fit,time_axis)
		bds = [[0,20,-4*np.pi],[np.inf,40,4*np.pi]]
		guess = [1,29,max(-4*np.pi,min(4*np.pi,-np.pi*np.trapz((a[i]-baseline)[time_axis<1/29/2])*2/np.trapz(np.abs(a[i]-baseline)[time_axis<1/29])))]
		fit = curve_fit(sin_fun, time_axis,a[i]-baseline, p0=guess, bounds = bds, maxfev=100000000)
		# plt.figure()
		# plt.plot(time_axis,a[i])
		# plt.plot(time_axis,np.mean(a[i])+sin_fun(time_axis,*fit[0]))
		# plt.pause(0.001)
		b.append((data_per_digitizer[i].T-baseline-sin_fun(time_axis,*fit[0])).T)
	meancounttot.append([np.mean(x) for x in data_per_digitizer])
	meancountstdtot.append([np.mean(np.std(x,axis=0)) for x in b])


fileshot=np.array([files25,files29])
temperaturehot=np.array([temperature25,temperature29])
filescold=np.array([files27,files31])
temperaturecold=np.array([temperature27,temperature31])
inttime=1.5	# ms
framerate=383	# Hz
n=3
pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
temperature = temperaturehot.tolist()+temperaturecold.tolist()
files = fileshot.tolist()+filescold.tolist()
while np.shape(temperature[0])!=():
	temperature=np.concatenate(temperature)
	files=np.concatenate(files)

for i_file,file in enumerate(files):
	full_saved_file_dict=np.load(file+'.npz')
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	if i_file==0:
		digitizer_ID = np.array(uniques_digitizer_ID)
	if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
		print('ERROR: problem with the ID of the digitizer in \n' + file)
		exit()
	# this data contains the pesky constant oscillation, that will is removed when the data is processed and I need to remofe from here too
	a = [np.mean(x,axis=(-1,-2)) for x in data_per_digitizer]
	b = []
	for i in digitizer_ID:
		time_axis = np.arange(len(a[i]))*2*1 / framerate
		lin_fit = np.polyfit(time_axis,a[i],1)
		baseline = np.polyval(lin_fit,time_axis)
		bds = [[0,20,-4*np.pi],[np.inf,40,4*np.pi]]
		guess = [1,29,max(-4*np.pi,min(4*np.pi,-np.pi*np.trapz((a[i]-baseline)[time_axis<1/29/2])*2/np.trapz(np.abs(a[i]-baseline)[time_axis<1/29])))]
		fit = curve_fit(sin_fun, time_axis,a[i]-baseline, p0=guess, bounds = bds, maxfev=100000000)
		# plt.figure()
		# plt.plot(time_axis,a[i])
		# plt.plot(time_axis,np.mean(a[i])+sin_fun(time_axis,*fit[0]))
		# plt.pause(0.001)
		b.append((data_per_digitizer[i].T-baseline-sin_fun(time_axis,*fit[0])).T)
	meancounttot.append([np.mean(x) for x in data_per_digitizer])
	meancountstdtot.append([np.mean(np.std(x,axis=0)) for x in b])

meancounttot = np.array(meancounttot).flatten()
meancountstdtot = np.array(meancountstdtot).flatten()

plt.figure()
plt.plot(meancounttot,meancountstdtot/meancounttot,'+')
fit = np.polyfit(meancounttot,meancountstdtot/meancounttot,7)
plt.plot(np.sort(meancounttot),np.polyval(fit,np.sort(meancounttot)),'-',label='fit = '+str(fit))
plt.pause(0.001)

fit = np.array([-5.32486228e-31,  3.53268894e-26, -9.85692764e-22,  1.50423830e-17,-1.36726859e-13,  7.54763409e-10, -2.46270434e-06,  4.48742195e-03])



####################################



fileshot=np.array([files22])
temperaturehot=np.array([temperature22])
filescold=np.array([files23])
temperaturecold=np.array([temperature23])
inttime=1.0	# ms
framerate=994	# Hz
n=3
pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
temperature = temperaturehot.tolist()+temperaturecold.tolist()
files = fileshot.tolist()+filescold.tolist()
while np.shape(temperature[0])!=():
	temperature=np.concatenate(temperature)
	files=np.concatenate(files)

sin_fun = lambda x,A,f,p : A*np.sin(x*f*2*np.pi+p)
meancounttot=[]
meancountstdtot=[]
for i_file,file in enumerate(files):
	full_saved_file_dict=np.load(file+'.npz')
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	if i_file==0:
		digitizer_ID = np.array(uniques_digitizer_ID)
	if np.sum(digitizer_ID==uniques_digitizer_ID)<len(digitizer_ID):
		print('ERROR: problem with the ID of the digitizer in \n' + file)
		exit()
	# this data contains the pesky constant oscillation, that will is removed when the data is processed and I need to remofe from here too
	a = [np.mean(x,axis=(-1,-2)) for x in data_per_digitizer]
	b = []
	for i in digitizer_ID:
		time_axis = np.arange(len(a[i]))*2*1 / framerate
		lin_fit = np.polyfit(time_axis,a[i],1)
		baseline = np.polyval(lin_fit,time_axis)
		bds = [[0,20,-4*np.pi],[np.inf,40,4*np.pi]]
		guess = [1,29,max(-4*np.pi,min(4*np.pi,-np.pi*np.trapz((a[i]-baseline)[time_axis<1/29/2])*2/np.trapz(np.abs(a[i]-baseline)[time_axis<1/29])))]
		fit = curve_fit(sin_fun, time_axis,a[i]-baseline, p0=guess, bounds = bds, maxfev=100000000)
		# plt.figure()
		# plt.plot(time_axis,a[i]-baseline)
		# plt.plot(time_axis,sin_fun(time_axis,*fit[0]))
		# plt.plot(time_axis,sin_fun(time_axis,*guess),'--')
		# plt.pause(0.001)
		b.append((data_per_digitizer[i].T-baseline-sin_fun(time_axis,*fit[0])).T)
	meancounttot.append([np.mean(x) for x in data_per_digitizer])
	meancountstdtot.append([np.mean(np.std(x,axis=0)) for x in b])

meancounttot = np.array(meancounttot).flatten()
meancountstdtot = np.array(meancountstdtot).flatten()

plt.plot(meancounttot,meancountstdtot/meancounttot,'+')
fit = np.polyfit(meancounttot,meancountstdtot/meancounttot,7)
plt.plot(np.sort(meancounttot),np.polyval(fit,np.sort(meancounttot)),'-',label='fit = '+str(fit))
plt.pause(0.001)

fit = np.array([2.02656820e-29, -9.13097909e-25,  1.74306947e-20, -1.82476234e-16,1.12726963e-12, -4.07021315e-09,  7.70300710e-06, -4.64653103e-03])

# in reality the difference between 383Hz and 994Hz is so small that I can absolutely ignore it.
