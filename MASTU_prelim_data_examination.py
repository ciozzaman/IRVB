# Created 13/01/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 14

# added to reat the .ptw
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
print(os.path.abspath(os.getcwd()))
import pyradi.ryptw as ryptw

# degree of polynomial of choice
n=3
# folder of the parameters path
pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']

path = '/home/ffederic/work/irvb/MAST-U/'
to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04']
to_do = np.flip(to_do,axis=0)
# to_do = ['2021-06-03']
# path = '/home/ffederic/work/irvb/MAST-U/preliminaly_shots/'
# to_do = ['2021-05-13','2021-05-12','2021-04-28','2021-04-29','2021-04-30']
f = []
for (dirpath, dirnames, filenames) in os.walk(path):
	f.append(dirnames)
days_available = f[0]
shot_available = []
for i_day,day in enumerate(to_do):
	f = []
	for (dirpath, dirnames, filenames) in os.walk(path+day+'/'):
		f.append(filenames)
	shot_available.append([])
	for name in f[0]:
		if name[-3:]=='ats' or name[-3:]=='ptw':
			shot_available[i_day].append(name)


every_pixel_independent = False
for i_day,day in enumerate(to_do):
# for i_day,day in enumerate(np.flip(to_do,axis=0)):
	# for name in shot_available[i_day]:
	for name in np.flip(shot_available[i_day],axis=0):
		laser_to_analyse=path+day+'/'+name

		try:
			laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
		except:
			print('missing '+laser_to_analyse[:-4]+'.npz'+' file. rigenerated')
			if laser_to_analyse[-4:]=='.ats':
				full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse)
			else:
				full_saved_file_dict = coleval.ptw_to_dict(laser_to_analyse)
			np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
			laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
		full_saved_file_dict = dict(laser_dict)

		try:

			laser_counts, laser_digitizer_ID = coleval.separate_data_with_digitizer(laser_dict)
			time_of_experiment = laser_dict['time_of_measurement']
			laser_digitizer = laser_dict['digitizer_ID']
			time_of_experiment_digitizer_ID, laser_digitizer_ID = coleval.generic_separate_with_digitizer(time_of_experiment,laser_digitizer)
			laser_framerate = 1e6/np.mean(np.sort(np.diff(time_of_experiment))[2:-2])
			laser_int_time = laser_dict['IntegrationTime']

			try:
				full_correction_coefficients = laser_dict['full_correction_coeff']
				full_correction_coefficients_present = True
			except:
				full_correction_coefficients_present = False
				full_correction_coefficients = np.zeros((2,*np.shape(laser_counts[0])[1:],5))*np.nan

			try:
				aggregated_correction_coefficients = laser_dict['aggregated_correction_coeff']
				aggregated_correction_coefficients_present = True
			except:
				aggregated_correction_coefficients_present = False
				aggregated_correction_coefficients = np.zeros((2,5))*np.nan

			fig, ax = plt.subplots( 4,1,figsize=(7, 15), squeeze=False, sharex=True)
			plot_index = 0
			horizontal_coord = np.arange(np.shape(laser_counts[0])[2])
			vertical_coord = np.arange(np.shape(laser_counts[0])[1])
			horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
			select = np.logical_or(np.logical_or(vertical_coord<30,vertical_coord>240),np.logical_or(horizontal_coord<30,horizontal_coord>290))
			startup_correction = []
			laser_counts_corrected = np.array(laser_counts)
			temp=0
			exponential = lambda t,c1,c2,c3,c4,t0: -c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))
			external_clock_marker = 0
			for i in range(len(laser_digitizer_ID)):
				reference = np.mean(laser_counts[i][:,select],axis=-1)
				if np.abs(reference[-1]-reference[10])>30:
					external_clock_marker+=1
				laser_counts_corrected[i] = laser_counts_corrected[i].astype(np.float)
				temp = max(temp,np.max(np.mean(laser_counts[i][10:],axis=(-1,-2))))
				time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6
				bds = [[0,0,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf]]
				guess=[100,1,0,0,time_of_experiment_digitizer_ID_seconds[10]]
				select_time = np.logical_or(time_of_experiment_digitizer_ID_seconds<2.4,time_of_experiment_digitizer_ID_seconds>8)	# I avoid the period with the plasma
				select_time[:8]=False	# to avoid the very firsts frames that can have unreasonable counts
				ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds,np.mean(laser_counts[i],axis=(-1,-2)),color=color[i],label='all foil mean DIG'+str(laser_digitizer_ID[i]))
				ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds,reference,'--',color=color[i],label='out of foil area DIG'+str(laser_digitizer_ID[i]))
				if every_pixel_independent:	# I do the startup correction for every pixel independently
					for v in range(np.shape(laser_counts[0])[1]):
						for h in range(np.shape(laser_counts[0])[2]):
							if full_correction_coefficients_present:
								laser_counts_corrected[i][:,v,h] += -exponential(time_of_experiment_digitizer_ID_seconds,*full_correction_coefficients[i,v,h])
							else:
								try:
									fit = curve_fit(exponential, time_of_experiment_digitizer_ID_seconds[select_time], laser_counts[i][select_time,v,h]-np.mean(laser_counts[i][-int(1*laser_framerate/len(laser_digitizer_ID)):,v,h]), p0=guess,bounds=bds,maxfev=int(1e6))
									laser_counts_corrected[i][:,v,h] += -exponential(time_of_experiment_digitizer_ID_seconds,*fit[0])
									full_correction_coefficients[i,v,h] = fit[0]
									guess = fit[0]
								except:
									print('%.3gv,%.3gh failed' %(v,h))
									continue
				else:
					fit = curve_fit(exponential, time_of_experiment_digitizer_ID_seconds[select_time], reference[select_time]-np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]), p0=guess,bounds=bds,maxfev=int(1e6))
					aggregated_correction_coefficients[i] = fit[0]
					ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds,exponential(time_of_experiment_digitizer_ID_seconds,*fit[0])+np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]),':',color=color[i],label='fit of out of foil area')
					# startup_correction.append(np.mean(reference[(time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6>12]) - reference)
					# startup_correction.append(-exponential((time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6,*fit[0]))
					laser_counts_corrected[i] = (laser_counts_corrected[i].T + -exponential((time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6,*fit[0])).T
					ax[plot_index,0].axhline(y=np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]),linestyle=':',color=color[i])
			ax[plot_index,0].set_ylabel('mean counts [au]')
			fig.suptitle(day+'/'+name+'\nint time %.3gms, framerate %.3gHz' %(laser_int_time/1000,laser_framerate))
			ax[plot_index,0].set_ylim(top=temp)
			ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID)))
			ax[plot_index,0].grid()
			ax[plot_index,0].legend(loc='best', fontsize='x-small')

			plot_index += 1
			ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6,laser_dict['SensorTemp_0'])
			ax[plot_index,0].set_ylabel('ambient temp [K]\n(SensorTemp_0)')
			ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID)))
			ax[plot_index,0].grid()
			plot_index += 1
			ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6,laser_dict['SensorTemp_3'])
			ax[plot_index,0].set_ylabel('detector temp [K]\n(SensorTemp_3)')
			ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID)))
			ax[plot_index,0].grid()
			plot_index += 1
			ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6,laser_dict['DetectorTemp'])
			ax[plot_index,0].set_ylabel('detector temp [K]\n(DetectorTemp)')
			ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID)))
			ax[plot_index,0].grid()
			ax[plot_index,0].set_xlabel('time [s]')

			if every_pixel_independent:	# I do the startup correction for every pixel independently
				if full_correction_coefficients_present:
					if np.sum((full_saved_file_dict['full_correction_coeff']!=full_correction_coefficients))>0:
						full_saved_file_dict['full_correction_coeff'] = full_correction_coefficients
						full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))'
						np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
				else:
					full_saved_file_dict['full_correction_coeff'] = full_correction_coefficients
					full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))'
					np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
			else:
				if aggregated_correction_coefficients_present:
					if np.sum((full_saved_file_dict['aggregated_correction_coeff']!=aggregated_correction_coefficients))>1:
						full_saved_file_dict['aggregated_correction_coeff'] = aggregated_correction_coefficients
						full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))'
						np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
				else:
					full_saved_file_dict['aggregated_correction_coeff'] = aggregated_correction_coefficients
					full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))'
					np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
				plt.savefig(laser_to_analyse[:-4]+'_1.eps', bbox_inches='tight')
			plt.close('all')

			ROI_horizontal = [220,235,50,100]
			# ROI_horizontal = [35,45,68,73]
			ROI_vertical = [100,170,25,45]
			plt.figure(figsize=(10, 5))
			for i in range(len(laser_digitizer_ID)):
				# horizontal_displacement = np.diff((np.sum(laser_counts[i][:,ROI_horizontal[0]:ROI_horizontal[1],ROI_horizontal[2]:ROI_horizontal[3]],axis=2).T/(np.sum(laser_counts[i][:,ROI_horizontal[0]:ROI_horizontal[1],ROI_horizontal[2]:ROI_horizontal[3]],axis=2)[:,0])).T,axis=1).argmin(axis=1)
				horizontal_displacement = np.diff((np.sum(laser_counts[i][:,ROI_horizontal[0]:ROI_horizontal[1],ROI_horizontal[2]:ROI_horizontal[3]],axis=2)),axis=1).argmin(axis=1)
				horizontal_displacement = horizontal_displacement-horizontal_displacement[10]
				# vertical_displacement = np.diff((np.sum(laser_counts[i][:,ROI_vertical[0]:ROI_vertical[1],ROI_vertical[2]:ROI_vertical[3]],axis=1).T/(np.sum(laser_counts[i][:,ROI_vertical[0]:ROI_vertical[1],ROI_vertical[2]:ROI_vertical[3]],axis=1)[:,0])).T,axis=1).argmax(axis=1)
				vertical_displacement = np.diff((np.sum(laser_counts[i][:,ROI_vertical[0]:ROI_vertical[1],ROI_vertical[2]:ROI_vertical[3]],axis=1)),axis=1).argmax(axis=1)
				vertical_displacement = vertical_displacement-vertical_displacement[10]
				time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6
				if i==0:
					plt.plot(time_of_experiment_digitizer_ID_seconds,horizontal_displacement,label='horizontal',colplt.cloor=color[0])
					plt.plot(time_of_experiment_digitizer_ID_seconds,vertical_displacement,label='vertical',color=color[1])
				else:
					plt.plot(time_of_experiment_digitizer_ID_seconds,horizontal_displacement,color=color[0])
					plt.plot(time_of_experiment_digitizer_ID_seconds,vertical_displacement,color=color[1])
			plt.xlabel('time [s]')
			plt.ylabel('pixels of displacement [au]')
			plt.title('Displacement of the image\nlooking for the effect of disruptions')
			plt.legend(loc='best', fontsize='x-small')
			plt.savefig(laser_to_analyse[:-4]+'_2.eps', bbox_inches='tight')
			plt.close('all')

			if every_pixel_independent:	# I do the startup correction for every pixel independently
				for i in range(len(laser_digitizer_ID)):
					fig, ax = plt.subplots( 5,1,figsize=(5, 13), squeeze=False, sharex=True)
					fig.suptitle(day+'/'+name+'\nint time %.3gms, framerate %.3gHz' %(laser_int_time/1000,laser_framerate) + '\ndigitizer '+str(laser_digitizer_ID[i]))

					plot_index = 0
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,0],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('c1 [au]')
					ax[plot_index,0].set_title('correction coefficient c1 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')

					plot_index += 1
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,1],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('c2 [1/s]')
					ax[plot_index,0].set_title('correction coefficient c2 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')

					plot_index += 1
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,2],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('c3 [1/s^2]')
					ax[plot_index,0].set_title('correction coefficient c3 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')

					plot_index += 1
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,3],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('c4 [1/s^2]')
					ax[plot_index,0].set_title('correction coefficient c4 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')

					plot_index += 1
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,4],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('t0 [s]')
					ax[plot_index,0].set_title('correction coefficient t0 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_xlabel('Horizontal axis [pixles]')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')
					plt.savefig(laser_to_analyse[:-4]+'_corr_coeff_dig'+str(laser_digitizer_ID[i])+'.eps', bbox_inches='tight')
					plt.close('all')


			# plt.figure(figsize=(8, 5))
			# plt.imshow(laser_counts[0][3],'rainbow',origin='lower')
			# plt.colorbar().set_label('counts [au]')
			# # plt.title('Only foil in frame '+str(0)+' in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
			# plt.xlabel('Horizontal axis [pixles]')
			# plt.ylabel('Vertical axis [pixles]')
			# plt.pause(0.01)

			for i in laser_digitizer_ID:
				laser_counts_corrected[i] = laser_counts_corrected[i].tolist()
			laser_counts_corrected = list(laser_counts_corrected)

			if not every_pixel_independent:
				for i in range(len(laser_digitizer_ID)):
					laser_counts_corrected[i] = np.flip(np.transpose(laser_counts_corrected[i],(0,2,1)),axis=2).astype(np.float)
					plt.figure(figsize=(8, 5))
					plt.imshow(np.mean(laser_counts_corrected[i][-int(1*(laser_framerate/len(laser_digitizer_ID))):],axis=0),'rainbow',origin='lower')
					temp = np.shape(laser_counts_corrected[i])[2]
					plt.plot([temp-ROI_horizontal[0],temp-ROI_horizontal[0],temp-ROI_horizontal[1],temp-ROI_horizontal[1],temp-ROI_horizontal[0]],[ROI_horizontal[2],ROI_horizontal[3],ROI_horizontal[3],ROI_horizontal[2],ROI_horizontal[2]],'--k')
					plt.plot([temp-ROI_vertical[0],temp-ROI_vertical[0],temp-ROI_vertical[1],temp-ROI_vertical[1],temp-ROI_vertical[0]],[ROI_vertical[2],ROI_vertical[3],ROI_vertical[3],ROI_vertical[2],ROI_vertical[2]],'--k')
					plt.plot([0,temp],[220]*2,'--k')
					plt.plot([temp-30,temp-240,temp-240,temp-30,temp-30],[30,30,290,290,30],'--k')
					plt.colorbar().set_label('counts [au]')
					plt.xlabel('Horizontal axis [pixles]')
					plt.ylabel('Vertical axis [pixles]')
					plt.title('Counts Background digitizer '+str(laser_digitizer_ID[i]))
					plt.savefig(laser_to_analyse[:-4]+'_back_dig_'+str(laser_digitizer_ID[i])+'.eps', bbox_inches='tight')
					plt.close('all')
					laser_counts_corrected[i]-=np.mean(laser_counts_corrected[i][-int(1*(laser_framerate/len(laser_digitizer_ID))):],axis=0)
					# laser_counts_corrected[i] = median_filter(laser_counts_corrected[i],size=[1,3,3])
					# if not(os.path.exists(laser_to_analyse[:-4]+'_digitizer'+str(i) + '.mp4')):
					if external_clock_marker:
						select_time = np.logical_and((time_of_experiment_digitizer_ID[i] - time_of_experiment_digitizer_ID[i][0])*1e-6-aggregated_correction_coefficients[i,4]>2.5,(time_of_experiment_digitizer_ID[i] - time_of_experiment_digitizer_ID[i][0])*1e-6-aggregated_correction_coefficients[i,4]<3+2.5)
						ani = coleval.movie_from_data(np.array([laser_counts_corrected[i][select_time]]), laser_framerate/len(laser_digitizer_ID), integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='adjusted counts digitizer '+str(i)+' [au]')
					else:
						select_time = np.logical_and((time_of_experiment_digitizer_ID[i] - time_of_experiment_digitizer_ID[i][0])*1e-6>0,(time_of_experiment_digitizer_ID[i] - time_of_experiment_digitizer_ID[i][0])*1e-6<3+2.5)
						ani = coleval.movie_from_data(np.array([laser_counts_corrected[i][select_time]]), laser_framerate/len(laser_digitizer_ID), integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='adjusted counts digitizer '+str(i)+' [au]',prelude='internal clocks, unset t=0\n')
					# plt.pause(0.01)
					ani.save(laser_to_analyse[:-4]+'_digitizer'+str(i) + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
					plt.close('all')
				laser_counts_corrected_merged = np.array([(laser_counts_corrected[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_counts_corrected[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)])
				laser_counts_merged = np.array([(laser_counts[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_counts[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)])
				# if not(os.path.exists(laser_to_analyse[:-4] + '.mp4')):
				if external_clock_marker:
					select_time = np.logical_and((time_of_experiment - time_of_experiment[0])*1e-6>2.5,(time_of_experiment - time_of_experiment[0])*1e-6<3+2.5)
					ani = coleval.movie_from_data(np.array([laser_counts_corrected_merged[select_time]]), laser_framerate, integration=laser_int_time/1000,extvmin=0,extvmax=laser_counts_corrected_merged[10:,:220].max(),xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='adjusted counts [au]')
				else:
					select_time = np.logical_and((time_of_experiment - time_of_experiment[0])*1e-6>0,(time_of_experiment - time_of_experiment[0])*1e-6<3+2.5)
					ani = coleval.movie_from_data(np.array([laser_counts_corrected_merged[select_time]]), laser_framerate, integration=laser_int_time/1000,extvmin=0,extvmax=laser_counts_corrected_merged[10:,:220].max(),xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='adjusted counts [au]',prelude='internal clocks, unset t=0\n')
				# plt.pause(0.01)
				ani.save(laser_to_analyse[:-4]+ '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
				ani = coleval.movie_from_data(np.array([laser_counts_merged]), laser_framerate, integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='raw counts [au]')
				# plt.pause(0.01)
				ani.save(laser_to_analyse[:-4]+ '_raw.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
				plt.close('all')

			print('completed ' + laser_to_analyse)
		except Exception as e:
			print('FAILED ' + laser_to_analyse)
			print('with error: ' + str(e))
