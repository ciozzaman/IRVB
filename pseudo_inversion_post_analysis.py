# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())


# is_this_extra = False
# # foil_resolution = 47
grid_resolution = 2  # in cm
with_noise = True
# foil_resolution_max = 187
weigthed_best_search = False
# eigenvalue_cut_vertical = False
# enable_mask2 = False
# enable_mask1 = True
treshold_method_try_to_search = True
residuals_on_power_on_voxels = True

spatial_averaging_all = [1, 2,3, 4,5,6,7,8,9,10]



time_averaging_tries = [1,2,3, 4,5,6]

first = 0
for is_this_extra in [False]:
	for flag_mask1_present, flag_mask2_present in [[False, False], [True, False], [True, True]]:
		# for residuals_on_power_on_voxels in [True]:
		radiator_location_present = []
		power_discretisation_fraction_present = []
		fraction_total_power_detected_present = []
		alpha_present = []
		treshold_present = []
		score_x_record_present = []
		spatial_and_time_averaging_present = []
		spatial_averaging_present = []
		time_averaging_present = []
		x = []
		x.append(0)
		for spatial_averaging in spatial_averaging_all:
			for time_averaging in time_averaging_tries:

				if spatial_averaging > 1:
					foil_resolution = str(spatial_averaging) + 'x' + str(spatial_averaging)
				else:
					foil_resolution = '187'

				if is_this_extra:
					foil_res = '_foil_pixel_h_' + str(foil_resolution) + '_extra'
				else:
					foil_res = '_foil_pixel_h_' + str(foil_resolution)
				grid_type = 'core_res_' + str(grid_resolution) + 'cm'
				path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[
																								5:] + foil_res + '_power'
				path_sensitivity_original = copy.deepcopy(path_sensitivity)
				path_for_plots = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res + '_power'

				if flag_mask2_present:
					path_for_plots = path_for_plots + '/mask2'
					print(
						'Sensitivity matrix filtered for voxels that do not see the foil and pixels that are not shone by any plasma')
					mod = 'mask2'
				elif (flag_mask1_present and not flag_mask2_present):
					path_for_plots = path_for_plots + '/mask1'
					print('Sensitivity matrix filtered for voxels that do not see the foil')
					mod = 'mask1'
				else:
					path_for_plots = path_for_plots + '/unmasked'
					print('Sensitivity matrix not filtered')
					mod = 'unmasked'

				if treshold_method_try_to_search:
					path_for_plots = path_for_plots + '/SVG_search_best_cut'
					print('Cut of the eigenvalues defined based minimizing the residuals')
				else:
					if eigenvalue_cut_vertical:
						path_for_plots = path_for_plots + '/SVG_vertical_cut'
						print('Cut of the eigenvalues in the vertical direction')
					else:
						path_for_plots = path_for_plots + '/SVG_horizontal_cut'
						print('Cut of the eigenvalues in the horizontal direction')

				if weigthed_best_search:
					path_for_plots = path_for_plots + '/weighted_best_alpha_search'
					print('Estimation of the residuals weighted on the expected signals')
				else:
					path_for_plots = path_for_plots + '/no_weighted_best_alpha_search'
					print('Estimation of the residuals not weighted')

				if residuals_on_power_on_voxels:
					path_for_plots = path_for_plots + '/residuals_on_power_on_voxels'
					print('Estimation of the residuals on the expected versus inverted power on voxels')
				else:
					path_for_plots = path_for_plots + '/residuals_on_power_on_foil_pixels'
					print('Estimation of the residuals on the expected versus inverted power on foil pixels')

				if with_noise:
					path_for_plots = path_for_plots + '/with_noise_time_averaging_' + str(time_averaging)
				if os.path.exists(path_for_plots + '/stats.csv'):

					type = '.csv'
					filenames = coleval.all_file_names(path_for_plots, type)[0]
					with open(os.path.join(path_for_plots, filenames)) as csv_file:
						csv_reader = csv.reader(csv_file, delimiter=',')
						for row in csv_reader:
							if row[0] == 'with noise, std of noise on power, time averaging ':
								with_noise, noise_on_power, time_averaging = row[1][1:-1].split(', ')
								noise_on_power = float(noise_on_power)
								time_averaging = int(time_averaging)
							if row[
								0] == 'relative to the best alpha: value, eigenvalue threshold, residuals on solution, residuals on smoothing':
								alpha, treshold, score_x_record, score_y_record = row[1][1:-1].split(', ')
								alpha = float(alpha)
								treshold = float(treshold)
								score_x_record = float(score_x_record)
								score_y_record = float(score_y_record)
							if row[0] == 'total input radiated power':
								total_input_radiated_power = float(row[1])
							if row[0] == 'total estimated radiated power':
								total_estimated_radiated_power = float(row[1])
							if row[0] == 'fraction of the total power detected':
								fraction_total_power_detected = float(row[1])


					fraction_total_power_detected_present.append(fraction_total_power_detected)
					alpha_present.append(alpha)
					treshold_present.append(treshold)
					score_x_record_present.append(score_x_record)
					spatial_and_time_averaging_present.append([spatial_averaging,time_averaging])
					spatial_averaging_present.append(spatial_averaging)
					time_averaging_present.append(time_averaging)
					x.append(x[-1] + 1)
				else:
					print(path_for_plots + '/stat.csv \n missing')
					x[-1] = (x[-1] + 1)

		if mod == 'unmasked':
			line_type = '-'
			marker_type = 'o'
		elif mod == 'mask1':
			line_type = '--'
			marker_type = '+'
		elif mod == 'mask2':
			line_type = '-.'
			marker_type = 'v'
		color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm', 'pink']
		plt.figure(1)
		for time_index in range(1,1+max(time_averaging_present)):
			x_to_plot=[]
			y_to_plot = []
			for index in range(len(time_averaging_present)):
				if time_averaging_present[index]==time_index:
					x_to_plot.append(spatial_averaging_present[index])
					y_to_plot.append(score_x_record_present[index])
			plt.plot(x_to_plot,y_to_plot,line_type+color[time_index],label=mod+' time averaging '+str(time_index))
		plt.legend(loc='best')
		plt.grid()
		plt.xlabel('spatial averaging')
		plt.ylabel('residuals')
		plt.pause(0.01)

		plt.figure(2)
		for time_index in range(1,1+max(time_averaging_present)):
			x_to_plot=[]
			y_to_plot = []
			for index in range(len(time_averaging_present)):
				if time_averaging_present[index]==time_index:
					x_to_plot.append(spatial_averaging_present[index])
					y_to_plot.append(fraction_total_power_detected_present[index])
			plt.plot(x_to_plot,y_to_plot,line_type+color[time_index],label=mod+' time averaging '+str(time_index))
		plt.legend(loc='best')
		plt.grid()
		plt.xlabel('spatial averaging')
		plt.ylabel('fraction of input power in the inversion')
		plt.pause(0.01)

		plt.figure(3)
		for time_index in range(1,1+max(time_averaging_present)):
			x_to_plot=[]
			y_to_plot1 = []
			y_to_plot2 = []
			for index in range(len(time_averaging_present)):
				if time_averaging_present[index]==time_index:
					x_to_plot.append(spatial_averaging_present[index])
					y_to_plot1.append(alpha_present[index])
					y_to_plot2.append(treshold_present[index])
			plt.plot(x_to_plot,y_to_plot1,line_type+color[time_index],label=mod+' time averaging '+str(time_index)+' regularisation')
			plt.plot(x_to_plot,y_to_plot2,marker_type+color[time_index],label=mod+' time averaging '+str(time_index)+' SVG treshold')
		plt.title('parameter that minimise the residuals')
		plt.legend(loc='best')
		plt.grid()
		plt.yscale('log')
		plt.xlabel('spatial averaging')
		# plt.ylabel('fraction of input power in the inversion')
		plt.pause(0.01)



print('J O B   D O N E !')

