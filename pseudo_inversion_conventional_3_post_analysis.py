# Created 29/09/2019
# Fabio Federici


# #this is if working on a pc, use pc printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

#this is if working in batch, use predefined NOT visual printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())




grid_resolution = 2  # in cm
foil_resolution_max = 187
foil_resolution = '187'
weigthed_best_search = False
eigenvalue_cut_vertical = False
enable_mask2 = False
enable_mask1 = True
flag_mask2_present = False
flag_mask1_present = True

foil_res = '_foil_pixel_h_' + str(foil_resolution)
grid_type = 'core_res_' + str(grid_resolution) + 'cm'

alpha_exponents_to_test = np.linspace(-13, -1, num=49)

spatial_averaging_all = [1, 2, 3, 4, 5, 6, 8, 10]
time_averaging_all = [1]

from raysect.optical import World, ConstantSF
world = World()
from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
from cherab.mastu.bolometry import load_default_bolometer_config, load_standard_voxel_grid
from cherab.tools.inversions import ToroidalVoxelGrid
if enable_mask1:
	core_voxel_grid = load_standard_voxel_grid(grid_type + '_masked1', parent=world)
else:
	core_voxel_grid = load_standard_voxel_grid(grid_type, parent=world)
num_voxels = core_voxel_grid.count

# for ref_number in [*mdsnos_cd15H,*mdsnos_cd2H,*mdsnos_sxd2L,*mdsnos_sxd15H,*mdsnos_sxd2H,'seeding_10']:
for ref_number in ['seeding_10']:
	try:
		print('results for SOPLS simulation '+str(ref_number))


		path_for_plots = '/home/ffederic/work/analysis_scripts'+'/SOLPS'+str(ref_number)+'/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'

		if flag_mask2_present:
			path_for_plots = path_for_plots + '/mask2'
			print('Sensitivity matrix filtered for voxels that do not see the foil and pixels that are not shone by any plasma')
			mod = '_mask2'
		elif (flag_mask1_present and not flag_mask2_present):
			path_for_plots = path_for_plots + '/mask1'
			print('Sensitivity matrix filtered for voxels that do not see the foil')
			mod = '_mask1'
		else:
			path_for_plots = path_for_plots + '/unmasked'
			print('Sensitivity matrix not filtered')
			mod = ''

		path_for_plots = path_for_plots + '/only_Tikhonov'
		print('Only Tikhonov regularization performed')

		if weigthed_best_search:
			path_for_plots = path_for_plots + '/weighted_best_alpha_search'
			print('Estimation of the residuals weighted on the expected signals')
		else:
			path_for_plots = path_for_plots + '/no_weighted_best_alpha_search'
			print('Estimation of the residuals not weighted')





		x_point = Point2D(0.55,-1.25)	# m
		x_point_radious_small = 0.05	# m
		x_point_radious_med = 0.1  # m
		x_point_radious_large = 0.15  # m

		zone_indicator = np.zeros((num_voxels))
		for index in range(num_voxels):
			p1, p2, p3, p4 = core_voxel_grid._voxels[index].vertices
			voxel_centre = Point2D((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4)
			if voxel_centre.distance_to(x_point)<x_point_radious_small:	# small x-point	-->> 0 <<--
				zone_indicator[index]+=100
			if voxel_centre.distance_to(x_point)<x_point_radious_med:	# med x-point	-->> 1 <<--
				zone_indicator[index]+=10
			if voxel_centre.distance_to(x_point)<x_point_radious_large:	# large x-point	-->> 2 <<--
				zone_indicator[index]+=1
			if (voxel_centre.x<0.55 and voxel_centre.y<(-0.96*voxel_centre.x-0.822)):	# inner strike point	-->> 3 <<--
				zone_indicator[index] += 1000
			if (voxel_centre.x > 0.55 and voxel_centre.x <= 1.09 and voxel_centre.y < -1.35):  # conventional divertor	-->> 4 <<--
				zone_indicator[index] += 10000
			if (voxel_centre.x>1.09 and voxel_centre.y<-1.35):	# super-x divertor	-->> 5 <<--
				zone_indicator[index] += 100000
			if (voxel_centre.y>-1.15):	# core	-->> 6 <<--
				zone_indicator[index] += 0.1
		core_voxel_grid.plot(voxel_values=zone_indicator, colorbar=['rainbow', 'voxel division','log'])
		plt.title('Voxel regions identification\ncore=10^-1, large x-point=10^0, med x-point=10^1, small x-point=10^2,\ninner strike point=10^3, conventional divertor=10^4,\nsuper-x divertor=10^5')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.rcParams["figure.figsize"] = [8, 10]
		plt.savefig(path_for_plots + '/voxel_regions_identification.eps')
		plt.close()

		input_emissivity = np.load(path_for_plots + '/residuals_on_power_on_voxels' + '/input_emissivity.npy')

		# total_power_1 = 0
		# total_power_2 = 0
		# total_power_3 = 0
		# total_power_4 = 0
		# total_power_5 = 0
		# total_power_6 = 0
		# total_power_7 = 0
		input_total_power = np.zeros((8))

		total_rad_power = 0
		for index in range(num_voxels):
			p1, p2, p3, p4 = core_voxel_grid._voxels[index].vertices
			r_maj = max([p1.x, p2.x, p3.x, p4.x])
			r_min = min([p1.x, p2.x, p3.x, p4.x])
			thickness = max([p1.y, p2.y, p3.y, p4.y]) - min([p1.y, p2.y, p3.y, p4.y])
			voxel_centre = Point2D((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4)
			volume = np.pi * thickness * (r_maj ** 2 - r_min ** 2)
			# volume = core_voxel_grid._voxels[index].volume
			if voxel_centre.distance_to(x_point)<x_point_radious_small:	# small x-point	-->> 0 <<--
				input_total_power[0] += input_emissivity[index] * volume
			if voxel_centre.distance_to(x_point)<x_point_radious_med:	# med x-point	-->> 1 <<--
				input_total_power[1] += input_emissivity[index] * volume
			if voxel_centre.distance_to(x_point)<x_point_radious_large:	# large x-point	-->> 2 <<--
				input_total_power[2] += input_emissivity[index] * volume
			if (voxel_centre.x<0.55 and voxel_centre.y<(-0.96*voxel_centre.x-0.822)):	# inner strike point	-->> 3 <<--
				input_total_power[3] += input_emissivity[index] * volume
			if (voxel_centre.x > 0.55 and voxel_centre.x <= 1.09 and voxel_centre.y < -1.35):  # conventional divertor	-->> 4 <<--
				input_total_power[4] += input_emissivity[index] * volume
			if (voxel_centre.x>1.09 and voxel_centre.y<-1.35):	# super-x divertor	-->> 5 <<--
				input_total_power[5] += input_emissivity[index] * volume
			if (voxel_centre.y>-1.15):	# core	-->> 6 <<--
				input_total_power[6] += input_emissivity[index] * volume
			input_total_power[7] += input_emissivity[index] * volume
			total_rad_power += input_emissivity[index] * volume

		inverted_total_power = np.zeros((2,2,len(time_averaging_all), len(spatial_averaging_all), 8))
		all_alpha = np.zeros((2,2,len(time_averaging_all), len(spatial_averaging_all)))
		for i_r,residuals_on_power_on_voxels in enumerate([True, False]):
			if residuals_on_power_on_voxels:
				label_res = 'res on voxels'
			else:
				label_res = 'res on foil'
			for i_n,with_noise in enumerate([False,True]):
				if with_noise:
					label_noise='with noise'
					time_averaging_tries = cp.deepcopy(time_averaging_all)
				else:
					label_noise = 'no noise'
					time_averaging_tries = [1]
				for i_t, time_averaging in enumerate(time_averaging_tries):
					for i_s,spatial_averaging in enumerate(spatial_averaging_all):

						if residuals_on_power_on_voxels:
							path_for_plots_internal = path_for_plots + '/residuals_on_power_on_voxels'
							print('Estimation of the residuals on the expected versus inverted power on voxels')
						else:
							path_for_plots_internal = path_for_plots + '/residuals_on_power_on_foil_pixels'
							print('Estimation of the residuals on the expected versus inverted power on foil pixels')

						if spatial_averaging > 1:
							path_for_plots_internal = path_for_plots_internal + '/pixel_binning_of_' + str(spatial_averaging) + 'x' + str(spatial_averaging)
							print('Binning of foil pixels of ' + str(spatial_averaging) + 'x' + str(spatial_averaging))
						else:
							print('No pixel binning')

						if with_noise:
							path_for_plots_internal = path_for_plots_internal + '/with_noise_time_averaging_' + str(time_averaging)

						input_emissivity = np.load(path_for_plots_internal+'/inverted_emissivity.npy')

						for index in range(num_voxels):
							p1, p2, p3, p4 = core_voxel_grid._voxels[index].vertices
							r_maj = max([p1.x, p2.x, p3.x, p4.x])
							r_min = min([p1.x, p2.x, p3.x, p4.x])
							thickness = max([p1.y, p2.y, p3.y, p4.y]) - min([p1.y, p2.y, p3.y, p4.y])
							voxel_centre = Point2D((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4)
							volume = np.pi * thickness * (r_maj ** 2 - r_min ** 2)
							# volume = core_voxel_grid._voxels[index].volume
							if voxel_centre.distance_to(x_point) < x_point_radious_small:  # small x-point	-->> 0 <<--
								inverted_total_power[i_r,i_n,i_t,i_s,0] += input_emissivity[index] * volume
							if voxel_centre.distance_to(x_point) < x_point_radious_med:  # med x-point	-->> 1 <<--
								inverted_total_power[i_r,i_n,i_t,i_s,1] += input_emissivity[index] * volume
							if voxel_centre.distance_to(x_point) < x_point_radious_large:  # large x-point	-->> 2 <<--
								inverted_total_power[i_r,i_n,i_t,i_s,2] += input_emissivity[index] * volume
							if (voxel_centre.x < 0.55 and voxel_centre.y < (-0.96 * voxel_centre.x - 0.822)):  # inner strike point	-->> 3 <<--
								inverted_total_power[i_r,i_n,i_t,i_s,3] += input_emissivity[index] * volume
							if (voxel_centre.x > 0.55 and voxel_centre.x <= 1.09 and voxel_centre.y < -1.35):  # conventional divertor	-->> 4 <<--
								inverted_total_power[i_r,i_n,i_t,i_s,4] += input_emissivity[index] * volume
							if (voxel_centre.x > 1.09 and voxel_centre.y < -1.35):  # super-x divertor	-->> 5 <<--
								inverted_total_power[i_r,i_n,i_t,i_s,5] += input_emissivity[index] * volume
							if (voxel_centre.y > -1.15):  # core	-->> 6 <<--
								inverted_total_power[i_r,i_n,i_t,i_s,6] += input_emissivity[index] * volume
							inverted_total_power[i_r, i_n, i_t, i_s, 7] += input_emissivity[index] * volume

						with open(path_for_plots_internal + '/stats.csv') as csv_file:
							csv_reader = csv.reader(csv_file, delimiter=',')
							# csv_reader = csv.reader(csv_file,delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
							line_count = 0
							for row in csv_reader:
								if row[0] == 'relative to the best alpha: value, residuals on solution, residuals on smoothing':
									all_alpha[i_r,i_n,i_t,i_s] = float(row[1].split(',')[0][1:])


		line_type = [ '--',':', 'v', 'o','x', '+', '*', '>', 'D', '<']
		sum_type = ['small x-point d='+str(2*x_point_radious_small)+'m','med x-point d='+str(2*x_point_radious_med)+'m','large x-point d='+str(2*x_point_radious_large)+'m','inner strike point','conventional divertor','super-x divertor','core','total rad power']
		plt.figure(figsize=(20, 10))
		for index in range(len(inverted_total_power[0,0,0,0])):
			plt.plot(np.array(spatial_averaging_all) ** 2, np.ones_like(spatial_averaging_all)*input_total_power[index],'C' + str(index) + '-',label=sum_type[index])
			for i_r,residuals_on_power_on_voxels in enumerate([True, False]):
				if residuals_on_power_on_voxels:
					label_res = 'res on voxels'
				else:
					label_res = 'res on foil'
				for i_n,with_noise in enumerate([False,True]):
					if with_noise:
						label_noise = 'with noise'
						time_averaging_tries = cp.deepcopy(time_averaging_all)
					else:
						label_noise = 'no noise'
						time_averaging_tries = [1]
					for i_t, time_averaging in enumerate(time_averaging_tries):
						# plt.plot(np.array(spatial_averaging_all)**2,inverted_total_power[i_r,i_n,i_t,:,index],'C'+str(index)+line_type[int((i_r*2+i_n)*len(time_averaging_tries)+i_t)],label=sum_type[index]+', time ave='+str(time_averaging)+', '+label_noise+', '+label_res)
						plt.plot(np.array(spatial_averaging_all)**2,inverted_total_power[i_r,i_n,i_t,:,index],'C'+str(index)+line_type[int((i_r*2+i_n)*len(time_averaging_tries)+i_t)])
		plt.title("Total power measured in different areas for SOLPS "+str(ref_number)+"\nsolid=from phantom, '"+line_type[0]+"'=no noise-res on voxels, \n'"+line_type[1]+"'=with noise-res on voxels, '"+line_type[2]+"'=no noise-res on foil, '"+line_type[3]+"'=with noise-res on foil")
		plt.xlabel('Number of pixels averaged [pixles]')
		plt.ylabel('Power [W]')
		plt.xscale('log')
		plt.yscale('log')
		plt.grid()
		plt.legend(loc='best')
		plt.savefig(path_for_plots + '/power_per_region.eps')
		plt.close()




		plt.figure(figsize=(20, 10))
		for i_r,residuals_on_power_on_voxels in enumerate([True, False]):
			if residuals_on_power_on_voxels:
				label_res = 'res on voxels'
			else:
				label_res = 'res on foil'
			for i_n,with_noise in enumerate([False,True]):
				if with_noise:
					label_noise = 'with noise'
					time_averaging_tries = cp.deepcopy(time_averaging_all)
				else:
					label_noise = 'no noise'
					time_averaging_tries = [1]
				for i_t, time_averaging in enumerate(time_averaging_tries):
					plt.plot(np.array(spatial_averaging_all)**2,all_alpha[i_r,i_n,i_t],'C'+str(int((i_r*2+i_n)*len(time_averaging_tries)+i_t)),label='time ave='+str(time_averaging)+', '+label_noise+', '+label_res)
		plt.title("Tikhonov regularization parameter with averaging for SOLPS "+str(ref_number)+' total input radiated power '+ '%.3g' % total_rad_power + 'W')
		plt.xlabel('Number of pixels averaged [pixles]')
		plt.ylabel('Regularisation parameter [au]')
		plt.xscale('log')
		plt.yscale('log')
		plt.grid()
		plt.legend(loc='best')
		plt.savefig(path_for_plots + '/Tikhonov_regularisation_param.eps')
		plt.close()

		print('SOLPS'+str(ref_number)+" worked")

	except:
		print('SOLPS'+str(ref_number)+" didn't work")


