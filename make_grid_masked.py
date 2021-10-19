# Created 13/12/2018
# Fabio Federici


# #this is if working on a pc, use pc printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

#this is if working in batch, use predefined NOT visual printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

import pickle
import cherab.mastu.bolometry.grid_construction








# 2019-03-08 I want to try to improve the inversion by:
#
#   1   reducing the voxels to just the one that see the foil
#   2   reducing the pixels to only the one that can see a decent plasma (using the test case I used in the beginning)
#
#
#   step 1
#

is_this_extra = False
averaging = 6

# for averaging in [7,8,9,10,15]:
for averaging in [1]:
	for is_this_extra in [True,False]:
		grid_resolution = 4  # in cm
		with_noise = True
		foil_resolution_max = 187
		weigthed_best_search = True

		if averaging>1:
			foil_resolution = str(averaging)+'x'+str(averaging)
		else:
			foil_resolution = '187'

		if is_this_extra:
			foil_res = '_foil_pixel_h_' + str(foil_resolution) + '_extra'
		else:
			foil_res = '_foil_pixel_h_' + str(foil_resolution)

		grid_type = 'core_res_' + str(grid_resolution) + 'cm'
		path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res + '_power'

		if not os.path.exists(path_sensitivity + '/sensitivity.npz'):
			sensitivities = np.load(path_sensitivity + '/sensitivity.npy')
			scipy.sparse.save_npz(path_sensitivity + '/sensitivity.npz', scipy.sparse.csr_matrix(sensitivities))
		else:
			sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity.npz')).todense())

		filenames = coleval.all_file_names(path_sensitivity, '.csv')[0]
		with open(os.path.join(path_sensitivity, filenames)) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if row[0] == 'foil horizontal pixels ':
					pixel_h = int(row[1])
				if row[0] == 'pipeline type ':
					pipeline = row[1]
				if row[0] == 'type of volume grid ':
					grid_type = row[1]
			# print(row)



		directory = '/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction'
		grid_file = os.path.join(
			directory,
			'{}_rectilinear_grid.pickle'.format(grid_type)
		)
		with open(grid_file, 'rb') as f:
			grid_data_all = pickle.load(f)
		grid_laplacian = grid_data_all['laplacian']
		grid_mask = grid_data_all['mask']
		grid_data = grid_data_all['voxels']
		grid_index_2D_to_1D_map = grid_data_all['index_2D_to_1D_map']
		grid_index_1D_to_2D_map = grid_data_all['index_1D_to_2D_map']



		grid_data_masked1=[]
		sensitivities_masked1=[]
		voxels_to_erase=[]
		for index,sens in enumerate(sensitivities.T):
			sens_sum = np.sum(sens)
			# print(sens_sum)
			if sens_sum==0:
				voxels_to_erase.append(index)
			else:
				grid_data_masked1.append(grid_data[index])
				sensitivities_masked1.append(sensitivities[:,index])
		grid_data_masked1 = np.array(grid_data_masked1)
		sensitivities_masked1 = np.array(sensitivities_masked1).T


		grid_index_2D_to_1D_map_masked1 = copy.deepcopy(grid_index_2D_to_1D_map)
		grid_index_1D_to_2D_map_masked1 = copy.deepcopy(grid_index_1D_to_2D_map)
		grid_mask_masked1 = copy.deepcopy(grid_mask)
		grid_laplacian_masked1 = copy.deepcopy(grid_laplacian)

		voxels_already_deleted = 0
		for index in voxels_to_erase:
			index_relative = index -voxels_already_deleted
			voxel_2D_location = grid_index_1D_to_2D_map[index]
			if not (np.sum(sensitivities.T[index]) == 0):
				print('some indexing is wrong in voxel '+str(index))
				exit()
			if not (grid_index_2D_to_1D_map_masked1.pop(voxel_2D_location)==index):
				print('Error in handling the dictionary grid_index_2D_to_1D_map_masked1')
				exit()
			if not (grid_index_1D_to_2D_map_masked1.pop(index)==voxel_2D_location):
				print('Error in handling the dictionary grid_index_1D_to_2D_map_masked1')
				exit()
			grid_mask_masked1[voxel_2D_location]=False
			grid_laplacian_masked1 = np.delete(grid_laplacian_masked1 , (index_relative) , axis=1)
			grid_laplacian_masked1 = np.delete(grid_laplacian_masked1, (index_relative), axis=0)
			voxels_already_deleted+=1





		grid_masked1 = {
			'voxels': grid_data_masked1,
			'index_2D_to_1D_map': grid_index_2D_to_1D_map_masked1,
			'index_1D_to_2D_map': grid_index_1D_to_2D_map_masked1,
			'mask': grid_mask_masked1,
			'laplacian': grid_laplacian_masked1,
		}

		# Save the files in the same directory as the loader module
		directory = os.path.split(cherab.mastu.bolometry.grid_construction.__file__)[0]
		file_name = 'core_res_'+str(grid_resolution)+'cm_masked1_rectilinear_grid.pickle'
		file_path = os.path.join(directory, file_name)
		if not os.path.exists(file_path):
			with open(file_path, "wb") as f:
				pickle.dump(grid_masked1, f)

		scipy.sparse.save_npz(path_sensitivity + '/sensitivity_masked1.npz', scipy.sparse.csr_matrix(sensitivities_masked1))










		#
		#
		#   step 2
		#











		from cherab.core.math.mask import PolygonMask2D
		import cherab.mastu.bolometry.grid_construction

		_MASTU_CORE_GRID_POLYGON = np.array([
			(1.49, -0.0),
			(1.49, -1.007),
			(1.191, -1.007),
			(0.893, -1.304),
			(0.868, -1.334),
			(0.847, -1.368),
			(0.832, -1.404),
			(0.823, -1.442),
			(0.82, -1.481),
			(0.82, -1.49),
			(0.825, -1.522),
			(0.84, -1.551),
			(0.864, -1.573),
			(0.893, -1.587),
			(0.925, -1.59),
			(1.69, -1.552),
			(2.0, -1.56),
			(2.0, -2.169),
			(1.319, -2.169),
			(1.769, -1.719),
			(1.73, -1.68),
			(1.35, -2.06),
			(1.09, -2.06),
			(0.9, -1.87),
			(0.36, -1.33),
			(0.333, -1.303),
			(0.333, -1.1),
			(0.261, -0.5),
			(0.261, 0.0),
			(0.261, 0.2),     # point added just to cut the unnecessary voxels the 04/02/2018
			# (0.261, 0.5),
			# (0.333, 0.8),     # point added just to cut the unnecessary voxels	# Replaced 04/02/2018
			# (0.333, 1.1),
			# (0.333, 1.303),
			# (0.36, 1.33),
			# (0.9, 1.87),
			# (1.09, 2.06),
			# (1.35, 2.06),
			# (1.73, 1.68),
			# (1.769, 1.719),
			# (1.319, 2.169),
			# (2.0, 2.169),
			# (2.0, 1.56),
			# (1.69, 1.552),
			# (0.925, 1.59),
			# (0.893, 1.587),
			# (0.864, 1.573),
			# (0.84, 1.551),
			# (0.825, 1.522),
			# (0.82, 1.49),
			# (0.82, 1.481),
			# (0.823, 1.442),
			# (0.832, 1.404),
			# (0.847, 1.368),
			# (0.868, 1.334),
			# (0.893, 1.304),
			# (1.191, 1.007),
			# (1.49, 1.007)
			# (1.49, 0.8)     # point added just to cut the unnecessary voxels	# Replaced 04/02/2018
			(1.49, 0.75)  # point added just to cut the unnecessary voxels the 04/02/2018
		])

		CORE_POLYGON_MASK = PolygonMask2D(_MASTU_CORE_GRID_POLYGON)


		_MASTU_PLASMA_REGION = np.array([
			(-0.9000, 0.0000),
			(-0.8994, 0.0464),
			(-0.8977, 0.0920),
			(-0.8949, 0.1367),
			(-0.8910, 0.1805),
			(-0.8861, 0.2235),
			(-0.8802, 0.2656),
			(-0.8734, 0.3069),
			(-0.8656, 0.3473),
			(-0.8570, 0.3868),
			(-0.8475, 0.4256),
			(-0.8372, 0.4635),
			(-0.8262, 0.5005),
			(-0.8144, 0.5368),
			(-0.8019, 0.5722),
			(-0.7888, 0.6068),
			(-0.7750, 0.6406),
			(-0.7607, 0.6736),
			(-0.7458, 0.7058),
			(-0.7303, 0.7372),
			(-0.7145, 0.7678),
			(-0.6981, 0.7976),
			(-0.6814, 0.8267),
			(-0.6643, 0.8549),
			(-0.6469, 0.8824),
			(-0.6292, 0.9091),
			(-0.6112, 0.9351),
			(-0.5930, 0.9603),
			(-0.5746, 0.9847),
			(-0.5561, 1.0084),
			(-0.5375, 1.0313),
			(-0.5187, 1.0535),
			(-0.5000, 1.0750),
			(-0.4813, 1.0957),
			(-0.4625, 1.1157),
			(-0.4439, 1.1350),
			(-0.4254, 1.1536),
			(-0.4070, 1.1714),
			(-0.3888, 1.1885),
			(-0.3708, 1.2050),
			(-0.3531, 1.2207),
			(-0.3357, 1.2357),
			(-0.3186, 1.2501),
			(-0.3019, 1.2637),
			(-0.2855, 1.2767),
			(-0.2697, 1.2890),
			(-0.2542, 1.3006),
			(-0.2393, 1.3116),
			(-0.2250, 1.3219),
			(-0.2112, 1.3315),
			(-0.1981, 1.3405),
			(-0.1856, 1.3488),
			(-0.1738, 1.3565),
			(-0.1628, 1.3635),
			(-0.1525, 1.3699),
			(-0.1430, 1.3757),
			(-0.1344, 1.3809),
			(-0.1266, 1.3854),
			(-0.1198, 1.3893),
			(-0.1139, 1.3926),
			(-0.1090, 1.3953),
			(-0.1051, 1.3973),
			(-0.1023, 1.3988),
			(-0.1006, 1.3997),
			(-0.1000, 1.4000),
			(-0.1000, 1.4000),
			(-0.0999, 1.3993),
			(-0.0997, 1.3974),
			(-0.0994, 1.3942),
			(-0.0989, 1.3898),
			(-0.0983, 1.3841),
			(-0.0975, 1.3773),
			(-0.0967, 1.3694),
			(-0.0957, 1.3604),
			(-0.0946, 1.3502),
			(-0.0934, 1.3390),
			(-0.0922, 1.3268),
			(-0.0908, 1.3136),
			(-0.0893, 1.2995),
			(-0.0877, 1.2844),
			(-0.0861, 1.2684),
			(-0.0844, 1.2516),
			(-0.0826, 1.2339),
			(-0.0807, 1.2153),
			(-0.0788, 1.1961),
			(-0.0768, 1.1760),
			(-0.0748, 1.1552),
			(-0.0727, 1.1338),
			(-0.0705, 1.1116),
			(-0.0684, 1.0889),
			(-0.0661, 1.0655),
			(-0.0639, 1.0416),
			(-0.0616, 1.0171),
			(-0.0593, 0.9921),
			(-0.0570, 0.9666),
			(-0.0547, 0.9406),
			(-0.0523, 0.9143),
			(-0.0500, 0.8875),
			(-0.0477, 0.8604),
			(-0.0453, 0.8329),
			(-0.0430, 0.8051),
			(-0.0407, 0.7771),
			(-0.0384, 0.7488),
			(-0.0361, 0.7203),
			(-0.0339, 0.6916),
			(-0.0316, 0.6627),
			(-0.0295, 0.6337),
			(-0.0273, 0.6046),
			(-0.0252, 0.5755),
			(-0.0232, 0.5463),
			(-0.0212, 0.5171),
			(-0.0193, 0.4879),
			(-0.0174, 0.4587),
			(-0.0156, 0.4297),
			(-0.0139, 0.4007),
			(-0.0123, 0.3719),
			(-0.0107, 0.3433),
			(-0.0092, 0.3149),
			(-0.0078, 0.2867),
			(-0.0066, 0.2587),
			(-0.0054, 0.2311),
			(-0.0043, 0.2037),
			(-0.0033, 0.1767),
			(-0.0025, 0.1501),
			(-0.0017, 0.1239),
			(-0.0011, 0.0981),
			(-0.0006, 0.0728),
			(-0.0003, 0.0480),
			(-0.0001, 0.0237),
			(-0.0000, -0.0000),
			(0.0000, 0.0000),
			(-0.0001, -0.0237),
			(-0.0003, -0.0480),
			(-0.0006, -0.0728),
			(-0.0011, -0.0981),
			(-0.0017, -0.1239),
			(-0.0025, -0.1501),
			(-0.0033, -0.1767),
			(-0.0043, -0.2037),
			(-0.0054, -0.2311),
			(-0.0066, -0.2587),
			(-0.0078, -0.2867),
			(-0.0092, -0.3149),
			(-0.0107, -0.3433),
			(-0.0123, -0.3719),
			(-0.0139, -0.4007),
			(-0.0156, -0.4297),
			(-0.0174, -0.4587),
			(-0.0193, -0.4879),
			(-0.0212, -0.5171),
			(-0.0232, -0.5463),
			(-0.0252, -0.5755),
			(-0.0273, -0.6046),
			(-0.0295, -0.6337),
			(-0.0316, -0.6627),
			(-0.0339, -0.6916),
			(-0.0361, -0.7203),
			(-0.0384, -0.7488),
			(-0.0407, -0.7771),
			(-0.0430, -0.8051),
			(-0.0453, -0.8329),
			(-0.0477, -0.8604),
			(-0.0500, -0.8875),
			(-0.0523, -0.9143),
			(-0.0547, -0.9406),
			(-0.0570, -0.9666),
			(-0.0593, -0.9921),
			(-0.0616, -1.0171),
			(-0.0639, -1.0416),
			(-0.0661, -1.0655),
			(-0.0684, -1.0889),
			(-0.0705, -1.1116),
			(-0.0727, -1.1338),
			(-0.0748, -1.1552),
			(-0.0768, -1.1760),
			(-0.0788, -1.1961),
			(-0.0807, -1.2153),
			(-0.0826, -1.2339),
			(-0.0844, -1.2516),
			(-0.0861, -1.2684),
			(-0.0877, -1.2844),
			(-0.0893, -1.2995),
			(-0.0908, -1.3136),
			(-0.0922, -1.3268),
			(-0.0934, -1.3390),
			(-0.0946, -1.3502),
			(-0.0957, -1.3604),
			(-0.0967, -1.3694),
			(-0.0975, -1.3773),
			(-0.0983, -1.3841),
			(-0.0989, -1.3898),
			(-0.0994, -1.3942),
			(-0.0997, -1.3974),
			(-0.0999, -1.3993),
			(-0.1000, -1.4000),
			(-0.1000, -1.4000),
			(-0.1007, -1.4004),
			(-0.1026, -1.4017),
			(-0.1057, -1.4038),
			(-0.1101, -1.4067),
			(-0.1156, -1.4104),
			(-0.1222, -1.4148),
			(-0.1299, -1.4200),
			(-0.1387, -1.4258),
			(-0.1484, -1.4323),
			(-0.1591, -1.4394),
			(-0.1706, -1.4471),
			(-0.1831, -1.4554),
			(-0.1963, -1.4642),
			(-0.2104, -1.4736),
			(-0.2251, -1.4834),
			(-0.2406, -1.4937),
			(-0.2568, -1.5045),
			(-0.2735, -1.5157),
			(-0.2909, -1.5272),
			(-0.3087, -1.5392),
			(-0.3271, -1.5514),
			(-0.3459, -1.5640),
			(-0.3652, -1.5768),
			(-0.3848, -1.5898),
			(-0.4047, -1.6031),
			(-0.4249, -1.6166),
			(-0.4454, -1.6303),
			(-0.4661, -1.6440),
			(-0.4869, -1.6579),
			(-0.5079, -1.6719),
			(-0.5289, -1.6859),
			(-0.5500, -1.7000),
			(-0.5711, -1.7141),
			(-0.5921, -1.7281),
			(-0.6131, -1.7421),
			(-0.6339, -1.7560),
			(-0.6546, -1.7697),
			(-0.6751, -1.7834),
			(-0.6953, -1.7969),
			(-0.7152, -1.8102),
			(-0.7348, -1.8232),
			(-0.7541, -1.8360),
			(-0.7729, -1.8486),
			(-0.7913, -1.8608),
			(-0.8091, -1.8728),
			(-0.8265, -1.8843),
			(-0.8432, -1.8955),
			(-0.8594, -1.9062),
			(-0.8749, -1.9166),
			(-0.8896, -1.9264),
			(-0.9037, -1.9358),
			(-0.9169, -1.9446),
			(-0.9294, -1.9529),
			(-0.9409, -1.9606),
			(-0.9516, -1.9677),
			(-0.9613, -1.9742),
			(-0.9701, -1.9800),
			(-0.9778, -1.9852),
			(-0.9844, -1.9896),
			(-0.9899, -1.9933),
			(-0.9943, -1.9962),
			(-0.9974, -1.9983),
			(-0.9993, -1.9996),
			(-1.0000, -2.0000),
			(-1.0000, -2.0000),
			(-1.0001, -2.0000),
			(-1.0003, -2.0000),
			(-1.0006, -2.0000),
			(-1.0011, -2.0000),
			(-1.0018, -2.0000),
			(-1.0026, -2.0000),
			(-1.0035, -2.0000),
			(-1.0045, -2.0000),
			(-1.0057, -2.0000),
			(-1.0069, -2.0000),
			(-1.0084, -2.0000),
			(-1.0099, -2.0000),
			(-1.0115, -2.0000),
			(-1.0133, -2.0000),
			(-1.0152, -2.0000),
			(-1.0172, -2.0000),
			(-1.0193, -2.0000),
			(-1.0215, -2.0000),
			(-1.0238, -2.0000),
			(-1.0262, -2.0000),
			(-1.0288, -2.0000),
			(-1.0314, -2.0000),
			(-1.0341, -2.0000),
			(-1.0369, -2.0000),
			(-1.0398, -2.0000),
			(-1.0428, -2.0000),
			(-1.0459, -2.0000),
			(-1.0490, -2.0000),
			(-1.0523, -2.0000),
			(-1.0556, -2.0000),
			(-1.0590, -2.0000),
			(-1.0625, -2.0000),
			(-1.0661, -2.0000),
			(-1.0697, -2.0000),
			(-1.0734, -2.0000),
			(-1.0771, -2.0000),
			(-1.0809, -2.0000),
			(-1.0848, -2.0000),
			(-1.0888, -2.0000),
			(-1.0928, -2.0000),
			(-1.0968, -2.0000),
			(-1.1009, -2.0000),
			(-1.1051, -2.0000),
			(-1.1093, -2.0000),
			(-1.1136, -2.0000),
			(-1.1178, -2.0000),
			(-1.1222, -2.0000),
			(-1.1266, -2.0000),
			(-1.1310, -2.0000),
			(-1.1354, -2.0000),
			(-1.1399, -2.0000),
			(-1.1444, -2.0000),
			(-1.1489, -2.0000),
			(-1.1535, -2.0000),
			(-1.1581, -2.0000),
			(-1.1627, -2.0000),
			(-1.1673, -2.0000),
			(-1.1720, -2.0000),
			(-1.1766, -2.0000),
			(-1.1813, -2.0000),
			(-1.1859, -2.0000),
			(-1.1906, -2.0000),
			(-1.1953, -2.0000),
			(-1.2000, -2.0000),
			(-1.2000, -2.0000),
			(-1.2094, -1.9997),
			(-1.2190, -1.9990),
			(-1.2287, -1.9978),
			(-1.2386, -1.9961),
			(-1.2485, -1.9939),
			(-1.2586, -1.9913),
			(-1.2687, -1.9884),
			(-1.2789, -1.9850),
			(-1.2892, -1.9812),
			(-1.2995, -1.9770),
			(-1.3100, -1.9725),
			(-1.3204, -1.9677),
			(-1.3309, -1.9625),
			(-1.3414, -1.9571),
			(-1.3520, -1.9513),
			(-1.3625, -1.9453),
			(-1.3730, -1.9390),
			(-1.3836, -1.9325),
			(-1.3941, -1.9258),
			(-1.4046, -1.9188),
			(-1.4150, -1.9117),
			(-1.4255, -1.9044),
			(-1.4358, -1.8969),
			(-1.4461, -1.8893),
			(-1.4563, -1.8815),
			(-1.4664, -1.8736),
			(-1.4765, -1.8657),
			(-1.4864, -1.8576),
			(-1.4963, -1.8495),
			(-1.5060, -1.8414),
			(-1.5156, -1.8332),
			(-1.5250, -1.8250),
			(-1.5343, -1.8168),
			(-1.5434, -1.8086),
			(-1.5524, -1.8005),
			(-1.5612, -1.7924),
			(-1.5699, -1.7843),
			(-1.5783, -1.7764),
			(-1.5865, -1.7685),
			(-1.5945, -1.7607),
			(-1.6023, -1.7531),
			(-1.6099, -1.7456),
			(-1.6172, -1.7383),
			(-1.6243, -1.7312),
			(-1.6311, -1.7242),
			(-1.6377, -1.7175),
			(-1.6440, -1.7110),
			(-1.6500, -1.7047),
			(-1.6557, -1.6987),
			(-1.6611, -1.6929),
			(-1.6662, -1.6875),
			(-1.6710, -1.6823),
			(-1.6754, -1.6775),
			(-1.6796, -1.6730),
			(-1.6833, -1.6688),
			(-1.6867, -1.6650),
			(-1.6898, -1.6616),
			(-1.6924, -1.6587),
			(-1.6947, -1.6561),
			(-1.6966, -1.6539),
			(-1.6981, -1.6522),
			(-1.6991, -1.6510),
			(-1.6998, -1.6503),
			(-1.7000, -1.6500),
			(-1.7000, -1.6500),
			(-1.6996, -1.6500),
			(-1.6983, -1.6500),
			(-1.6962, -1.6500),
			(-1.6933, -1.6500),
			(-1.6897, -1.6500),
			(-1.6853, -1.6500),
			(-1.6802, -1.6500),
			(-1.6743, -1.6500),
			(-1.6678, -1.6500),
			(-1.6606, -1.6500),
			(-1.6528, -1.6500),
			(-1.6443, -1.6500),
			(-1.6353, -1.6500),
			(-1.6257, -1.6500),
			(-1.6155, -1.6500),
			(-1.6048, -1.6500),
			(-1.5936, -1.6500),
			(-1.5819, -1.6500),
			(-1.5697, -1.6500),
			(-1.5571, -1.6500),
			(-1.5441, -1.6500),
			(-1.5307, -1.6500),
			(-1.5169, -1.6500),
			(-1.5028, -1.6500),
			(-1.4883, -1.6500),
			(-1.4735, -1.6500),
			(-1.4584, -1.6500),
			(-1.4431, -1.6500),
			(-1.4275, -1.6500),
			(-1.4117, -1.6500),
			(-1.3957, -1.6500),
			(-1.3795, -1.6500),
			(-1.3631, -1.6500),
			(-1.3467, -1.6500),
			(-1.3301, -1.6500),
			(-1.3134, -1.6500),
			(-1.2967, -1.6500),
			(-1.2799, -1.6500),
			(-1.2631, -1.6500),
			(-1.2463, -1.6500),
			(-1.2295, -1.6500),
			(-1.2127, -1.6500),
			(-1.1960, -1.6500),
			(-1.1795, -1.6500),
			(-1.1630, -1.6500),
			(-1.1466, -1.6500),
			(-1.1304, -1.6500),
			(-1.1144, -1.6500),
			(-1.0986, -1.6500),
			(-1.0830, -1.6500),
			(-1.0676, -1.6500),
			(-1.0525, -1.6500),
			(-1.0377, -1.6500),
			(-1.0232, -1.6500),
			(-1.0090, -1.6500),
			(-0.9952, -1.6500),
			(-0.9818, -1.6500),
			(-0.9687, -1.6500),
			(-0.9561, -1.6500),
			(-0.9439, -1.6500),
			(-0.9322, -1.6500),
			(-0.9209, -1.6500),
			(-0.9102, -1.6500),
			(-0.9000, -1.6500),
			(-0.9000, -1.6500),
			(-0.8911, -1.6500),
			(-0.8824, -1.6500),
			(-0.8737, -1.6500),
			(-0.8651, -1.6500),
			(-0.8566, -1.6500),
			(-0.8482, -1.6500),
			(-0.8398, -1.6500),
			(-0.8316, -1.6499),
			(-0.8234, -1.6499),
			(-0.8153, -1.6498),
			(-0.8073, -1.6496),
			(-0.7993, -1.6495),
			(-0.7914, -1.6493),
			(-0.7835, -1.6490),
			(-0.7757, -1.6487),
			(-0.7680, -1.6484),
			(-0.7603, -1.6480),
			(-0.7527, -1.6476),
			(-0.7451, -1.6471),
			(-0.7375, -1.6465),
			(-0.7300, -1.6459),
			(-0.7225, -1.6451),
			(-0.7151, -1.6444),
			(-0.7077, -1.6435),
			(-0.7003, -1.6425),
			(-0.6929, -1.6415),
			(-0.6856, -1.6404),
			(-0.6783, -1.6392),
			(-0.6710, -1.6378),
			(-0.6637, -1.6364),
			(-0.6564, -1.6349),
			(-0.6491, -1.6332),
			(-0.6418, -1.6315),
			(-0.6345, -1.6296),
			(-0.6272, -1.6276),
			(-0.6199, -1.6255),
			(-0.6126, -1.6233),
			(-0.6053, -1.6209),
			(-0.5979, -1.6184),
			(-0.5906, -1.6157),
			(-0.5832, -1.6129),
			(-0.5758, -1.6100),
			(-0.5683, -1.6069),
			(-0.5609, -1.6036),
			(-0.5534, -1.6002),
			(-0.5458, -1.5966),
			(-0.5382, -1.5929),
			(-0.5306, -1.5890),
			(-0.5229, -1.5849),
			(-0.5152, -1.5806),
			(-0.5074, -1.5762),
			(-0.4996, -1.5715),
			(-0.4917, -1.5667),
			(-0.4837, -1.5617),
			(-0.4757, -1.5565),
			(-0.4676, -1.5510),
			(-0.4594, -1.5454),
			(-0.4512, -1.5396),
			(-0.4429, -1.5335),
			(-0.4345, -1.5273),
			(-0.4260, -1.5208),
			(-0.4174, -1.5141),
			(-0.4088, -1.5072),
			(-0.4000, -1.5000),
			(-0.4000, -1.5000),
			(-0.3969, -1.4967),
			(-0.3947, -1.4929),
			(-0.3933, -1.4886),
			(-0.3929, -1.4836),
			(-0.3933, -1.4782),
			(-0.3946, -1.4721),
			(-0.3966, -1.4655),
			(-0.3994, -1.4583),
			(-0.4029, -1.4505),
			(-0.4071, -1.4421),
			(-0.4120, -1.4332),
			(-0.4175, -1.4236),
			(-0.4237, -1.4134),
			(-0.4304, -1.4027),
			(-0.4378, -1.3913),
			(-0.4456, -1.3794),
			(-0.4540, -1.3668),
			(-0.4628, -1.3536),
			(-0.4721, -1.3397),
			(-0.4818, -1.3253),
			(-0.4919, -1.3102),
			(-0.5024, -1.2945),
			(-0.5132, -1.2781),
			(-0.5243, -1.2611),
			(-0.5357, -1.2434),
			(-0.5474, -1.2251),
			(-0.5593, -1.2061),
			(-0.5714, -1.1864),
			(-0.5836, -1.1661),
			(-0.5960, -1.1451),
			(-0.6085, -1.1235),
			(-0.6211, -1.1011),
			(-0.6337, -1.0781),
			(-0.6464, -1.0543),
			(-0.6591, -1.0299),
			(-0.6717, -1.0048),
			(-0.6843, -0.9790),
			(-0.6968, -0.9525),
			(-0.7092, -0.9252),
			(-0.7215, -0.8973),
			(-0.7336, -0.8686),
			(-0.7454, -0.8392),
			(-0.7571, -0.8090),
			(-0.7685, -0.7782),
			(-0.7796, -0.7466),
			(-0.7905, -0.7142),
			(-0.8009, -0.6811),
			(-0.8110, -0.6473),
			(-0.8208, -0.6127),
			(-0.8300, -0.5773),
			(-0.8389, -0.5412),
			(-0.8473, -0.5043),
			(-0.8551, -0.4666),
			(-0.8624, -0.4282),
			(-0.8692, -0.3890),
			(-0.8754, -0.3489),
			(-0.8809, -0.3081),
			(-0.8858, -0.2665),
			(-0.8900, -0.2241),
			(-0.8935, -0.1809),
			(-0.8963, -0.1369),
			(-0.8983, -0.0921),
			(-0.8996, -0.0465),
			# (-0.9000, 0.0000)
		])

			# This points were obtained from blender file radiator_all_core_and_divertor.stl
			# with this script
			#
			# spline = bpy.data.curves['full_tokamak'].splines[0]
			#
			# if len(spline.bezier_points) >= 2:
			#     r = spline.resolution_u + 1
			#     segments = len(spline.bezier_points)
			#     if not spline.use_cyclic_u:
			#         segments -= 1
			#     points = []
			#     for i in range(segments):
			#         inext = (i + 1) % len(spline.bezier_points)
			#         knot1 = spline.bezier_points[i].co
			#         handle1 = spline.bezier_points[i].handle_right
			#         handle2 = spline.bezier_points[inext].handle_left
			#         knot2 = spline.bezier_points[inext].co
			#         _points = mathutils.geometry.interpolate_bezier(knot1, handle1, handle2, knot2, r)
			#         points.extend(_points)
			#
			#
			# f = open('coordinates.txt','w')
			# for point in points:
			#     f.write(str(point)+'\n')
			# f.close()

		filtered=[]
		for index in range(len(_MASTU_PLASMA_REGION)-1):
			if (np.abs(_MASTU_PLASMA_REGION[index])!=np.abs(_MASTU_PLASMA_REGION[index+1])).all():
				filtered.append(_MASTU_PLASMA_REGION[index])
		_MASTU_PLASMA_REGION=np.array(filtered)
		for index in range(len(_MASTU_PLASMA_REGION)):
			_MASTU_PLASMA_REGION[index,0]=0.4-_MASTU_PLASMA_REGION[index,0]


		PLASMA_POLYGON_MASK = PolygonMask2D(_MASTU_PLASMA_REGION)
		# plt.plot(_MASTU_PLASMA_REGION[:, 0], _MASTU_PLASMA_REGION[:, 1], 'k')
		# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		# plt.axis('equal')
		# plt.show()

		# 	This is bad, it's massively imprecise. I calculate where radiation is not supposed to arrive and use that as a mask.











		mastu_path = "/home/ffederic/work/SOLPS/seeding/seed_10"
		if not mastu_path in sys.path:
			sys.path.append(mastu_path)
		ds_puff8 = xr.open_dataset('/home/ffederic/work/SOLPS/seeding/seed_10/balance.nc', autoclose=True).load()

		grid_x = ds_puff8.crx.mean(dim='4')
		grid_y = ds_puff8.cry.mean(dim='4')
		impurity_radiation = ds_puff8.b2stel_she_bal.sum('ns')
		hydrogen_radiation = ds_puff8.eirene_mc_eael_she_bal.sum('nstra') - ds_puff8.eirene_mc_papl_sna_bal.isel(ns=1).sum('nstra') * 13.6 * 1.6e-19   # ds_puff8.eirene_mc_eael_she_bal.sum('nstra') is the total electron energy sink due to plasma / atoms interactions, including ionisation/excitation (dominant) and charge exchange (negligible). Here we assume all not used for ionisation goes to radiation, including the CX bit.
		total_radiation = -hydrogen_radiation + impurity_radiation
		total_radiation_density = -np.divide(hydrogen_radiation + impurity_radiation,ds_puff8.vol)

		# fig, ax = plt.subplots()
		# # grid_x.plot.line(ax=ax, x='nx_plus2')
		# # grid_y.plot.line(ax=ax, x='ny_plus2')
		# # plt.pcolormesh(grid_x.values, grid_y.values, impurity_radiation.values)
		#
		#
		# plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values), norm=LogNorm(vmin=1000, vmax=total_radiation_density.values.max()),cmap='rainbow')
		# # plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values),cmap='rainbow')
		# ax.set_ylim(top=-0.5)
		# ax.set_xlim(left=0.)
		# plt.title('Emissivity profile as imported directly from SOLPS')
		# plt.colorbar().set_label('Emissivity [W/m^3]')
		# plt.xlabel('R [m]')
		# plt.ylabel('Z [m]')
		#
		# x=np.linspace(0.55-0.075,0.55+0.075,10)
		# y=-1.2+np.sqrt(0.08**2-(x-0.55)**2)
		# y_=-1.2-np.sqrt(0.08**2-(x-0.55)**2)
		# plt.plot(x,y,'k')
		# plt.plot(x,y_,'k')
		# plt.show()


		# # 10/01/2019 for Matt, send him this data in ASCII format
		#
		# os.chdir("/home/ffederic/work/analysis scripts/")
		# grid_R_flat = grid_x.values.flatten()
		# np.savetxt('grid_R.txt', grid_R_flat)
		# grid_Z_flat = grid_y.values.flatten()
		# np.savetxt('grid_Z.txt', grid_Z_flat)
		# total_radiation_density_flat = np.abs(total_radiation_density.values).flatten()
		# np.savetxt('total_radiation_density.txt', total_radiation_density_flat)





		# CHERAB section




		# os.chdir("/home/ffederic/work/cherab/cherab_mastu/diagnostics/bolometry/irvb/")
		from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
		from raysect.primitive import import_stl, Sphere, Mesh, Cylinder
		from raysect.optical import World, ConstantSF
		from raysect.optical.material import NullMaterial
		from raysect.optical.material.absorber import AbsorbingSurface
		from raysect.optical.material.emitter import UniformVolumeEmitter,InhomogeneousVolumeEmitter
		from raysect.optical.observer import TargettedCCDArray, PowerPipeline2D
		from raysect.core.math.interpolators import Discrete2DMesh


		os.chdir("/home/ffederic/work/cherab/cherab_mastu/")
		from cherab.mastu.machine import MASTU_FULL_MESH


		os.chdir("/home/ffederic/work/cherab/cherab_core/")
		from cherab.core.math.mappers import AxisymmetricMapper


		world = World()


		for cad_file in MASTU_FULL_MESH:
			directory, filename = os.path.split(cad_file[0])
			name, ext = filename.split('.')
			print("importing {} ...".format(filename))
			Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)

		os.chdir("/home/ffederic/work/analysis_scripts/irvb/")
		irvb_cad = import_stl("IRVB_camera_no_backplate_4mm.stl", parent=world, material=AbsorbingSurface(), name="IRVB")



		# create radiator


		cr_r = np.transpose(ds_puff8.crx.values)
		cr_z = np.transpose(ds_puff8.cry.values)
		radiated_power=np.transpose(total_radiation_density.values)

		def make_solps_power_function(cr_r, cr_z, radiated_power):

			import numpy as np
			from cherab.core.math.mappers import AxisymmetricMapper
			from raysect.core.math.interpolators import Discrete2DMesh

			nx = cr_r.shape[0]
			ny = cr_r.shape[1]

			# Iterate through the arrays from MDS plus to pull out unique vertices
			unique_vertices = {}
			vertex_id = 0
			for i in range(nx):
				for j in range(ny):
					for k in range(4):
						vertex = (cr_r[i, j, k], cr_z[i, j, k])
						try:
							unique_vertices[vertex]
						except KeyError:
							unique_vertices[vertex] = vertex_id
							vertex_id += 1

			# Load these unique vertices into a numpy array for later use in Raysect's mesh interpolator object.
			num_vertices = len(unique_vertices)
			vertex_coords = np.zeros((num_vertices, 2), dtype=np.float64)
			for vertex, vertex_id in unique_vertices.items():
				vertex_coords[vertex_id, :] = vertex

			# Number of triangles must be equal to number of rectangle centre points times 2.
			num_tris = nx * ny * 2
			triangles = np.zeros((num_tris, 3), dtype=np.int32)

			_triangle_to_grid_map = np.zeros((nx * ny * 2, 2), dtype=np.int32)
			tri_index = 0
			for i in range(nx):
				for j in range(ny):
					# Pull out the index number for each unique vertex in this rectangular cell.
					# Unusual vertex indexing is based on SOLPS output, see Matlab code extract from David Moulton.
					# cell_r = [r(i,j,1),r(i,j,3),r(i,j,4),r(i,j,2)];
					v1_id = unique_vertices[(cr_r[i, j, 0], cr_z[i, j, 0])]
					v2_id = unique_vertices[(cr_r[i, j, 2], cr_z[i, j, 2])]
					v3_id = unique_vertices[(cr_r[i, j, 3], cr_z[i, j, 3])]
					v4_id = unique_vertices[(cr_r[i, j, 1], cr_z[i, j, 1])]

					# Split the quad cell into two triangular cells.
					# Each triangle cell is mapped to the tuple ID (ix, iy) of its parent mesh cell.
					triangles[tri_index, :] = (v1_id, v2_id, v3_id)
					_triangle_to_grid_map[tri_index, :] = (i, j)
					tri_index += 1
					triangles[tri_index, :] = (v3_id, v4_id, v1_id)
					_triangle_to_grid_map[tri_index, :] = (i, j)
					tri_index += 1

			radiated_power=radiated_power.flatten()
			rad_power = np.zeros(radiated_power.shape[0]*2)
			for i in range(radiated_power.shape[0]):
				rad_power[i*2] = radiated_power[i]
				rad_power[i*2 + 1] = radiated_power[i]

			return AxisymmetricMapper(Discrete2DMesh(vertex_coords, triangles, rad_power,limit=False))




		class RadiatedPower(InhomogeneousVolumeEmitter):

			def __init__(self, radiation_function):
				super().__init__()
				self.radiation_function = radiation_function

			def emission_function(self, point, direction, spectrum, world, ray, primitive, to_local, to_world):

				p = point.transform(to_world)
				spectrum.samples[0] += self.radiation_function(p.x, p.y, p.z)

				return spectrum


		radiation_function = make_solps_power_function(cr_r, cr_z, radiated_power)


		# plt.figure()
		# X = np.arange(0, grid_x.values.max(), 0.001)
		# Y = np.arange(grid_y.values.min(), -0.6, 0.001)
		# rad_test = np.zeros((len(X), len(Y)))
		# grid_x2 = np.zeros((len(X), len(Y)))
		# grid_y2 = np.zeros((len(X), len(Y)))
		# for ix, x in enumerate(X):
		# 	for jy, y in enumerate(Y):
		# 		rad_test[ix, jy] = radiation_function(x, 0, y)
		# 		grid_x2[ix, jy] = x
		# 		grid_y2[ix, jy] = y


		# plt.pcolor(grid_x2, grid_y2, rad_test, norm=LogNorm(vmin=1000, vmax=total_radiation_density.values.max()),cmap='rainbow')
		# # plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values),cmap='rainbow')
		# ax.set_ylim(top=-0.5)
		# ax.set_xlim(left=0.)
		# plt.title('Emissivity profile as imported from SOLPS using CHERAB utilities')
		# plt.colorbar().set_label('Emissivity [W/m^3]')
		# plt.xlabel('R [m]')
		# plt.ylabel('Z [m]')
		#
		# x=np.linspace(0.55-0.075,0.55+0.075,10)
		# y=-1.2+np.sqrt(0.08**2-(x-0.55)**2)
		# y_=-1.2-np.sqrt(0.08**2-(x-0.55)**2)
		# plt.plot(x,y,'k')
		# plt.plot(x,y_,'k')
		#
		# plt.show()



		# # THIS IS FOR UNIFORM EMITTED POWER DENSITY
		# total_power=500000
		# center_radiator=[0.55,-1.2]
		#
		# # TOROIDAL RADIATOR
		# minor_radius_radiator=0.08
		# volume_radiator=2*(np.pi**2)*center_radiator[0]*minor_radius_radiator**2
		#
		# # SQUARE ANULAR RADIATOR
		# # side_radiator=0.14
		# # volume_radiator=np.pi*((center_radiator[0]+side_radiator/2)**2-(center_radiator[0]-side_radiator/2)**2)*side_radiator
		# # x_point_lower = Point2D(center_radiator[0] - side_radiator/2, center_radiator[1] - side_radiator/2)
		# # x_point_upper = Point2D(center_radiator[0] + side_radiator/2, center_radiator[1] + side_radiator/2)
		#
		# power_density = total_power / volume_radiator / (4*np.pi)
		# power_density_core=50000 / (4*np.pi)
		#
		# from cherab.tools.primitives.annulus_mesh import generate_annulus_mesh_segments
		# # generate_annulus_mesh_segments(x_point_lower, x_point_upper, 360, world, material=UniformVolumeEmitter(ConstantSF(power_density)))
		# # radiator = import_stl('radiator_R0.55_Z-1.2_r0.08.stl', material=UniformVolumeEmitter(ConstantSF(power_density)), parent=world)
		# # radiator = import_stl('radiator_all_closed_surface.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
		# # radiator = import_stl('radiator_all_super_x_divertor.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
		# # radiator = import_stl('radiator_all_core.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
		# radiator = import_stl('radiator_all_core_and_divertor.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
		# # radiator = import_stl('test.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)



		emitter_material = RadiatedPower(radiation_function)
		outer_radius = cr_r.max() + 0.01
		plasma_height = cr_z.max() - cr_z.min()
		lower_z = cr_z.min()
		radiator = Cylinder(outer_radius, plasma_height, material=emitter_material, parent=world,transform=translate(0, 0, lower_z))


		pinhole_centre = Point3D(1.491933, 0, -0.7198).transform(rotate_z(135-0.76004))
		pinhole_target = Sphere(0.005, transform=translate(*pinhole_centre), parent=world, material=NullMaterial())


		stand_off=0.045
		CCD_radius=1.50467+stand_off
		CCD_angle=135*(np.pi*2)/360

		ccd_centre = Point3D(CCD_radius*np.cos(CCD_angle), CCD_radius*np.sin(CCD_angle), -0.699522)
		ccd_normal = Vector3D(-CCD_radius*np.cos(CCD_angle), -CCD_radius*np.sin(CCD_angle),0).normalise()
		ccd_y_axis = Vector3D(0,0,1).normalise()
		ccd_x_axis = ccd_y_axis.cross(ccd_normal)

		# if is_this_extra:
		# 	pixel_h = foil_resolution_max
		# else:
		# 	pixel_h = np.ceil(foil_resolution_max/averaging).astype('int')

		pixel_h = foil_resolution_max
		pixel_v=(pixel_h*9)//7

		plt.ion()
		power = PowerPipeline2D()
		detector = TargettedCCDArray(targets=[pinhole_target], width=0.07, pixels=(pixel_h, pixel_v), targetted_path_prob=1.0,
									 parent=world, pipelines=[power],
									 transform=translate(*ccd_centre)*rotate_basis(ccd_normal, ccd_y_axis))
		detector.max_wavelength = 601
		detector.min_wavelength = 600
		# detector.pixel_samples = 500//(1/(5*5))
		detector.pixel_samples = 500

		detector.observe()
		# plt.imshow(np.flip(np.transpose(power.frame.mean),axis=0), origin='lower')
		# plt.show()


		detection = coleval.flatten_full( np.flip(np.transpose(power.frame.mean),axis=0))
		if is_this_extra:
			detection = coleval.foil_measurement_averaging_foil_pixels_extra_loseless(detection,foil_resolution_max,averaging,averaging)
		else:
			detection = coleval.foil_measurement_averaging_foil_pixels_loseless(detection,foil_resolution_max,averaging,averaging)

		mask2 = np.ones(np.shape(detection))
		for index,value in enumerate(detection):
			if value!=0:
				mask2[index]=0

		scipy.sparse.save_npz(path_sensitivity + '/mask2_on_camera_pixels.npz', scipy.sparse.csr_matrix(mask2))

		sensitivities_masked2 = []
		for index, masked in enumerate(mask2):
			if not masked:
				sensitivities_masked2.append(sensitivities_masked1[index])
		sensitivities_masked2=np.array(sensitivities_masked2)

		scipy.sparse.save_npz(path_sensitivity + '/sensitivity_masked2.npz', scipy.sparse.csr_matrix(sensitivities_masked2))



		# 	# I might go even further and define a treshold for the minimum reading that I consider valid.
		# 	# this anyway makes unclear what I mean by that signal leve, and this will also change for different plasma configurations
		#
		# power_treshold = 1e-5
		#
		# detector = TargettedCCDArray(targets=[pinhole_target], width=0.07, pixels=(pixel_h, pixel_v), targetted_path_prob=1.0,
		# 							 parent=world, pipelines=[power],
		# 							 transform=translate(*ccd_centre)*rotate_basis(ccd_normal, ccd_y_axis))
		# detector.max_wavelength = 601
		# detector.min_wavelength = 600
		# detector.pixel_samples = 5000
		# detector.observe()
		# detection = coleval.flatten_full( np.flip(np.transpose(power.frame.mean),axis=0))
		# if is_this_extra:
		# 	detection = coleval.foil_measurement_averaging_foil_pixels_extra_loseless(detection,foil_resolution_max,averaging,averaging)
		#
		# mask3 = np.ones(np.shape(detection))
		# for index,value in enumerate(detection):
		# 	if value<power_treshold:
		# 		mask3[index]=0
		#
		# scipy.sparse.save_npz(path_sensitivity + '/mask3_on_camera_pixels.npz', scipy.sparse.csr_matrix(mask2))
		#
		#
		# sensitivities_masked3 = []
		# for index,masked in enumerate(mask3):
		# 	if not masked:
		# 		sensitivities_masked3.append(sensitivities_masked1[index])
		# sensitivities_masked3=np.array(sensitivities_masked3)
		#
		# scipy.sparse.save_npz(path_sensitivity + '/sensitivity_masked3.npz', scipy.sparse.csr_matrix(sensitivities_masked3))
		#
