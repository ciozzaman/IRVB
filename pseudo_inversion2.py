# Created 13/12/2018
# Fabio Federici


# #this is if working on a pc, use pc printer
# exec(open("/home/ffederic/work/analysis_scripts/preamble_import_pc.py").read())

#this is if working in batch, use predefined NOT visual printer
exec(open("/home/ffederic/work/analysis_scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/preamble_indexing.py").read())













# def _run_(foil_resolution):

# Load emission profile

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















foil_resolution_all = [93, 62, 47, 37, 31, 26, 19]
for foil_resolution in foil_resolution_all:

	# data analysis

	is_this_extra = True
	# foil_resolution = 19
	grid_resolution = 2		# in cm
	with_noise = True
	foil_resolution_max = 187
	weigthed_best_search= True


	if with_noise:
		time_averaging_tries=[1,2,3,4,5,6]
	else:
		time_averaging_tries=[1]

	noise_on_temporal_comp = 190	#this is increased  from 76 to account for the correction of diffusivity from laser measurements
	noise_on_laplacian_comp = 223
	noise_on_bb_comp = 0.46

	for time_averaging in time_averaging_tries:
		if not is_this_extra:
			noise_on_power = np.sqrt( (noise_on_temporal_comp/(np.sqrt(time_averaging)))**2 + (noise_on_laplacian_comp/int(np.around(foil_resolution_max/foil_resolution)))**2 + noise_on_bb_comp**2 )
		else:
			noise_on_power = np.sqrt((noise_on_temporal_comp / (np.sqrt(time_averaging))) ** 2 + (noise_on_laplacian_comp) ** 2 + noise_on_bb_comp ** 2)

		if is_this_extra:
			foil_res = '_foil_pixel_h_'+str(foil_resolution)+'_extra'
		else:
			foil_res = '_foil_pixel_h_'+str(foil_resolution)
		grid_type = 'core_res_'+str(grid_resolution)+'cm'
		path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
		path_sensitivity_original = copy.deepcopy(path_sensitivity)
		if not os.path.exists(path_sensitivity + '/sensitivity.npz'):
			sensitivities = np.load(path_sensitivity + '/sensitivity.npy')
			scipy.sparse.save_npz(path_sensitivity + '/sensitivity.npz', scipy.sparse.csr_matrix(sensitivities))
		else:
			sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity.npz')).todense())
		type='.csv'
		filenames = coleval.all_file_names(path_sensitivity, type)[0]
		with open(os.path.join(path_sensitivity,filenames)) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if row[0]=='foil horizontal pixels ':
					pixel_h=int(row[1])
				if row[0]=='pipeline type ':
					pipeline =row[1]
				if row[0]=='type of volume grid ':
					grid_type =row[1]
				# print(row)

		# with open(path_sensitivity+'/svd_decomposition.pickle', 'rb') as f:
		# 	decomposition = pickle.load(f)

		from raysect.optical import World, ConstantSF
		# world = World()
		from cherab.mastu.bolometry import load_default_bolometer_config, load_standard_voxel_grid
		from cherab.tools.inversions import ToroidalVoxelGrid
		core_voxel_grid = load_standard_voxel_grid(grid_type,parent=world)
		# sensitivities = np.random.rand(500,155)
		# print(scipy.sparse.issparse(scipy.sparse.csr_matrix(sensitivities)))
		# import scipy.sparse
		# scipy.sparse.save_npz(path_sensitivity+'/sensitivity.npz',scipy.sparse.csr_matrix(sensitivities))
		# sensitivities_sparse=scipy.sparse.load_npz(path_sensitivity+'/sensitivity.npz')
		# print(np.allclose(sensitivities, sensitivities_sparse.todense()))

		directory = '/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction'
		grid_file = os.path.join(
			directory,
			'{}_rectilinear_grid.pickle'.format(grid_type)
		)
		with open(grid_file, 'rb') as f:
			grid_data = pickle.load(f)
		laplacian = grid_data['laplacian']

		# check=[]
		# alpha_record=[]
		# for alpha in np.linspace(0.000001,0.0000001,num=20):
		# 	# print(repr(sensitivities))
		# 	# shape=np.shape(sensitivities)
		# 	# trashold=tre
		# 	test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
		#
		# 	# U, s, Vh = np.linalg.svd(sensitivities)
		# 	U, s, Vh = np.linalg.svd(test_matrix)
		# 	# s, Vh = scipy.sparse.linalg.eigs(sensitivities,k=min(np.shape(sensitivities))-4)
		# 	sgna = np.linalg.cond(test_matrix)
		# 	check.append(sgna)
		# 	alpha_record.append(alpha)
		# 	# zero=[]
		# 	print('alpha='+str(alpha))
		# 	print('max s='+str(s[0]))
		# 	print('conditioning='+str(sgna))
		# 	# for i in range(min(shape)):
		# 	# 	if s[i]<trashold:
		# 	# 		zero.append(i)
		# 	# # if zero==[]:
		# 	# # 	zero.append(min(shape))
		# 	# U=U[:,:zero[0]]
		# 	# Vh=Vh[:zero[0]]
		# 	# sigma = np.zeros((zero[0],zero[0]))
		# 	# for i in range(zero[0]):
		# 	# 	if s[i]<trashold:
		# 	# 		sigma[i, i]=0
		# 	# 	else:
		# 	# 		sigma[i, i] = s[i]
		# 	# a1 = np.dot(U, np.dot(sigma, Vh))
		# 	# # np.allclose(sensitivities, a1)
		# 	# inv_sigma = np.zeros((zero[0],zero[0]))
		# 	# for i in range(zero[0]):
		# 	# 	inv_sigma[i, i] = 1/s[i]
		# 	# # SGNAPPO = np.dot(Vh.T, np.dot(inv_sigma, U.T))
		# 	#
		# 	# print('tre='+str(trashold))
		# 	# print(str((np.dot(sensitivities,np.linspace(0,155-1,155)) - np.dot(a1,np.linspace(0,155-1,155))).max()))
		# 	# # # del a
		# 	# BLUE=0
		# 	# # del SGNAPPO
		# 	# del sigma
		# 	# del U,Vh,s,zero,shape,inv_sigma
		# 	del U, Vh, s
		# print(coleval.find_nearest_index(check,0))
		# alpha=alpha_record[coleval.find_nearest_index(check,0)[0]]


		#Here there are different types of radiator

		if False:
			test_voxel=43
			power_on_voxels = np.zeros((core_voxel_grid.count))
			power_on_voxels[test_voxel] = 2
		elif False:
			power_on_voxels=np.linspace(1E+08,1,np.shape( sensitivities)[1])
		elif True:
			power_on_voxels = np.zeros((core_voxel_grid.count))
			num_voxels = core_voxel_grid.count

			i = np.linspace(0, num_voxels - 1, num_voxels, dtype=int)
			for index in i:
				p1, p2, p3, p4 = core_voxel_grid._voxels[index].vertices
				voxel_centre = Point2D((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4)
				power_on_voxels[index] = radiation_function(voxel_centre.x, 0, voxel_centre.y)


		if is_this_extra:
			d = np.load(path_sensitivity + '/foil_power.npy')
			d_original = copy.deepcopy(d)
		else:
			d = np.dot(sensitivities, power_on_voxels)
			d_original = copy.deepcopy(d)

		if weigthed_best_search:
			path_sensitivity = path_sensitivity + '/weighted_best_alpha_search'
			if not os.path.exists(path_sensitivity):
				os.makedirs(path_sensitivity)
		else:
			path_sensitivity = path_sensitivity + '/no_weighted_best_alpha_search'
			if not os.path.exists(path_sensitivity):
				os.makedirs(path_sensitivity)

		if with_noise:
			path_sensitivity=path_sensitivity+'/with_noise_time_averaging_'+str(time_averaging)
			if not os.path.exists(path_sensitivity):
				os.makedirs(path_sensitivity)
			print('with noise, power std='+str(noise_on_power))
			if is_this_extra:
				foil_res_max = '_foil_pixel_h_' + str(foil_resolution_max)
				path_sensitivity_max_res = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res_max + '_power'
				if not os.path.exists(path_sensitivity_max_res + '/sensitivity.npz'):
					sensitivities_max_res = np.load(path_sensitivity_max_res + '/sensitivity.npy')
					scipy.sparse.save_npz(path_sensitivity_max_res + '/sensitivity.npz', scipy.sparse.csr_matrix(sensitivities_max_res))
				else:
					sensitivities_max_res = np.array((scipy.sparse.load_npz(path_sensitivity_max_res + '/sensitivity.npz')).todense())
				type = '.csv'
				filenames = coleval.all_file_names(path_sensitivity_max_res, type)[0]
				with open(os.path.join(path_sensitivity_max_res, filenames)) as csv_file:
					csv_reader = csv.reader(csv_file, delimiter=',')
					for row in csv_reader:
						if row[0] == 'foil horizontal pixels ':
							pixel_h_max_res = int(row[1])
				d_high_res = np.dot(sensitivities_max_res, power_on_voxels)
				d_original = coleval.foil_measurement_averaging_foil_pixels_extra(d_high_res,pixel_h_max_res,int(np.around(foil_resolution_max/foil_resolution)))
				d_high_res = d_high_res + np.random.normal(0, noise_on_power, len(d_high_res))
				d = coleval.foil_measurement_averaging_foil_pixels_extra(d_high_res, pixel_h_max_res, int(np.around(foil_resolution_max / foil_resolution)))

			else:
				d_original = copy.deepcopy(d)
				d = d_original + np.random.normal(0, noise_on_power, len(d_original))

			for index,value in enumerate(d):
				if value<=0:
					d[index]=0

		# Here there are different weighting functions
		if False:
			weight = np.ones(len(power_on_voxels))
		elif False:
			weight = power_on_voxels / (power_on_voxels).max()

		if weigthed_best_search:
			weight1 = d_original / (d_original).max()
			weight2 = power_on_voxels / (power_on_voxels).max()
		else:
			weight1 = np.ones((np.shape(sensitivities)[0]))
			weight2 = np.ones((np.shape(sensitivities)[1]))


		# treshold=3E-22	# this for 2cm 7 h pixels foil resolution
		# treshold=2E-21	# this for 2cm 19 h pixels foil resolution
		# treshold=1E-18	# this for 2cm full foil resolution
		check=[]
		s_record = []
		alpha_record=[]
		U_record = []
		Vh_record = []
		score_x_record=[]
		score_y_record=[]
		treshold_record = []

		find_minimum = min(np.shape(sensitivities))

		for exp in np.linspace(-30,-6,num=73):
			alpha=1*10**(exp)
			if exp==-50:
				alpha=0
			test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian

			if not os.path.exists(path_sensitivity_original+'/svg_decomp_alpha_'+str(alpha)+'.npz'):
				try:
					U, s, Vh = np.linalg.svd(test_matrix)
				except:
					print('the SVG decomposition of alpha = ' + str(alpha) + ' did not converge')
					continue
				# U, s, Vh = np.linalg.svd(test_matrix)
				np.savez_compressed(path_sensitivity_original+'/svg_decomp_alpha_'+str(alpha),U=U, s=s, Vh=Vh)
			else:
				U = np.load(path_sensitivity_original+'/svg_decomp_alpha_'+str(alpha)+'.npz')['U']
				s = np.load(path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha) + '.npz')['s']
				Vh = np.load(path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha) + '.npz')['Vh']

			if False:
				treshold = 1.5*s[50+coleval.find_nearest_index(s[100:]-s[:-100],0)]	#	the 1.5 is just to make sure to catch all of the negligible terms
			else:
				s_scaling = np.log(np.array(s)) - min(np.log(np.array(s)) - 1)
				s_scaling = s_scaling * 1 * (len(s) / max(s_scaling))
				s_scaling = s_scaling * np.linspace(int(min(s_scaling)), int(min(s_scaling)) + len(s), len(s))
				where1 = coleval.find_nearest_index(s_scaling, 1E20)
				for index in range(101, len(s_scaling) - 1, 1):
					# if ( np.mean(s_scaling[index-100:index-10])>s_scaling[index] and np.mean(s_scaling[index+10:index+100])>s_scaling[index] ):
					if (np.mean(s_scaling[index - 100:index - 10]) > s_scaling[index] and np.mean(s_scaling[index + 10:index + 100]) > s_scaling[index]):
						find_minimum = index +coleval.find_nearest_index(s_scaling[index - 100:index + 101],0)-100
						break
				treshold = s[find_minimum]
			treshold_record.append(treshold)
			shape = np.shape(test_matrix)
			U_record.append(U)
			Vh_record.append(Vh)
			zero = []
			for i in range(shape[0]):
				if s[i] <= treshold:
					zero.append(i)
			if zero == []:
				zero.append(shape[0])
			sigma = np.zeros((zero[0], zero[0]))
			U = U[:, :zero[0]]
			Vh = Vh[:zero[0]]
			for i in range(zero[0]):
				if s[i] <= treshold:
					sigma[i, i] = 0
				else:
					sigma[i, i] = s[i]
			a1 = np.dot(U, np.dot(sigma, Vh))
			inv_sigma = np.zeros((zero[0], zero[0]))
			for i in range(zero[0]):
				inv_sigma[i, i] = 1 / s[i]
			a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
			m = np.dot(a1_inv, np.dot(sensitivities.T, d))
			for index,value in enumerate(m):
				if value<=0:
					m[index]=0
			print('alpha='+str(alpha))
			print('treshold='+str(treshold))
			print('zero='+str(zero[0]))
			if False:
				score=np.sum(((power_on_voxels-m)**2)*weight)
				print('check=' + str(score))
				check.append(score)
			else:
				score_x = np.sum(((np.dot(sensitivities, m) - d) ** 2) * weight1)
				score_y = np.sum(((np.dot(laplacian,m)) ** 2) * weight2)
				print('||Gm-d||2=' + str(score_x))
				print('||Laplacian(m)||2=' + str(score_y))
				score_x_record.append(score_x)
				score_y_record.append(score_y)


			# print('conditioning=' + str(np.linalg.cond(test_matrix)))

			alpha_record.append(alpha)
			s_record.append(s)

			plt.plot(s, 'o')
			plt.plot(s)
			plt.plot([0,len(s)-1],[treshold, treshold])
			plt.yscale('log')
			plt.xlabel('i')
			plt.ylabel('eigenvalues')
			plt.grid()
			plt.savefig(path_sensitivity + '/eigenvalues_for_alpha_'+str(alpha)+'.eps')
			plt.close()


		index_best_fit=coleval.find_nearest_index(score_x_record,0)
		best_alpha=alpha_record[index_best_fit]
		plt.title('Best regularization coefficient is alpha='+str(best_alpha)+'\nwith residuals ||Gm-d||2='+str(score_x_record[index_best_fit]))
		plt.plot(score_x_record,score_y_record, 'x')
		plt.plot(score_x_record,score_y_record)
		plt.yscale('log')
		plt.xscale('log')
		plt.xlabel('||Gm-d||2')
		plt.ylabel('||Laplacian(m)||2')
		plt.grid()
		plt.savefig(path_sensitivity + '/L_curve.eps')
		plt.close()


		# print('alpha')
		# print(alpha_record)
		# print('check')
		# print(check)
		if False:
			alpha=alpha_record[coleval.find_nearest_index(check,0)[0]]
		elif False:
			alpha=0
		else:
			alpha = alpha_record[index_best_fit]
			treshold = treshold_record[index_best_fit]

		print('Best alpha='+str(alpha))
		test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
		# print(np.dot(sensitivities.T, sensitivities))

		if False:
			# check that SVD decomposition is consistent
			for i in range(10):
				U, s, Vh = np.linalg.svd(test_matrix)
				print('The next number should not change')
				print(s[0])
		elif False:
			U, s, Vh = np.linalg.svd(test_matrix)
		else:
			U = np.load(path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha) + '.npz')['U']
			s = np.load(path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha) + '.npz')['s']
			Vh = np.load(path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha) + '.npz')['Vh']

		print(s)

		shape = np.shape(test_matrix)

		zero = []
		for i in range(shape[0]):
			if s[i] <= treshold:
				zero.append(i)
		if zero == []:
			zero.append(shape[0])
		print(zero)
		sigma = np.zeros((zero[0], zero[0]))
		U = U[:, :zero[0]]
		Vh = Vh[:zero[0]]
		for i in range(zero[0]):
			if s[i] <= treshold:
				sigma[i, i] = 0
			else:
				sigma[i, i] = s[i]
		a1 = np.dot(U, np.dot(sigma, Vh))
		print(np.allclose(test_matrix, a1))

		inv_sigma = np.zeros((zero[0], zero[0]))
		for i in range(zero[0]):
			inv_sigma[i, i] = 1 / s[i]
		a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))

		scipy.sparse.save_npz(path_sensitivity + '/inverse_sensitivity_alpha' + str(alpha) + '.npz',scipy.sparse.csr_matrix(a1_inv))

		# print(np.trace(np.dot(a1,a1_inv)))
		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).max())
		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).min())

		core_voxel_grid.plot(voxel_values=power_on_voxels, colorbar=['rainbow', 'Emissivity [W/m3]', 'log'])
		# print(path_sensitivity+'/gnappo.eps')
		plt.title('Emission prifle form SOLPS simulation')
		plt.savefig(path_sensitivity + '/input_emission_profile.eps')
		plt.close()

		if not is_this_extra:
			if with_noise:
				plt.figure()
				cmap = plt.cm.rainbow
				cmap.set_under(color='white')
				vmin = 0.0000000000000000000000001
				plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
				plt.title('Power density on the foil via sensitivity matrix\nnoise std of ' + str(noise_on_power))
				plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off ' + str(vmin) + 'W/m^2')
				plt.xlabel('Horizontal axis [pixles]')
				plt.ylabel('Vertical axis [pixles]')
				plt.savefig(path_sensitivity + '/power_distribution_on_foil.eps')
				plt.close()
			else:
				plt.figure()
				cmap = plt.cm.rainbow
				cmap.set_under(color='white')
				vmin = 0.0000000000000000000000001
				plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
				plt.title('Power density on the foil via sensitivity matrix')
				plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off ' + str(vmin) + 'W/m^2')
				plt.xlabel('Horizontal axis [pixles]')
				plt.ylabel('Vertical axis [pixles]')
				plt.savefig(path_sensitivity + '/power_distribution_on_foil.eps')
				plt.close()

		# def l_curve_sample(x,*params):
		# 	import numpy as np
		# 	treshold=params[0]
		# 	max=1
		# 	if len(x)==():
		# 		if x>treshold:
		# 			return max
		# 		else:
		# 			return treshold
		# 	else:
		# 		out=[]
		# 		for i in range(len(x)):
		# 			if x[i] > treshold:
		# 				out.append(max)
		# 			else:
		# 				out.append(treshold)
		# 		return np.array(out)
		#
		# guess=np.array([1E-10])
		# curve_fit(l_curve_sample, np.linspace(0,len(s)-1,len(s)), s, p0=guess, maxfev=100000000)

		m = np.dot(a1_inv, np.dot(sensitivities.T, d))
		for index, value in enumerate(m):
			if value <= 0:
				m[index] = 0
		print('np.allclose(power_on_voxels, m) = ' + str(np.allclose(power_on_voxels, m)))
		if False:
			score = np.sum(((power_on_voxels - m) ** 2) * weight)
			print('check=' + str(score))
		else:
			score_x = np.sum(((np.dot(sensitivities, m) - d) ** 2) * weight1)
			score_y = np.sum(((np.dot(laplacian, m)) ** 2) * weight2)
			print('||Gm-d||2=' + str(score_x))
			print('||Laplacian(m)||2=' + str(score_y))
		# print('m')
		# print(m)
		# print('power_on_voxels')
		# print(power_on_voxels)

		plt.title('Input vs estimated emission profile ||Gm-d||2=' + str(
			np.around(score_x, decimals=2)) + '||Laplacian(m)||2=' + str(np.around(score_y, decimals=2)))
		plt.plot(m, label='estimation')
		plt.plot(power_on_voxels, label='input')
		plt.legend(loc='best')
		plt.savefig(path_sensitivity + '/emission_profile_compare.eps')
		plt.close()

		maximum_original = max(power_on_voxels)
		minimum_original = min([x for x in power_on_voxels if x != 0])
		to_print = np.zeros(np.shape(m))
		maximum_new_record = 0
		for index, value in enumerate(m):
			if (value >= minimum_original and value <= maximum_original):
				to_print[index] = value
			elif (value > maximum_original):
				to_print[index] = maximum_original
				if value > maximum_new_record:
					maximum_new_record = value
			elif (value < minimum_original and value != 0):
				to_print[index] = minimum_original
		to_print[-1] = maximum_original
		core_voxel_grid.plot(voxel_values=to_print, colorbar=['rainbow', 'Emissivity [W/m3]', 'log'])
		if maximum_new_record > 0:
			plt.title('Estimated emissivity\nmaximum of the inverted emissivity = ' + str(
				maximum_new_record) + '\n instead of ' + str(maximum_original))
		else:
			plt.title('Estimated emissivity\n')
		plt.savefig(path_sensitivity + '/estimated_emission.eps')
		plt.close()

		difference = m - power_on_voxels
		core_voxel_grid.plot(voxel_values=difference, colorbar=['rainbow', 'Emissivity [W/m3]'])
		plt.title('Difference of emission profile')
		plt.savefig(path_sensitivity + '/emission_difference.eps')
		plt.close()

		# min_greater_zero = min([x for x in m if x > 0])
		# max_below_zero = max([x for x in m if x < 0])
		# reference_min=min(min_greater_zero,-max_below_zero)
		reference_min = 1
		max_greater_zero = max([x for x in m if x > 0])
		# min_below_zero = min([x for x in m if x < 0])
		# reference_max = max(max_greater_zero, -min_below_zero)
		reference_max = max_greater_zero
		to_print = np.zeros(np.shape(difference))
		for index, value in enumerate(difference):
			if value > reference_min:
				to_print[index] = np.log(value / reference_min)
			elif value < -reference_min:
				to_print[index] = -np.log(-value / reference_min)
		core_voxel_grid.plot(voxel_values=to_print, colorbar=['rainbow',
															  'Logaritm of the difference with sign scaled on ' + str(
																  reference_min) + 'W/m3'])
		plt.savefig(path_sensitivity + '/logaritmic_estimated_difference.eps')
		plt.close()

		compare = np.zeros(np.shape(difference))
		for index, value in enumerate(difference):
			if power_on_voxels[index] > 0:
				compare[index] = value / power_on_voxels[index]
		core_voxel_grid.plot(voxel_values=compare, colorbar=['rainbow', 'Emissivity [W/m3]'])
		plt.title('Relative difference of emission profile')
		plt.savefig(path_sensitivity + '/relative_estimated_difference.eps')
		plt.close()

		# I do this bit to evaluate what is the standard deviation on the power on the voxels with noise

		if with_noise:
			number_of_tries = 1000
			m_all = []
			for i in range(number_of_tries):
				d = d_original + np.random.normal(0, noise_on_power, len(d_original))
				for index, value in enumerate(d):
					if value <= 0:
						d[index] = 0
				m = np.dot(a1_inv, np.dot(sensitivities.T, d))
				for index, value in enumerate(m):
					if value <= 0:
						m[index] = 0
			m_all = np.array(m_all)
			m_std = np.std(m_all, axis=(0))

			core_voxel_grid.plot(voxel_values=m_std, colorbar=['rainbow', 'Emissivity std [W/m3]'])
			plt.title('std of the inverted emissivity\nwith noise on foil power of ' + str(
				noise_on_power) + '\n mean of the voxel power std of ' + str(np.mean(m_std)))
			plt.savefig(path_sensitivity + '/noise.eps')
			plt.close()

		# I need to print stats to know what was going on

		header = ['# Record of what was found in this simulation of inversion']
		to_write = [['grid resolution in cm', str(grid_resolution)],
					['foil horizontal resolution in pixels', str(pixel_h)],
					['shape of sensitivity matrix ', str(np.shape(sensitivities))],
					['with noise, std of noise on power, time averaging ',
					 str([with_noise, noise_on_power, time_averaging])],
					['extra informations in the sensitivity matrix ', str(is_this_extra)],
					['Alpha that made possible SVG decomposition ', str(alpha_record)],
					['record of the values of the treshold on eigenvalues ', str(treshold_record)],
					['record of the residuals on solution ', str(score_x_record)],
					['record of the residuals on smoothing ', str(score_y_record)],
					[
						'relative to the best alpha: value, eigenvalue threshold, residuals on solution, residuals on smoothing',
						str([alpha, treshold, score_x_record[index_best_fit], score_y_record[index_best_fit]])]]
		# to_write='1, 1, '+str(foil_fake_corner1)[8:-1]+', '+str(foil_fake_corner2)[8:-1]+', '+str(foil_fake_corner3)[8:-1]+', '+str(foil_fake_corner4)[8:-1]

		with open(path_sensitivity + '/stats.csv', mode='w') as f:
			writer = csv.writer(f)
			writer.writerow(header)
			for row in to_write:
				writer.writerow(row)
		f.close()

	# foil_resolution_all = [187, 93, 62, 47, 37, 31, 26, 19]
	#
	# import concurrent.futures as cf
	# with cf.ProcessPoolExecutor() as executor:
	# 	executor.map(_run_, foil_resolution_all)

# exit()
#
# # I skip the rest because less relevant
#
#
#
#
# plt.plot(alpha_record,check,'o')
# plt.plot(alpha_record,check)
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('alpha')
# plt.ylabel('check')
# plt.savefig(path_sensitivity+'/gnappo5.eps')
# plt.close()
#
#
#
#
#
# if False:
# 	# This bit is to check different tresholds
#
# 	# print('alpha')
# 	# print(alpha_record)
# 	# print('check')
# 	# print(check)
# 	alpha=0
# 	test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
# 	# print(np.dot(sensitivities.T, sensitivities))
#
# 	shape=np.shape(test_matrix)
# 	U, s, Vh = np.linalg.svd(test_matrix)
#
# 	for exp in np.linspace(-19,-11,num=20):
# 		treshold=1*10**(exp)
# 		print('treshold ' + str(treshold))
# 		zero=[]
# 		for i in range(shape[0]):
# 			if s[i]<treshold	:
# 				zero.append(i)
# 		if zero==[]:
# 			zero.append(shape[0])
# 		print('zero' + str(zero[0]))
# 		sigma = np.zeros((zero[0],zero[0]))
# 		U=U[:,:zero[0]]
# 		Vh=Vh[:zero[0]]
# 		for i in range(zero[0]):
# 			if s[i]<treshold:
# 				sigma[i, i]=0
# 			else:
# 				sigma[i, i] = s[i]
# 		a1 = np.dot(U, np.dot(sigma, Vh))
#
# 		inv_sigma = np.zeros((zero[0],zero[0]))
# 		for i in range(zero[0]):
# 			inv_sigma[i, i] = 1/s[i]
# 		a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
#
# 		# print(np.trace(np.dot(a1,a1_inv)))
# 		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).max())
# 		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).min())
#
# 		d=np.dot(sensitivities,power_on_voxels)
# 		m=np.dot(a1_inv,np.dot(sensitivities.T,d))
# 		score = np.sum((power_on_voxels - m) ** 2)
# 		print('check='+str(score))
# 		# print('m')
# 		# print(m)
# 		# print('power_on_voxels')
# 		# print(power_on_voxels)
#
#
# 		core_voxel_grid.plot(voxel_values=m,colorbar=['rainbow','Emissivity [W/m3]','log'])
# 		plt.title('residuals = '+str(score))
# 		plt.savefig(path_sensitivity+'/treshold_'+str(treshold)+'.eps')
# 		plt.close()
#
# elif True:
#
# 	# I do this to verify which voxel inversion is better than the other
#
# 	back=0
# 	spike=1
# 	path=path_sensitivity + '/voxel_by_voxel_back'+str(back)+'-spike'+str(spike)
# 	if not os.path.exists(path):
# 		os.makedirs(path)
#
# 	check=[]
# 	a1_inv = np.array((scipy.sparse.load_npz(path_sensitivity + '/inverse_sensitivity.npz')).todense())
# 	for test_voxel in range(core_voxel_grid.count):
# 		power_on_voxels = np.ones((core_voxel_grid.count))*back
# 		power_on_voxels[test_voxel] = spike
#
# 		if False:
# 			weight = np.ones(len(power_on_voxels))
# 			weight_string='uniform'
# 		elif True:
# 			weight = power_on_voxels / (power_on_voxels).max()
# 			weight_string = 'linear on input emissivity'
#
# 		if (test_voxel % 20) == 0:
# 			core_voxel_grid.plot(voxel_values=power_on_voxels, colorbar=['rainbow', 'Emissivity [W/m3]'])
# 			plt.title('Input emissivity')
# 			plt.savefig(path + '/voxel_'+str(test_voxel)+'input.eps')
# 			plt.close()
# 		d = np.dot(sensitivities, power_on_voxels)
# 		m = np.dot(a1_inv, np.dot(sensitivities.T, d))
# 		score = np.sum(((power_on_voxels - m) ** 2) * weight)
# 		if (test_voxel % 20) == 0:
# 			core_voxel_grid.plot(voxel_values=m,colorbar=['rainbow','Emissivity [W/m3]'])
# 			plt.title('Estimated emissivity \n Score '+str(score))
# 			plt.savefig(path + '/voxel_'+str(test_voxel)+'estimation.eps')
# 			plt.close()
# 		check.append(score)
# 	check=np.array(check)
# 	print(repr(check))
# 	print(np.shape(check))
# 	core_voxel_grid.plot(voxel_values=check, colorbar=['rainbow', 'score [au]'])
# 	plt.title(weight_string+' weight score to find one single pixel')
# 	plt.savefig(path + '/all_scores.eps')
# 	plt.close()