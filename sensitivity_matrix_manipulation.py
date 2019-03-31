# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())


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

















# data analysis

foil_res = '_foil_pixel_h_187'
grid_type = 'core_res_2cm'
path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
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

# plt.figure()
# test_pixel=1000
num_voxels = core_voxel_grid.count
# i_all=np.linspace(0,num_voxels-1,num_voxels,dtype=int)
# import copy
# sample = copy.deepcopy(i_all); sample[test_pixel]=10000
# core_voxel_grid.plot(voxel_values=sample)
# plt.pause(0.0001)
#
# plt.figure()
# plt.imshow(coleval.split_fixed_length(sensitivities[:,1000],pixel_h),origin='lower')
# plt.pause(0.0001)
#
# sensitivities_averaged=sensitivities_matrix_averaging_foil_pixels(sensitivities,pixel_h,2,2)
# plt.figure()
# plt.imshow(np.array(coleval.split_fixed_length(sensitivities_averaged[:,1000],93)),origin='lower')
# plt.pause(0.0001)



averaging_all = [2,3,4,5,6,7,8,9,10,15]
for averaging_h in averaging_all:

	# averaging_h=2
	averaging_v=averaging_h
	shapeorig = np.shape(sensitivities)
	npixels = shapeorig[0]
	h_pixels = pixel_h
	v_pixels = npixels // h_pixels
	h_end_pixels = np.ceil(h_pixels / averaging_h).astype('int')
	v_end_pixels = np.ceil(v_pixels / averaging_v).astype('int')
	foil_res = '_foil_pixel_h_'+str(averaging_h)+'x'+str(averaging_v)+'_extra'
	grid_type = 'core_res_2cm'

	print(coleval.sensitivities_matrix_averaging_foil_pixels_extra_loseless)

	sensitivities_averaged=coleval.sensitivities_matrix_averaging_foil_pixels_extra_loseless(sensitivities,pixel_h,averaging_h,averaging_v)




	path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
	if not os.path.exists(path_sensitivity):
		os.makedirs(path_sensitivity)
	scipy.sparse.save_npz(path_sensitivity + '/sensitivity.npz', scipy.sparse.csr_matrix(sensitivities_averaged))

	import csv
	header = ['# Sensitivity matrix generated with:','# ']
	to_write=[['foil horizontal pixels ',str(h_end_pixels)],['foil vertical pixels ',str(v_end_pixels)],['type of volume grid ',grid_type],['number of voxels ',num_voxels],['pipeline type ','pipeline'],['detector.pixel_samples',averaging_v*averaging_h*1000],['this sensitivity matrix was generated loseless from the full frame sensitivity']]
	# to_write='1, 1, '+str(foil_fake_corner1)[8:-1]+', '+str(foil_fake_corner2)[8:-1]+', '+str(foil_fake_corner3)[8:-1]+', '+str(foil_fake_corner4)[8:-1]

	with open(path_sensitivity+'/sensitivity_matrix_info.csv', mode='w') as f:
		writer = csv.writer(f)
		writer.writerow(header[0].split(','))
		writer.writerow([header[1]])
		for row in to_write:
			writer.writerow(row)
	f.close()











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



	d = np.dot(sensitivities, power_on_voxels)
	d = coleval.foil_measurement_averaging_foil_pixels_extra_loseless(d,pixel_h,averaging_h,averaging_v)

	# foil_res = '_foil_pixel_h_93_extra'
	# grid_type = 'core_res_2cm'
	# path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
	np.save(path_sensitivity+'/foil_power',d)





	foil_res = foil_res[:-6]


	# sensitivities_averaged=coleval.sensitivities_matrix_averaging_foil_pixels(sensitivities,pixel_h,averaging_h,averaging_v)
	sensitivities_averaged=coleval.sensitivities_matrix_averaging_foil_pixels_loseless(sensitivities,pixel_h,averaging_h,averaging_v)


	path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
	if not os.path.exists(path_sensitivity):
		os.makedirs(path_sensitivity)
	scipy.sparse.save_npz(path_sensitivity + '/sensitivity.npz', scipy.sparse.csr_matrix(sensitivities_averaged))


	with open(path_sensitivity+'/sensitivity_matrix_info.csv', mode='w') as f:
		writer = csv.writer(f)
		writer.writerow(header[0].split(','))
		writer.writerow([header[1]])
		for row in to_write:
			writer.writerow(row)
	f.close()



