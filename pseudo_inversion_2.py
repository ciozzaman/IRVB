# Created 13/12/2018
# Fabio Federici


# #this is if working on a pc, use pc printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

#this is if working in batch, use predefined NOT visual printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())





from multiprocessing import Pool,cpu_count
try:
	number_cpu_available = open('/proc/cpuinfo').read().count('processor\t:')
except:
	number_cpu_available = cpu_count()
print('Number of cores available: '+str(number_cpu_available))









# def _run_(foil_resolution):

# Load emission profile
ref_number = 'seeding_10'
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
	try:
		Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)
	except:
		print('FAIL')

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






is_this_extra = False
# foil_resolution = 47
grid_resolution = 2  # in cm
for with_noise in [False,True]:
	# with_noise = True
	foil_resolution_max = 187
	weigthed_best_search = False
	eigenvalue_cut_vertical = False
	enable_mask2 = False
	enable_mask1 = True
	# treshold_method_try_to_search = True
	# residuals_on_power_on_voxels = False
	for residuals_on_power_on_voxels in [True,False]:
		alpha_exponents_to_test = np.linspace(-13, -1, num=49)

		spatial_averaging_all = [1, 2, 3, 4, 5, 6, 8, 10]
		# time_averaging_all = [1,2,3,4,5]
		time_averaging_all = [1]

		# # foil_resolution_all = [37, 31, 26, 19]
		# foil_resolution_all = [47, 37, 31, 26, 19]
		# # foil_resolution_all = [93]
		# for foil_resolution in foil_resolution_all:
		# 	if (is_this_extra and foil_resolution==187):
		# 		continue

		exec(open("/home/ffederic/work/analysis_scripts/scripts/pseudo_inversion_Tikhonov.py").read())


