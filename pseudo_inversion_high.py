# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/preamble_indexing.py").read())


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


grid_type = 'core_res_2cm'
path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+'_power'
if False:
	sensitivities=np.load(path_sensitivity+'/sensitivity.npy')
elif True:
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


# Here there are different weighting functions
if True:
	weight=np.ones(len(power_on_voxels))
elif False:
	weight=power_on_voxels/(power_on_voxels).max()


treshold=1E-18
check=[]
alpha_record=[]
for exp in np.linspace(-50,-50,num=1):
	alpha=1*10**(exp)
	if exp==-50:
		alpha=0
	test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
	U, s, Vh = np.linalg.svd(test_matrix)
	shape = np.shape(test_matrix)
	zero = []
	for i in range(shape[0]):
		if s[i] < treshold:
			zero.append(i)
	if zero == []:
		zero.append(shape[0])
	sigma = np.zeros((zero[0], zero[0]))
	U = U[:, :zero[0]]
	Vh = Vh[:zero[0]]
	for i in range(zero[0]):
		if s[i] < treshold:
			sigma[i, i] = 0
		else:
			sigma[i, i] = s[i]
	a1 = np.dot(U, np.dot(sigma, Vh))
	inv_sigma = np.zeros((zero[0], zero[0]))
	for i in range(zero[0]):
		inv_sigma[i, i] = 1 / s[i]
	a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
	d = np.dot(sensitivities, power_on_voxels)
	m = np.dot(a1_inv, np.dot(sensitivities.T, d))
	score=np.sum(((power_on_voxels-m)**2)*weight)
	print('alpha='+str(alpha))
	print('zero'+str(zero[0]))
	print('check='+str(score))
	print('conditioning=' + str(np.linalg.cond(test_matrix)))
	check.append(score)
	alpha_record.append(alpha)

	plt.plot(s, 'o')
	plt.plot(s)
	plt.yscale('log')
	plt.xlabel('i')
	plt.ylabel('eigenvalues')
	plt.grid()
	plt.savefig(path_sensitivity + '/eigenvalues_for_alpha_'+str(alpha)+'.eps')
	plt.close()


# print('alpha')
# print(alpha_record)
# print('check')
# print(check)
if False:
	alpha=alpha_record[coleval.find_nearest_index(check,0)[0]]
elif True:
	alpha=0
print('Best alpha='+str(alpha))
test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
# print(np.dot(sensitivities.T, sensitivities))

if False:
	# check that SVD decomposition is consistent
	for i in range(10):
		U, s, Vh = np.linalg.svd(test_matrix)
		print('The next number should not change')
		print(s[0])
else:
	U, s, Vh = np.linalg.svd(test_matrix)

print(s)

shape=np.shape(test_matrix)

zero=[]
for i in range(shape[0]):
	if s[i]<treshold	:
		zero.append(i)
if zero==[]:
	zero.append(shape[0])
print(zero)
sigma = np.zeros((zero[0],zero[0]))
U=U[:,:zero[0]]
Vh=Vh[:zero[0]]
for i in range(zero[0]):
	if s[i]<treshold:
		sigma[i, i]=0
	else:
		sigma[i, i] = s[i]
a1 = np.dot(U, np.dot(sigma, Vh))
print(np.allclose(test_matrix, a1))

inv_sigma = np.zeros((zero[0],zero[0]))
for i in range(zero[0]):
	inv_sigma[i, i] = 1/s[i]
a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))

scipy.sparse.save_npz(path_sensitivity+'/inverse_sensitivity.npz',scipy.sparse.csr_matrix(a1_inv))


# print(np.trace(np.dot(a1,a1_inv)))
# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).max())
# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).min())




core_voxel_grid.plot(voxel_values=power_on_voxels,colorbar=['rainbow','Emissivity [W/m3]','log'])
print(path_sensitivity+'/gnappo.eps')
plt.savefig(path_sensitivity+'/gnappo.eps')
plt.close()

d=np.dot(sensitivities,power_on_voxels)
plt.figure()
cmap = plt.cm.rainbow
cmap.set_under(color='white')
vmin=0.0000000000000000000000001
plt.imshow(coleval.split_fixed_length(d,pixel_h),origin='lower', cmap=cmap, vmin=vmin)
plt.title('Power density on the foil via sensitivity matrix')
plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off '+str(vmin)+'W/m^2')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.savefig(path_sensitivity+'/gnappo2.eps')
plt.close()




m=np.dot(a1_inv,np.dot(sensitivities.T,d))
print('np.allclose(power_on_voxels, m) = '+str(np.allclose(power_on_voxels, m)))
score = np.sum((power_on_voxels - m) ** 2)
print('check='+str(score))
# print('m')
# print(m)
# print('power_on_voxels')
# print(power_on_voxels)

plt.plot(m,label='estimation, check='+str(np.around(score,decimals=3)))
plt.plot(power_on_voxels,label='input')
plt.legend(loc='best')
plt.savefig(path_sensitivity+'/gnappo3.eps')
plt.close()


core_voxel_grid.plot(voxel_values=m,colorbar=['rainbow','Emissivity [W/m3]','log'])
plt.savefig(path_sensitivity+'/gnappo4.eps')
plt.close()

plt.plot(alpha_record,check,'o')
plt.plot(alpha_record,check)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('check')
plt.savefig(path_sensitivity+'/gnappo5.eps')
plt.close()





if False:
	# This bit is to check different tresholds

	# print('alpha')
	# print(alpha_record)
	# print('check')
	# print(check)
	alpha=0
	test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
	# print(np.dot(sensitivities.T, sensitivities))

	shape=np.shape(test_matrix)
	U, s, Vh = np.linalg.svd(test_matrix)

	for exp in np.linspace(-19,-11,num=20):
		treshold=1*10**(exp)
		print('treshold ' + str(treshold))
		zero=[]
		for i in range(shape[0]):
			if s[i]<treshold	:
				zero.append(i)
		if zero==[]:
			zero.append(shape[0])
		print('zero' + str(zero[0]))
		sigma = np.zeros((zero[0],zero[0]))
		U=U[:,:zero[0]]
		Vh=Vh[:zero[0]]
		for i in range(zero[0]):
			if s[i]<treshold:
				sigma[i, i]=0
			else:
				sigma[i, i] = s[i]
		a1 = np.dot(U, np.dot(sigma, Vh))

		inv_sigma = np.zeros((zero[0],zero[0]))
		for i in range(zero[0]):
			inv_sigma[i, i] = 1/s[i]
		a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))

		# print(np.trace(np.dot(a1,a1_inv)))
		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).max())
		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).min())

		d=np.dot(sensitivities,power_on_voxels)
		m=np.dot(a1_inv,np.dot(sensitivities.T,d))
		score = np.sum((power_on_voxels - m) ** 2)
		print('check='+str(score))
		# print('m')
		# print(m)
		# print('power_on_voxels')
		# print(power_on_voxels)


		core_voxel_grid.plot(voxel_values=m,colorbar=['rainbow','Emissivity [W/m3]','log'])
		plt.title('residuals = '+str(score))
		plt.savefig(path_sensitivity+'/treshold_'+str(treshold)+'.eps')
		plt.close()

elif True:

	# I do this to verify which voxel inversion is better than the other

	back=0
	spike=1
	path=path_sensitivity + '/voxel_by_voxel_back'+str(back)+'-spike'+str(spike)
	if not os.path.exists(path):
		os.makedirs(path)

	check=[]
	a1_inv = np.array((scipy.sparse.load_npz(path_sensitivity + '/inverse_sensitivity.npz')).todense())
	for test_voxel in range(core_voxel_grid.count):
		power_on_voxels = np.ones((core_voxel_grid.count))*back
		power_on_voxels[test_voxel] = spike

		if True:
			weight = np.ones(len(power_on_voxels))
			weight_string='uniform'
		elif False:
			weight = power_on_voxels / (power_on_voxels).max()
			weight_string = 'linear on input emissivity'

		if (test_voxel % 20) == 0:
			core_voxel_grid.plot(voxel_values=power_on_voxels, colorbar=['rainbow', 'Emissivity [W/m3]'])
			plt.title('Input emissivity')
			plt.savefig(path + '/voxel_'+str(test_voxel)+'input.eps')
			plt.close()
		d = np.dot(sensitivities, power_on_voxels)
		m = np.dot(a1_inv, np.dot(sensitivities.T, d))
		score = np.sum(((power_on_voxels - m) ** 2) * weight)
		if (test_voxel % 20) == 0:
			core_voxel_grid.plot(voxel_values=m,colorbar=['rainbow','Emissivity [W/m3]'])
			plt.title('Estimated emissivity \n Score '+str(score))
			plt.savefig(path + '/voxel_'+str(test_voxel)+'estimation.eps')
			plt.close()
		check.append(score)
	check=np.array(check)
	print(repr(check))
	print(np.shape(check))
	core_voxel_grid.plot(voxel_values=check, colorbar=['rainbow', 'score [au]'])
	plt.title(weight_string+' weight score to find one single pixel')
	plt.savefig(path + '/all_scores.eps')
	plt.close()