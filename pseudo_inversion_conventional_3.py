# Created 29/09/2019
# Fabio Federici


# #this is if working on a pc, use pc printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

#this is if working in batch, use predefined NOT visual printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())



from multiprocessing import Pool,cpu_count,set_start_method
# set_start_method('spawn',force=True)
try:
	number_cpu_available = open('/proc/cpuinfo').read().count('processor\t:')
except:
	number_cpu_available = cpu_count()
print('Number of cores available: '+str(number_cpu_available))



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


def import_radiator_from_solps_sim_on_triangular_mesh(sim):

	import numpy as np
	from cherab.core.math.mappers import AxisymmetricMapper
	from raysect.core.math.interpolators import Discrete2DMesh

	radiated_power=sim.total_radiation.flatten()
	rad_power = np.zeros(radiated_power.shape[0]*2)
	for i in range(radiated_power.shape[0]):
		rad_power[i*2] = radiated_power[i]
		rad_power[i*2 + 1] = radiated_power[i]

	return AxisymmetricMapper(Discrete2DMesh(sim.mesh.vertex_coords, sim.mesh.triangles, rad_power,limit=False))




class RadiatedPower(InhomogeneousVolumeEmitter):

	def __init__(self, radiation_function):
		super().__init__()
		self.radiation_function = radiation_function

	def emission_function(self, point, direction, spectrum, world, ray, primitive, to_local, to_world):

		p = point.transform(to_world)
		spectrum.samples[0] += self.radiation_function(p.x, p.y, p.z)

		return spectrum





# def _run_(foil_resolution):

# Load emission profile

# mastu_path = "/home/ffederic/work/SOLPS/seeding/seed_10"
# if not mastu_path in sys.path:
# 	sys.path.append(mastu_path)
# ds_puff8 = xr.open_dataset('/home/ffederic/work/SOLPS/seeding/seed_10/balance.nc', autoclose=True).load()

os.chdir("/home/ffederic/work/cherab/cherab_solps/cherab/solps/formats")
from mdsplus import load_solps_from_mdsplus,load_mesh_from_mdsplus


# ref_number = 69592	#from mdsnos_cd2H
# # ref_number = 69570	#from mdsnos_cd15H
# # ref_number = 69566	#from mdsnos_cd1H
# sim = load_solps_from_mdsplus(mds_server, ref_number)

# for ref_number in mdsnos_cd2H:
# 	sim = load_solpstime_averaging_all_from_mdsplus(mds_server, ref_number)

# ref_number = mdsnos_sxd2H[0]
# ref_number = mdsnos_cd2H[0]


for ref_number in np.flip([*mdsnos_cd15H,*mdsnos_cd2H,*mdsnos_sxd2L,*mdsnos_sxd15H,*mdsnos_sxd2H],axis=0):
# for ref_number in np.flip(mdsnos_sxd2H[:-2],axis=0):
	print('loading SOPLS simulation '+str(ref_number))
	sim = load_solps_from_mdsplus(mds_server, ref_number)
	tot_pow = 0
	for i,vol in enumerate(sim.mesh.vol.flatten()):
		tot_pow+=vol*(sim.total_radiation.flatten())[i]

	fig, ax = plt.subplots()
	grid_x = sim.mesh.cr
	grid_y = sim.mesh.cz
	total_radiation_density = sim.total_radiation
	plt.pcolor(grid_x, grid_y, total_radiation_density, norm=LogNorm(vmin=1000, vmax=total_radiation_density.max()),cmap='rainbow')
	plt.title('SOLPS simulation '+str(ref_number)+'\n total emitted power '+str(tot_pow)+'W')
	plt.colorbar().set_label('Emissivity [W/m^3]')
	plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	plt.xlabel('R [m]')
	plt.ylabel('Z [m]')
	plt.axis('equal')
	ax.set_ylim(top=-0.5)
	ax.set_xlim(left=0.)
	if not os.path.exists('/home/ffederic/work/analysis_scripts'+'/SOLPS'+str(ref_number)):
		os.makedirs('/home/ffederic/work/analysis_scripts'+'/SOLPS'+str(ref_number))
	plt.savefig('/home/ffederic/work/analysis_scripts'+'/SOLPS'+str(ref_number) + '/SOLPS_simulation_'+str(ref_number)+'_emissivity_distr.eps')
	plt.close()
	# plt.pause(0.01)



	radiation_function = import_radiator_from_solps_sim_on_triangular_mesh(sim)



	# fig, ax = plt.subplots()
	# X = np.arange(0, grid_x.max(), 0.01)
	# Y = np.arange(grid_y.min(), grid_y.max(), 0.01)
	# rad_test = np.zeros((len(X), len(Y)))
	# grid_x2 = np.zeros((len(X), len(Y)))
	# grid_y2 = np.zeros((len(X), len(Y)))
	# tot_power = 0
	# for ix, x in enumerate(X):
	# 	for jy, y in enumerate(Y):
	# 		rad_test[ix, jy] = radiation_function(x, 0, y)
	# 		grid_x2[ix, jy] = x
	# 		grid_y2[ix, jy] = y
	# 		# if y<-1.42:
	# 		tot_power+=radiation_function(x, 0, y)*0.001*0.001*2*np.pi*x
	#
	#
	# plt.pcolor(grid_x2, grid_y2, rad_test, norm=LogNorm(vmin=100),cmap='rainbow')
	# # plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values),cmap='rainbow')
	# plt.title('Emissivity profile as imported from SOLPS using CHERAB utilities')
	# plt.colorbar().set_label('Emissivity [W/m^3]')
	# plt.xlabel('R [m]')
	# plt.ylabel('Z [m]')
	# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	#
	# x=np.linspace(0.55-0.075,0.55+0.075,10)
	# y=-1.2+np.sqrt(0.08**2-(x-0.55)**2)
	# y_=-1.2-np.sqrt(0.08**2-(x-0.55)**2)
	# plt.plot(x,y,'k')
	# plt.plot(x,y_,'k')
	# plt.axis('equal')
	# ax.set_ylim(top=-0.5)
	# ax.set_xlim(left=0.)
	# plt.pause(0.0001)









	is_this_extra = False
	# foil_resolution = 47
	grid_resolution = 2  # in cm
	for with_noise in [True,False]:
	# for with_noise in [True]:
	# with_noise = True
		foil_resolution_max = 187
		weigthed_best_search = False
		eigenvalue_cut_vertical = False
		enable_mask2 = False
		enable_mask1 = True
		# treshold_method_try_to_search = True
		# residuals_on_power_on_voxels = False
		for residuals_on_power_on_voxels in [True, False]:
		# for residuals_on_power_on_voxels in [False]:
			alpha_exponents_to_test = np.linspace(-13,-1,num=49)



			spatial_averaging_all = [1,2,3,4,5,6,8,10]
			time_averaging_all = [1,2,3,4,5]


			# # foil_resolution_all = [37, 31, 26, 19]
			# foil_resolution_all = [47, 37, 31, 26, 19]
			# # foil_resolution_all = [93]
			# for foil_resolution in foil_resolution_all:
			# 	if (is_this_extra and foil_resolution==187):
			# 		continue


			exec(open("/home/ffederic/work/analysis_scripts/scripts/pseudo_inversion_Tikhonov.py").read())
