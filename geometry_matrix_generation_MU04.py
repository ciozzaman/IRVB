# Created 13/12/2018
# Fabio Federici


# #this is if working on a pc, use pc printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

#this is if working in batch, use predefined NOT visual printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
import copy




# Load voxels
from raysect.optical import World
if False:	# not used because in source /home/jlovell/venvs/cherab/bin/activate, that has the last stable version of cherab, the option 'path' is not available
	from cherab.mastu.bolometry import load_standard_voxel_grid
elif False:	# used because in source /home/jlovell/venvs/cherab/bin/activate, that has the last stable version of cherab, the option 'path' was not available
	# os.chdir("/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction")
	from load_voxel_grids import load_standard_voxel_grid
else:	# It seems that now 2024/02/29, using my own venv source /home/ffederic/venvs/cherab_20230727/bin/activate the old path works
	from cherab.mastu.bolometry import load_standard_voxel_grid
world = World()
grid_type = 'core_res_2cm'
# grid_type = 'core_high_res'
core_voxel_grid = load_standard_voxel_grid(grid_type,parent=world, grid_file='/home/ffederic/work/analysis_scripts/grid_construction/'+grid_type+'_rectilinear_grid.pickle')
# core_voxel_grid = load_standard_voxel_grid(grid_type,parent=world, path='/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction')



# Load IRVB geometry
from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
from raysect.primitive import Mesh
from raysect.optical.material import NullMaterial
from raysect.optical import World
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.observer import TargettedCCDArray, PowerPipeline2D,RadiancePipeline2D
from raysect.optical import UnityVolumeEmitter
from raysect.core import SerialEngine, MulticoreEngine
from cherab.tools.observers import BolometerIRVB,BolometerSlit



os.chdir("/home/ffederic/work/cherab/cherab_mastu/")
from cherab.mastu.machine import MASTU_FULL_MESH






for cad_file in MASTU_FULL_MESH:
	directory, filename = os.path.split(cad_file[0])
	name, ext = filename.split('.')
	print("importing {} ...".format(filename))
	Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)


# now I want to build a box around the IRVB foil, following the description in
# https://www.cherab.info/demonstrations/bolometry/camera_from_primitives.html#bolometer-from-primitives

# First let's define the real position of all the important geometrical features as per the measured pinhole and flange positions
pinhole_size = 4	# mm
stand_off=0.060	# 0.045 / 0.060 / 0.075
pinhole_centre = Point3D(-1.0424941 ,  1.07123647, -0.7168958)
detector_flange_centre = Point3D(-1.05649,1.05731,-0.69705)
ccd_normal = Vector3D(0.4439085,-0.43815247,0.0021807).normalise()
pinhole_to_foil_vertical = 0.01 + 0.003 + 0.002 + stand_off
ccd_centre = detector_flange_centre -pinhole_to_foil_vertical*ccd_normal
ccd_x_axis = -ccd_normal.cross(Vector3D(0,0,1)).normalise()	# this is basically a horizonthal vector. I add the minus in front only so I have the same orientation as previously
ccd_y_axis = -ccd_x_axis.cross(ccd_normal).normalise()	# this is basically a vertical vector. I add the minus in front only so I have the same orientation as previously
IRVB_cylinder_diameter = 0.18	# m
IRVB_cylinder_length = 0.63	# m
pinhole_offset = np.array([-0.0198,-0.0198])	# x,y, assuming here that y coincides with the vertical axis. configuration up to MU04
foil_size = [0.07,0.09]


from raysect.primitive import import_stl,Sphere,Cylinder,Subtract

'''The cylinder is defined by a radius and height. It lies along the z-axis
and extends over the z range [0, height]. The ends of the cylinder are
capped with disks forming a closed surface.'''

IRVB_cylinder = Cylinder(radius=IRVB_cylinder_diameter/2,height=IRVB_cylinder_length, parent=world)
IRVB_cylinder_outside = Cylinder(radius=IRVB_cylinder.radius+1e-5,height=IRVB_cylinder.height+1e-5*2, parent=world,transform=translate(0,0,-1e-5))
IRVB_cylinder = Subtract(IRVB_cylinder_outside,IRVB_cylinder)
aperture = Cylinder(radius=pinhole_size*1e-3/2,height=6e-5, parent=world,transform=translate(*pinhole_offset,-3e-5))
IRVB_cylinder = Subtract(IRVB_cylinder,aperture)
IRVB_cylinder.material = AbsorbingSurface()

IRVB_cylinder.transform = translate(*detector_flange_centre) * rotate_basis(-ccd_normal, ccd_y_axis)	# cylinder oriented and translated to correct position

# pinhole_slit = Cylinder(radius=pinhole_size*1e-3/2+0.001,height=6e-5, transform=translate(0,0,-3e-5), parent=world, material=NullMaterial())
# pinhole_slit.transform = translate(*pinhole_centre) * rotate_basis(-ccd_normal, ccd_y_axis)
slit = BolometerSlit(slit_id="IRVB_MU04_HE11-2", centre_point=Point3D(0,0,0),basis_x=Vector3D(1,0,0), dx=pinhole_size*1e-3/(2**0.5), basis_y=Vector3D(0,1,0), dy=pinhole_size*1e-3/(2**0.5),parent=world)	# even if it looks as making a rectangle, it makes a circular aperture that contains the rectangle
slit.transform = translate(*pinhole_centre) * rotate_basis(-ccd_normal, ccd_y_axis)


pixel_h=187
pixel_v=int((pixel_h*foil_size[1])//foil_size[0])
pixel_area = foil_size[0]*foil_size[1]/(pixel_h*pixel_v)

# plt.ion()
# radiance = RadiancePipeline2D()
# detector = TargettedCCDArray(targets=[pinhole_target], width=0.07, pixels=(pixel_h, pixel_v), targetted_path_prob=1.0,
# 							 parent=world, pipelines=[radiance],
# 							 transform=translate(*ccd_centre)*rotate_basis(ccd_normal, ccd_y_axis))
# detector.max_wavelength = 601
# detector.min_wavelength = 600
# detector.spectral_bins = 1
# detector.pixel_samples = 500

num_voxels = core_voxel_grid.count
num_pixels = pixel_h * pixel_v
# sensitivities = np.zeros((num_voxels, num_pixels))


n_samples_per_pixel=2000# 5000	# for low resolution I used 5000, 10000 for high
path_sensitivity = '/home/ffederic/work/analysis_scripts/MU04_sensitivity_matrix_'+grid_type[5:]+'_foil_pixel_h_'+str(pixel_h)+'_power'+'_stand_off_'+str(stand_off)+'_pinhole_'+str(pinhole_size)
if not os.path.exists(path_sensitivity):
	os.makedirs(path_sensitivity)

# I add this just to superimpose MAST-U outline to the voxels

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



# pipeline='radiance'
pipeline='power'

IRVB = BolometerIRVB(name='IRVB_MU04_HE11-2',width=foil_size[0],pixels=(pixel_h,pixel_v),slit=slit,transform = translate(*ccd_centre) * rotate_basis(ccd_normal, ccd_y_axis),parent=world,units=pipeline)
IRVB.max_wavelength = 601
IRVB.min_wavelength = 600
IRVB.spectral_bins = 1
IRVB.pixel_samples = n_samples_per_pixel
# plt.ion()
# plt.figure()
IRVB.observe()
IRVB.calculate_sensitivity(core_voxel_grid)





















# def sensitivity(world,pinhole_target,pixel_h, pixel_v,ccd_centre,ccd_normal,ccd_y_axis,core_voxel_grid):

override = True
if False:
	# Use this bit if you want to split the total number of processes
	total_number = num_voxels
	launch=6
	number_of_launches = 6

	start=int((launch-1)*total_number/number_of_launches-0.999999)
	end = int(launch*total_number/number_of_launches+0.999999)
	if (end>(num_voxels-1)):
		end=num_voxels-1
	intervals = end-start+1
	# i=np.linspace(0,num_voxels-1,num_voxels,dtype=int)
	i=np.linspace(start,end,intervals,dtype=int)
elif False:
	# Use this for single process
	i=np.linspace(0,num_voxels-1,num_voxels,dtype=int)
else:
	# use this to process only missing .npy files
	filenames=coleval.all_file_names(path_sensitivity,'.npy')
	done_ones = []
	for filename in filenames:
		done_ones.append(int(filename[filename.find('voxel')+5:filename.find('.')]))
	i_all=np.linspace(0,num_voxels-1,num_voxels,dtype=int)
	to_do = []
	for value in i_all:
		if not (value in done_ones):
			to_do.append(value)
	i = np.array(to_do)
	random_sequence = np.random.random(len(i))
	i = np.array([y for _, y in sorted(zip(random_sequence, i))])
	override = False
i_all=np.linspace(0,num_voxels-1,num_voxels,dtype=int)



for index in i:
# for index in np.flip(i,axis=0):
	index = int(index)
	path = path_sensitivity+'/sensitivity_voxel'+str(index)
	if not override:
		if os.path.exists(path+'.npy'):	# if I launch multiple of thius together I want to avoid the same file over written
			continue
	# def sensitivity(i):
	# 	global world
	# 	global pinhole_target
	# 	global pixel_h
	# 	global pixel_v
	# 	global cNSLOTScd_centre
	# 	global ccd_normal
	# 	global ccd_y_axis
	# 	global core_voxel_grid
	# 	global path_sensitivity

	# def sensitivity_internal(i):
	if pipeline=='radiance':
		radiance = RadiancePipeline2D()
		detector = TargettedCCDArray(targets=[pinhole_target], width=0.07, pixels=(pixel_h, pixel_v),
									 targetted_path_prob=1.0,
									 parent=world, pipelines=[radiance],
									 transform=translate(*ccd_centre) * rotate_basis(ccd_normal, ccd_y_axis))
		# width corresponds to pixels[0], so this is correct
	else:
		power = PowerPipeline2D()
		detector = TargettedCCDArray(targets=[pinhole_target], width=0.07, pixels=(pixel_h, pixel_v),
									 targetted_path_prob=1.0,
									 parent=world, pipelines=[power],
									 transform=translate(*ccd_centre) * rotate_basis(ccd_normal, ccd_y_axis))

	detector.max_wavelength = 601
	detector.min_wavelength = 600
	detector.spectral_bins = 1
	detector.pixel_samples = n_samples_per_pixel
	# if index>0:
	# 	core_voxel_grid._voxels[index-1].parent = None
	# del core_voxel_grid
	# core_voxel_grid = load_standard_voxel_grid('core_low_res', parent=world)
	core_voxel_grid.set_active(index)
	# core_voxel_grid._voxels[index].parent = world
	# core_voxel_grid._voxels[index].material = UnityVolumeEmitter()

	try:
		nslots = int(os.environ['NSLOTS'])
		if nslots == 1:
			detector.render_engine = SerialEngine()
		else:
			detector.render_engine = MulticoreEngine(nslots)
		print('using '+str(nslots)+' cores')
	except KeyError:
		pass

	detector.observe()

	# note from 2021-10-08
	# core_voxel_grid.set_active(index) creates a unitary volume emitter with 1W/str/m^3/ x nm
	# assuming that the frequency interval is 1nm (it should be from detector.max_wavelength = 601, detector.min_wavelength = 600, detector.spectral_bins = 1)
	# then the total radiated power from the voxel is 1*4*pi, not just 1, as it is assumed here
	# I don't want to change the geometry matrices, to I'll need to carry this 4*pi factor forward
	if pipeline=='radiance':
		results = np.flip(np.transpose(radiance.frame.mean), axis=-2)/pixel_area
	else:
		results = np.flip(np.transpose(power.frame.mean), axis=-2)/pixel_area	# W/m^2/str /(W/m^3) of emitter

	plt.figure()
	cmap = plt.cm.rainbow
	cmap.set_under(color='white')
	vmin = 0.0000000000000000000000001
	plt.imshow(results, origin='lower', cmap=cmap, vmin=vmin)
	plt.title('Voxel n.'+str(index))
	plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off ' + str(vmin) + 'W/m^2')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	if (index%20)==0:	# done 04/02/2018 to reduce the number of images store for high resolution grids
		plt.savefig(path_sensitivity+'/sensitivity_voxel '+str(index)+'camera.eps')
	plt.close()

	sample = copy.deepcopy(i_all)
	sample[index] = 100000
	plt.figure(2)
	core_voxel_grid.plot(voxel_values=sample,title='Location of voxel '+str(index))
	plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	plt.axis('equal')
	if (index%20)==0:	# done 04/02/2018 to reduce the number of images store for high resolution grids
		plt.savefig(path_sensitivity + '/sensitivity_voxel ' + str(index) + 'location.eps')
	plt.close()
	plt.close()

	sens_row = results.flatten()
	np.save(path,sens_row)
	print('saved '+path)
	# return sensitivity_internal

#
#
# nslots = int(os.environ['NSLOTS'])
# print('nslots ',nslots)



# import concurrent.futures as cf
# with cf.ProcessPoolExecutor() as executor:
# 	executor.map(sensitivity,i)

# # Process to be done afterwards to put all together
for index in i_all:
	sensitivities[index, :] = np.load(path_sensitivity+'/sensitivity_voxel'+str(int(index))+'.npy')
np.save(path_sensitivity+'/sensitivity',sensitivities.T)

import csv
header = ['# Sensitivity matrix generated with:','# ']
to_write=[['foil horizontal pixels ',str(pixel_h)],['foil vertical pixels ',pixel_v],['type of volume grid ',grid_type],['number of voxels ',num_voxels],['pipeline type ',pipeline],['detector.pixel_samples',detector.pixel_samples]]
# to_write='1, 1, '+str(foil_fake_corner1)[8:-1]+', '+str(foil_fake_corner2)[8:-1]+', '+str(foil_fake_corner3)[8:-1]+', '+str(foil_fake_corner4)[8:-1]

with open(path_sensitivity+'/sensitivity_matrix_info.csv', mode='w') as f:
	writer = csv.writer(f)
	writer.writerow(header[0].split(','))
	writer.writerow([header[1]])
	for row in to_write:
		writer.writerow(row)
f.close()






# Piece for later analysis
#
# sensitivities=np.load(path_sensitivity+'/sensitivity.npy')
#
#
# test_pixel = 12
#
# plt.figure()
# cmap = plt.cm.rainbow
# cmap.set_under(color='white')
# vmin=0.0000000000000000000000001
# plt.imshow(coleval.split_fixed_length(sensitivities[test_pixel,:],pixel_h),origin='lower', cmap=cmap, vmin=vmin)
# plt.title('Power density on the foil generated via CHERAB simulation')
# plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off '+str(vmin)+'W/m^2')
# plt.xlabel('Horizontal axis [pixles]')
# plt.ylabel('Vertical axis [pixles]')
#
#
import copy
sample = copy.deepcopy(i_all); sample[test_pixel]=10000
core_voxel_grid.plot(voxel_values=sample)
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.axis('equal')
plt.show()
#
#
#
#
# test_pixel = 0
#
#
# core_voxel_grid.plot(voxel_values=sensitivities[:,test_pixel],title='Power on foil pixel n.'+str(test_pixel)+' for all voxels',colorbar=['rainbow','Sensitivity [m]','log'])
# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
# plt.axis('equal')
#
#
#
# plt.figure()
# X = np.arange(0, pixel_v)
# Y = np.arange(0, pixel_h)
# foil_position = np.zeros((len(X)*len(Y)))
# grid_x2 = np.zeros((len(X), len(Y)))
# grid_y2 = np.zeros((len(X), len(Y)))
# for ix, x in enumerate(X):
# 	for jy, y in enumerate(Y):
# 		# rad_test[ix, jy] = radiation_function(x, 0, y)
# 		grid_x2[ix, jy] = x
# 		grid_y2[ix, jy] = y
# foil_position[test_pixel] = 1
# plt.imshow(coleval.split_fixed_length(foil_position,len(Y)),origin='lower')
# plt.title('Position of the pixel on the foil n.'+str(test_pixel)+'\n(from the camera point of view)')
# plt.xlabel('Horizontal axis [pixles]')
# plt.ylabel('Vertical axis [pixles]')
# plt.show()
