# Created 10/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())


# just to import _MASTU_CORE_GRID_POLYGON
calculate_tangency_angle_for_poloidal_section=coleval.calculate_tangency_angle_for_poloidal_section
exec(open("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval/collect_and_eval/MASTU_structure.py").read())




# CHERAB section




# os.chdir("/home/ffederic/work/cherab/cherab_mastu/diagnostics/bolometry/irvb/")
from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
from raysect.primitive import import_stl, Sphere, Mesh, Cylinder
from raysect.optical import World, ConstantSF
from raysect.optical.material import NullMaterial
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.material.emitter import UniformVolumeEmitter,InhomogeneousVolumeEmitter
from raysect.optical.observer import TargettedCCDArray, PowerPipeline2D
# from raysect.core.math.interpolators import Discrete2DMesh
# this should be the same, just different because of change in CHERAB version
from raysect.core.math.function.float.function2d.interpolate.discrete2dmesh import Discrete2DMesh


# os.chdir("/home/ffederic/work/cherab/cherab_mastu/")
from cherab.mastu.machine import MASTU_FULL_MESH


# os.chdir("/home/ffederic/work/cherab/cherab_core/")
from cherab.core.math.mappers import AxisymmetricMapper


world = World()


for cad_file in MASTU_FULL_MESH:
	directory, filename = os.path.split(cad_file[0])
	name, ext = filename.split('.')
	print("importing {} ...".format(filename))
	Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)

os.chdir("/home/ffederic/work/analysis_scripts/irvb/")
irvb_cad = import_stl("IRVB_camera_no_backplate_4mm.stl", parent=world, material=AbsorbingSurface(), name="IRVB")



# load emissivity

# name='IRVB-MASTU_shot-45401.ptw'
# i_day,day = 0,'2021-10-21'
name='IRVB-MASTU_shot-45473.ptw'

path = '/home/ffederic/work/irvb/MAST-U/'
shot_list = get_data(path+'shot_list2.ods')
temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
for i in range(1,len(shot_list['Sheet1'])):
	if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
		date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
		break
i_day,day = 0,str(date.date())
laser_to_analyse=path+day+'/'+name

full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
full_saved_file_dict_FAST.allow_pickle=True
full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
try:
	full_saved_file_dict_FAST['multi_instrument'] = full_saved_file_dict_FAST['multi_instrument'].all()
except:
	full_saved_file_dict_FAST['multi_instrument'] = dict([])

try:
	temporary = gna___
	try:
		full_saved_file_dict_FAST['second_pass'] = full_saved_file_dict_FAST['second_pass'].all()
		inverted_dict = full_saved_file_dict_FAST['second_pass']['inverted_dict']
		foil_power = full_saved_file_dict_FAST['second_pass'].all()['FAST_powernoback']
	except:
		full_saved_file_dict_FAST['second_pass'] = full_saved_file_dict_FAST['second_pass']
		inverted_dict = full_saved_file_dict_FAST['second_pass']['inverted_dict']
		foil_power = full_saved_file_dict_FAST['second_pass']['FAST_powernoback']
except:
	try:
		full_saved_file_dict_FAST['first_pass'] = full_saved_file_dict_FAST['first_pass'].all()
		inverted_dict = full_saved_file_dict_FAST['first_pass']['inverted_dict']
		foil_power = full_saved_file_dict_FAST['first_pass'].all()['FAST_powernoback']
	except:
		full_saved_file_dict_FAST['first_pass'] = full_saved_file_dict_FAST['first_pass']
		inverted_dict = full_saved_file_dict_FAST['first_pass']['inverted_dict']
		foil_power = full_saved_file_dict_FAST['first_pass']['FAST_powernoback']
grid_resolution = 2	# cm
time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
inverted_data_sigma = inverted_dict[str(grid_resolution)]['inverted_data_sigma']
binning_type = inverted_dict[str(grid_resolution)]['binning_type']
# foil_power = inverted_dict[str(grid_resolution)]['foil_power']
foil_power_std = inverted_dict[str(grid_resolution)]['foil_power_std']
x_optimal_ext = inverted_dict[str(grid_resolution)]['x_optimal_ext']
grid_data_masked_crop = inverted_dict[str(grid_resolution)]['grid_data_masked_crop']
try:
	temperature_minus_background_crop_dt = full_saved_file_dict_FAST['first_pass'].all()['temperature_minus_background_crop_dt']	# this exists only for the first pass
except:
	temperature_minus_background_crop_dt = full_saved_file_dict_FAST['first_pass']['temperature_minus_background_crop_dt']	# this exists only for the first pass
stand_off_length = 0.045	# m
# Rf=1.54967	# m	radius of the centre of the foil
Rf=1.48967 + 0.01 + 0.003 + 0.002 + stand_off_length	# m	radius of the centre of the foil
plane_equation = np.array([1,-1,0,2**0.5 * Rf])	# plane of the foil
centre_of_foil = np.array([-Rf/(2**0.5), Rf/(2**0.5), -0.7])	# x,y,z
foil_size = [0.07,0.09]
structure_alpha=0.5

i_t = 18
start_from_pantom_from_emissivity = True
use_phantom_from_SOLPS = False
SART_inversion = False
override_phantom = False
print('start_from_pantom_from_emissivity')
print(start_from_pantom_from_emissivity)
print('use_phantom_from_SOLPS')
print(use_phantom_from_SOLPS)
print('override_phantom')
print(override_phantom)
print('SART_inversion')
print(SART_inversion)

EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
i_time = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
pinhole_offset = np.array([-0.0198,-0.0198])	# toroidal direction parallel to the place surface, z
# pinhole_offset_extra = np.array([+0.012/(2**0.5),-0.012/(2**0.5)])
pinhole_offset_extra = np.array([0,0])
stand_off_length = 0.045	# m
# Rf=1.54967	# m	radius of the centre of the foil
Rf=1.48967 + 0.01 + 0.003 + 0.002 + stand_off_length	# m	radius of the centre of the foil
plane_equation = np.array([1,-1,0,2**0.5 * Rf])	# plane of the foil
centre_of_foil = np.array([-Rf/(2**0.5), Rf/(2**0.5), -0.7])	# x,y,z
all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,centre_of_foil=centre_of_foil)
all_time_x_point_location = coleval.return_all_time_x_point_location(efit_reconstruction,plane_equation=plane_equation,centre_of_foil=centre_of_foil)
all_time_strike_points_location,all_time_strike_points_location_rot = coleval.return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,centre_of_foil=centre_of_foil)
fueling_point_location_on_foil = coleval.return_fueling_point_location_on_foil(plane_equation=plane_equation,centre_of_foil=centre_of_foil)
structure_point_location_on_foil = coleval.return_structure_point_location_on_foil(plane_equation=plane_equation,centre_of_foil=centre_of_foil)
all_time_mag_axis_location = coleval.return_all_time_mag_axis_location(efit_reconstruction,plane_equation=plane_equation,centre_of_foil=centre_of_foil)
core_poloidal_location_on_foil = coleval.return_core_poloidal_location_on_foil(plane_equation=plane_equation,centre_of_foil=centre_of_foil)

# Load sensitivity matrix


grid_resolution = 2
foil_resolution = '187'

foil_res = '_foil_pixel_h_' + str(foil_resolution)

grid_type = 'core_res_' + str(grid_resolution) + 'cm'
path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res + '_power'
try:
	sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity.npz')).todense())
except:
	sensitivities = np.load(path_sensitivity + '/sensitivity.npy')

filenames = coleval.all_file_names(path_sensitivity, '.csv')[0]
with open(os.path.join(path_sensitivity, filenames)) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if row[0] == 'foil vertical pixels ':
			pixel_v = int(row[1])
		if row[0] == 'foil horizontal pixels ':
			pixel_h = int(row[1])
		if row[0] == 'pipeline type ':
			pipeline = row[1]
		if row[0] == 'type of volume grid ':
			grid_type = row[1]
	# print(row)

directory = '/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction'
grid_file = os.path.join(directory,'{}_rectilinear_grid.pickle'.format(grid_type))
with open(grid_file, 'rb') as f:
	grid_data_all = pickle.load(f)
grid_laplacian = grid_data_all['laplacian']
grid_mask = grid_data_all['mask']
grid_data = grid_data_all['voxels']
grid_index_2D_to_1D_map = grid_data_all['index_2D_to_1D_map']
grid_index_1D_to_2D_map = grid_data_all['index_1D_to_2D_map']

sensitivities_reshaped = sensitivities.reshape((pixel_v,pixel_h,len(grid_laplacian)))
sensitivities_reshaped = np.transpose(sensitivities_reshaped , (1,0,2))

if grid_resolution==8:
	# temp=1e-3
	temp=1e-7
	temp2=0
elif grid_resolution==2:
	temp=0#1e-4
	temp2=0#1e-4
elif grid_resolution==4:
	temp=0
	temp2=0
# sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = coleval.reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,sum_treshold=temp2,std_treshold = temp,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False)
sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = coleval.reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,sum_treshold=temp2,std_treshold = temp,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False,restrict_polygon=_MASTU_CORE_GRID_POLYGON)

shrink_factor_t=7
shrink_factor_x=3
# this step is to adapt the matrix to the size of the foil I measure, that can be slightly different
binning_type = 'bin' + str(1) + 'x' + str(1) + 'x' + str(1)
shape = list(temperature_minus_background_crop_dt.shape[1:])	# +2 spatially because I remove +/- 1 pixel when I calculate the laplacian of the temperature
if shape!=list(sensitivities_reshaped_masked.shape[:-1]):
	shape.extend([len(grid_laplacian_masked)])
	def mapping(output_coords):
		return(output_coords[0]/shape[0]*pixel_h,output_coords[1]/shape[1]*pixel_v,output_coords[2])
	sensitivities_reshaped_masked2 = geometric_transform(sensitivities_reshaped_masked,mapping,output_shape=shape)
else:
	sensitivities_reshaped_masked2 = cp.deepcopy(sensitivities_reshaped_masked)

sensitivities_binned = coleval.proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

# additional cropping of the foil to exlude regions without plasma LOS, the frame of the foil and gas puff
# ROI = np.array([[0.2,0.85],[0.1,0.9]])
# ROI = np.array([[0.05,0.95],[0.05,0.95]])
# ROI = np.array([[0.2,0.95],[0.1,1]])
ROI1 = np.array([[0.03,0.82],[0.03,0.90]])	# horizontal, vertical
ROI2 = np.array([[0.03,0.68],[0.03,0.91]])
ROI_beams = np.array([[0.,0.32],[0.42,1]])
# # suggestion from Jack: keep the view as poloidal as I can, remove the central colon bit
# ROI1 = np.array([[0.03,0.65],[0.03,0.90]])	# horizontal, vertical
# ROI2 = np.array([[0.03,0.65],[0.03,0.91]])
ROI_beams = np.array([[0.,0.32],[0.42,1]])
sensitivities_binned_crop,selected_ROI,ROI1,ROI2,ROI_beams = coleval.cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_binned,ROI1,ROI2,ROI_beams,laser_to_analyse,additional_output=True)

additional_polygons_dict = dict([])
additional_polygons_dict['time'] = np.array([0])	# in this case I plot the same polygon for the whole movie
additional_polygons_dict['0'] = np.array([[[ROI1[0,0],ROI1[0,1],ROI1[0,1],ROI1[0,0],ROI1[0,0]],[ROI1[1,0],ROI1[1,0],ROI1[1,1],ROI1[1,1],ROI1[1,0]]]])
additional_polygons_dict['1'] = np.array([[[ROI2[0,0],ROI2[0,1],ROI2[0,1],ROI2[0,0],ROI2[0,0]],[ROI2[1,0],ROI2[1,0],ROI2[1,1],ROI2[1,1],ROI2[1,0]]]])
additional_polygons_dict['2'] = np.array([[[ROI_beams[0,0],ROI_beams[0,1],ROI_beams[0,1],ROI_beams[0,0],ROI_beams[0,0]],[ROI_beams[1,0],ROI_beams[1,0],ROI_beams[1,1],ROI_beams[1,1],ROI_beams[1,0]]]])
additional_polygons_dict['number_of_polygons'] = 3
additional_polygons_dict['marker'] = ['--k','--k','--k']

if grid_resolution==8:
	# temp=1e-3
	temp=1e-7
	temp2=0
elif grid_resolution==2:
	temp=0#1e-4
	temp2=0#5e-5
	# temp2=np.sum(sensitivities_binned_crop,axis=(0,1)).max()*1e-4
elif grid_resolution==4:
	temp=0
	temp2=0
sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = coleval.reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked,sum_treshold=temp2,std_treshold = temp,restrict_polygon=_MASTU_CORE_GRID_POLYGON,chop_corner_close_to_baffle=False)
# sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked,sum_treshold=temp2,std_treshold = temp,chop_corner_close_to_baffle=False)

# I add another step of filtering purely fo eliminate cells too isolated from the bulk and can let the laplacian grow
select = np.max(grid_laplacian_masked_crop,axis=0)<2.9
grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=select)
select = np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9
grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=select)
select = np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9
grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=select)
select = np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9
sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = coleval.reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,sum_treshold=0,std_treshold = 0,restrict_polygon=[],chop_corner_close_to_baffle=False,cells_to_exclude=select)
selected_edge_cells_for_laplacian = np.sum(sensitivities_binned_crop,axis=(0,1))>np.sum(sensitivities_binned_crop,axis=(0,1)).max()*0.2
grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=selected_edge_cells_for_laplacian)
select = np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9
grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=select)
select = np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9
grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=select)
select = np.logical_and(np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9,np.logical_not(selected_edge_cells_for_laplacian))
sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = coleval.reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,sum_treshold=0,std_treshold = 0,restrict_polygon=[],chop_corner_close_to_baffle=False,cells_to_exclude=select)
selected_edge_cells_for_laplacian = selected_edge_cells_for_laplacian[np.logical_not(select)]

selected_super_x_cells = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>0.85,np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.65)
# select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-3)	# this does not work if you change the ROI
select_foil_region_with_plasma = coleval.select_region_with_plasma(sensitivities_binned_crop,selected_ROI)
selected_ROI_no_plasma = np.logical_and(selected_ROI,np.logical_not(select_foil_region_with_plasma))
select_foil_region_with_plasma = select_foil_region_with_plasma.flatten()

x1 = [1.55,0.25]	# r,z
x2 = [1.1,-0.15]
interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
select = np.mean(grid_data_masked_crop,axis=1)[:,1]>interp(np.mean(grid_data_masked_crop,axis=1)[:,0])
selected_central_border_cells = np.logical_and(select,np.logical_and(np.max(grid_Z_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,1]>-0.5))
selected_central_border_cells = np.dot(grid_laplacian_masked_crop,selected_central_border_cells*np.random.random(selected_central_border_cells.shape))!=0

selected_central_column_border_cells = np.logical_and(np.logical_and(np.max(grid_R_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)
selected_central_column_border_cells = np.logical_and(np.logical_and(np.dot(grid_laplacian_masked_crop,selected_central_column_border_cells*np.random.random(selected_central_column_border_cells.shape))!=0,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)

# selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1,np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6))
# selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6))
selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1,np.mean(grid_data_masked_crop,axis=1)[:,1]<2))
selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<2))

if False:
	selected_edge_cells_for_laplacian = np.logical_and(np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6),np.dot(grid_laplacian_masked_crop,selected_edge_cells*np.random.random(selected_edge_cells.shape))!=0)
	if grid_resolution<8:
		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
	if grid_resolution<4:
		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
		# selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)

	# def temp_func():	# I package it only so it doesn't mess up other variables
	# 	from shapely.geometry import Point
	# 	from shapely.geometry.polygon import Polygon
	# 	select = np.zeros_like(selected_edge_cells_for_laplacian).astype(bool)
	# 	polygon = Polygon(FULL_MASTU_CORE_GRID_POLYGON)
	# 	for i_e in range(len(grid_data_masked_crop)):
	# 		if np.sum([polygon.contains(Point((grid_data_masked_crop[i_e][i__e,0],grid_data_masked_crop[i_e][i__e,1]))) for i__e in range(4)])==0:
	# 			select[i_e] = True
	# 	selected_cells_to_exclude = np.logical_or(selected_edge_cells_for_laplacian,select)
	# 	return selected_cells_to_exclude
	# selected_cells_to_exclude = temp_func()
	selected_cells_to_exclude = np.zeros_like(selected_edge_cells_for_laplacian)

else:	# strange thing. I want less regularisation where the sensitivity is much higher than the rest of the image
	# selected_edge_cells_for_laplacian = np.sum(sensitivities_binned_crop,axis=(0,1))>np.sum(sensitivities_binned_crop,axis=(0,1)).max()*0.2
	grid_laplacian_masked_crop = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=selected_edge_cells_for_laplacian)
	grid_Z_derivate_masked_crop = coleval.build_Z_derivate(grid_data_masked_crop,cells_to_exclude=selected_edge_cells_for_laplacian)
	grid_R_derivate_masked_crop = coleval.build_R_derivate(grid_data_masked_crop,cells_to_exclude=selected_edge_cells_for_laplacian)

	def temp_func():	# I package it only so it doesn't mess up other variables
		from shapely.geometry import Point
		from shapely.geometry.polygon import Polygon
		select = np.zeros_like(selected_edge_cells_for_laplacian).astype(bool)
		polygon = Polygon(FULL_MASTU_CORE_GRID_POLYGON)
		for i_e in range(len(grid_data_masked_crop)):
			if np.sum([polygon.contains(Point((grid_data_masked_crop[i_e][i__e,0],grid_data_masked_crop[i_e][i__e,1]))) for i__e in range(4)])==0:
				select[i_e] = True
		selected_cells_to_exclude = np.logical_or(selected_edge_cells_for_laplacian,select)
		return selected_cells_to_exclude
	selected_cells_to_exclude = temp_func()

sensitivities_binned_crop_shape = sensitivities_binned_crop.shape
sensitivities_binned_crop = sensitivities_binned_crop.reshape((sensitivities_binned_crop.shape[0]*sensitivities_binned_crop.shape[1],sensitivities_binned_crop.shape[2]))

if shrink_factor_x > 1:
	foil_resolution = str(shrink_factor_x) + 'x' + str(shrink_factor_x)
else:
	foil_resolution = str(shape[0])

foil_res = '_foil_pixel_h_' + str(foil_resolution)
path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
path_sensitivity_original = cp.deepcopy(path_sensitivity)

binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
print('starting '+binning_type)



# find power on the foil
foil_position_dict = dict([('angle',0.7),('foilcenter',[157,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',240)])	# modified 2021/09/21 to match sensitivity matrix
dx=foil_position_dict['foilhorizw']/foil_position_dict['foilhorizwpixel']
horizontal_coord = np.arange(np.shape(foil_power[0])[1])
vertical_coord = np.arange(np.shape(foil_power[0])[0])
horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
horizontal_coord = (horizontal_coord+1+0.5)*dx	# +1 because in the process of calculating the power I eliminate the first and last pixel in spatial coordinates, +0.5 do be the centre of the pixel
vertical_coord = (vertical_coord+1+0.5)*dx
horizontal_coord -= foil_position_dict['foilhorizw']*0.5+0.0198
vertical_coord -= foil_position_dict['foilvertw']*0.5-0.0198
distance_from_vertical = (horizontal_coord**2+vertical_coord**2)**0.5
pinhole_to_foil_vertical = 0.008 + 0.003 + 0.002 + 0.045	# pinhole holder, washer, foil holder, stand_off
pinhole_to_pixel_distance = (pinhole_to_foil_vertical**2 + distance_from_vertical**2)**0.5

etendue = np.ones_like(foil_power[0]) * (np.pi*(0.002**2)) / (pinhole_to_pixel_distance**2)	# I should include also the area of the pixel, but that is already in the w/m2 power
etendue *= (pinhole_to_foil_vertical/pinhole_to_pixel_distance)**2	 # cos(a)*cos(b). for pixels not directly under the pinhole both pinhole and pixel are tilted respect to the vertical, with same angle.

homogeneous_scaling=1e-4

if start_from_pantom_from_emissivity:
	if False:
		# taken from https://git.ccfe.ac.uk/MAST-U_Scheduler/abm/-/blob/master/abm/__main__.py
		source = '45401'
		import pyuda
		_client = pyuda.Client()
		geom_data=_client.geometry('/bolo', source)
		foils = [foil for foil in geom_data.data['core/foils/data']]
		channel_numbers = [int(foil.id[foil.id.find('CH')+2:]) for foil in foils]  # type: ignore
		detector_centres = [(foil["centre_point"].r, foil["centre_point"].z) for foil in foils]
		slit_centres = []
		for foil in foils:
			slit = geom_data.get_slit_by_id(foil.slit_id)
			slit_centres.append((slit["centre_point"].r, slit["centre_point"].z))
		axis = [0.9, 0.0]
		slit_axis_vectors = np.subtract(np.asarray(axis)[None, :], slit_centres)
		los_vectors = np.subtract(slit_centres, detector_centres)
		# Get the signed angle between the vectors
		los_axis_angles = np.arctan2(
			slit_axis_vectors[:, 0] * los_vectors[:, 1] - slit_axis_vectors[:, 1] * los_vectors[:, 0],
			slit_axis_vectors[:, 0] * los_vectors[:, 0] + slit_axis_vectors[:, 1] * los_vectors[:, 1],
		)
		impact_parameters = np.linalg.norm(slit_axis_vectors, axis=-1) * np.sin(los_axis_angles)
		# from mastu_exhaust_analysis.pyBolometer import Bolometer
		# abm = _client.get('/abm', source)
		brightness = _client.get('/abm/core/brightness', source).data.T
		good = _client.get('/abm/core/good', source).data
		select = np.logical_and(np.abs(los_vectors[:,1])>1e-4,good)
		select_13_14 = np.logical_and(select,np.logical_or(np.array(channel_numbers)==13,np.array(channel_numbers)==14))
		time = _client.get('/abm/core/time', source).data
		# temp = brightness[:,np.abs(time-1).argmin()]
		temp = np.array([y for _, y in sorted(zip(impact_parameters, np.arange(len(impact_parameters))))])
		temp = brightness[temp]
		select = np.array([y for _, y in sorted(zip(impact_parameters, select))])
		select_13_14 = np.array([y for _, y in sorted(zip(impact_parameters, select_13_14))])
		prad = 2 * np.pi * axis[0] * np.trapz(temp[select],x=np.sort(impact_parameters)[select],axis=0)
		temp_13_14 = cp.deepcopy(temp)
		temp_13_14[np.logical_not(select_13_14)] = 0
		prad_13_14 = 2 * np.pi * axis[0] * np.trapz(temp_13_14[select],x=np.sort(impact_parameters)[select],axis=0)
		plt.figure()
		plt.plot(time,prad*1e-3,'--',label='whole core')
		plt.plot(time,prad_13_14*1e-3,label='only CH13/14')
		plt.legend(loc='best')
		plt.grid()
		plt.ylabel('brightness [kW]')
		plt.xlabel('time [s]')
		plt.pause(0.001)



		EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
		efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+'45401'+'.nc',pulse_ID='45401')
		i_time = np.abs(efit_reconstruction.time-1.00).argmin()
		pinhole_offset = np.array([-0.0198,-0.0198])	# toroidal direction parallel to the place surface, z
		# pinhole_offset_extra = np.array([+0.012/(2**0.5),-0.012/(2**0.5)])
		pinhole_offset_extra = np.array([0,0])
		stand_off_length = 0.045	# m
		# Rf=1.54967	# m	radius of the centre of the foil
		Rf=1.48967 + 0.01 + 0.003 + 0.002 + stand_off_length	# m	radius of the centre of the foil
		plane_equation = np.array([1,-1,0,2**0.5 * Rf])	# plane of the foil
		centre_of_foil = np.array([-Rf/(2**0.5), Rf/(2**0.5), -0.7])	# x,y,z
		all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
		all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,centre_of_foil=centre_of_foil)
		all_time_x_point_location = coleval.return_all_time_x_point_location(efit_reconstruction,plane_equation=plane_equation,centre_of_foil=centre_of_foil)
		all_time_strike_points_location,all_time_strike_points_location_rot = coleval.return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,centre_of_foil=centre_of_foil)
		fueling_point_location_on_foil = coleval.return_fueling_point_location_on_foil(plane_equation=plane_equation,centre_of_foil=centre_of_foil)
		structure_point_location_on_foil = coleval.return_structure_point_location_on_foil(plane_equation=plane_equation,centre_of_foil=centre_of_foil)
		all_time_mag_axis_location = coleval.return_all_time_mag_axis_location(efit_reconstruction,plane_equation=plane_equation,centre_of_foil=centre_of_foil)
		core_poloidal_location_on_foil = coleval.return_core_poloidal_location_on_foil(plane_equation=plane_equation,centre_of_foil=centre_of_foil)
		cv0 = np.zeros((61,78)).T
		foil_size = [0.07,0.09]
		structure_alpha=0.5

		r = np.mean(grid_data_masked,axis=1)[:,0]
		z = np.mean(grid_data_masked,axis=1)[:,1]
		phantom_radious = 0.08
		phantom = ((r-efit_reconstruction.lower_xpoint_r[i_time]+0.06)**2 + (z-efit_reconstruction.lower_xpoint_z[i_time]-0.23)**2)**0.5 < phantom_radious
		phantom = phantom * (0.16)*1e6/np.sum(phantom * 2*np.pi*((grid_resolution*1e-2)**2))/(4*np.pi)
		foil_power_phantom = np.sum(sensitivities_binned * phantom,axis=-1)

		brightness = 4*np.pi*foil_power_phantom.reshape(np.shape(foil_power[0]))/etendue

		plt.figure()
		for i in range(len(fueling_point_location_on_foil)):
			plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
			plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
		for i in range(len(structure_point_location_on_foil)):
			plt.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
		plt.plot(all_time_x_point_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
		plt.plot(all_time_mag_axis_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--r')
		for __i in range(len(all_time_strike_points_location_rot[i_time])):
			plt.plot(all_time_strike_points_location_rot[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
		for __i in range(len(all_time_separatrix[i_time])):
			plt.plot(all_time_separatrix[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--b')
		for i in range(len(core_poloidal_location_on_foil)):
			plt.plot(np.array(core_poloidal_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(core_poloidal_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--r',alpha=1)

		plt.gca().set_aspect('equal')
		plt.imshow(np.flip(brightness,axis=(0)).T*1e-3,'rainbow',origin='bottom')
		plt.colorbar().set_label('Brightness [kW/m^2]')
		plt.pause(0.01)
	else:
		pass

	# solps 69568, 69592
	if use_phantom_from_SOLPS:
		type_of_test = 'SOLPS_pantom_test'
		from cherab.solps import load_solps_from_mdsplus
		# ref_number = mdsnos_cd15H[5]
		ref_number = mdsnos_cd2H[8]
		sim = load_solps_from_mdsplus('solps-mdsplus.aug.ipp.mpg.de:8001',ref_number)
		tot_pow = 0
		for i,vol in enumerate(sim.mesh.vol.flatten()):
			tot_pow+=vol*(sim.total_radiation.flatten())[i]

		x_point = [0.5188,-1.261]
		from cherab.tools.equilibrium import import_eqdsk
		eq = import_eqdsk('/home/jlovell/Bolometer/mastu-analysis/pyMASTU/Equilibrium/Data/MAST-U_conv.geqdsk')
		grid_x = sim.mesh.cr
		grid_y = sim.mesh.cz
		# plt.figure(figsize=(6,6))
		# total_radiation_density = sim.total_radiation
		# plt.pcolor(grid_x, grid_y, total_radiation_density, norm=LogNorm(vmin=total_radiation_density.max()*1e-3, vmax=total_radiation_density.max()),cmap='rainbow')
		# plt.title('SOLPS simulation '+str(ref_number)+'\n total radiated power %.4gkW' %(tot_pow/1000))
		# plt.colorbar().set_label('Emissivity [W/m^3]')
		# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		# plt.plot(x_point[0],x_point[1],'xr')
		# plt.plot(eq.lcfs_polygon[:,0],eq.lcfs_polygon[:,1],'--b')
		# plt.xlabel('R [m]')
		# plt.ylabel('Z [m]')
		# plt.axis('equal')
		# plt.ylim(top=0.,bottom=-2.1)
		# plt.xlim(left=0.23,right=1.45)
		# plt.pause(0.01)

		def import_radiator_from_solps_sim_on_triangular_mesh(sim):
			import numpy as np
			from cherab.core.math.mappers import AxisymmetricMapper
			from raysect.core.math.function.float.function2d.interpolate import Discrete2DMesh
			radiated_power=sim.total_radiation.flatten()
			rad_power = np.zeros(radiated_power.shape[0]*2)
			for i in range(radiated_power.shape[0]):
				rad_power[i*2] = radiated_power[i]
				rad_power[i*2 + 1] = radiated_power[i]
			return AxisymmetricMapper(Discrete2DMesh(sim.mesh.vertex_coordinates, sim.mesh.triangles, rad_power,limit=False))

		radiation_function = import_radiator_from_solps_sim_on_triangular_mesh(sim)	# W/m^3

		phantom_int = np.zeros((len(selected_edge_cells_for_laplacian)))
		for index in range(len(selected_edge_cells_for_laplacian)):
			phantom_int[index] = radiation_function(np.mean(grid_data_masked_crop,axis=1)[:,0][index], 0, np.mean(grid_data_masked_crop,axis=1)[:,1][index])	# W/m^3

		phantom_int /= 4*np.pi	# W/m^3/st

		# plot for the paper
		# plt.figure(figsize=(6,6))
		# # plt.pcolor(grid_x, grid_y, total_radiation_density, norm=LogNorm(vmin=1000, vmax=total_radiation_density.max()),cmap='rainbow')
		# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],s=10,c=phantom_int,cmap='rainbow',norm=LogNorm(),vmin=phantom_int.max()*1e-4)
		# plt.title('SOLPS simulation '+str(ref_number)+'\n total radiated power %.4gkW' %(np.sum(phantom_int*2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*((grid_resolution/100)**2))/1000))
		# plt.colorbar().set_label('Emissivity [W/m^3]')
		# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		# plt.plot(x_point[0],x_point[1],'xr')
		# plt.plot(eq.lcfs_polygon[:,0],eq.lcfs_polygon[:,1],'--b')
		# plt.xlabel('R [m]')
		# plt.ylabel('Z [m]')
		# plt.axis('equal')
		# plt.ylim(top=0.,bottom=-2.1)
		# plt.xlim(left=0.23,right=1.45)
		# plt.pause(0.01)

	else:
		type_of_test = 'pantom_test'
		phantom_int = x_optimal_ext[i_t,:-2]
		# phantom_int = phantom_int*np.logical_not(selected_cells_to_exclude)
		phantom_int[selected_cells_to_exclude] = 0
		phantom_int[phantom_int<0]=0	# W/m^3/st


	try:
		full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz',allow_pickle=True)
		full_saved_file_dict_FAST.allow_pickle=True
		full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
		full_saved_file_dict_FAST[type_of_test] = full_saved_file_dict_FAST[type_of_test].all()
		powernoback_full_orig = full_saved_file_dict_FAST[type_of_test]['powernoback_full_orig']
		sigma_powernoback_full = full_saved_file_dict_FAST[type_of_test]['sigma_powernoback_full']

		if override_phantom:
			Temporary = I_want_it_to_fail
	except:
		full_saved_file_dict_FAST[type_of_test] = dict([])

		foil_position_dict = dict([('angle',0.7),('foilcenter',[157,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',240)])	# modified 2021/09/21 to match sensitivity matrix

		diffusivity =1.347180363576353e-05
		thickness =2.6854735451543735e-06
		emissivity =0.9999999999
		Ptthermalconductivity=71.6 #[W/(m·K)]
		zeroC=273.15 #K / C
		sigmaSB=5.6704e-08 #[W/(m2 K4)]
		# 2022/10/14 Laser_data_analysis3_3.py, I separate well the digitizers now
		sigma_diffusivity = 0.13893985595434794
		sigma_emissivity_foil =0.08610352405403708
		sigma_thickness =0.1357454922343387
		sigma_rec_diffusivity = sigma_diffusivity

		dt = 1/383*2	# the *2 from the 2 digitizers that were averaged
		dx=foil_position_dict['foilhorizw']/foil_position_dict['foilhorizwpixel']
		proportional = 3.4e-14
		additive = 1600
		params_BB = np.ones((*(sensitivities_reshaped.shape[:-1]),2))
		params_BB[:,:,0] = proportional
		params_BB[:,:,1] = additive
		errparams_BB = np.zeros((*np.shape(params_BB),2))
		errparams_BB[:,:,:,0] = 6.435877532718499e-35
		photon_dict = coleval.calc_interpolators_BB(wavewlength_top=5.1,wavelength_bottom=1.5,inttime=2)

		temperature_evolution_with_noise,temperature_std_crop,peak_oscillation_freq,ref_temperature,counts_full_resolution_std,time_full_res_int = coleval.calculate_temperature_from_phantom2(phantom_int,grid_data_masked_crop,grid_data,sensitivities_reshaped,params_BB,errparams_BB,dt,dx,foil_position_dict,diffusivity,thickness,emissivity,Ptthermalconductivity,photon_dict,shrink_factor_t)

		time_binned,powernoback_full_extra,powernoback_full_std_extra = coleval.from_temperature_to_power_on_foil(temperature_evolution_with_noise,counts_full_resolution_std,peak_oscillation_freq,dt,shrink_factor_t,shrink_factor_x,params_BB,errparams_BB,emissivity,thickness,diffusivity,Ptthermalconductivity,sigma_thickness,sigma_rec_diffusivity,sigma_emissivity_foil,ref_temperature,temperature_std_crop,time_full_res_int,foil_position_dict,photon_dict)

		# i_t = temp_save['i_t']
		i_t_new = len(time_binned)-1
		print(time_binned)

		powernoback_full_orig = cp.deepcopy(powernoback_full_extra[i_t_new])
		sigma_powernoback_full = cp.deepcopy(powernoback_full_std_extra[i_t_new])

else:
	type_of_test = 'real_data_test'
	full_saved_file_dict_FAST[type_of_test] = dict([])

	powernoback_full_orig = foil_power[i_t]
	sigma_powernoback_full = foil_power_std[i_t]
	phantom_int = []
full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
full_saved_file_dict_FAST.allow_pickle=True
full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
try:
	full_saved_file_dict_FAST[type_of_test] = full_saved_file_dict_FAST[type_of_test].all()
except:
	full_saved_file_dict_FAST[type_of_test] = dict([])
full_saved_file_dict_FAST[type_of_test]['powernoback_full_orig'] = powernoback_full_orig
full_saved_file_dict_FAST[type_of_test]['sigma_powernoback_full'] = sigma_powernoback_full
full_saved_file_dict_FAST[type_of_test]['phantom_int'] = phantom_int #	W/m^3/st
np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)

powernoback_full_orig[np.logical_not(selected_ROI)] = 0
sigma_powernoback_full[np.logical_not(selected_ROI)] = np.nan
reference_sigma_powernoback = np.nanmedian(sigma_powernoback_full[selected_ROI])
sigma_powernoback_full[np.isnan(sigma_powernoback_full)] = 1e10
powernoback = powernoback_full_orig.flatten()
sigma_powernoback = sigma_powernoback_full.flatten()
sigma_powernoback_2 = sigma_powernoback**2

grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
grid_Z_derivate_masked_crop_scaled = grid_Z_derivate_masked_crop/((1e-2*grid_resolution)**1)
grid_R_derivate_masked_crop_scaled = grid_R_derivate_masked_crop/((1e-2*grid_resolution)**1)
number_cells_ROI = np.sum(selected_ROI)
number_cells_plasma = np.sum(select_foil_region_with_plasma)

sigma_emissivity = 1e6	# 2e3	# this is completely arbitrary
sigma_emissivity_2 = sigma_emissivity**2
r_int = np.mean(grid_data_masked_crop,axis=1)[:,0]
r_int_2 = r_int**2

selected_ROI_internal = selected_ROI.flatten()
not_selected_super_x_cells = np.logical_not(selected_super_x_cells)


if SART_inversion:
	type_of_inversion = 'SART'
	beta_range = 10**np.arange(0,-5-0.06,-0.06)

	SART_x_optimal_all = []
	SART_conv_all = []
	# from cherab.tools.inversions import invert_constrained_sart
	# I import my own version that has more controls
	os.chdir("/home/ffederic/work/cherab/added_by_Fabio")
	from sart import invert_constrained_sart

	from mwi_dp.inv.SART import SART_v7
	# in SART_v7 GM has to be pixels x voxels
	inv_settings = {}
	inv_settings['lambda_r'] = 0.95
	inv_settings['beta'] = 1E-9
	# inv_settings['iterations'] = 10000
	inv_settings['rel_change_stop'] = 1e-7
	inv_settings['rel_im_change_thresh'] = 1
	inv_settings['inv_module'] = 'SART'
	inv_settings['inv_method'] = 'SART_v7'
	inv_settings['grid_verts'] = None
	if False:
		gna = []
		for value in grid_data_masked_crop:
			gna.append([str(x) for x in value])
		gna = np.array(gna)
		gna_univoque = np.unique(gna)
		gna_index = np.arange(len(gna_univoque)).astype(int)
		gna2 = []
		for value in gna:
			gna2.append([ gna_index[gna_univoque==x][0] for x in value ])
		grid_cells = np.array(gna2)
		inv_settings['grid_cells'] = grid_cells
		inv_settings['laplacian'] = None
	else:
		inv_settings['grid_cells'] = None
		inv_settings['laplacian'] = -grid_laplacian_masked_crop_scaled/np.max(grid_laplacian_masked_crop_scaled,axis=0)
		inv_settings['laplacian'][grid_laplacian_masked_crop_scaled==0] = 0
	inv_settings['EFIT_shot_nb'] = None
	inv_settings['no_negatives'] = True
	inv_settings['guess'] = None
	from scipy.sparse import csr_matrix

	try:
		full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
		full_saved_file_dict_FAST.allow_pickle=True
		full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
		full_saved_file_dict_FAST[type_of_test] = full_saved_file_dict_FAST[type_of_test].all()
		previous_SART_x_optimal_all = full_saved_file_dict_FAST[type_of_test][type_of_inversion]['SART_x_optimal_all']
		guess = np.zeros((len(beta_range),np.shape(sensitivities_binned_crop)[1],1))
		guess[:np.shape(previous_SART_x_optimal_all)[0],:np.shape(previous_SART_x_optimal_all)[1],0] = previous_SART_x_optimal_all
	except:
		guess = []

	for i_beta,beta in enumerate(beta_range):
		inv_settings['beta'] = beta**2
		inv_settings['lambda_r'] = 0.5
		inv_settings['iterations'] = 100
		# inv_settings['guess'] = np.ones((len(grid_laplacian_masked_crop_scaled),1))
		if len(guess)==0:
			if i_beta==0:
				inv_settings['guess'] = None
			else:
				inv_settings['guess'] = eps
		else:
			inv_settings['guess'] = guess[i_beta]
		eps, inv_data = SART_v7(powernoback_full_orig,csr_matrix(sensitivities_binned_crop),np.ones_like(powernoback_full_orig).astype(bool),inv_settings=inv_settings, rotate_image=False)
		inv_settings['lambda_r'] = 0.95
		inv_settings['iterations'] = 10000
		inv_settings['guess'] = eps
		eps, inv_data = SART_v7(powernoback_full_orig,csr_matrix(sensitivities_binned_crop),np.ones_like(powernoback_full_orig).astype(bool),inv_settings=inv_settings, rotate_image=False,track_performance=True)
		SART_x_optimal_all.append(np.ravel(eps))
		SART_conv_all.append(inv_data)


	# for i_beta,beta in enumerate(beta_range):
	# 	# beta=0.00001
	# 	if i_beta==0:
	# 		SART_x_optimal, conv = invert_constrained_sart(sensitivities_binned_crop.astype(np.float64),grid_laplacian_masked_crop_scaled.astype(np.float64), powernoback.astype(np.float64),conv_tol=1.0E-5,beta_laplace=beta, max_iterations=250,relaxation=0.1,allow_negativity=True)
	# 		SART_x_optimal, conv = invert_constrained_sart(sensitivities_binned_crop.astype(np.float64),grid_laplacian_masked_crop_scaled.astype(np.float64), powernoback.astype(np.float64),initial_guess=SART_x_optimal,conv_tol=1.0E-10 ,beta_laplace=beta, max_iterations=250)
	# 	SART_x_optimal, conv = invert_constrained_sart(sensitivities_binned_crop.astype(np.float64),grid_laplacian_masked_crop_scaled.astype(np.float64), powernoback.astype(np.float64),initial_guess=SART_x_optimal,relaxation=0.1,conv_tol=1.0E-10 ,beta_laplace=beta, max_iterations=1000)
	# 	SART_x_optimal_all.append(SART_x_optimal)
	# 	SART_conv_all.append(conv)
	output_dict = dict([])
	output_dict['beta_range'] = beta_range
	output_dict['SART_x_optimal_all'] = SART_x_optimal_all
	output_dict['SART_conv_all'] = SART_conv_all

else:
	type_of_inversion = 'Bayes'

	# target_chi_square = sensitivities_binned_crop.shape[1]	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
	# target_chi_square_sigma = 200	# this should be tight, because for such a high number of degrees of freedom things should average very well
	# regolarisation_coeff_edge = 10
	# regolarisation_coeff_edge_multiplier = 100
	regolarisation_coeff_central_border_Z_derivate_multiplier = 0
	regolarisation_coeff_central_column_border_R_derivate_multiplier = 0
	regolarisation_coeff_edge_laplacian_multiplier = 0
	regolarisation_coeff_divertor_multiplier = 1
	regolarisation_coeff_non_negativity_multiplier = 200
	regolarisation_coeff_offsets_multiplier = 0#1e-10
	regolarisation_coeff_edge_laplacian = 0.01#0.001#0.02
	regolarisation_coeff_edge = 0

	guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2

	prob_and_gradient,calc_hessian = coleval.define_fitting_functions(homogeneous_scaling,regolarisation_coeff_divertor_multiplier,regolarisation_coeff_central_column_border_R_derivate_multiplier,regolarisation_coeff_central_border_Z_derivate_multiplier,regolarisation_coeff_edge_laplacian_multiplier,sensitivities_binned_crop,selected_ROI_internal,select_foil_region_with_plasma,grid_laplacian_masked_crop_scaled,not_selected_super_x_cells,selected_edge_cells_for_laplacian,selected_super_x_cells,selected_central_column_border_cells,selected_central_border_cells,regolarisation_coeff_non_negativity_multiplier,selected_edge_cells,r_int,regolarisation_coeff_edge,regolarisation_coeff_offsets_multiplier,number_cells_ROI,reference_sigma_powernoback,number_cells_plasma,r_int_2)

	regolarisation_coeff_range = 10**np.linspace(-0.5,-5,num=77)
	# regolarisation_coeff_range = [0.004618210073061655]

	print(np.shape(sensitivities_binned_crop))

	x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,voxels_centre,recompose_voxel_emissivity_excluded_all = coleval.loop_fit_over_regularisation(prob_and_gradient,regolarisation_coeff_range,guess,grid_data_masked_crop,powernoback,sigma_powernoback,sigma_emissivity,x_optimal_all_guess=guess,factr=1e9,excluded_cells = selected_edge_cells_for_laplacian)
	# if first_guess == []:

	print(np.shape(x_optimal_all))

	output_dict = dict([])
	output_dict['regolarisation_coeff_range'] = regolarisation_coeff_range
	output_dict['x_optimal_all'] = x_optimal_all

full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
full_saved_file_dict_FAST.allow_pickle=True
full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
full_saved_file_dict_FAST[type_of_test] = full_saved_file_dict_FAST[type_of_test].all()
full_saved_file_dict_FAST[type_of_test][type_of_inversion] = output_dict
np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
print('done ok')
exit()










######   examine output






full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz',allow_pickle=True)
full_saved_file_dict_FAST.allow_pickle=True
full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)


# type_of_test = 'pantom_test'
# type_of_test = 'SOLPS_pantom_test'
type_of_test = 'real_data_test'

powernoback_full_orig = full_saved_file_dict_FAST[type_of_test].all()['powernoback_full_orig']
sigma_powernoback_full = full_saved_file_dict_FAST[type_of_test].all()['sigma_powernoback_full']
if type_of_test in ['pantom_test','SOLPS_pantom_test']:
	forward_model_residuals = True
else:
	forward_model_residuals = False
try:
	phantom_int = full_saved_file_dict_FAST[type_of_test].all()['phantom_int']	# W/m^3/st
except:
	phantom_int = []

if False:	# examine SART solution
	type_of_inversion = 'SART'
	output_dict = full_saved_file_dict_FAST[type_of_test].all()[type_of_inversion]
	beta_range = output_dict['beta_range']
	SART_x_optimal_all = output_dict['SART_x_optimal_all']
	SART_conv_all = output_dict['SART_conv_all']
	if phantom_int == []:
		pass
		# phantom_int = x_optimal_ext[i_t,:-2]*np.logical_not(selected_cells_to_exclude)
		# phantom_int[phantom_int<0]=0	# W/m^3/st
	else:
		trash,recompose_voxel_emissivity_phantom = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.array(phantom_int.tolist()+[0]+[0]),np.mean(grid_data_masked_crop,axis=1))
	SART_x_optimal_all = np.array(SART_x_optimal_all)

	if phantom_int != []:
		plt.figure(figsize=(6,8))
		plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_phantom*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],vmin=np.nanmin(recompose_voxel_emissivity_phantom)*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
		# plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_phantom*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],norm=LogNorm(),vmin=recompose_voxel_emissivity_phantom[recompose_voxel_emissivity_phantom>0].min())
		if type_of_test == 'SOLPS_pantom_test':
			from cherab.solps import load_solps_from_mdsplus
			ref_number = mdsnos_cd2H[8]
			sim = load_solps_from_mdsplus('solps-mdsplus.aug.ipp.mpg.de:8001',ref_number)
			x_point = [0.5188,-1.261]
			from cherab.tools.equilibrium import import_eqdsk
			eq = import_eqdsk('/home/jlovell/Bolometer/mastu-analysis/pyMASTU/Equilibrium/Data/MAST-U_conv.geqdsk')
			grid_x = sim.mesh.cr
			grid_y = sim.mesh.cz
			plt.plot(x_point[0],x_point[1],'xr')
			plt.plot(eq.lcfs_polygon[:,0],eq.lcfs_polygon[:,1],'--b')
			plt.plot([0.38,0.87],[-1.35,-1.84],'xg')	# strike points
		else:
			for i in range(len(all_time_sep_r[i_time])):
				plt.plot(r_fine[all_time_sep_r[i_time][i]],z_fine[all_time_sep_z[i_time][i]],'--b')
			plt.plot(efit_reconstruction.lower_xpoint_r[i_time],efit_reconstruction.lower_xpoint_z[i_time],'xr')
			plt.plot(efit_reconstruction.strikepointR[i_time],efit_reconstruction.strikepointZ[i_time],'xr')
		plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		structure_radial_profile = coleval.return_structure_radial_profile()
		for i in range(len(structure_radial_profile)):
			plt.plot(structure_radial_profile[i][:,0],structure_radial_profile[i][:,1],'--k',alpha=0.5)
		plt.plot(1.4918014,-0.7198,'xg')	# pinhole
		plt.colorbar().set_label('Emissivity kW/m^3')
		plt.title('Emissivity phantom from 45473 500ms\n radiated power %.4gkW' %(4*np.pi*np.sum(phantom_int*2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*((grid_resolution/100)**2))/1000))
		plt.ylim(bottom=-2.1,top=0.20)
		plt.xlim(left=0.2,right=1.6)
		plt.grid()
		# plt.pause(0.01)
		plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_phantom_'+type_of_test+'.png', bbox_inches='tight')
		plt.close()
	else:
		pass

	plt.figure(figsize=(8,8))
	# plt.imshow(np.flip((4*np.pi*powernoback_full_orig/etendue).T,axis=1)*1e-3,'rainbow',origin='bottom')
	# plt.colorbar().set_label('Brightness kW/m^2')
	plt.imshow(np.flip((powernoback_full_orig).T,axis=1).astype(float),'rainbow',origin='bottom')
	plt.colorbar().set_label('Power density W/m^2')
	cv0 = np.zeros_like(powernoback_full_orig.T)
	for i in range(len(fueling_point_location_on_foil)):
		plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
		plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
	for i in range(len(structure_point_location_on_foil)):
		plt.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
	plt.plot(all_time_x_point_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
	plt.plot(all_time_mag_axis_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--r')
	for __i in range(len(all_time_strike_points_location_rot[i_time])):
		plt.plot(all_time_strike_points_location_rot[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
	for __i in range(len(all_time_separatrix[i_time])):
		plt.plot(all_time_separatrix[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--b')
	# for i in range(len(core_poloidal_location_on_foil)):
	# 	plt.plot(np.array(core_poloidal_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(core_poloidal_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--r',alpha=1)
	plt.title('Power density phantom '+type_of_test+'\n with added noise')
	# plt.pause(0.01)
	plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_foil_power_'+type_of_test+'.png', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8,8))
	# plt.imshow(np.flip((4*np.pi*powernoback_full_orig/etendue).T,axis=1)*1e-3,'rainbow',origin='bottom')
	# plt.colorbar().set_label('Brightness kW/m^2')
	plt.imshow(np.flip((sigma_powernoback_full).T,axis=1),'rainbow',origin='bottom')
	plt.colorbar().set_label('Power density std W/m^2')
	cv0 = np.zeros_like(powernoback_full_orig.T)
	for i in range(len(fueling_point_location_on_foil)):
		plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
		plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
	for i in range(len(structure_point_location_on_foil)):
		plt.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
	plt.plot(all_time_x_point_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
	plt.plot(all_time_mag_axis_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--r')
	for __i in range(len(all_time_strike_points_location_rot[i_time])):
		plt.plot(all_time_strike_points_location_rot[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
	for __i in range(len(all_time_separatrix[i_time])):
		plt.plot(all_time_separatrix[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--b')
	# for i in range(len(core_poloidal_location_on_foil)):
		# plt.plot(np.array(core_poloidal_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(core_poloidal_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--r',alpha=1)
	plt.title('Power density std phantom '+type_of_test+'\n with added noise')
	# plt.pause(0.01)
	plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_foil_power_std_'+type_of_test+'.png', bbox_inches='tight')
	plt.close()




	# plt.figure()
	# index=10
	# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=SART_x_optimal_all[index]*np.logical_not(selected_cells_to_exclude),s=50,marker='s',cmap='rainbow',vmin=SART_x_optimal_all[index][np.logical_not(selected_cells_to_exclude)].min(),vmax=SART_x_optimal_all[index][np.logical_not(selected_cells_to_exclude)].max())#,vmin=min(SART_x_optimal1.min(),SART_x_optimal.min()),vmax=max(SART_x_optimal1.max(),SART_x_optimal.max()))
	# # plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=phantom_int*np.logical_not(selected_cells_to_exclude),s=50,marker='s',cmap='rainbow')#,vmin=min(SART_x_optimal1.min(),SART_x_optimal.min()),vmax=max(SART_x_optimal1.max(),SART_x_optimal.max()))
	# # plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_cells_to_exclude,s=50,marker='s',cmap='rainbow')
	# # plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=eps.T[0],s=50,marker='s',cmap='rainbow',vmin=eps.T[0][np.logical_not(selected_cells_to_exclude)].min(),vmax=eps.T[0][np.logical_not(selected_cells_to_exclude)].max())#,vmin=min(SART_x_optimal1.min(),SART_x_optimal.min()),vmax=max(SART_x_optimal1.max(),SART_x_optimal.max()))
	# plt.gca().set_aspect('equal')
	# plt.colorbar()
	# plt.pause(0.01)


	# plt.figure()
	# for i in range(len(SART_conv_all)):
	# 	plt.plot(SART_conv_all[i]['image_error'],label=str(i))
	# plt.semilogy()
	# plt.legend()
	# plt.pause(0.01)


	# plt.figure()
	# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.max(grid_laplacian_masked_crop,axis=0),s=50,marker='s',cmap='rainbow')#,vmin=eps[np.logical_not(selected_cells_to_exclude)].min(),vmax=eps[np.logical_not(selected_cells_to_exclude),0].max())#,vmin=min(SART_x_optimal1.min(),SART_x_optimal.min()),vmax=max(SART_x_optimal1.max(),SART_x_optimal.max()))
	# plt.colorbar()
	# plt.pause(0.01)

	selected_ROI_internal = selected_ROI.flatten()
	sigma_emissivity = 1e6	# 2e3	# this is completely arbitrary
	sigma_emissivity_2 = sigma_emissivity**2
	extended_SART_x_optimal_all = np.array(SART_x_optimal_all)
	extended_SART_x_optimal_all = np.array(extended_SART_x_optimal_all.T.tolist() + np.zeros((len(extended_SART_x_optimal_all), 2)).T.tolist()).T
	score_x = np.nansum(((np.dot(sensitivities_binned_crop,np.array(extended_SART_x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(extended_SART_x_optimal_all)).T*np.array(extended_SART_x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(extended_SART_x_optimal_all)).T*np.array(extended_SART_x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback_full_orig.flatten()) ** 2) / (sigma_powernoback_full.flatten()**2),axis=1)
	score_y = np.nansum(((np.dot(grid_laplacian_masked_crop,np.array(extended_SART_x_optimal_all)[:,:-2].T).T) ** 2) / (sigma_emissivity**2),axis=1)
	score_x_orig = np.flip(score_x,axis=0)
	score_y_orig = np.flip(score_y,axis=0)
	regolarisation_coeff_range_orig = np.flip(beta_range**2,axis=0)
	regolarisation_coeff_range = np.flip(beta_range**2,axis=0)
	SART_x_optimal_all = np.flip(SART_x_optimal_all,axis=0)

	# plt.figure()
	# plt.plot(score_x_orig,score_y_orig)
	# plt.plot(score_x_orig,score_y_orig,'+')
	# plt.semilogx()
	# plt.semilogy()
	# plt.xlabel('||Gm-d||2')
	# plt.ylabel('||Laplacian(m)||2')
	# plt.pause(0.01)
	#
	# plt.figure()
	# plt.plot(beta_range,score_x,'--')
	# plt.plot(beta_range,score_x,'+')
	# plt.plot(beta_range,score_y,'-')
	# plt.plot(beta_range,score_y,'+')
	# plt.semilogx()
	# plt.semilogy()
	# plt.pause(0.01)

	regolarisation_coeff_upper_limit = 1
	regolarisation_coeff_lower_limit = 0
	curvature_fit_regularisation_interval = 0.25
	fraction_of_L_curve_for_fit = 0.08
	score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,x_optimal_SART,points_removed,regolarisation_coeff,regolarisation_coeff_range,curvature_range_left_all,curvature_range_right_all,peaks_all,best_index =coleval.find_optimal_regularisation_minimal(score_x_orig,score_y_orig,regolarisation_coeff_range,SART_x_optimal_all,regolarisation_coeff_upper_limit = regolarisation_coeff_upper_limit,regolarisation_coeff_lower_limit = regolarisation_coeff_lower_limit,curvature_fit_regularisation_interval=curvature_fit_regularisation_interval,fraction_of_L_curve_for_fit=fraction_of_L_curve_for_fit,forward_model_residuals=forward_model_residuals)
	trash,recompose_voxel_emissivity_SART = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.array((x_optimal_SART*np.logical_not(selected_edge_cells_for_laplacian)).tolist()+[0]+[0]),np.mean(grid_data_masked_crop,axis=1))
	trash,recompose_voxel_emissivity_SART_excluded = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.array((x_optimal_SART*selected_edge_cells_for_laplacian).tolist()+[0]+[0]),np.mean(grid_data_masked_crop,axis=1))

	plt.figure(figsize=(8,6))
	plt.plot(score_x_orig,score_y_orig)
	plt.plot(score_x_orig,score_y_orig,'+')
	plt.plot(np.exp(score_x[best_index]),np.exp(score_y[best_index]),'o',color='b')
	plt.plot(score_x_orig[np.abs(regolarisation_coeff_range_orig-0.1).argmin()],score_y_orig[np.abs(regolarisation_coeff_range_orig-0.1).argmin()],'or')
	plt.plot(score_x_orig[np.abs(regolarisation_coeff_range_orig-1e-5).argmin()],score_y_orig[np.abs(regolarisation_coeff_range_orig-1e-5).argmin()],'og')
	plt.semilogx()
	plt.semilogy()
	plt.xlabel("||Wm'-q||")
	plt.ylabel("||Lm'||")
	plt.title('Phantom '+type_of_test+'\n SART L-curve')
	plt.grid()
	# plt.pause(0.01)
	plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_L_curve_'+type_of_test+'-'+type_of_inversion+'.png', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8,4))
	plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature)
	plt.plot(regolarisation_coeff,Lcurve_curvature[best_index-curvature_range],'ob')
	plt.plot(regolarisation_coeff_range[np.abs(regolarisation_coeff_range-0.1).argmin()],Lcurve_curvature[np.abs(regolarisation_coeff_range-0.1).argmin()-curvature_range],'or')
	plt.plot(regolarisation_coeff_range[np.abs(regolarisation_coeff_range-1e-5).argmin()],Lcurve_curvature[np.abs(regolarisation_coeff_range-1e-5).argmin()-curvature_range],'og')
	# plt.axvline(x=regolarisation_coeff_upper_limit,color='r',linestyle='--')
	# plt.axvline(x=regolarisation_coeff_lower_limit,color='g',linestyle='--')
	plt.semilogx()
	plt.xlabel(r'$\alpha$')
	plt.ylabel("L-curve curvature")
	plt.grid()
	# plt.pause(0.01)
	plt.title('Phantom '+type_of_test+'\n SART L-curve')
	plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_L_curve_curvature_'+type_of_test+'-'+type_of_inversion+'.png', bbox_inches='tight')
	plt.close()

else:
	type_of_inversion = 'Bayes'
	output_dict = full_saved_file_dict_FAST[type_of_test].all()[type_of_inversion]
	regolarisation_coeff_range = output_dict['regolarisation_coeff_range']
	x_optimal_all = output_dict['x_optimal_all']
	x_optimal_all = np.array(x_optimal_all)
	# if phantom_int !=[]:
	# 	phantom_int = x_optimal_ext[i_t,:-2]*np.logical_not(selected_cells_to_exclude)
	# 	phantom_int[phantom_int<0]=0	# W/m^3/st
	# 	trash,recompose_voxel_emissivity_phantom = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.array(list(phantom_int)+[0]+[0]),np.mean(grid_data_masked_crop,axis=1))
	# else:
	# 	recompose_voxel_emissivity_phantom = np.zeros((2,2))

	regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
	x_optimal_all = np.flip(x_optimal_all,axis=0)

	selected_ROI_internal = selected_ROI.flatten()
	sigma_emissivity = 1e6	# 2e3	# this is completely arbitrary
	sigma_emissivity_2 = sigma_emissivity**2
	grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
	score_x_orig = np.sum(((np.dot(sensitivities_binned_crop,np.array(x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback_full_orig.flatten()) ** 2) / (sigma_powernoback_full.flatten()**2),axis=1)
	score_y_orig = np.sum(((np.dot(grid_laplacian_masked_crop_scaled,(np.logical_not(selected_edge_cells_for_laplacian)*np.array(x_optimal_all)[:,:-2]).T).T) ** 2) / (sigma_emissivity**2),axis=1)
	regolarisation_coeff_range_orig = cp.deepcopy(regolarisation_coeff_range)
	# plt.figure()
	# plt.plot(score_x_orig,score_y_orig)
	# plt.plot(score_x_orig,score_y_orig,'+')
	# plt.semilogx()
	# plt.semilogy()
	# plt.xlabel('||Gm-d||2')
	# plt.ylabel('||Laplacian(m)||2')
	# plt.pause(0.01)

	regolarisation_coeff_upper_limit = 0.1
	regolarisation_coeff_upper_limit = regolarisation_coeff_range[np.abs(regolarisation_coeff_range - regolarisation_coeff_upper_limit).argmin()]
	regolarisation_coeff_lower_limit = 2e-4
	regolarisation_coeff_lower_limit = regolarisation_coeff_range[np.abs(regolarisation_coeff_range - regolarisation_coeff_lower_limit).argmin()]
	curvature_fit_regularisation_interval = 0.05
	fraction_of_L_curve_for_fit = 0.08
	score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,x_optimal_Bayes,points_removed,regolarisation_coeff,regolarisation_coeff_range,curvature_range_left_all,curvature_range_right_all,peaks_all,best_index = coleval.find_optimal_regularisation_minimal(score_x_orig,score_y_orig,regolarisation_coeff_range,x_optimal_all,regolarisation_coeff_upper_limit = regolarisation_coeff_upper_limit,regolarisation_coeff_lower_limit = regolarisation_coeff_lower_limit,curvature_fit_regularisation_interval=curvature_fit_regularisation_interval,forward_model_residuals=forward_model_residuals,avoid_score_y_rise=False)
	trash,recompose_voxel_emissivity_Bayes = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.array((x_optimal_Bayes[:-2]*np.logical_not(selected_edge_cells_for_laplacian)).tolist()+[0]+[0]),np.mean(grid_data_masked_crop,axis=1))
	trash,recompose_voxel_emissivity_Bayes_excluded = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.array((x_optimal_Bayes[:-2]*selected_edge_cells_for_laplacian).tolist()+[x_optimal_Bayes[-2]]+[x_optimal_Bayes[-1]]),np.mean(grid_data_masked_crop,axis=1))

	foil_fitted_excluded = np.dot(sensitivities_binned_crop,np.array(x_optimal_Bayes[:-2]*selected_edge_cells_for_laplacian).T)  + selected_ROI_internal*(x_optimal_Bayes[-1])*homogeneous_scaling + select_foil_region_with_plasma*(x_optimal_Bayes[-2])*homogeneous_scaling
	foil_fitted_excluded = foil_fitted_excluded.reshape(np.shape(powernoback_full_orig))

	if False:	# only plotted if needed
		plt.figure(figsize=(8,8))
		# plt.imshow(np.flip((4*np.pi*powernoback_full_orig/etendue).T,axis=1)*1e-3,'rainbow',origin='bottom')
		# plt.colorbar().set_label('Brightness kW/m^2')
		plt.imshow(np.flip((foil_fitted_excluded).T,axis=1),'rainbow',origin='bottom')
		plt.colorbar().set_label('Power density excluded W/m^2')
		cv0 = np.zeros_like(powernoback_full_orig.T)
		for i in range(len(fueling_point_location_on_foil)):
			plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
			plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
		for i in range(len(structure_point_location_on_foil)):
			plt.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
		# plt.plot(all_time_x_point_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
		# plt.plot(all_time_mag_axis_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--r')
		# for __i in range(len(all_time_strike_points_location_rot[i_time])):
		# 	plt.plot(all_time_strike_points_location_rot[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
		# for __i in range(len(all_time_separatrix[i_time])):
		# 	plt.plot(all_time_separatrix[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--b')
		# for i in range(len(core_poloidal_location_on_foil)):
			# plt.plot(np.array(core_poloidal_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(core_poloidal_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--r',alpha=1)
		plt.title('Power density excluded phantom '+type_of_test+'\n Bayes L-curve')
		# plt.pause(0.01)
		plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_foil_power_excluded_'+type_of_test+'-'+type_of_inversion+'.png', bbox_inches='tight')
		plt.close()
	else:
		pass

	plt.figure(figsize=(8,6))
	plt.plot(score_x_orig,score_y_orig)
	plt.plot(score_x_orig,score_y_orig,'+')
	plt.plot(np.exp(score_x[best_index]),np.exp(score_y[best_index]),'o',color='b')
	plt.plot(score_x_orig[np.abs(regolarisation_coeff_range_orig-regolarisation_coeff_upper_limit).argmin()],score_y_orig[np.abs(regolarisation_coeff_range_orig-regolarisation_coeff_upper_limit).argmin()],'or')
	plt.plot(score_x_orig[np.abs(regolarisation_coeff_range_orig-regolarisation_coeff_lower_limit).argmin()],score_y_orig[np.abs(regolarisation_coeff_range_orig-regolarisation_coeff_lower_limit).argmin()],'og')
	plt.semilogx()
	plt.semilogy()
	plt.xlabel("||Wm'-q||")
	plt.ylabel("||Lm'||")
	plt.title('Phantom '+type_of_test+'\n Bayes L-curve')
	plt.grid()
	# plt.pause(0.01)
	plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_L_curve_'+type_of_test+'-'+type_of_inversion+'.png', bbox_inches='tight')
	plt.close()

	regolarisation_coeff_range_small = regolarisation_coeff_range[curvature_range:-curvature_range]

	plt.figure(figsize=(8,4))
	plt.plot(regolarisation_coeff_range_small,Lcurve_curvature)
	plt.plot(regolarisation_coeff,Lcurve_curvature[best_index-curvature_range],'ob')
	plt.axvline(x=regolarisation_coeff_upper_limit,color='r',linestyle='--')
	plt.axvline(x=regolarisation_coeff_lower_limit,color='g',linestyle='--')
	plt.plot(regolarisation_coeff_range_small[np.abs(regolarisation_coeff_range_small-regolarisation_coeff_upper_limit).argmin()],Lcurve_curvature[np.abs(regolarisation_coeff_range_small-regolarisation_coeff_upper_limit).argmin()],'or')
	plt.plot(regolarisation_coeff_range_small[np.abs(regolarisation_coeff_range_small-regolarisation_coeff_lower_limit).argmin()],Lcurve_curvature[np.abs(regolarisation_coeff_range_small-regolarisation_coeff_lower_limit).argmin()],'og')
	plt.semilogx()
	plt.xlabel(r'$\alpha$')
	plt.ylabel("L-curve curvature")
	plt.grid()
	plt.title('Phantom '+type_of_test+'\n Bayes L-curve')
	# plt.pause(0.01)
	plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_L_curve_curvature_'+type_of_test+'-'+type_of_inversion+'.png', bbox_inches='tight')
	plt.close()

	args = [powernoback_full_orig.flatten(),sigma_powernoback_full.flatten(),sigma_emissivity,regolarisation_coeff,(sigma_powernoback_full.flatten())**2,sigma_emissivity**2]
	hessian=calc_hessian(x_optimal_Bayes,*args)
	covariance = np.linalg.inv(hessian)
	correlation = ((covariance/(np.diag(covariance)**0.5)).T/(np.diag(covariance)**0.5)).T
	x_optimal_Bayes_sigma = np.diag(covariance)**0.5
	trash,recompose_voxel_Bayes_sigma = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.array((x_optimal_Bayes_sigma[:-2]*np.logical_not(selected_edge_cells_for_laplacian)).tolist()+[0]+[0]),np.mean(grid_data_masked_crop,axis=1))

if phantom_int == []:
	recompose_voxel_emissivity_phantom = np.mean([recompose_voxel_emissivity_SART,recompose_voxel_emissivity_Bayes],axis=0)
plt.figure(figsize=(6,8))
# plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_SART,(1,0)),axis=1),axis=1),axis=0)*1e-3,'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],vmin=np.nanmin(recompose_voxel_emissivity_phantom)*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
# plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_SART_excluded,(1,0)),axis=1),axis=1),axis=0)*1e-3,'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])#,vmin=np.nanmin(recompose_voxel_emissivity_phantom)*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_SART+recompose_voxel_emissivity_SART_excluded,(1,0)),axis=1),axis=1),axis=0)*1e-3,'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],norm=LogNorm(),vmin=recompose_voxel_emissivity_phantom[recompose_voxel_emissivity_phantom>0].min(),vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
structure_radial_profile = coleval.return_structure_radial_profile()
for i in range(len(structure_radial_profile)):
	plt.plot(structure_radial_profile[i][:,0],structure_radial_profile[i][:,1],'--k',alpha=0.5)
plt.plot(1.4918014,-0.7198,'xg')	# pinhole
if type_of_test == 'SOLPS_pantom_test':
	plt.plot(x_point[0],x_point[1],'xr')
	plt.plot(eq.lcfs_polygon[:,0],eq.lcfs_polygon[:,1],'--b')
	plt.plot([0.38,0.87],[-1.35,-1.84],'xg')	# strike points
else:
	for i in range(len(all_time_sep_r[i_time])):
		plt.plot(r_fine[all_time_sep_r[i_time][i]],z_fine[all_time_sep_z[i_time][i]],'--b')
	plt.plot(efit_reconstruction.lower_xpoint_r[i_time],efit_reconstruction.lower_xpoint_z[i_time],'xr')
	plt.plot(efit_reconstruction.strikepointR[i_time],efit_reconstruction.strikepointZ[i_time],'xr')
plt.colorbar().set_label('Emissivity kW/m^3')
plt.title('Emissivity phantom from 45473 500ms\n SART inversion')
plt.ylim(bottom=-2.1,top=0.20)
plt.xlim(left=0.2,right=1.6)
plt.grid()
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_emissivity_'+type_of_test+'-'+'SART'+'.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(6,8))
plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_Bayes,(1,0)),axis=1),axis=1),axis=0)*1e-3,'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],vmin=np.nanmin(recompose_voxel_emissivity_phantom)*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
# plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_Bayes_excluded,(1,0)),axis=1),axis=1),axis=0)*1e-3,'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])#,vmin=np.nanmin(recompose_voxel_emissivity_phantom)*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
# plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_Bayes,(1,0)),axis=1),axis=1),axis=0)*1e-3,'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],norm=LogNorm(),vmin=recompose_voxel_emissivity_phantom[recompose_voxel_emissivity_phantom>0].min(),vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
# plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_Bayes_sigma,(1,0)),axis=1),axis=1),axis=0)*1e-3,'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],norm=LogNorm(),vmin=recompose_voxel_emissivity_phantom[recompose_voxel_emissivity_phantom>0].min(),vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
# plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_Bayes/recompose_voxel_Bayes_sigma,(1,0)),axis=1),axis=1),axis=0)*1e-3,'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],norm=LogNorm(),vmin=1e-5)#,vmin=recompose_voxel_emissivity_phantom[recompose_voxel_emissivity_phantom>0].min(),vmax=np.nanmax(recompose_voxel_emissivity_phantom)*1e-3)
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
structure_radial_profile = coleval.return_structure_radial_profile()
for i in range(len(structure_radial_profile)):
	plt.plot(structure_radial_profile[i][:,0],structure_radial_profile[i][:,1],'--k',alpha=0.5)
plt.plot(1.4918014,-0.7198,'xg')	# pinhole
if type_of_test == 'SOLPS_pantom_test':
	plt.plot(x_point[0],x_point[1],'xr')
	plt.plot(eq.lcfs_polygon[:,0],eq.lcfs_polygon[:,1],'--b')
	plt.plot([0.38,0.87],[-1.35,-1.84],'xg')	# strike points
else:
	for i in range(len(all_time_sep_r[i_time])):
		plt.plot(r_fine[all_time_sep_r[i_time][i]],z_fine[all_time_sep_z[i_time][i]],'--b')
	plt.plot(efit_reconstruction.lower_xpoint_r[i_time],efit_reconstruction.lower_xpoint_z[i_time],'xr')
	plt.plot(efit_reconstruction.strikepointR[i_time],efit_reconstruction.strikepointZ[i_time],'xr')
plt.colorbar().set_label('Emissivity kW/m^3')
plt.title('Emissivity phantom from 45473 500ms\n Bayesian inversion')
plt.ylim(bottom=-2.1,top=0.20)
plt.xlim(left=0.2,right=1.6)
plt.grid()
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/inversion_comparison_emissivity_'+type_of_test+'-'+'Bayes'+'.png', bbox_inches='tight')
plt.close()


fitted_power_SART = (((np.dot(sensitivities_binned_crop,np.array(x_optimal_SART).T).T - powernoback_full_orig.flatten()) ** 2)).reshape(np.shape(sigma_powernoback_full))
fitted_power_Bayes = ((np.dot(sensitivities_binned_crop,np.array(x_optimal_Bayes)[:-2].T).T  + (np.array(selected_ROI_internal)*(x_optimal_Bayes[-1]))*homogeneous_scaling + (np.array(select_foil_region_with_plasma)*(x_optimal_Bayes[-2]))*homogeneous_scaling  - powernoback_full_orig.flatten()) ** 2).reshape(np.shape(sigma_powernoback_full))
plt.figure()
plt.imshow(fitted_power_SART,'rainbow',vmin=np.nanmin(np.concatenate([fitted_power_SART,fitted_power_Bayes])),vmax=np.nanmax(np.concatenate([fitted_power_SART,fitted_power_Bayes])))

plt.figure()
plt.imshow(fitted_power_Bayes,'rainbow',vmin=np.nanmin(np.concatenate([fitted_power_SART,fitted_power_Bayes])),vmax=np.nanmax(np.concatenate([fitted_power_SART,fitted_power_Bayes])))


if phantom_int!=[]:
	# total std
	np.nanmean((x_optimal_SART-phantom_int)**2)**0.5
	np.nanmean((x_optimal_Bayes[:-2]*np.logical_not(selected_edge_cells_for_laplacian)-phantom_int)**2)**0.5
	# below x-point no swd std
	select = np.logical_and((np.mean(grid_data_masked_crop,axis=1)[:,1])<-1,(np.mean(grid_data_masked_crop,axis=1)[:,0])<1)
	np.nanstd((x_optimal_SART-phantom_int)[select])
	np.nanstd((x_optimal_Bayes[:-2]-phantom_int)[select])

	(np.nansum((x_optimal_SART-phantom_int)**2))**0.5
	(np.nansum((x_optimal_Bayes[:-2]-phantom_int)**2))**0.5
	np.nanmean(x_optimal_SART-phantom_int)/np.nanmean(phantom_int)
	np.nanmean(x_optimal_Bayes[:-2]-phantom_int)/np.nanmean(phantom_int)
	np.nansum((x_optimal_SART-phantom_int)**2)/np.nansum(phantom_int**2)
	np.nansum((x_optimal_Bayes[:-2]-phantom_int)**2)/np.nansum(phantom_int**2)
	# total radiated power
	np.nansum((x_optimal_SART-phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)) / np.nansum((phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi))
	np.nansum((x_optimal_Bayes[:-2]*np.logical_not(selected_edge_cells_for_laplacian)-phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)) / np.nansum((phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi))
	np.nansum((x_optimal_SART)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi))
	np.nansum((x_optimal_Bayes[:-2]*np.logical_not(selected_edge_cells_for_laplacian))*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi))
	np.nansum(((x_optimal_Bayes_sigma[:-2]*np.logical_not(selected_edge_cells_for_laplacian))*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi))**2)**0.5
	# below x-point no swd total power
	if type_of_test == 'SOLPS_pantom_test':
		select = np.logical_and((np.mean(grid_data_masked_crop,axis=1)[:,1])<x_point[1],(np.mean(grid_data_masked_crop,axis=1)[:,0])<0.8)
	else:
		select = np.logical_and((np.mean(grid_data_masked_crop,axis=1)[:,1])<efit_reconstruction.lower_xpoint_z[i_time],(np.mean(grid_data_masked_crop,axis=1)[:,0])<0.8)
	np.nansum((x_optimal_SART-phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select) / np.nansum((phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select)
	np.nansum((x_optimal_Bayes[:-2]*np.logical_not(selected_edge_cells_for_laplacian)-phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select) / np.nansum((phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select)
	np.nansum((phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select)
	np.nansum((x_optimal_SART)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select)
	np.nansum((x_optimal_Bayes[:-2]*np.logical_not(selected_edge_cells_for_laplacian))*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select)
	np.nansum(((x_optimal_Bayes_sigma[:-2]*np.logical_not(selected_edge_cells_for_laplacian))*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select)**2)**0.5
	temp = np.logical_not(selected_edge_cells_for_laplacian)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select
	np.nansum(((covariance[:-2,:-2]*temp).T*temp).T)**0.5
	if type_of_test == 'SOLPS_pantom_test':
		select = ((x_point[1]-(np.mean(grid_data_masked_crop,axis=1)[:,1]))**2 + (x_point[0]-(np.mean(grid_data_masked_crop,axis=1)[:,0]))**2)**0.5 < 0.1
	else:
		select = ((efit_reconstruction.lower_xpoint_z[i_time]-(np.mean(grid_data_masked_crop,axis=1)[:,1]))**2 + (efit_reconstruction.lower_xpoint_r[i_time]-(np.mean(grid_data_masked_crop,axis=1)[:,0]))**2)**0.5 < 0.1
	np.nansum((x_optimal_SART-phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select) / np.nansum((phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select)
	np.nansum((x_optimal_Bayes[:-2]-phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select) / np.nansum((phantom_int)*(np.mean(grid_data_masked_crop,axis=1)[:,0])*((grid_resolution*1e-2)**2)*2*np.pi*(4*np.pi)*select)

	plt.figure()
	select = (efit_reconstruction.lower_xpoint_z[i_time]<(np.mean(grid_data_masked_crop,axis=1)[:,1]))
	plt.plot(phantom_int[select],x_optimal_SART[select],'+',label='core')
	select = ((efit_reconstruction.lower_xpoint_z[i_time]-(np.mean(grid_data_masked_crop,axis=1)[:,1]))**2 + (efit_reconstruction.lower_xpoint_r[i_time]-(np.mean(grid_data_masked_crop,axis=1)[:,0]))**2)**0.5 < 0.1
	plt.plot(phantom_int[select],x_optimal_SART[select],'+',label='x-point')
	select = (efit_reconstruction.lower_xpoint_z[i_time]>(np.mean(grid_data_masked_crop,axis=1)[:,1]))*(efit_reconstruction.lower_xpoint_r[i_time]<(np.mean(grid_data_masked_crop,axis=1)[:,0]))
	plt.plot(phantom_int[select],x_optimal_SART[select],'+',label='outer leg')
	select = (efit_reconstruction.lower_xpoint_z[i_time]>(np.mean(grid_data_masked_crop,axis=1)[:,1]))*(efit_reconstruction.lower_xpoint_r[i_time]>(np.mean(grid_data_masked_crop,axis=1)[:,0]))
	plt.plot(phantom_int[select],x_optimal_SART[select],'+',label='inner leg')
	plt.plot([np.min([phantom_int,x_optimal_SART]),np.max([phantom_int,x_optimal_SART])],[np.min([phantom_int,x_optimal_SART]),np.max([phantom_int,x_optimal_SART])],'--k')
	plt.legend(loc='best')
	plt.grid()
	# plt.ylabel('brightness [kW]')
	# plt.xlabel('time [s]')
	plt.pause(0.001)

	plt.figure()
	select = (efit_reconstruction.lower_xpoint_z[i_time]<(np.mean(grid_data_masked_crop,axis=1)[:,1]))
	plt.plot(phantom_int[select],x_optimal_Bayes[:-2][select],'+',label='core')
	select = ((efit_reconstruction.lower_xpoint_z[i_time]-(np.mean(grid_data_masked_crop,axis=1)[:,1]))**2 + (efit_reconstruction.lower_xpoint_r[i_time]-(np.mean(grid_data_masked_crop,axis=1)[:,0]))**2)**0.5 < 0.1
	plt.plot(phantom_int[select],x_optimal_Bayes[:-2][select],'+',label='x-point')
	select = (efit_reconstruction.lower_xpoint_z[i_time]>(np.mean(grid_data_masked_crop,axis=1)[:,1]))*(efit_reconstruction.lower_xpoint_r[i_time]<(np.mean(grid_data_masked_crop,axis=1)[:,0]))
	plt.plot(phantom_int[select],x_optimal_Bayes[:-2][select],'+',label='outer leg')
	select = (efit_reconstruction.lower_xpoint_z[i_time]>(np.mean(grid_data_masked_crop,axis=1)[:,1]))*(efit_reconstruction.lower_xpoint_r[i_time]>(np.mean(grid_data_masked_crop,axis=1)[:,0]))
	plt.plot(phantom_int[select],x_optimal_Bayes[:-2][select],'+',label='inner leg')
	plt.plot([np.min([phantom_int,x_optimal_Bayes[:-2]]),np.max([phantom_int,x_optimal_Bayes[:-2]])],[np.min([phantom_int,x_optimal_Bayes[:-2]]),np.max([phantom_int,x_optimal_Bayes[:-2]])],'--k')
	plt.legend(loc='best')
	plt.grid()
	# plt.ylabel('brightness [kW]')
	# plt.xlabel('time [s]')
	plt.pause(0.001)



# This is to start to compare expected and inverted emission with a simpler emission profile



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

# THIS IS FOR UNIFORM EMITTED POWER DENSITY
total_power=500000
center_radiator=[0.55,-1.2]

# # TOROIDAL RADIATOR
# minor_radius_radiator=0.08
# volume_radiator=2*(np.pi**2)*center_radiator[0]*minor_radius_radiator**2

# SQUARE ANULAR RADIATOR
side_radiator=0.25
volume_radiator=np.pi*((center_radiator[0]+side_radiator/2)**2-(center_radiator[0]-side_radiator/2)**2)*side_radiator
x_point_lower = Point2D(center_radiator[0] - side_radiator/2, center_radiator[1] - side_radiator/2)
x_point_upper = Point2D(center_radiator[0] + side_radiator/2, center_radiator[1] + side_radiator/2)

power_density = total_power / volume_radiator









# Load sensitivity matrix

import csv
path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_high_res_power'
sensitivities=np.load(path_sensitivity+'/sensitivity.npy')
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



# Load voxels
from cherab.mastu.bolometry import load_default_bolometer_config, load_standard_voxel_grid
from cherab.tools.inversions import ToroidalVoxelGrid
core_voxel_grid = load_standard_voxel_grid(grid_type,parent=world)


power_on_voxels=np.zeros((core_voxel_grid.count))
num_voxels=core_voxel_grid.count

i=np.linspace(0,num_voxels-1,num_voxels,dtype=int)
for index in i:
	p1,p2,p3,p4 = core_voxel_grid._voxels[index].vertices
	voxel_centre = Point2D((p1.x+p2.x+p3.x+p4.x)/4,(p1.y+p2.y+p3.y+p4.y)/4)
	if (voxel_centre.x<x_point_upper.x and voxel_centre.x>x_point_lower.x and voxel_centre.y<x_point_upper.y and voxel_centre.y>x_point_lower.y):
		power_on_voxels[index] = power_density
core_voxel_grid.plot(voxel_values=power_on_voxels,title='Emissivity adapted to the grid used to calculate sensitivity',colorbar=['rainbow','Emissivity [W/m^3]'])
# core_voxel_grid.plot(voxel_values=power_on_voxels,title='Emissivity adapted to the grid used to calculate sensitivity \n max='+str(np.max(power_on_voxels))+'[W/m^3]')


power_on_foil = np.dot(sensitivities.T,power_on_voxels)


pixel_v = (pixel_h * 9) // 7
pixel_area = 0.07*(0.07*pixel_v/pixel_h)/(pixel_h*pixel_v)

plt.figure()
cmap = plt.cm.rainbow
cmap.set_under(color='white')
vmin=0.0000000000000000000000001
plt.imshow(coleval.split_fixed_length(power_on_foil,pixel_h),origin='lower', cmap=cmap, vmin=vmin)
plt.title('Power density on the foil via sensitivity matrix')
plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off '+str(vmin)+'W/m^2')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.show()





import pickle
import os
directory = '/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction'
grid_file = os.path.join(
	directory,
	'{}_rectilinear_grid.pickle'.format(grid_type)
)

print('Used grid file:\n' + grid_file)

with open(grid_file, 'rb') as f:
	grid_data = pickle.load(f)
laplacian = grid_data['laplacian']



beta=0.000001
from cherab.tools.inversions import invert_constrained_sart
inverted_emission, conv = invert_constrained_sart(sensitivities.T,laplacian, power_on_foil,conv_tol=1.0E-05 ,beta_laplace=beta, max_iterations=250)
inverted_emission, conv = invert_constrained_sart(sensitivities.T,laplacian, power_on_foil,initial_guess=inverted_emission,relaxation=0.5,conv_tol=1.0E-10 ,beta_laplace=beta, max_iterations=250)
inverted_emission, conv = invert_constrained_sart(sensitivities.T,laplacian, power_on_foil,initial_guess=inverted_emission,relaxation=0.1,conv_tol=1.0E-10 ,beta_laplace=beta, max_iterations=1000)
core_voxel_grid.plot(voxel_values=inverted_emission,title='Thomographic inversion using laplacian and beta_laplace='+str(beta)+' \n convergence= '+str(conv[-1]),colorbar=['rainbow','Emissivity [W/m^3]'])



diff = power_on_voxels-inverted_emission
# diff=np.zeros((len(power_on_voxels)))
for index,value in enumerate(diff):
	if value<0:
		# diff[index]=np.abs((inverted_emission[index] - power_on_voxels[index])/power_on_voxels[index] )
		diff[index] = 0
core_voxel_grid.plot(voxel_values=diff,title='Emissivity difference between original and inverted prifile',colorbar=['rainbow','Emissivity [W/m^3]'])



from cherab.tools.inversions import invert_sart
inverted_emission, conv = invert_sart(sensitivities.T, power_on_foil, max_iterations=250)
core_voxel_grid.plot(voxel_values=inverted_emission,title='Emissivity adapted to the grid used to calculate sensitivity',colorbar=['rainbow','Emissivity [W/m^3]','log'])
plt.show()
