# Created 28/09/2022
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())


"""
Write out the processed IRVB data to netCDF file, using putdata.
Started from https://git.ccfe.ac.uk/MAST-U_Scheduler/abm/-/blob/development/abm/write_data.py
"""
# import logging
# from pathlib import Path
# from typing import Dict
# import numpy as np
import xarray as xr
from mast.mast_client import MastClient
# from abm.config import Config
# from abm.util import Status, System


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.getLogger("abm").level)

shot = 45401
pass_number	=0

def write_data(shot: int,
			   pass_number: int
			   ) -> None:
	"""
	Write the processed IRVB data to netCDF file.

	:param shot: the shot number
	:param pass_no: the pass number
	"""


	name='IRVB-MASTU_shot-'+str(shot)+'.ptw'
	path = '/home/ffederic/work/irvb/MAST-U/'

	shot_list = get_data(path+'shot_list2.ods')
	temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
	for i in range(1,len(shot_list['Sheet1'])):
		if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
			date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
			break
	day = str(date.date())

	laser_to_analyse=path+day+'/'+name

	full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
	full_saved_file_dict_FAST.allow_pickle=True
	full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
	if pass_number==0:
		full_saved_file_dict_FAST_pass = full_saved_file_dict_FAST['first_pass'].all()
	elif pass_number==1:
		full_saved_file_dict_FAST_pass = full_saved_file_dict_FAST['second_pass'].all()
	else:
		print('Only allowed pass number are 0 or 1')
		exit()
	inverted_dict = full_saved_file_dict_FAST_pass['inverted_dict']
	grid_resolution = 2	# cm
	time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
	inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
	inverted_data_sigma = inverted_dict[str(grid_resolution)]['inverted_data_sigma']
	binning_type = inverted_dict[str(grid_resolution)]['binning_type']
	processing_start_time = full_saved_file_dict_FAST_pass['processing_start_time']

	time_binned = full_saved_file_dict_FAST_pass['FAST_time_binned']
	powernoback = full_saved_file_dict_FAST_pass['FAST_powernoback']
	brightness = full_saved_file_dict_FAST_pass['FAST_brightness']
	FAST_counts_minus_background_crop = full_saved_file_dict_FAST_pass['FAST_counts_minus_background_crop']
	FAST_counts_minus_background_crop = FAST_counts_minus_background_crop[1:-1,1:-1,1:-1]

	R = inverted_dict[str(grid_resolution)]['geometry']['R']
	Z = inverted_dict[str(grid_resolution)]['geometry']['Z']

	try:
		foil_properties_used = full_saved_file_dict_FAST_pass['foil_properties_used']
		params_BB = inverted_dict[str(grid_resolution)]['params_BB_used']
		errparams_BB = inverted_dict[str(grid_resolution)]['errparams_BB_used']
	except:
		print('shot '+str(shot)+' foil properties missing, relaunch the pre-processing')
		foil_properties_used = 'NA'
		params_BB = 'NA'
		errparams_BB = 'NA'

	# If shot number is negative, treat it as a zshot.
	filename = 'rvb' + str(shot) + '.nc'
	output_dir = os.path.split(laser_to_analyse)[0]
	full_filename = output_dir +'/'+ filename
	client = MastClient(None)

	# Create file.
	client.put(
		str(full_filename),
		step_id="create", conventions="Fusion-1.1", data_class="analysed data",
		title="Analysed IRVB data", shot=shot, pass_number=pass_number,
		status=processing_start_time,
	)

	# Write configuration parameters
	group = "/rvb/config"
	client.put(foil_properties_used, step_id="attribute", group=group,
			   name="foil_properties_used")
	client.put(params_BB, step_id="attribute", group=group,
			   name="params_BB_used")
	client.put(errparams_BB, step_id="attribute", group=group,
			   name="errparams_BB_used")
	client.put(binning_type, step_id="attribute", group=group,
			   name="binning_type")
	client.put(processing_start_time, step_id="attribute", group=group,
			   name="processing_start_time")
	client.put(grid_resolution, step_id="attribute", group=group,
			   name="inversion_grid_resolution_cm")

	group = 'rvb/line_integrated'
	# Write dimension and coordinate data.
	# Time
	time = processed_data[system].time.values.astype("float32")
	client.put(len(time_binned), step_id="dimension", name="time", group=group, units="s")
	client.put(time_binned, step_id="coordinate", name="time", group=group,
			   label="Time", coord_class="time", units="s")
	# h_axis
	h_axis = np.flip(np.arange(powernoback.shape[1]),axis=0)
	client.put(len(h_axis), step_id="dimension", name="h_axis", group=group)
	client.put(h_axis, step_id="coordinate", name="h_axis", group=group,
			   label="Horiz axis", units="", comment="sequence of horizontal foil coordinate")

	# v_axis
	v_axis = np.arange(powernoback.shape[2])
	client.put(len(v_axis), step_id="dimension", name="v_axis", group=group)
	client.put(v_axis, step_id="coordinate", name="v_axis", group=group,
			   label="Vert axis", units="", comment="sequence of vertical foil coordinate")

	# Write signal data.
	# powernoback
	client.put(
		powernoback.astype("float32"),
		step_id="variable", name="foil_power", group=group,
		dimensions="time,h_axis,v_axis", units="W/m**2", label="Foil power density",
		comment="Power density absorbed by the foil"
	)

	# brightness
	client.put(
		brightness.astype("float32"),
		step_id="variable", name="brightness", group=group,
		dimensions="time,h_axis,v_axis", units="W/m**2", label="Line integral brightness",
		comment="Brightness at the pinhole"
	)
	# FAST_counts_minus_background_crop
	client.put(
		FAST_counts_minus_background_crop.astype("float32"),
		step_id="variable", name="delta_coults", group=group,
		dimensions="time,h_axis,v_axis", units="au", label="Count increase",
		comment="Binned foil counts minus background"
	)

	group = 'rvb/thomographic_inverted'
	# Write dimension and coordinate data.
	# Time
	client.put(len(time_full_binned_crop), step_id="dimension", name="time", group=group, units="s")
	client.put(time_full_binned_crop, step_id="coordinate", name="time", group=group,
			   label="Time", coord_class="time", units="s")
	# R
	client.put(len(R), step_id="dimension", name="r", group=group)
	client.put(R, step_id="coordinate", name="r", group=group,
			   label="Radius", units="m", comment="Radius")

	# Z
	client.put(len(Z), step_id="dimension", name="z", group=group)
	client.put(Z, step_id="coordinate", name="z", group=group,
			   label="Vert position", units="m", comment="Vertical position")

	# Write signal data.
	# inverted_data
	client.put(
		inverted_data.astype("float32"),
		step_id="variable", name="emissivity", group=group,
		dimensions="time,r,z", units="W/m**3", label="Plasma emissivity",
		comment="Plasma emissivity"
	)

	# inverted_data_sigma
	client.put(
		inverted_data_sigma.astype("float32"),
		step_id="variable", name="emissivity_sigma", group=group,
		dimensions="time,r,z", units="W/m**3", label="Plasma emissivity uncertainty",
		comment="Plasma emissivity uncertainty"
	)

	# Close the file.
	client.put(step_id="close")
