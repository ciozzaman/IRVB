# Created 2021/07/20 by Fabio Federici with contributions from Yoshika Terada

# stuff that is required to install
# pip3 install QtRangeSlider --user
# pip3 install pyqtgraph==0.12.2 --user

# significantly modified /home/ffederic/.local/lib/python3.7/site-packages/qtrangeslider/_generic_range_slider.py
# adding
# # ###############  2021/07/21 test  #######################
# def setMinimum(self, min: float):
#     if isinstance(min, (list, tuple)):
#         return type(min)(self.setMinimum(v) for v in min)
#     pos = super().setRange(min, max(super().maximum(), min))
#
# def setMaximum(self, max: float):
#     if isinstance(max, (list, tuple)):
#         return type(min)(self.setMaximum(v) for v in max)
#     pos = super().setRange(min(super().minimum(), max), max)


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, QVBoxLayout, QWidget
from qtrangeslider import QRangeSlider
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

# import pyqtgraph.console
import numpy as np

from pyqtgraph.dockarea import *

# do fint automatically the date
import pyuda
client=pyuda.Client()

#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())

#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

# to come back at the original directory
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')

# data = np.load('/home/ff645/Downloads/GUI/IRVB-MASTU_shot-043979.npz')
# data = data['data'][100:200]
#
# gna_dict = dict([])
# gna_dict['data'] = data
# np.savez_compressed('/home/ff645/Downloads/GUI/IRVB-MASTU_shot-043979_short',**gna_dict)

data = np.zeros((100,100,100))
data_shape = np.shape(data)
efit_reconstruction=None
# data = np.load('/home/ffederic/work/irvb/MAST-U/2021-07-01/IRVB-MASTU_shot-44391.npz')['data']
framerate = 383 # Hz
shot_ID = '043979'
time_array = np.arange(len(data))*1/framerate

class RangeSlider(QWidget):
	def __init__(self, minimum, maximum, parent=None):
		super(RangeSlider, self).__init__(parent=parent)
		self.verticalLayout = QVBoxLayout(self)
		self.label = QLabel(self)
		self.verticalLayout.addWidget(self.label)
		self.horizontalLayout = QVBoxLayout()
		# spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
		# self.horizontalLayout.addItem(spacerItem)
		self.slider = QRangeSlider(self)
		# QRangeSlider forces me to a range float 0 - 99 anyway
		self.slider.setValue([0., 99/2,99.])
		# next 2 lines just don't work
		self.slider.setMinimum(minimum)
		self.slider.setMaximum(maximum)
		self.slider.setValue([minimum, (maximum+minimum)//2,maximum])
		self.slider.setSingleStep(0.001)
		self.slider.setTickInterval(0.001)
		self.slider.setOrientation(Qt.Horizontal)
		self.horizontalLayout.addWidget(self.slider)
		self.shotID = ''
		# spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
		# self.horizontalLayout.addItem(spacerItem1)
		self.verticalLayout.addLayout(self.horizontalLayout)
		self.resize(self.sizeHint())

		self.minimum = minimum
		self.maximum = maximum
		self.slider.valueChanged.connect(self.setLabelValue)
		self.x = None
		self.setLabelValue(self.slider.value())

	def setLabelValue(self, value):
		self.x = self.minimum + (float(value[1]) / (self.slider.maximum() - self.slider.minimum())) * (
		self.maximum - self.minimum)
		self.left = self.minimum + (float(value[0]) / (self.slider.maximum() - self.slider.minimum())) * (
		self.maximum - self.minimum)
		self.right = self.minimum + (float(value[2]) / (self.slider.maximum() - self.slider.minimum())) * (
		self.maximum - self.minimum)
		self.label.setText('shot '+str(self.shotID)+' from%.4g to %.4g ms : %.4g ms' %(1e3*time_array[round(self.left)],1e3*time_array[round(self.right)],1e3*time_array[round(self.x)]))


app = pg.mkQApp()
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(900,700)
win.setWindowTitle('pyqtgraph example: dockarea')

## Create docks, place them into the window one at a time.
## Note that size arguments are only a suggestion; docks will still have to
## fill the entire dock area and obey the limits of their internal widgets.
d2 = Dock("", size=(500,50))
d7 = Dock("", size=(400, 50))
d5 = Dock("", size=(500,450))
d1 = Dock("", size=(150, 450))
d4 = Dock("", size=(250,450))
d3 = Dock("", size=(400,150))
d6 = Dock("", size=(500,150))
area.addDock(d1, 'left')
area.addDock(d2, 'top', d1)
area.addDock(d5, 'left', d1)
area.addDock(d3, 'bottom')
area.addDock(d4, 'right', d1)
area.addDock(d6, 'right', d3)
area.addDock(d7, 'right', d2)


## Add widgets into each dock

# range slider
# horizontalLayout = QVBoxLayout()
# w1 = Slider(0, len(data)-1)
w2 = RangeSlider(0, len(time_array)-1)
d2.addWidget(w2)

# main image

# w5 = pg.ImageView()
# w5.setImage(data[0])
# d5.addWidget(w5)

horizontalLayout = QVBoxLayout()
w5 = pg.GraphicsLayoutWidget()
horizontalLayout.addWidget(w5)
image1 = pg.ImageItem()
image1.setImage(data[0])
p5 = w5.addPlot()
# p6.setTitle(title="shot " + str(w2.shotID))
p5.addItem(image1)
p5.setAspectLocked()
# p5.invertX()
image_frame = p5.plot([],[], pen='r')
image_frame_export = p5.plot([],[], pen='r')
image_frame_roi = pg.ROI([0, 0], [np.shape(image1.image)[0], np.shape(image1.image)[1]])
image_frame_roi.addScaleHandle([1, 0], [0,1])
image_frame_roi.addScaleHandle([1, 1], [0,0])
image_frame_roi.addScaleHandle([0, 1], [1,0])
image_frame_roi.addScaleHandle([0, 0], [1,1])
p5.addItem(image_frame_roi)
image_frame_roi.setZValue(10)  # make sure ROI is drawn above image
image_frame_roi.setPen(pen='r')

d5.addWidget(w5)
curve = p5.plot(pen='r')

# Contrast/color control
hist = pg.HistogramLUTItem()
hist.setImageItem(image1)
# hist.gradient.setColorMode('rgb')
# hist.setLevels(np.min(data[0]), np.max(data[0]))
histogram_level_high = np.max(data[0])
histogram_level_low = np.min(data[0])
p8 = w5.addItem(hist)


# Draggable line for setting isocurve level
isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(isoLine)
hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
isoLine.setValue(np.mean(data[0]))
isoLine.setZValue(1000) # bring iso line above contrast controls

# Custom ROI for selecting an image region
roi = pg.ROI([np.shape(image1.image)[0]*3//8, np.shape(image1.image)[1]*3//8], [np.shape(image1.image)[0]//4, np.shape(image1.image)[1]//4])
# roi.addScaleHandle([0.5, 1], [0.5, 0.5])
# roi.addScaleHandle([0, 0.5], [0.5, 0.5])
roi.addScaleHandle([1, 0], [0,1])
roi.addScaleHandle([1, 1], [0,0])
roi.addScaleHandle([0, 1], [1,0])
roi.addScaleHandle([0, 0], [1,1])
# roi.addRotateHandle([0, 1], [0.5, 0.5])  # works but better not use it
p5.addItem(roi)
roi.setZValue(10)  # make sure ROI is drawn above image
roi.setPen(pen='m')
roi2 = pg.ROI([np.shape(image1.image)[0]*1//8, np.shape(image1.image)[1]*1//8], [np.shape(image1.image)[0]//4, np.shape(image1.image)[1]//4])
# roi.addScaleHandle([0.5, 1], [0.5, 0.5])
# roi.addScaleHandle([0, 0.5], [0.5, 0.5])
roi2.addScaleHandle([1, 0], [0,1])
roi2.addScaleHandle([0, 1], [1,0])
roi2.addScaleHandle([0, 0], [1,1])
roi2.addScaleHandle([1, 1], [0,0])
# roi2.addRotateHandle([0, 1], [0.5, 0.5])  # works but better not use it
p5.addItem(roi2)
roi2.setZValue(10)  # make sure ROI is drawn above image
roi2.setPen(pen='c')

# Isocurve drawing
iso = pg.IsocurveItem(level=0.8, pen='g')
iso.setParentItem(image1)
iso.setZValue(5)

fueling_point_location_on_foil = coleval.return_fueling_point_location_on_foil()
structure_point_location_on_foil = coleval.return_structure_point_location_on_foil()
core_tangential_location_on_foil = coleval.return_core_tangential_location_on_foil()
core_poloidal_location_on_foil = coleval.return_core_poloidal_location_on_foil()
divertor_poloidal_location_on_foil = coleval.return_divertor_poloidal_location_on_foil()
structure_radial_profile = coleval.return_structure_radial_profile()
core_poloidal = coleval.return_core_poloidal()
divertor_poloidal = coleval.return_divertor_poloidal()
foil_size = [0.07,0.09]
flag_radial_profile = False
inversion_R = 0
inversion_Z = 0
dr = 0
dz = 0

# # image1
# 		for i in range(len(fueling_point_location_on_foil)):
# 			ax.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
# 			ax.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
# 		for i in range(len(structure_point_location_on_foil)):
# 			ax.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)


# build isocurves from smoothed data
iso.setData(pg.gaussianFilter(image1.image, (2, 2)))


# Another plot area for displaying ROI data
w3 = pg.GraphicsLayoutWidget()
p3 = w3.addPlot(title="horizontal average")
p3.showGrid(x=1,y=1,alpha=0.5)
d3.addWidget(w3)

# Another plot area for displaying ROI data
w1 = pg.GraphicsLayoutWidget()
p1 = w1.addPlot(title="vert. average")
p1.showGrid(x=1,y=1,alpha=0.5)
d1.addWidget(w1)

# Another plot area for displaying ROI data
w6 = pg.GraphicsLayoutWidget()
p6 = w6.addPlot(title="time average of selected area")
p6.showGrid(x=1,y=1,alpha=0.5)
p6_1 = p6.plot([],[], pen='m')
p6_2 = p6.plot([],[], pen='c')
p6_time_mark = p6.plot([],[], pen='g')
d6.addWidget(w6)



quantity_options = {'counts [au]': 'counts', 'FAST counts [au]': 'FAST_counts_minus_background_crop', 'FAST power [W/m2]': 'FAST_powernoback', 'FAST brightness [W/m2]': 'FAST_brightness', 'FAST foil fit [W/m2]': 'FAST/inverted/fitted_foil_power', 'FAST foil fit error [W/m2]': 'FAST/inverted/foil_power_residuals', 'FAST emissivity [W/m3]': 'FAST/inverted/inverted_data', "Temperature [°C]": 'laser_temperature_crop_binned_full', "Relative temp [K]": 'laser_temperature_minus_background_crop_binned_full', "tot power [W/m2]": 'powernoback_full', "tot power std [W/m2]": 'powernoback_std_full', "BB power [W/m2]": 'BBrad_full', "diff power [W/m2]": 'diffusion_full', "dt power [W/m2]": 'timevariation_full', "brightness [W/m2]": 'brightness_full', "emissivity [W/m3]": 'inverted/inverted_data', "fitted power [W/m2]": 'inverted/fitted_foil_power', "fitted power residuals [W/m2]": 'inverted/foil_power_residuals'}

# creating all the flags and accessories needed
params = [
	{'name': 'ROI', 'type': 'group', 'children': [
		# {'name': 'ROI magenta hor', 'type': 'int', 'value': 10},
		# {'name': 'ROI magenta d hor', 'type': 'int', 'value': 10},
		# {'name': 'ROI magenta ver', 'type': 'int', 'value': 10},
		# {'name': 'ROI magenta d ver', 'type': 'int', 'value': 10},
		{'name': 'ROI magenta hor', 'type': 'float', 'value': 10, 'step': 1e-3},
		{'name': 'ROI magenta d hor', 'type': 'float', 'value': 10, 'step': 1e-3},
		{'name': 'ROI magenta ver', 'type': 'float', 'value': 10, 'step': 1e-3},
		{'name': 'ROI magenta d ver', 'type': 'float', 'value': 10, 'step': 1e-3},
		# {'name': 'ROI magenta angle', 'type': 'int', 'value': 0},
		# {'name': 'ROI cyan hor', 'type': 'int', 'value': 10},
		# {'name': 'ROI cyan d hor', 'type': 'int', 'value': 10},
		# {'name': 'ROI cyan ver', 'type': 'int', 'value': 10},
		# {'name': 'ROI cyan d ver', 'type': 'int', 'value': 10},
		{'name': 'ROI cyan hor', 'type': 'float', 'value': 10, 'step': 1e-3},
		{'name': 'ROI cyan d hor', 'type': 'float', 'value': 10, 'step': 1e-3},
		{'name': 'ROI cyan ver', 'type': 'float', 'value': 10, 'step': 1e-3},
		{'name': 'ROI cyan d ver', 'type': 'float', 'value': 10, 'step': 1e-3},
		# {'name': 'ROI cyan angle', 'type': 'int', 'value': 0},
		{'name': 'Time start [ms]', 'type': 'float', 'value': time_array[int(w2.left)]*1e3, 'step': 1, 'finite': False},
		{'name': 'Time end [ms]', 'type': 'float', 'value': time_array[int(w2.right)]*1e3, 'step': 1, 'finite': False},
		{'name': 'Time [ms]', 'type': 'float', 'value': time_array[int(w2.x)]*1e3, 'step': 1, 'finite': False},
		{'name': 'Histogram auto', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
		{'name': 'Hist lev high', 'type': 'float', 'value': histogram_level_high, 'step': np.min(np.diff(np.sort(data.flatten()))), 'finite': False},
		{'name': 'Hist lev low', 'type': 'float', 'value': histogram_level_low, 'step': np.min(np.diff(np.sort(data.flatten()))), 'finite': False},
	]},
	{'name': 'Set display', 'type': 'group', 'children': [
		{'name': 'Shot +', 'type': 'action'},
		{'name': 'Shot -', 'type': 'action'},
		{'name': 'Quantity', 'type': 'list', 'values': quantity_options, 'value': 'FAST_counts_minus_background_crop'},
		{'name': 'Binning', 'type': 'list', 'values': {'bin1x1x1': 'bin1x1x1', 'bin1x3x3': 'bin1x3x3', 'bin1x5x5': 'bin1x5x5', "bin1x10x10": 'bin1x10x10', "bin2x1x1": 'bin2x1x1', "bin2x3x3": 'bin2x3x3', "bin2x5x5": 'bin2x5x5', "bin2x10x10": 'bin2x10x10', "bin3x1x1": 'bin3x1x1', "bin3x3x3": 'bin3x3x3', "bin3x5x5": 'bin3x5x5', "bin3x10x10": 'bin3x10x10', "bin5x1x1": 'bin5x1x1', "bin5x3x3": 'bin5x3x3', "bin5x5x5": 'bin5x5x5', "bin5x10x10": 'bin5x10x10'}, 'value': 1},
		{'name': 'Voxel res', 'type': 'list', 'values': {'2': '2', '4': '4'}, 'value': '4'},
		{'name': 'Load data', 'type': 'action'},
		{'name': 'Load EFIT', 'type': 'action'},
		{'name': 'Play', 'type': 'action'},
		{'name': 'Rewind', 'type': 'action'},
		{'name': 'Pause', 'type': 'action'},
		{'name': 'Export video', 'type': 'action'},
		{'name': 'Export image', 'type': 'action'},
		{'name': 'Export left', 'type': 'float', 'value': 10, 'step': 1e-3, 'finite': False},
		{'name': 'Export right', 'type': 'float', 'value': 10, 'step': 1e-3, 'finite': False},
		{'name': 'Export up', 'type': 'float', 'value': 10, 'step': 1e-3, 'finite': False},
		{'name': 'Export down', 'type': 'float', 'value': 10, 'step': 1e-3, 'finite': False},
		{'name': 'Include prelude', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
		{'name': 'Export size', 'type': 'float', 'value': 15, 'step': 1e-3, 'finite': False},
	]},
	{'name': 'Overlays', 'type': 'group', 'children': [
		{'name': 'Structure', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
		{'name': 'Mag axis', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
		{'name': 'X-point', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
		{'name': 'Separatrix', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
		{'name': 'Core Resistive bol', 'type': 'bool', 'value': False, 'tip': "This is a checkbox"},
		{'name': 'Div Resistive bol', 'type': 'bool', 'value': False, 'tip': "This is a checkbox"},
	]}
]
# creating all the flags and accessories needed
params1 = [
		{'name': 'Shot number', 'type': 'int', 'value': 0},
		{'name': 'File Path', 'type': 'str', 'value': "insert full path", 'tip': 'try'},
		{'name': 'Navigate Path', 'type': 'action'},
]

## Create tree of Parameter objects
param_ext = Parameter.create(name='params', type='group', children=params)
param_ext1 = Parameter.create(name='params', type='group', children=params1)

## If anything changes in the tree, print a message
def change(param, changes):
	global histogram_level_low,histogram_level_high,overlay_structure,overlay_fueling_point,overlay_x_point,overlay_mag_axis,overlay_strike_points_1,overlay_strike_points_2,overlay_separatrix,overlay_Core_Resistive_bol,overlay_Div_Resistive_bol,efit_reconstruction,all_time_mag_axis_location,all_time_x_point_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix,time_array,flag_radial_profile,all_time_sep_r,all_time_sep_z,r_fine,z_fine,data_shape
	print("tree changes:")
	for param, change, data in changes:
		path = param_ext.childPath(param)
		if path is not None:
			childName = '.'.join(path)
		else:
			childName = param.name()
		print('  parameter: %s'% childName)
		print('  change:	%s'% change)
		print('  data:	  %s'% str(data))
		print('  ----------')
	roi.setPos((param_ext['ROI', 'ROI magenta hor'],param_ext['ROI', 'ROI magenta ver']))
	roi.setSize((param_ext['ROI', 'ROI magenta d hor'],param_ext['ROI', 'ROI magenta d ver']))
	# roi.setAngle((param_ext['ROI', 'ROI magenta angle']))
	roi2.setPos((param_ext['ROI', 'ROI cyan hor'],param_ext['ROI', 'ROI cyan ver']))
	roi2.setSize((param_ext['ROI', 'ROI cyan d hor'],param_ext['ROI', 'ROI cyan d ver']))
	image_frame_roi.setPos((param_ext['Set display', 'Export left'],param_ext['Set display', 'Export down']))
	image_frame_roi.setSize((param_ext['Set display', 'Export right']-param_ext['Set display', 'Export left'],param_ext['Set display', 'Export up']-param_ext['Set display', 'Export down']))
	# roi2.setAngle((param_ext['ROI', 'ROI cyan angle']))
	w2.slider.setValue([np.abs(param_ext['ROI', 'Time start [ms]']*1e-3-time_array).argmin()/(w2.maximum-w2.minimum)*(w2.slider.maximum()-w2.slider.minimum()),np.abs(param_ext['ROI', 'Time [ms]']*1e-3-time_array).argmin()/(w2.maximum-w2.minimum)*(w2.slider.maximum()-w2.slider.minimum()),np.abs(param_ext['ROI', 'Time end [ms]']*1e-3-time_array).argmin()/(w2.maximum-w2.minimum)*(w2.slider.maximum()-w2.slider.minimum())])
	histogram_level_high,histogram_level_low = param_ext['ROI', 'Hist lev high'],param_ext['ROI', 'Hist lev low']
	hist.setLevels(histogram_level_high,histogram_level_low)
	# hist.sigLevelChangeFinished.emit(True)
	if param_ext['Overlays','Structure']==False:
		for i in range(len(overlay_structure)):
			overlay_structure[i].setAlpha(0,False)
	else:
		for i in range(len(overlay_structure)):
			overlay_structure[i].setAlpha(0.5,False)
	if param_ext['Overlays','Structure']==False or flag_radial_profile:
		try:
			for i in range(len(overlay_fueling_point)):
				overlay_fueling_point[i].setAlpha(0,False)
		except:
			pass
	else:
		for i in range(len(overlay_fueling_point)):
			overlay_fueling_point[i].setAlpha(0.5,False)
	if param_ext['Overlays','Core Resistive bol']==False:
		for i in range(len(overlay_Core_Resistive_bol)):
			overlay_Core_Resistive_bol[i].setAlpha(0,False)
	else:
		for i in range(len(overlay_Core_Resistive_bol)):
			overlay_Core_Resistive_bol[i].setAlpha(1,False)
	if param_ext['Overlays','Div Resistive bol']==False:
		for i in range(len(overlay_Div_Resistive_bol)):
			overlay_Div_Resistive_bol[i].setAlpha(0,False)
	else:
		for i in range(len(overlay_Div_Resistive_bol)):
			overlay_Div_Resistive_bol[i].setAlpha(1,False)
	try:
		if (param_ext['Overlays','Mag axis'] or param_ext['Overlays','X-point'] or param_ext['Overlays','Separatrix']):
			i_time = np.abs(time_array[round(w2.x)]-efit_reconstruction.time).argmin()
		if param_ext['Overlays','Mag axis']==False:
			overlay_mag_axis.setAlpha(0,False)
		if param_ext['Overlays','X-point']==False:
			overlay_x_point.setAlpha(0,False)
		if flag_radial_profile:
			if param_ext['Overlays','Mag axis']==True:
				overlay_mag_axis.setData([efit_reconstruction.mag_axis_r[i_time]],[efit_reconstruction.mag_axis_z[i_time]])
				overlay_mag_axis.setAlpha(1,False)
			if param_ext['Overlays','X-point']==True:
				overlay_x_point.setData([efit_reconstruction.lower_xpoint_r[i_time]],[efit_reconstruction.lower_xpoint_z[i_time]])
				overlay_x_point.setAlpha(1,False)
			if param_ext['Overlays','Separatrix']==False:
				overlay_strike_points_1.setAlpha(0,False)
				# for __i in range(len(overlay_strike_points_2)):
				# 	overlay_strike_points_2[__i].setAlpha(0,False)
				for __i in range(len(overlay_separatrix)):
					overlay_separatrix[__i].setAlpha(0,False)
			else:
				temp = efit_reconstruction.strikepointR[i_time]>0.1	# to avoid that strike points apre placed at 0,0
				overlay_strike_points_1.setData(efit_reconstruction.strikepointR[i_time][temp].tolist()*2,efit_reconstruction.strikepointZ[i_time][temp].tolist()+(-efit_reconstruction.strikepointZ[i_time][temp]).tolist())
				overlay_strike_points_1.setAlpha(1,False)
				# for __i in range(len(overlay_strike_points_2)):
				# 	overlay_strike_points_2[__i].setData(all_time_strike_points_location_rot[i_time][__i][:,0]*(data_shape[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(data_shape[2]-1)/foil_size[1])
				# 	overlay_strike_points_2[__i].setAlpha(1,False)
				for __i in range(len(overlay_separatrix)):
					overlay_separatrix[__i].setData(r_fine[all_time_sep_r[i_time][__i]],z_fine[all_time_sep_z[i_time][__i]])
					overlay_separatrix[__i].setAlpha(1,False)
		else:
			if param_ext['Overlays','Mag axis']==True:
				overlay_mag_axis.setData(all_time_mag_axis_location[i_time][:,0]*(data_shape[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(data_shape[2]-1)/foil_size[1])
				overlay_mag_axis.setAlpha(1,False)
			if param_ext['Overlays','X-point']==True:
				overlay_x_point.setData(all_time_x_point_location[i_time][:,0]*(data_shape[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(data_shape[2]-1)/foil_size[1])
				overlay_x_point.setAlpha(1,False)
			if param_ext['Overlays','Separatrix']==False:
				overlay_strike_points_1.setAlpha(0,False)
				for __i in range(len(overlay_strike_points_2)):
					overlay_strike_points_2[__i].setAlpha(0,False)
				for __i in range(len(overlay_separatrix)):
					overlay_separatrix[__i].setAlpha(0,False)
			else:
				overlay_strike_points_1.setData(all_time_strike_points_location[i_time][:,0]*(data_shape[1]-1)/foil_size[0],all_time_strike_points_location[i_time][:,1]*(data_shape[2]-1)/foil_size[1])
				overlay_strike_points_1.setAlpha(1,False)
				for __i in range(len(overlay_strike_points_2)):
					overlay_strike_points_2[__i].setData(all_time_strike_points_location_rot[i_time][__i][:,0]*(data_shape[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(data_shape[2]-1)/foil_size[1])
					overlay_strike_points_2[__i].setAlpha(1,False)
				for __i in range(len(overlay_separatrix)):
					overlay_separatrix[__i].setData(all_time_separatrix[i_time][__i][:,0]*(data_shape[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(data_shape[2]-1)/foil_size[1])
					overlay_separatrix[__i].setAlpha(1,False)
	except:
		print('no legacy plots')

inhibit_update_like_video = True
inhibit_update_like_video_rew = True
def update_like_video():
	global inhibit_update_like_video,inhibit_update_like_video_rew,time_array
	if inhibit_update_like_video == False:
		inhibit_update_like_video_rew = True
		print('update_like_video')
		now = np.abs(param_ext['ROI', 'Time [ms]']-time_array*1e3).argmin()
		if now == len(time_array)-1:
			now = -1
		future = now+1
		param_ext['ROI', 'Time [ms]'] = time_array[future]*1e3
		# now = w2.x
		# if now == len(time_array)-1:
		# 	now = -1
		# future = now+1
		# w2.x = future
		print('update_plot_inhibit '+str(update_plot_inhibit))
		update_plot()
	elif inhibit_update_like_video_rew == False:
		inhibit_update_like_video = True
		print('update_like_video')
		now = np.abs(param_ext['ROI', 'Time [ms]']-time_array*1e3).argmin()
		if now == 0:
			now = len(time_array)
		future = now-1
		param_ext['ROI', 'Time [ms]'] = time_array[future]*1e3
		# now = w2.x
		# if now == len(time_array)-1:
		# 	now = -1
		# future = now+1
		# w2.x = future
		print('update_plot_inhibit '+str(update_plot_inhibit))
		update_plot()

def start_video():
	global inhibit_update_like_video,inhibit_update_like_video_rew
	inhibit_update_like_video = False
	inhibit_update_like_video_rew = True

def rew_video():
	global inhibit_update_like_video,inhibit_update_like_video_rew
	inhibit_update_like_video_rew = False
	inhibit_update_like_video = True

def pause_video():
	global inhibit_update_like_video,inhibit_update_like_video_rew
	inhibit_update_like_video = True
	inhibit_update_like_video_rew = True

# create timer for play / pause function
timer = pg.QtCore.QTimer()
timer.timeout.connect(update_like_video)
timer.start(500)	# refresh time in ms

def export_video():
	global framerate,data,time_array,histogram_level_high,histogram_level_low,w2,flag_radial_profile,inversion_R,inversion_Z,dr,dz,efit_reconstruction,param_ext,data_shape
	path_mother = os.path.split(param_ext1['File Path'])[0]
	filenames = coleval.all_file_names(path_mother,os.path.split(param_ext1['File Path'])[1]+'_export_')
	if len(filenames)==0:
		next_export = 1
	else:
		done_ones = []
		for filename in filenames:
			done_ones.append(int(filename[filename.find('export_')+len('export_'):filename.find('.')]))
		next_export = np.max(done_ones) + 1
	start_time = np.abs(param_ext['ROI', 'Time start [ms]']*1e-3-time_array).argmin()
	end_time = np.abs(param_ext['ROI', 'Time end [ms]']*1e-3-time_array).argmin()

	barlabel=list(quantity_options.keys())[(param_ext['Set display','Quantity']==np.array(list(quantity_options.values()))).argmax()]
	if param_ext['ROI', 'Histogram auto']:
		histogram_level_high_for_plot = 'auto'
		histogram_level_low_for_plot = 'auto'
	else:
		histogram_level_high_for_plot = cp.deepcopy(histogram_level_high)
		histogram_level_low_for_plot = cp.deepcopy(histogram_level_low)

	# here I set the boundaries of the export
	if flag_radial_profile:
		full_range_hor = np.sort((inversion_R-dr/2).tolist()+[inversion_R.max()+dr/2])
		full_range_ver = np.sort((inversion_Z-dz/2).tolist()+[inversion_Z.max()+dz/2])
	else:
		full_range_hor = np.arange(0,data_shape[1]+1)
		full_range_ver = np.arange(0,data_shape[2]+1)
	limit_left = np.abs(param_ext['Set display','Export left'] - full_range_hor).argmin()
	limit_right = np.abs(param_ext['Set display','Export right'] - full_range_hor).argmin()
	limit_bottom = np.abs(param_ext['Set display','Export down'] - full_range_ver).argmin()
	limit_top = np.abs(param_ext['Set display','Export up'] - full_range_ver).argmin()
	if flag_radial_profile:
		extent = [inversion_R.min()-dr/2, inversion_R.max()+dr/2, inversion_Z.min()-dz/2, inversion_Z.max()+dz/2]
		image_extent = [full_range_hor[limit_left], full_range_hor[limit_right], full_range_ver[limit_bottom], full_range_ver[limit_top]]
		ani = coleval.movie_from_data_radial_profile(np.array([np.transpose(np.flip(data,axis=1)[start_time:end_time],(0,2,1))]), framerate,timesteps=time_array[start_time:end_time],integration=2,time_offset=time_array[start_time],extvmin=histogram_level_low_for_plot,extvmax=histogram_level_high_for_plot, extent = extent, image_extent=image_extent,xlabel='R [m]', ylabel='Z [m]',barlabel=barlabel, prelude='shot '  + w2.shotID + '\n'+str(param_ext['Set display','Binning'])+'\n'+'grid resolution %.3gcm\n' %(int(param_ext['Set display','Voxel res'])) ,overlay_structure=param_ext['Overlays','Structure'],include_EFIT=True,efit_reconstruction=efit_reconstruction,pulse_ID=w2.shotID,overlay_x_point=param_ext['Overlays','X-point'],overlay_mag_axis=param_ext['Overlays','Mag axis'],overlay_strike_points=param_ext['Overlays','Separatrix'],overlay_separatrix=param_ext['Overlays','Separatrix'])#,extvmin=0,extvmax=4e4)
	else:
		image_extent = [full_range_hor[limit_left]-0.5, full_range_hor[limit_right]-1+0.5, full_range_ver[limit_bottom]-0.5, full_range_ver[limit_top]-1+0.5]
		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(np.flip(data,axis=1)[start_time:end_time],(0,2,1)),axis=2)]), framerate,image_extent=image_extent,timesteps=time_array[start_time:end_time],integration=2,time_offset=time_array[start_time],extvmin=histogram_level_low_for_plot,extvmax=histogram_level_high_for_plot,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel=barlabel, prelude='shot ' + w2.shotID + '\n'+str(param_ext['Set display','Binning'])+'\n',overlay_structure=param_ext['Overlays','Structure'],include_EFIT=True,efit_reconstruction=efit_reconstruction,pulse_ID=w2.shotID,overlay_x_point=param_ext['Overlays','X-point'],overlay_mag_axis=param_ext['Overlays','Mag axis'],overlay_strike_points=param_ext['Overlays','Separatrix'],overlay_separatrix=param_ext['Overlays','Separatrix'])
	ani.save(param_ext1['File Path'] + '_export_' + str(next_export) + '.mp4', fps=5*framerate/383, writer='ffmpeg',codec='mpeg4')
	plt.close('all')
	print('\n'+'\n'+param_ext1['File Path'] + '_export_' + str(next_export) + '.mp4 generated'+'\n'+'\n')

def export_image():
	global framerate,data,time_array,histogram_level_high,histogram_level_low,w2,flag_radial_profile,inversion_R,inversion_Z,dr,dz,efit_reconstruction,param_ext,data_shape
	path_mother = os.path.split(param_ext1['File Path'])[0]
	filenames = coleval.all_file_names(path_mother,os.path.split(param_ext1['File Path'])[1]+'_export_')
	if len(filenames)==0:
		next_export = 1
	else:
		done_ones = []
		for filename in filenames:
			done_ones.append(int(filename[filename.find('export_')+len('export_'):filename.find('.')]))
		next_export = np.max(done_ones) + 1

	barlabel=list(quantity_options.keys())[(param_ext['Set display','Quantity']==np.array(list(quantity_options.values()))).argmax()]
	if param_ext['ROI', 'Histogram auto']:
		histogram_level_high_for_plot = 'auto'
		histogram_level_low_for_plot = 'auto'
	else:
		if barlabel[-6:] == '[W/m3]':
			histogram_level_high_for_plot = cp.deepcopy(histogram_level_high)*1e-3
			histogram_level_low_for_plot = cp.deepcopy(histogram_level_low)*1e-3
		else:
			histogram_level_high_for_plot = cp.deepcopy(histogram_level_high)
			histogram_level_low_for_plot = cp.deepcopy(histogram_level_low)
	a = w2.x
	if barlabel[-6:] == '[W/m3]':
		barlabel = barlabel[:-6] + '[kW/m3]'
		to_plot = data[round(a)]*1e-3
	else:
		to_plot = data[round(a)]
	# here I set the boundaries of the export
	if flag_radial_profile:
		full_range_hor = np.sort((inversion_R-dr/2).tolist()+[inversion_R.max()+dr/2])
		full_range_ver = np.sort((inversion_Z-dz/2).tolist()+[inversion_Z.max()+dz/2])
	else:
		full_range_hor = np.arange(0,data_shape[1]+1)
		full_range_ver = np.arange(0,data_shape[2]+1)
	limit_left = np.abs(param_ext['Set display','Export left'] - full_range_hor).argmin()
	limit_right = np.abs(param_ext['Set display','Export right'] - full_range_hor).argmin()
	limit_bottom = np.abs(param_ext['Set display','Export down'] - full_range_ver).argmin()
	limit_top = np.abs(param_ext['Set display','Export up'] - full_range_ver).argmin()
	if param_ext['Overlays','X-point'] or param_ext['Overlays','Mag axis'] or param_ext['Overlays','Separatrix']:
		include_EFIT = True
	else:
		include_EFIT = False
	if flag_radial_profile:
		if param_ext['Set display','Include prelude']:
			prelude = 'shot '  + w2.shotID + '\n'+str(param_ext['Set display','Binning'])+'\n'+'grid resolution %.3gcm\nTime %.3gms\n' %(int(param_ext['Set display','Voxel res']),param_ext['ROI', 'Time [ms]'])
		else:
			prelude = ''
		extent = [inversion_R.min()-dr/2, inversion_R.max()+dr/2, inversion_Z.min()-dz/2, inversion_Z.max()+dz/2]
		image_extent = [full_range_hor[limit_left], full_range_hor[limit_right], full_range_ver[limit_bottom], full_range_ver[limit_top]]
		fig,efit_reconstruction = coleval.image_from_data_radial_profile(np.array([np.transpose(np.flip([to_plot],axis=1),(0,2,1))]),form_factor_size=param_ext['Set display','Export size'],ref_time=param_ext['ROI', 'Time [ms]']*1e-3,extvmin=histogram_level_low_for_plot,extvmax=histogram_level_high_for_plot, extent = extent, image_extent=image_extent,xlabel='R [m]', ylabel='Z [m]',barlabel=barlabel, prelude=prelude ,overlay_structure=param_ext['Overlays','Structure'],include_EFIT=include_EFIT,efit_reconstruction=efit_reconstruction,pulse_ID=w2.shotID,overlay_x_point=param_ext['Overlays','X-point'],overlay_mag_axis=param_ext['Overlays','Mag axis'],overlay_strike_points=param_ext['Overlays','Separatrix'],overlay_separatrix=param_ext['Overlays','Separatrix'],EFIT_output_requested=True)#,extvmin=0,extvmax=4e4)
	else:
		if param_ext['Set display','Include prelude']:
			prelude = 'shot ' + w2.shotID + '\n'+str(param_ext['Set display','Binning'])+'\n'+'Time %.3gms\n' %(param_ext['ROI', 'Time [ms]'])
		else:
			prelude = ''
		image_extent = [full_range_hor[limit_left]-0.5, full_range_hor[limit_right]-1+0.5, full_range_ver[limit_bottom]-0.5, full_range_ver[limit_top]-1+0.5]
		fig,efit_reconstruction = coleval.image_from_data(np.array([np.flip(np.transpose(np.flip([to_plot],axis=1),(0,2,1)),axis=2)]),form_factor_size=param_ext['Set display','Export size'],image_extent=image_extent,ref_time=param_ext['ROI', 'Time [ms]']*1e-3,extvmin=histogram_level_low_for_plot,extvmax=histogram_level_high_for_plot,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel=barlabel, prelude=prelude,overlay_structure=param_ext['Overlays','Structure'],include_EFIT=include_EFIT,efit_reconstruction=efit_reconstruction,pulse_ID=w2.shotID,overlay_x_point=param_ext['Overlays','X-point'],overlay_mag_axis=param_ext['Overlays','Mag axis'],overlay_strike_points=param_ext['Overlays','Separatrix'],overlay_separatrix=param_ext['Overlays','Separatrix'],EFIT_output_requested=True)
	plt.savefig(param_ext1['File Path'] + '_export_' + str(next_export) + '.png', bbox_inches='tight')
	plt.close('all')
	print('\n'+'\n'+param_ext1['File Path'] + '_export_' + str(next_export) + '.mp4 generated'+'\n'+'\n')


param_ext.param('Set display','Play').sigActivated.connect(start_video)
param_ext.param('Set display','Rewind').sigActivated.connect(rew_video)
param_ext.param('Set display','Pause').sigActivated.connect(pause_video)
param_ext.param('Set display','Export video').sigActivated.connect(export_video)
param_ext.param('Set display','Export image').sigActivated.connect(export_image)


def NavigatePath():
	global all_time_x_point_location,all_time_mag_axis_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix,efit_reconstruction
	dialog = QtGui.QFileDialog()
	laser_to_analyse=dialog.getOpenFileName(caption="Open the short archive", filter="Archive file (*.npz)", directory="/home/ffederic/work/irvb/MAST-U")[0][:-4]
	if laser_to_analyse[-5:] == 'short':
		laser_to_analyse = laser_to_analyse[:-6]
	elif laser_to_analyse[-4:] == 'FAST':
		laser_to_analyse = laser_to_analyse[:-5]
	print(laser_to_analyse)
	param_ext1['File Path'] = laser_to_analyse
	w2.shotID = laser_to_analyse[-5:]
	param_ext1['Shot number'] = int(w2.shotID)
	# p6.setTitle(title="shot " + str(w2.shotID))
	# Load_EFIT()
	Load()

path_to_explore = '/home/ffederic/work/irvb/MAST-U'
f = []
for (dirpath, dirnames, filenames) in os.walk(path_to_explore):
	f.append(dirnames)
dates = []
for value in f[0]:
	if value[:4] == '2021':	# select within the same year
		dates.append(value)
dates_searchable = [int(value.replace('-','')) for value in dates]
dates = np.array([value for _, value in sorted(zip(dates_searchable, dates))])
# dates_searchable = np.sort(dates_searchable)
def Select_previous_pulse():
	global all_time_x_point_location,all_time_mag_axis_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix,efit_reconstruction
	laser_to_analyse = param_ext1['File Path']
	if w2.shotID != '':
		shotID = int(w2.shotID)
		shotID -= 1
		laser_to_analyse = laser_to_analyse[:-5] + str(shotID)
		if os.path.exists(laser_to_analyse+'.npz'):
			param_ext1['File Path'] = laser_to_analyse
			w2.shotID = str(shotID)
			print(laser_to_analyse)
			Load()
		else:
			date_found = (dates == os.path.split(os.path.split(laser_to_analyse)[0])[1]).argmax()
			neighbour_date = dates[max(0,date_found-1)]
			test_path = os.path.split(os.path.split(laser_to_analyse)[0])[0] + '/' + neighbour_date + '/' + os.path.split(laser_to_analyse)[1][:-5] + str(shotID)
			w2.shotID = str(shotID)
			if os.path.exists(test_path + '.npz'):
				param_ext1['File Path'] = test_path
				print(laser_to_analyse)
				Load()
			else:
				print('SHOT  ' + str(shotID) + '  MISSING')

def Select_next_pulse():
	global all_time_x_point_location,all_time_mag_axis_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix,efit_reconstruction
	laser_to_analyse = param_ext1['File Path']
	if w2.shotID != '':
		shotID = int(w2.shotID)
		shotID += 1
		laser_to_analyse = laser_to_analyse[:-5] + str(shotID)
		if os.path.exists(laser_to_analyse+'.npz'):
			param_ext1['File Path'] = laser_to_analyse
			w2.shotID = str(shotID)
			print(laser_to_analyse)
			Load()
		else:
			date_found = (dates == os.path.split(os.path.split(laser_to_analyse)[0])[1]).argmax()
			neighbour_date = dates[min(len(dates)-1,date_found+1)]
			test_path = os.path.split(os.path.split(laser_to_analyse)[0])[0] + '/' + neighbour_date + '/' + os.path.split(laser_to_analyse)[1][:-5] + str(shotID)
			if os.path.exists(test_path + '.npz'):
				param_ext1['File Path'] = test_path
				w2.shotID = str(shotID)
				print(laser_to_analyse)
				Load()
			else:
				print('SHOT  ' + str(shotID) + '  MISSING')

def Select_specific_pulse():
	global all_time_x_point_location,all_time_mag_axis_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix,efit_reconstruction
	laser_to_analyse = param_ext1['File Path']
	if param_ext1['Shot number'] != 0:
		shotID = int(param_ext1['Shot number'])
		date = client.get_shot_date_time(shotID)[0]
		laser_to_analyse = '/home/ffederic/work/irvb/MAST-U/' + date + '/IRVB-MASTU_shot-' + str(shotID)
		if os.path.exists(laser_to_analyse+'.npz'):
			param_ext1['File Path'] = laser_to_analyse
			w2.shotID = str(shotID)
			print(laser_to_analyse)
			Load()
		else:
			print('SHOT  ' + str(shotID) + '  MISSING')


def Load_EFIT():
	global all_time_x_point_location,all_time_mag_axis_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix,efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine
	try:
		EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
		print('reading '+EFIT_path_default+'/epm0'+str(w2.shotID)+'.nc')
		efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+str(w2.shotID)+'.nc',w2.shotID)
		all_time_x_point_location = coleval.return_all_time_x_point_location(efit_reconstruction)
		all_time_mag_axis_location = coleval.return_all_time_mag_axis_location(efit_reconstruction)
		all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
		all_time_strike_points_location,all_time_strike_points_location_rot = coleval.return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
		all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
		param_ext['Overlays', 'Mag axis'] = True
		param_ext['Overlays', 'X-point'] = True
		param_ext['Overlays', 'Separatrix'] = True
		initialise_plots_from_EFIT()
	except Exception as e:
		print('reading EFIT of '+str(w2.shotID)+' failed')
		logging.exception('with error: ' + str(e))
		param_ext['Overlays', 'Mag axis'] = False
		param_ext['Overlays', 'X-point'] = False
		param_ext['Overlays', 'Separatrix'] = False


update_plot_inhibit = False
def Load():
	global data,framerate,time_array,overlay_fueling_point,overlay_structure,overlay_x_point,overlay_mag_axis,overlay_strike_points_1,overlay_strike_points_2,overlay_separatrix,overlay_Core_Resistive_bol,overlay_Div_Resistive_bol,data_shape,etendue,update_plot_inhibit,flag_radial_profile,efit_reconstruction,inversion_R,inversion_Z,dr,dz
	flag_radial_profile = False
	update_plot_inhibit = True
	print(param_ext1['File Path'])
	if param_ext['Set display','Quantity'] == 'counts':
		laser_dict = np.load(param_ext1['File Path']+'.npz')
		data = (laser_dict['data'] + laser_dict['data_median']).astype(int)
		laser_dict.allow_pickle=True
		framerate = laser_dict['FrameRate']
		param_ext['Overlays', 'Structure'] = False
		param_ext['Overlays', 'Mag axis'] = False
		param_ext['Overlays', 'X-point'] = False
		param_ext['Overlays', 'Separatrix'] = False
		param_ext['Overlays', 'Core Resistive bol'] = False
		param_ext['Overlays', 'Div Resistive bol'] = False
		# laser_dict = np.load(param_ext1['File Path']+'_short.npz')
		# laser_dict.allow_pickle=True
		# etendue = laser_dict['bin1x1x1'].all()['etendue']
		param_ext['Set display','Binning'] = 'bin1x1x1'
		try:
			temp_dict = np.load(param_ext1['File Path']+'_FAST.npz')
			time_array = temp_dict['time_full_full']
		except:
			time_array = laser_dict['full_frame'].all()['time_full_full']
		param_ext['ROI', 'Time end [ms]'] = time_array[np.abs(time_array-1.5).argmin()]*1e3
		param_ext['ROI', 'Time [ms]'] = time_array[np.abs(time_array-0.7).argmin()]*1e3
		param_ext['ROI', 'Time start [ms]'] = time_array[np.abs(time_array-0).argmin()]*1e3
	elif param_ext['Set display','Quantity'][:4] == 'FAST':
		if param_ext['Set display','Quantity'][:13] == 'FAST/inverted':
			to_load = param_ext['Set display','Quantity'][14:]
			laser_dict = np.load(param_ext1['File Path']+'_FAST.npz')
			laser_dict.allow_pickle=True
			data = laser_dict['inverted_dict'].all()[param_ext['Set display','Voxel res']][to_load]
			# data[np.isnan(data)] = 0
			time_array = laser_dict['inverted_dict'].all()[param_ext['Set display','Voxel res']]['time_full_binned_crop']
			param_ext['Set display','Binning'] = laser_dict['inverted_dict'].all()[param_ext['Set display','Voxel res']]['binning_type']
			inversion_R = laser_dict['inverted_dict'].all()[param_ext['Set display','Voxel res']]['geometry']['R']
			inversion_Z = laser_dict['inverted_dict'].all()[param_ext['Set display','Voxel res']]['geometry']['Z']
			dr = np.median(np.diff(inversion_R))
			dz = np.median(np.diff(inversion_Z))
			if to_load == 'inverted_data':
				# data = np.transpose(data,(0,2,1))
				flag_radial_profile = True
				data = np.flip(data,axis=1)
		else:
			if os.path.exists(param_ext1['File Path']+'_FAST.npz'):
				laser_dict = np.load(param_ext1['File Path']+'_FAST.npz')
				laser_dict.allow_pickle=True
				data = laser_dict[param_ext['Set display','Quantity']]
				time_array = laser_dict['FAST_time_binned']
				param_ext['Set display','Binning'] = laser_dict['FAST_binning_type']
			else:
				laser_dict = np.load(param_ext1['File Path']+'.npz')
				laser_dict.allow_pickle=True
				data = laser_dict['only_foil'].all()[param_ext['Set display','Quantity']]
				time_array = laser_dict['only_foil'].all()['FAST_time_binned']
				param_ext['Set display','Binning'] = laser_dict['only_foil'].all()['FAST_binning_type']
		framerate = 1/np.median(np.diff(time_array))
		param_ext['ROI', 'Time start [ms]'] = time_array[0]*1e3
		param_ext['ROI', 'Time [ms]'] = np.median(time_array)*1e3
		param_ext['ROI', 'Time end [ms]'] = time_array[-1]*1e3
	elif param_ext['Set display','Quantity'][:8] == 'inverted':
		to_load = param_ext['Set display','Quantity'][9:]
		laser_dict = np.load(param_ext1['File Path']+'_inverted_baiesian.npz')
		laser_dict.allow_pickle=True
		shrink_factor_t = param_ext['Set display','Binning'][3:param_ext['Set display','Binning'].find('x')]
		shrink_factor_x = param_ext['Set display','Binning'][param_ext['Set display','Binning'].find('x')+1:param_ext['Set display','Binning'].find('x')+1+param_ext['Set display','Binning'][param_ext['Set display','Binning'].find('x')+1:].find('x')]
		data = laser_dict[param_ext['Set display','Voxel res']].all()[str(shrink_factor_x)][str(shrink_factor_t)][to_load]
		# data[np.isnan(data)] = 0
		time_array = laser_dict[param_ext['Set display','Voxel res']].all()[str(shrink_factor_x)][str(shrink_factor_t)]['time_full_binned_crop']
		inversion_R = laser_dict[param_ext['Set display','Voxel res']].all()[str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['R']
		inversion_Z = laser_dict[param_ext['Set display','Voxel res']].all()[str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['Z']
		dr = np.median(np.diff(inversion_R))
		dz = np.median(np.diff(inversion_Z))
		if to_load == 'inverted_data':
			# data = np.transpose(data,(0,2,1))
			flag_radial_profile = True
			data = np.flip(data,axis=1)
		framerate = 1/np.median(np.diff(time_array))
		param_ext['ROI', 'Time start [ms]'] = time_array[0]*1e3
		param_ext['ROI', 'Time [ms]'] = np.median(time_array)*1e3
		param_ext['ROI', 'Time end [ms]'] = time_array[-1]*1e3
	else:
		laser_dict = np.load(param_ext1['File Path']+'_short.npz')
		laser_dict.allow_pickle=True
		data = laser_dict[param_ext['Set display','Binning']].all()[param_ext['Set display','Quantity']]
		# etendue = laser_dict[param_ext['Set display','Binning']].all()['etendue']
		time_array = laser_dict[param_ext['Set display','Binning']].all()['time_full_binned']
		framerate = laser_dict[param_ext['Set display','Binning']].all()['laser_framerate_binned']
		param_ext['ROI', 'Time start [ms]'] = time_array[0]*1e3
		param_ext['ROI', 'Time [ms]'] = np.median(time_array)*1e3
		param_ext['ROI', 'Time end [ms]'] = time_array[-1]*1e3
	data = np.flip(data,axis=1)
	data_shape = np.shape(data)
	temp = cp.deepcopy(data[0])
	temp[np.isnan(temp)]=0
	image1.setImage(temp)
	isoLine.setValue(np.mean(data[0]))
	if flag_radial_profile:
		image1.setRect(QtCore.QRectF(inversion_R.min()-dr/2,inversion_Z.min()-dz/2,len(inversion_R)*dr,len(inversion_Z)*dz))
		image1.setBorder('b')
		image_frame.setData([inversion_R.min()-dr/2,inversion_R.max()+dr/2,inversion_R.max()+dr/2,inversion_R.min()-dr/2,inversion_R.min()-dr/2],[inversion_Z.min()-dz/2,inversion_Z.min()-dz/2,inversion_Z.max()+dz/2,inversion_Z.max()+dz/2,inversion_Z.min()-dz/2])
		param_ext['Set display', 'Export left'],param_ext['Set display', 'Export down'] = inversion_R.min()-dr/2, inversion_Z.min()-dz/2
		param_ext['Set display', 'Export right'],param_ext['Set display', 'Export up'] = inversion_R.max()+dr/2, inversion_Z.max()+dz/2
		param_ext['ROI', 'ROI magenta hor'],param_ext['ROI', 'ROI magenta ver'] = inversion_R.min()+(inversion_R.max()-inversion_R.min())*4/8, inversion_Z.min()+(inversion_Z.max()-inversion_Z.min())*4/8
		param_ext['ROI', 'ROI magenta d hor'],param_ext['ROI', 'ROI magenta d ver'] = (inversion_R.max()-inversion_R.min())/4, (inversion_Z.max()-inversion_Z.min())/4
		param_ext['ROI', 'ROI cyan hor'],param_ext['ROI', 'ROI cyan ver'] = inversion_R.min()+(inversion_R.max()-inversion_R.min())*2/8, inversion_Z.min()+(inversion_Z.max()-inversion_Z.min())*2/8
		param_ext['ROI', 'ROI cyan d hor'],param_ext['ROI', 'ROI cyan d ver'] = (inversion_R.max()-inversion_R.min())/4, (inversion_Z.max()-inversion_Z.min())/4
	else:
		image_frame.setData([0,data_shape[1],data_shape[1],0,0],[0,0,data_shape[2],data_shape[2],0])
		param_ext['Set display', 'Export left'],param_ext['Set display', 'Export down'] = 0, 0
		param_ext['Set display', 'Export right'],param_ext['Set display', 'Export up'] = np.shape(image1.image)[0], np.shape(image1.image)[1]
		param_ext['ROI', 'ROI magenta hor'],param_ext['ROI', 'ROI magenta ver'] = np.shape(image1.image)[0]*4//8, np.shape(image1.image)[1]*4//8
		param_ext['ROI', 'ROI magenta d hor'],param_ext['ROI', 'ROI magenta d ver'] = np.shape(image1.image)[0]//4, np.shape(image1.image)[1]//4
		param_ext['ROI', 'ROI cyan hor'],param_ext['ROI', 'ROI cyan ver'] = np.shape(image1.image)[0]*2//8, np.shape(image1.image)[1]*2//8
		param_ext['ROI', 'ROI cyan d hor'],param_ext['ROI', 'ROI cyan d ver'] = np.shape(image1.image)[0]//4, np.shape(image1.image)[1]//4
	w2.minimum = 0
	w2.maximum = len(data)-1
	w2.slider.setMinimum(w2.minimum)
	w2.slider.setMaximum(w2.maximum)
	param_ext['ROI', 'Histogram auto'] = True
	param_ext.children()[0].children()[8].setProperty('step',np.min(np.diff(np.sort(data.flatten()))))
	param_ext.children()[0].children()[9].setProperty('step',np.min(np.diff(np.sort(data.flatten()))))
	try:
		try:
			for i in range(len(overlay_fueling_point)):
				overlay_fueling_point[i].setAlpha(0,False)
				try:
					overlay_fueling_point[i].setData([],[])
				except:
					pass
			del overlay_fueling_point
		except:
			pass
		for i in range(len(overlay_structure)):
			overlay_structure[i].setAlpha(0,False)
			overlay_structure[i].setData([],[])
		del overlay_structure
		try:
			overlay_x_point.setAlpha(0,False)
			overlay_x_point.setData([],[])
			del overlay_x_point
		except:
			pass
		try:
			overlay_mag_axis.setAlpha(0,False)
			overlay_mag_axis.setData([],[])
			del overlay_mag_axis
		except:
			pass
		print('del overlay_mag_axis')
		try:
			overlay_strike_points_1.setAlpha(0,False)
			overlay_strike_points_1.setData([],[])
			del overlay_strike_points_1
		except:
			pass
		try:
			try:
				for i in range(len(overlay_strike_points_2)):
					overlay_strike_points_2[i].setAlpha(0,False)
					overlay_strike_points_2[i].setData([],[])
				del overlay_strike_points_2
			except:
				overlay_strike_points_2.setAlpha(0,False)
				overlay_strike_points_2.setData([],[])
				del overlay_strike_points_2
		except:
			pass
		try:
			for i in range(len(overlay_separatrix)):
				overlay_separatrix[i].setAlpha(0,False)
				overlay_separatrix[i].setData([],[])
			del overlay_separatrix
		except:
			pass
		print('del overlay_separatrix')
		try:
			for i in range(len(overlay_Core_Resistive_bol)):
				overlay_Core_Resistive_bol[i].setAlpha(0,False)
				overlay_Core_Resistive_bol[i].setData([],[])
			del overlay_Core_Resistive_bol
		except:
			pass
		try:
			for i in range(len(overlay_Div_Resistive_bol)):
				overlay_Div_Resistive_bol[i].setAlpha(0,False)
				overlay_Div_Resistive_bol[i].setData([],[])
			del overlay_Div_Resistive_bol
		except:
			pass
	except Exception as e:
		print('no legacy plots Load')
		logging.exception('with error: ' + str(e))
	if not flag_radial_profile:
		overlay_fueling_point = []
		for i in range(len(fueling_point_location_on_foil)):
			overlay_fueling_point.append(p5.plot(np.array(fueling_point_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],symbolBrush='g',symbolPen='g',symbol='+',symbolSize=20))
			overlay_fueling_point.append(p5.plot(np.array(fueling_point_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],symbolBrush='g',symbolPen='g',symbol='o',symbolSize=5))
	overlay_structure = []
	if flag_radial_profile:
		for i in range(len(structure_radial_profile)):
			overlay_structure.append(p5.plot(structure_radial_profile[i][:,0],structure_radial_profile[i][:,1],pen='g'))
	else:
		for i in range(len(structure_point_location_on_foil)):
			overlay_structure.append(p5.plot(np.array(structure_point_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],pen='g'))
	overlay_Core_Resistive_bol = []
	if  flag_radial_profile:
		for i in range(len(core_poloidal)):
			overlay_Core_Resistive_bol.append(p5.plot(core_poloidal[i][:,0],core_poloidal[i][:,1],pen='r'))
	else:
		for i in range(len(core_tangential_location_on_foil)):
			overlay_Core_Resistive_bol.append(p5.plot(np.array(core_tangential_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(core_tangential_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],pen='r'))
		for i in range(len(core_poloidal_location_on_foil)):
			overlay_Core_Resistive_bol.append(p5.plot(np.array(core_poloidal_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(core_poloidal_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],pen='r'))
	overlay_Div_Resistive_bol = []
	if  flag_radial_profile:
		for i in range(len(divertor_poloidal)):
			overlay_Div_Resistive_bol.append(p5.plot(divertor_poloidal[i][:,0],divertor_poloidal[i][:,1],pen='r'))
	else:
		for i in range(len(divertor_poloidal_location_on_foil)):
			overlay_Div_Resistive_bol.append(p5.plot(np.array(divertor_poloidal_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(divertor_poloidal_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],pen='r'))
	initialise_plots_from_EFIT()
	update_plot_inhibit = False
	update_plot()

def initialise_plots_from_EFIT():
	global overlay_x_point,overlay_mag_axis,overlay_strike_points_1,overlay_strike_points_2,overlay_strike_points_2,overlay_separatrix,all_time_strike_points_location_rot,all_time_separatrix,flag_radial_profile,all_time_sep_r,all_time_sep_z,r_fine,z_fine
	try:
		if flag_radial_profile:
			overlay_x_point = p5.plot([0],[0],symbolBrush='r',symbolPen='r',symbol='x',symbolSize=20,pen=None)
			overlay_mag_axis = p5.plot([0],[0],symbolBrush='r',symbolPen='r',symbol='x',symbolSize=20,pen=None)
			overlay_strike_points_2 = p5.plot([],[],symbolBrush='y',symbolPen='y',symbol='x',symbolSize=20,pen=None)
			overlay_separatrix = []
			for __i in range(len(all_time_sep_r[0])):
				overlay_separatrix.append(p5.plot([0],[0],pen='b'))
		else:
			overlay_x_point = p5.plot([0],[0],pen='r')
			overlay_mag_axis = p5.plot([0],[0],pen='r')
			overlay_strike_points_2 = []
			for __i in range(len(all_time_strike_points_location_rot[0])):
				overlay_strike_points_2.append(p5.plot([0],[0],pen='y'))
			overlay_separatrix = []
			for __i in range(len(all_time_separatrix[0])):
				overlay_separatrix.append(p5.plot([0],[0],pen='b'))
		overlay_strike_points_1 = p5.plot([0],[0],symbolBrush='y',symbolPen='y',symbol='x',symbolSize=20,pen=None)
	except:
		print('no EFIT loaded at initialise_plots_from_EFIT')

def reset_EFIT():
	global all_time_strike_points_location_rot,all_time_separatrix,overlay_x_point,overlay_mag_axis,efit_reconstruction
	try:
		del all_time_strike_points_location_rot
	except:
		pass
	try:
		del all_time_separatrix
	except:
		pass
	try:
		del overlay_x_point
	except:
		pass
	try:
		del overlay_mag_axis
	except:
		pass
	try:
		del efit_reconstruction
	except:
		pass
	efit_reconstruction=None


	# param_ext['Set display', 'File Path'] = anchive_path
param_ext1.param('Navigate Path').sigActivated.connect(NavigatePath)
param_ext1.param('File Path').sigValueChanged.connect(reset_EFIT)
param_ext.param('Set display','Load data').sigActivated.connect(Load)
param_ext.param('Set display','Load EFIT').sigActivated.connect(Load_EFIT)
param_ext.param('Set display','Shot +').sigActivated.connect(Select_next_pulse)
param_ext.param('Set display','Shot -').sigActivated.connect(Select_previous_pulse)
param_ext1.param('Shot number').sigValueChanged.connect(Select_specific_pulse)


update_hist_inhibit = False
def update_plot():
	global histogram_level_low, histogram_level_high,update_hist_inhibit,update_plot_inhibit,inhibit_update_like_video,inhibit_update_like_video_rew,flag_radial_profile,inversion_R,inversion_Z,data_shape
	if update_plot_inhibit==False:
		a = w2.x
		if (param_ext['ROI', 'Time [ms]']!=time_array[round(a)]*1e3) or inhibit_update_like_video==False or inhibit_update_like_video_rew==False:
			update_hist_inhibit = True
			temp = cp.deepcopy(data[round(a)])
			temp[np.isnan(temp)]=0
			image1.setImage(temp)
			# image1.setImage(data[round(a)])
			hist.setImageItem(image1)
			hist.setLevels(histogram_level_high,histogram_level_low)
		# self.image2.setImage(data[round(a)])
		iso.setLevel(isoLine.value())
		iso.setData(pg.gaussianFilter(image1.image, (2, 2)))
		if inhibit_update_like_video or inhibit_update_like_video_rew:
			param_ext['ROI', 'Time [ms]'] = time_array[round(a)]*1e3
		image_frame_pos = np.array(image_frame_roi.getState()['pos'])
		image_frame_size = np.array(image_frame_roi.getState()['size'])
		param_ext['Set display', 'Export left'] = image_frame_pos[0]
		param_ext['Set display', 'Export down'] = image_frame_pos[1]
		param_ext['Set display', 'Export right'] = image_frame_pos[0] + image_frame_size[0]
		param_ext['Set display', 'Export up'] = image_frame_pos[1] + image_frame_size[1]
		if not flag_radial_profile:
			selected = roi.getArrayRegion(image1.image, image1)
			pos = np.array(roi.getState()['pos']).astype(int)
			size = np.array(np.shape(selected))
			# indexes_horiz_axis = roi.getArrayRegion(image1.image, image1,returnMappedCoords=True)[1].astype(int)[0].flatten()
			# indexes_vert_axis = roi.getArrayRegion(image1.image, image1,returnMappedCoords=True)[1].astype(int)[1].flatten()
			param_ext['ROI', 'ROI magenta hor'] = pos[0]
			param_ext['ROI', 'ROI magenta ver'] = pos[1]
			param_ext['ROI', 'ROI magenta d hor'] = size[0]
			param_ext['ROI', 'ROI magenta d ver'] = size[1]
			selected2 = roi2.getArrayRegion(image1.image, image1)
			pos2 = np.array(roi2.getState()['pos']).astype(int)
			size2 = np.array(np.shape(selected2))
			param_ext['ROI', 'ROI cyan hor'] = pos2[0]
			param_ext['ROI', 'ROI cyan ver'] = pos2[1]
			param_ext['ROI', 'ROI cyan d hor'] = size2[0]
			param_ext['ROI', 'ROI cyan d ver'] = size2[1]
			param_ext['ROI', 'Time start [ms]'] = time_array[int(w2.left)]*1e3
			param_ext['ROI', 'Time end [ms]'] = time_array[int(w2.right)]*1e3
			p3.plot(np.arange(pos[0],pos[0]+size[0]),np.nanmean(selected,axis=1), clear=True)
			p1.plot(np.nanmean(selected,axis=0),np.arange(pos[1],pos[1]+size[1]), clear=True)
			temp = np.nanmean(data[int(w2.left):int(w2.right),max(0,pos[0]):pos[0]+size[0],max(0,pos[1]):pos[1]+size[1]].astype(float),axis=(1,2))
			# temp = np.nanmean(data[int(w2.left):int(w2.right),np.array([indexes_horiz_axis,indexes_vert_axis]).tolist()],axis=(1,2))
			# p6.plot(time_array[int(w2.left):int(w2.right)],temp, clear=True)
			# p6.plot([time_array[round(a)]]*2,[np.nanmin(temp),np.nanmax(temp)], pen='g')
			p6_time_mark.setData([time_array[round(a)]]*2,[np.nanmin(temp),np.nanmax(temp)])
			temp = np.nanmean(data[int(w2.left):int(w2.right),max(0,pos[0]):pos[0]+size[0],max(0,pos[1]):pos[1]+size[1]].astype(float),axis=(1,2))
			p6_1.setData(time_array[int(w2.left):int(w2.right)],temp)
			temp = np.nanmean(data[int(w2.left):int(w2.right),max(0,pos2[0]):pos2[0]+size2[0],max(0,pos2[1]):pos2[1]+size2[1]].astype(float),axis=(1,2))
			p6_2.setData(time_array[int(w2.left):int(w2.right)],temp)
			# hist.setImageItem(image1)
			# hist.sigLevelChangeFinished.emit(True)
	if flag_radial_profile:
		full_range_hor = np.sort((inversion_R-dr/2).tolist()+[inversion_R.max()+dr/2])
		full_range_ver = np.sort((inversion_Z-dz/2).tolist()+[inversion_Z.max()+dz/2])
	else:
		full_range_hor = np.arange(0,data_shape[1]+1)
		full_range_ver = np.arange(0,data_shape[2]+1)
		# param_ext['Set display','Export left'] = int(param_ext['Set display','Export left'])
		# param_ext['Set display','Export right'] = int(param_ext['Set display','Export right'])
		# param_ext['Set display','Export down'] = int(param_ext['Set display','Export down'])
		# param_ext['Set display','Export up'] = int(param_ext['Set display','Export up'])
	limit_left = np.abs(param_ext['Set display','Export left'] - full_range_hor).argmin()
	limit_right = np.abs(param_ext['Set display','Export right'] - full_range_hor).argmin()
	limit_bottom = np.abs(param_ext['Set display','Export down'] - full_range_ver).argmin()
	limit_top = np.abs(param_ext['Set display','Export up'] - full_range_ver).argmin()
	image_frame_export.setData([full_range_hor[limit_left],full_range_hor[limit_right],full_range_hor[limit_right],full_range_hor[limit_left],full_range_hor[limit_left]],[full_range_ver[limit_bottom],full_range_ver[limit_bottom],full_range_ver[limit_top],full_range_ver[limit_top],full_range_ver[limit_bottom]])


update_plot()

def update_hist():
	global histogram_level_low, histogram_level_high,update_hist_inhibit
	if update_hist_inhibit==False:
		histogram_level_low,histogram_level_high = hist.getLevels()
	update_hist_inhibit=False
	if param_ext['ROI', 'Histogram auto'] == True:
		histogram_level_high = np.max(image1.image)
		histogram_level_low = np.min(image1.image)
	hist.setLevels(histogram_level_high,histogram_level_low)
	param_ext['ROI', 'Hist lev high'] = histogram_level_high
	param_ext['ROI', 'Hist lev low'] = histogram_level_low

w2.slider.valueChanged.connect(update_plot)
isoLine.sigDragged.connect(update_plot)
roi.sigRegionChanged.connect(update_plot)
roi2.sigRegionChanged.connect(update_plot)
image_frame_roi.sigRegionChanged.connect(update_plot)
hist.sigLevelsChanged.connect(update_hist)

param_ext.sigTreeStateChanged.connect(change)

## Create two ParameterTree widgets, both accessing the same data
t = ParameterTree()
t.setParameters(param_ext, showTop=False)
t.setWindowTitle('pyqtgraph example: Parameter Tree')
d4.addWidget(t)
t1 = ParameterTree(showHeader=False)
t1.setParameters(param_ext1, showTop=False)
t1.setWindowTitle('pyqtgraph example: Parameter Tree')
d7.addWidget(t1)


win.show()
app.exec_()
