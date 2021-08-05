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
# data = np.load('/home/ffederic/work/irvb/MAST-U/2021-07-01/IRVB-MASTU_shot-44391.npz')['data']
framerate = 383 # Hz
shot_ID = '043979'
time_aray = np.arange(len(data))*1/framerate

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
		self.label.setText('shot '+str(self.shotID)+' from%.4g to %.4g ms : %.4g ms' %(1e3*time_aray[int(self.left)],1e3*time_aray[int(self.right)],1e3*time_aray[int(self.x)]))


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
w2 = RangeSlider(0, len(time_aray)-1)
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
roi.addScaleHandle([1, 0], [0.5, 0.5])
# roi.addRotateHandle([0, 1], [0.5, 0.5])  # works but better not use it
p5.addItem(roi)
roi.setZValue(10)  # make sure ROI is drawn above image
roi.setPen(pen='m')
roi2 = pg.ROI([np.shape(image1.image)[0]*1//8, np.shape(image1.image)[1]*1//8], [np.shape(image1.image)[0]//4, np.shape(image1.image)[1]//4])
# roi.addScaleHandle([0.5, 1], [0.5, 0.5])
# roi.addScaleHandle([0, 0.5], [0.5, 0.5])
roi2.addScaleHandle([1, 0], [0.5, 0.5])
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
foil_size = [0.07,0.09]

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





# creating all the flags and accessories needed
params = [
	{'name': 'ROI', 'type': 'group', 'children': [
		{'name': 'ROI magenta hor', 'type': 'int', 'value': 10},
		{'name': 'ROI magenta d hor', 'type': 'int', 'value': 10},
		{'name': 'ROI magenta ver', 'type': 'int', 'value': 10},
		{'name': 'ROI magenta d ver', 'type': 'int', 'value': 10},
		# {'name': 'ROI magenta angle', 'type': 'int', 'value': 0},
		{'name': 'ROI cyan hor', 'type': 'int', 'value': 10},
		{'name': 'ROI cyan d hor', 'type': 'int', 'value': 10},
		{'name': 'ROI cyan ver', 'type': 'int', 'value': 10},
		{'name': 'ROI cyan d ver', 'type': 'int', 'value': 10},
		# {'name': 'ROI cyan angle', 'type': 'int', 'value': 0},
		{'name': 'Time start [ms]', 'type': 'float', 'value': time_aray[int(w2.left)]*1e3, 'step': 1/framerate*1e3, 'finite': False},
		{'name': 'Time end [ms]', 'type': 'float', 'value': time_aray[int(w2.right)]*1e3, 'step': 1/framerate*1e3, 'finite': False},
		{'name': 'Time [ms]', 'type': 'float', 'value': time_aray[int(w2.x)]*1e3, 'step': 1/framerate*1e3, 'finite': False},
		{'name': 'Histogram auto', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
		{'name': 'Hist lev high', 'type': 'float', 'value': histogram_level_high, 'step': np.min(np.diff(np.sort(data.flatten()))), 'finite': False},
		{'name': 'Hist lev low', 'type': 'float', 'value': histogram_level_low, 'step': np.min(np.diff(np.sort(data.flatten()))), 'finite': False},
	]},
	{'name': 'Set display', 'type': 'group', 'children': [
		{'name': 'Quantity', 'type': 'list', 'values': {'counts': 'counts', 'FAST counts': 'FAST counts', "tepmerature": 'laser_temperature_crop_binned_full', "rel temp": 'laser_temperature_minus_background_crop_binned_full', "tot power": 'powernoback_full', "BB power": 'BBrad_full', "diff power": 'diffusion_full', "dt power": 'timevariation_full', "brightness": 'brightness_full'}, 'value': 'FAST counts'},
		{'name': 'Binning', 'type': 'list', 'values': {'bin1x1x1': 'bin1x1x1', 'bin1x3x3': 'bin1x3x3', 'bin1x5x5': 'bin1x5x5', "bin1x10x10": 'bin1x10x10', "bin2x1x1": 'bin2x1x1', "bin2x3x3": 'bin2x3x3', "bin2x5x5": 'bin2x5x5', "bin2x10x10": 'bin2x10x10', "bin3x1x1": 'bin3x1x1', "bin3x3x3": 'bin3x3x3', "bin3x5x5": 'bin3x5x5', "bin3x10x10": 'bin3x10x10'}, 'value': 1},
		{'name': 'Load data', 'type': 'action'},
		{'name': 'Load EFIT', 'type': 'action'},
		{'name': 'Play', 'type': 'action'},
		{'name': 'Rewind', 'type': 'action'},
		{'name': 'Pause', 'type': 'action'},
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
		{'name': 'File Path', 'type': 'str', 'value': "insert full path", 'tip': 'try'},
		{'name': 'Navigate Path', 'type': 'action'},
]

## Create tree of Parameter objects
param_ext = Parameter.create(name='params', type='group', children=params)
param_ext1 = Parameter.create(name='params', type='group', children=params1)

## If anything changes in the tree, print a message
def change(param, changes):
	global histogram_level_low,histogram_level_high,overlay_structure,overlay_fueling_point,overlay_x_point,overlay_mag_axis,overlay_strike_points_1,overlay_strike_points_2,overlay_separatrix,overlay_Core_Resistive_bol,overlay_Div_Resistive_bol,efit_reconstruction,all_time_mag_axis_location,all_time_x_point_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix
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
	# roi2.setAngle((param_ext['ROI', 'ROI cyan angle']))
	w2.slider.setValue([np.abs(param_ext['ROI', 'Time start [ms]']*1e-3-time_aray).argmin()/(w2.maximum-w2.minimum)*(w2.slider.maximum()-w2.slider.minimum()),np.abs(param_ext['ROI', 'Time [ms]']*1e-3-time_aray).argmin()/(w2.maximum-w2.minimum)*(w2.slider.maximum()-w2.slider.minimum()),np.abs(param_ext['ROI', 'Time end [ms]']*1e-3-time_aray).argmin()/(w2.maximum-w2.minimum)*(w2.slider.maximum()-w2.slider.minimum())])
	histogram_level_high,histogram_level_low = param_ext['ROI', 'Hist lev high'],param_ext['ROI', 'Hist lev low']
	hist.setLevels(histogram_level_high,histogram_level_low)
	# hist.sigLevelChangeFinished.emit(True)
	if param_ext['Overlays','Structure']==False:
		for i in range(len(overlay_fueling_point)):
			overlay_fueling_point[i].setAlpha(0,False)
		for i in range(len(overlay_structure)):
			overlay_structure[i].setAlpha(0,False)
	else:
		for i in range(len(overlay_fueling_point)):
			overlay_fueling_point[i].setAlpha(0.5,False)
		for i in range(len(overlay_structure)):
			overlay_structure[i].setAlpha(0.5,False)
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
			i_time = np.abs(time_aray[int(w2.x)]-efit_reconstruction.time).argmin()
		if param_ext['Overlays','Mag axis']==False:
			overlay_mag_axis.setAlpha(0,False)
		else:
			overlay_mag_axis.setData(all_time_mag_axis_location[i_time][:,0]*(data_shape[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(data_shape[2]-1)/foil_size[1])
			overlay_mag_axis.setAlpha(1,False)
		if param_ext['Overlays','X-point']==False:
			overlay_x_point.setAlpha(0,False)
		else:
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
	global inhibit_update_like_video,inhibit_update_like_video_rew
	if inhibit_update_like_video == False:
		inhibit_update_like_video_rew = True
		print('update_like_video')
		now = np.abs(param_ext['ROI', 'Time [ms]']-time_aray*1e3).argmin()
		if now == len(time_aray)-1:
			now = -1
		future = now+1
		param_ext['ROI', 'Time [ms]'] = time_aray[future]*1e3
		# now = w2.x
		# if now == len(time_aray)-1:
		# 	now = -1
		# future = now+1
		# w2.x = future
		print('update_plot_inhibit '+str(update_plot_inhibit))
		update_plot()
	elif inhibit_update_like_video_rew == False:
		inhibit_update_like_video = True
		print('update_like_video')
		now = np.abs(param_ext['ROI', 'Time [ms]']-time_aray*1e3).argmin()
		if now == 0:
			now = len(time_aray)
		future = now-1
		param_ext['ROI', 'Time [ms]'] = time_aray[future]*1e3
		# now = w2.x
		# if now == len(time_aray)-1:
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

param_ext.param('Set display','Play').sigActivated.connect(start_video)
param_ext.param('Set display','Rewind').sigActivated.connect(rew_video)
param_ext.param('Set display','Pause').sigActivated.connect(pause_video)


def NavigatePath():
	global all_time_x_point_location,all_time_mag_axis_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix,efit_reconstruction
	dialog = QtGui.QFileDialog()
	laser_to_analyse=dialog.getOpenFileName(caption="Open the short archive", filter="Archive file (*.npz)", directory="/home/ffederic/work/irvb/MAST-U")[0][:-4]
	if laser_to_analyse[-5:] == 'short':
		laser_to_analyse = laser_to_analyse[:-6]
	print(laser_to_analyse)
	param_ext1['File Path'] = laser_to_analyse
	w2.shotID = laser_to_analyse[-5:]
	# p6.setTitle(title="shot " + str(w2.shotID))
	Load_EFIT()
	Load()

def Load_EFIT():
	global all_time_x_point_location,all_time_mag_axis_location,all_time_strike_points_location,all_time_strike_points_location_rot,all_time_separatrix,efit_reconstruction
	try:
		EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
		print('reading '+EFIT_path_default+'/epm0'+str(w2.shotID)+'.nc')
		efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+str(w2.shotID)+'.nc')
		all_time_x_point_location = coleval.return_all_time_x_point_location(efit_reconstruction)
		all_time_mag_axis_location = coleval.return_all_time_mag_axis_location(efit_reconstruction)
		all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
		all_time_strike_points_location,all_time_strike_points_location_rot = coleval.return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
		all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
		param_ext['Overlays', 'Mag axis'] = True
		param_ext['Overlays', 'X-point'] = True
		param_ext['Overlays', 'Separatrix'] = True
	except Exception as e:
		print('reading EFIT of '+str(w2.shotID)+' failed')
		logging.exception('with error: ' + str(e))
		param_ext['Overlays', 'Mag axis'] = False
		param_ext['Overlays', 'X-point'] = False
		param_ext['Overlays', 'Separatrix'] = False


update_plot_inhibit = False
def Load():
	global data,framerate,time_aray,overlay_fueling_point,overlay_structure,overlay_x_point,overlay_mag_axis,overlay_strike_points_1,overlay_strike_points_2,overlay_separatrix,overlay_Core_Resistive_bol,overlay_Div_Resistive_bol,data_shape,etendue,update_plot_inhibit
	update_plot_inhibit = True
	print(param_ext1['File Path'])
	if param_ext['Set display','Quantity'] == 'counts':
		laser_dict = np.load(param_ext1['File Path']+'.npz')
		data = (laser_dict['data'] + laser_dict['data_median']).astype(int)
		laser_dict.allow_pickle=True
		time_aray = laser_dict['full_frame'].all()['time_full_full']
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
	elif param_ext['Set display','Quantity'] == 'FAST counts':
		laser_dict = np.load(param_ext1['File Path']+'.npz')
		laser_dict.allow_pickle=True
		data = laser_dict['only_foil'].all()['FAST_counts_minus_background_crop']
		time_aray = laser_dict['only_foil'].all()['FAST_counts_minus_background_crop_time']
		framerate = laser_dict['FrameRate']/len(np.unique(laser_dict['digitizer_ID']))
		# laser_dict = np.load(param_ext1['File Path']+'_short.npz')
		# laser_dict.allow_pickle=True
		# etendue = laser_dict['bin1x1x1'].all()['etendue']
		param_ext['Set display','Binning'] = 'bin1x1x1'
	else:
		laser_dict = np.load(param_ext1['File Path']+'_short.npz')
		laser_dict.allow_pickle=True
		data = laser_dict[param_ext['Set display','Binning']].all()[param_ext['Set display','Quantity']]
		# etendue = laser_dict[param_ext['Set display','Binning']].all()['etendue']
		time_aray = laser_dict[param_ext['Set display','Binning']].all()['time_full_binned']
		framerate = laser_dict[param_ext['Set display','Binning']].all()['laser_framerate_binned']
	data = np.flip(data,axis=1)
	image1.setImage(data[0])
	isoLine.setValue(np.mean(data[0]))
	param_ext['ROI', 'ROI magenta hor'],param_ext['ROI', 'ROI magenta ver'] = np.shape(image1.image)[0]*3//8, np.shape(image1.image)[1]*3//8
	param_ext['ROI', 'ROI magenta d hor'],param_ext['ROI', 'ROI magenta d ver'] = np.shape(image1.image)[0]//4, np.shape(image1.image)[1]//4
	param_ext['ROI', 'ROI cyan hor'],param_ext['ROI', 'ROI cyan ver'] = np.shape(image1.image)[0]*1//8, np.shape(image1.image)[1]*1//8
	param_ext['ROI', 'ROI cyan d hor'],param_ext['ROI', 'ROI cyan d ver'] = np.shape(image1.image)[0]//4, np.shape(image1.image)[1]//4
	w2.minimum = 0
	w2.maximum = len(data)-1
	w2.slider.setMinimum(w2.minimum)
	w2.slider.setMaximum(w2.maximum)
	param_ext['ROI', 'Time start [ms]'] = time_aray[0]*1e3
	param_ext['ROI', 'Time [ms]'] = np.mean(time_aray)*1e3
	param_ext['ROI', 'Time end [ms]'] = time_aray[-1]*1e3
	param_ext['ROI', 'Histogram auto'] = True
	param_ext.children()[0].children()[8].setProperty('step',np.min(np.diff(np.sort(data.flatten()))))
	param_ext.children()[0].children()[9].setProperty('step',np.min(np.diff(np.sort(data.flatten()))))
	data_shape = np.shape(data)
	try:
		for i in range(len(overlay_fueling_point)):
			overlay_fueling_point[i].setAlpha(0,False)
			overlay_fueling_point[i].setData([],[])
		del overlay_fueling_point
		for i in range(len(overlay_structure)):
			overlay_structure[i].setAlpha(0,False)
			overlay_structure[i].setData([],[])
		del overlay_structure
		overlay_x_point.setAlpha(0,False)
		overlay_x_point.setData([],[])
		del overlay_x_point
		overlay_mag_axis.setAlpha(0,False)
		overlay_mag_axis.setData([],[])
		del overlay_mag_axis
		overlay_strike_points_1.setAlpha(0,False)
		overlay_strike_points_1.setData([],[])
		del overlay_strike_points_1
		for i in range(len(overlay_strike_points_2)):
			overlay_strike_points_2[i].setAlpha(0,False)
			overlay_strike_points_2[i].setData([],[])
		del overlay_strike_points_2
		for i in range(len(overlay_separatrix)):
			overlay_separatrix[i].setAlpha(0,False)
			overlay_separatrix[i].setData([],[])
		del overlay_separatrix
		for i in range(len(overlay_Core_Resistive_bol)):
			overlay_Core_Resistive_bol[i].setAlpha(0,False)
			overlay_Core_Resistive_bol[i].setData([],[])
		del overlay_Core_Resistive_bol
		for i in range(len(overlay_Div_Resistive_bol)):
			overlay_Div_Resistive_bol[i].setAlpha(0,False)
			overlay_Div_Resistive_bol[i].setData([],[])
		del overlay_Div_Resistive_bol
	except:
		print('no legacy plots')
	overlay_fueling_point = []
	for i in range(len(fueling_point_location_on_foil)):
		overlay_fueling_point.append(p5.plot(np.array(fueling_point_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],symbolBrush='g',symbolPen='g',symbol='+',symbolSize=20))
		overlay_fueling_point.append(p5.plot(np.array(fueling_point_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],symbolBrush='g',symbolPen='g',symbol='o',symbolSize=5))
	overlay_structure = []
	for i in range(len(structure_point_location_on_foil)):
		overlay_structure.append(p5.plot(np.array(structure_point_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],pen='g'))
	overlay_Core_Resistive_bol = []
	for i in range(len(core_tangential_location_on_foil)):
		overlay_Core_Resistive_bol.append(p5.plot(np.array(core_tangential_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(core_tangential_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],pen='r'))
	for i in range(len(core_poloidal_location_on_foil)):
		overlay_Core_Resistive_bol.append(p5.plot(np.array(core_poloidal_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(core_poloidal_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],pen='r'))
	overlay_Div_Resistive_bol = []
	for i in range(len(divertor_poloidal_location_on_foil)):
		overlay_Div_Resistive_bol.append(p5.plot(np.array(divertor_poloidal_location_on_foil[i][:,0])*(data_shape[1]-1)/foil_size[0],np.array(divertor_poloidal_location_on_foil[i][:,1])*(data_shape[2]-1)/foil_size[1],pen='r'))
	try:
		overlay_x_point = p5.plot([0],[0],pen='r')
		overlay_mag_axis = p5.plot([0],[0],pen='r')
		overlay_strike_points_1 = p5.plot([0],[0],symbolBrush='y',symbolPen='y',symbol='x',symbolSize=20,pen=None)
		overlay_strike_points_2 = []
		for __i in range(len(all_time_strike_points_location_rot[0])):
			overlay_strike_points_2.append(p5.plot([0],[0],pen='y'))
		overlay_separatrix = []
		for __i in range(len(all_time_separatrix[0])):
			overlay_separatrix.append(p5.plot([0],[0],pen='b'))
	except:
		print('no EFIT loaded')
	update_plot_inhibit = False
	update_plot()


	# param_ext['Set display', 'File Path'] = anchive_path
param_ext1.param('Navigate Path').sigActivated.connect(NavigatePath)
param_ext.param('Set display','Load data').sigActivated.connect(Load)
param_ext.param('Set display','Load EFIT').sigActivated.connect(Load_EFIT)

update_hist_inhibit = False
def update_plot():
	global histogram_level_low, histogram_level_high,update_hist_inhibit,update_plot_inhibit,inhibit_update_like_video,inhibit_update_like_video_rew
	if update_plot_inhibit==False:
		a = w2.x
		if (param_ext['ROI', 'Time [ms]']!=time_aray[int(a)]*1e3) or inhibit_update_like_video==False or inhibit_update_like_video_rew==False:
			update_hist_inhibit = True
			image1.setImage(data[int(a)])
			hist.setImageItem(image1)
			hist.setLevels(histogram_level_high,histogram_level_low)
		# self.image2.setImage(data[int(a)])
		iso.setLevel(isoLine.value())
		iso.setData(pg.gaussianFilter(image1.image, (2, 2)))
		selected = roi.getArrayRegion(image1.image, image1)
		pos = np.array(roi.getState()['pos']).astype(int)
		size = np.array(np.shape(selected))
		indexes_horiz_axis = roi.getArrayRegion(image1.image, image1,returnMappedCoords=True)[1].astype(int)[0].flatten()
		indexes_vert_axis = roi.getArrayRegion(image1.image, image1,returnMappedCoords=True)[1].astype(int)[1].flatten()
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
		param_ext['ROI', 'Time start [ms]'] = time_aray[int(w2.left)]*1e3
		param_ext['ROI', 'Time end [ms]'] = time_aray[int(w2.right)]*1e3
		if inhibit_update_like_video or inhibit_update_like_video_rew:
			param_ext['ROI', 'Time [ms]'] = time_aray[int(a)]*1e3
		p3.plot(np.arange(pos[0],pos[0]+size[0]),np.nanmean(selected,axis=1), clear=True)
		p1.plot(np.nanmean(selected,axis=0),np.arange(pos[1],pos[1]+size[1]), clear=True)
		temp = np.nanmean(data[int(w2.left):int(w2.right),max(0,pos[0]):pos[0]+size[0],max(0,pos[1]):pos[1]+size[1]],axis=(1,2))
		# temp = np.nanmean(data[int(w2.left):int(w2.right),np.array([indexes_horiz_axis,indexes_vert_axis]).tolist()],axis=(1,2))
		# p6.plot(time_aray[int(w2.left):int(w2.right)],temp, clear=True)
		# p6.plot([time_aray[int(a)]]*2,[np.nanmin(temp),np.nanmax(temp)], pen='g')
		p6_time_mark.setData([time_aray[int(a)]]*2,[np.nanmin(temp),np.nanmax(temp)])
		temp = np.nanmean(data[int(w2.left):int(w2.right),max(0,pos[0]):pos[0]+size[0],max(0,pos[1]):pos[1]+size[1]],axis=(1,2))
		p6_1.setData(time_aray[int(w2.left):int(w2.right)],temp)
		temp = np.nanmean(data[int(w2.left):int(w2.right),max(0,pos2[0]):pos2[0]+size2[0],max(0,pos2[1]):pos2[1]+size2[1]],axis=(1,2))
		p6_2.setData(time_aray[int(w2.left):int(w2.right)],temp)
		# hist.setImageItem(image1)
		# hist.sigLevelChangeFinished.emit(True)

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
