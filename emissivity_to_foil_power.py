# Created 10/12/2018
# Fabio Federici

import inspect
from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
print(inspect.getfile(Point2D))
from cherab.core import Species, Maxwellian, Plasma, Line
print(inspect.getfile(Plasma))
from raysect.optical import Ray,Spectrum,World
print(inspect.getfile(Ray))


# FIRST!
# move into the proper virtual environment
# source /home/ffederic/venvs/cherab_20230727/bin/activate

#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

# just to import _MASTU_CORE_GRID_POLYGON
calculate_tangency_angle_for_poloidal_section=coleval.calculate_tangency_angle_for_poloidal_section
exec(open("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval/collect_and_eval/MASTU_structure.py").read())





if True:	# related to the SOLPS phantom and calculating the spectra produced by the plasma in one IRVB LOS
	# seeding scan
	# SOLPS_case = 'seed_1'
	# SOLPS_case = 'seed_5'
	# SOLPS_case = 'seed_10'
	# mastu_path = "/home/ffederic/work/SOLPS/seeding/" + SOLPS_case
	# fuelling scan
	# SOLPS_case = 'ramp_1'
	# SOLPS_case = 'ramp_3.3'
	SOLPS_case = 'ramp_11'
	mastu_path = "/home/ffederic/work/SOLPS/dscan/" + SOLPS_case
	if not mastu_path in sys.path:
		sys.path.append(mastu_path)
	ds_puff8 = xr.open_dataset(mastu_path+'/balance.nc', autoclose=True).load()
	# ds_puff8 = xr.open_dataset('/home/ffederic/work/SOLPS/dscan/ramp_10/balance.nc', autoclose=True).load()

	print(mastu_path)

	# switches!
	use_deuterium_lines = True
	use_carbon_lines = True
	use_nitrogen_lines = True
	use_core = True
	use_bremmstrahlung = True
	enables = [use_deuterium_lines,use_carbon_lines,use_nitrogen_lines,use_core,use_bremmstrahlung]
	enables = np.int32(enables)

	print(enables)


	if True:	# not really useful for calculating the spectra
		grid_x = ds_puff8.crx.mean(dim='4')
		grid_y = ds_puff8.cry.mean(dim='4')
		try:
			impurity_radiation = ds_puff8.b2stel_she_bal.sum('ns')
		except:
			impurity_radiation = 0
		hydrogen_radiation = ds_puff8.eirene_mc_eael_she_bal.sum('nstra') - ds_puff8.eirene_mc_papl_sna_bal.isel(ns=1).sum('nstra') * 13.6 * 1.6e-19   # ds_puff8.eirene_mc_eael_she_bal.sum('nstra') is the total electron energy sink due to plasma / atoms interactions, including ionisation/excitation (dominant) and charge exchange (negligible). Here we assume all not used for ionisation goes to radiation, including the CX bit.
		total_radiation = -hydrogen_radiation + impurity_radiation
		total_radiation_density = -np.divide(hydrogen_radiation + impurity_radiation,ds_puff8.vol)

		fig, ax = plt.subplots()
		# grid_x.plot.line(ax=ax, x='nx_plus2')
		# grid_y.plot.line(ax=ax, x='ny_plus2')
		# plt.pcolormesh(grid_x.values, grid_y.values, impurity_radiation.values)


		plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values), norm=LogNorm(vmin=1000, vmax=total_radiation_density.values.max()),cmap='rainbow')
		# plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values),cmap='rainbow')
		ax.set_ylim(top=0.1)
		ax.set_xlim(left=0.)
		plt.title('Emissivity profile as imported directly from SOLPS')
		plt.colorbar().set_label('Emissivity [W/m^3]')
		plt.xlabel('R [m]')
		plt.ylabel('Z [m]')

		x=np.linspace(0.55-0.075,0.55+0.075,10)
		y=-1.2+np.sqrt(0.08**2-(x-0.55)**2)
		y_=-1.2-np.sqrt(0.08**2-(x-0.55)**2)
		plt.plot(x,y,'k')
		plt.plot(x,y_,'k')
		plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		# plt.show()

		plt.plot(1.4918014 ,  -0.7198,'ro')	# pinhole
		plt.plot(1.56467,  -0.7,'r+')	# foil centre
		plt.plot(1.56467,  -0.7-0.045,'k+')	# foil bottom
		plt.plot(1.4918014 + (1.4918014-1.56467)*2,  -0.7198 + (-0.7198-(-0.7-0.045))*2,'k+')	# artificial start of the LOS foil bottom
		plt.plot(1.56467,  -0.7+0.045,'k+')	# foil top
		plt.plot(1.4918014 + (1.4918014-1.56467)*5,  -0.7198 + (-0.7198-(-0.7+0.045))*5,'k+')	# artificial start of the LOS foil top
		plt.plot(1.56467,  -0.7+0.015,'k+')	# x-point on foil
		plt.plot(1.4918014 + (1.4918014-1.56467)*4,  -0.7198 + (-0.7198-(-0.7+0.015))*4,'k+')	# artificial start of the LOS x-point on foil
		plt.plot(1.56467,  -0.7-0.045-0.012,'k+')	# foil bottom new geometry
		plt.plot(1.4918014 + (1.4918014-1.56467)*2,  -0.7198 + (-0.7198-(-0.7-0.045-0.012))*2,'k+')	# artificial start of the LOS foil bottom new geometry
		ax.set_aspect(1)
		plt.close()
	else:
		pass

	# now I need to read the equilibrium to create an artificial plasma at the core
	with open('/home/ffederic/work/SOLPS/seeding/seed_1/f_mastu_mastu_osp_scan_5.x4.equ', 'r') as f:
		lines = f.readlines()

	r = []
	for i_line in range(17,120):
		r.append(lines[i_line].strip().split())
	r = np.concatenate(r).astype(float)

	z = []
	for i_line in range(122,226):
		z.append(lines[i_line].strip().split())
	z = np.concatenate(z).astype(float)

	psi = []
	for i_line in range(227,52861):
		psi.append(lines[i_line].strip().split())
	psi = np.concatenate(psi).astype(float)
	psi = psi.reshape((len(r),len(z)))
	# plt.figure()
	# plt.pcolor(r, z, psi,cmap='rainbow')

	from scipy.interpolate import RegularGridInterpolator
	psi_interpolator = RegularGridInterpolator((r, z), psi.T, method='linear',bounds_error=False, fill_value=None)


	# Copyright 2014-2017 United Kingdom Atomic Energy Authority
	#
	# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
	# European Commission - subsequent versions of the EUPL (the "Licence");
	# You may not use this work except in compliance with the Licence.
	# You may obtain a copy of the Licence at:
	#
	# https://joinup.ec.europa.eu/software/page/eupl5
	#
	# Unless required by applicable law or agreed to in writing, software distributed
	# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
	# CONDITIONS OF ANY KIND, either express or implied.
	#
	# See the Licence for the specific language governing permissions and limitations
	# under the Licence.



	import matplotlib.pyplot as plt
	import numpy as np

	from cherab.core.atomic.elements import carbon, deuterium, nitrogen
	from cherab.solps import load_solps_from_balance

	from raysect.optical import Ray,Spectrum,World
	world = World()

	from raysect.optical.material import AbsorbingSurface
	from cherab.mastu.machine import import_mastu_mesh
	# import_mastu_mesh(world)
	# to have consistent results I have to inhibit reflections, setting perfectly absorbing surfaces
	import_mastu_mesh(world,override_material=AbsorbingSurface())

	from cherab.openadas import install
	from cherab.core.atomic.elements import *
	rates = {
		'adf15': (
			(carbon,	0, 'adf15/pec96#c/pec96#c_vsu#c0.dat'),
			(carbon,	0, 'adf15/pec96#c/pec96#c_pju#c0.dat'),
			(carbon,	1, 'adf15/pec96#c/pec96#c_vsu#c1.dat'),
			(carbon,	1, 'adf15/pec96#c/pec96#c_pju#c1.dat'),
			(carbon,	2, 'adf15/pec96#c/pec96#c_vsu#c2.dat'),
			(carbon,	2, 'adf15/pec96#c/pec96#c_pju#c2.dat'),
			(carbon,	3, 'adf15/pec96#c/pec96#c_pju#c3.dat'),
			(carbon,	4, 'adf15/pec96#c/pec96#c_pju#c4.dat'),
			(carbon,	5, 'adf15/pec96#c/pec96#c_pju#c5.dat'),
			# (neon,	  0, 'adf15/pec96#ne/pec96#ne_pju#ne0.dat'),	 #TODO: OPENADAS DATA CORRUPT
			# (neon,	  1, 'adf15/pec96#ne/pec96#ne_pju#ne1.dat'),	 #TODO: OPENADAS DATA CORRUPT
			# (nitrogen,  0, 'adf15/pec96#n/pec96#n_vsu#n0.dat'),
			# (nitrogen,  1, 'adf15/pec96#n/pec96#n_vsu#n1.dat'),
			(nitrogen,  0, 'adf15/pec96#n/pec96#n_vsu#n0.dat'),
			(nitrogen,  0, 'adf15/pec96#n/pec96#n_pju#n0.dat'),
			(nitrogen,  1, 'adf15/pec96#n/pec96#n_vsu#n1.dat'),
			(nitrogen,  1, 'adf15/pec96#n/pec96#n_pju#n1.dat'),
			(nitrogen,  2, 'adf15/pec96#n/pec96#n_vsu#n2.dat'),
			(nitrogen,  2, 'adf15/pec96#n/pec96#n_pju#n2.dat'),
			# (nitrogen,  3, 'adf15/pec96#n/pec96#n_vsu#n3.dat'),
			(nitrogen,  3, 'adf15/pec96#n/pec96#n_pju#n3.dat'),
			(nitrogen,  4, 'adf15/pec96#n/pec96#n_pju#n4.dat'),
			(nitrogen,  5, 'adf15/pec96#n/pec96#n_pju#n5.dat'),
			(nitrogen,  6, 'adf15/pec96#n/pec96#n_pju#n6.dat'),
		)
	}
	install.install_files(rates, download=True)


	plt.ion()

	xl, xu = (0.0, 2.0)
	yl, yu = (-2.0, 2.0)

	print('CHERAB solps_from_balance demo')
	print('Note: code assumes presence of deuterium and carbon species in SOLPS run')
	print('Enter name of balance.nc file:')
	# filename = input()
	filename = mastu_path+'/balance.nc'

	sim = load_solps_from_balance(filename)
	plasma = sim.create_plasma()
	plasma.parent=world
	mesh = sim.mesh

	d0 = plasma.composition.get(deuterium, 0)
	d1 = plasma.composition.get(deuterium, 1)
	try:
		c0 = plasma.composition.get(carbon, 0)
		c1 = plasma.composition.get(carbon, 1)
		c2 = plasma.composition.get(carbon, 2)
		c3 = plasma.composition.get(carbon, 3)
		c4 = plasma.composition.get(carbon, 4)
		c5 = plasma.composition.get(carbon, 5)
		c6 = plasma.composition.get(carbon, 6)
	except:
		use_carbon_lines = False
		enables = [use_deuterium_lines,use_carbon_lines,use_nitrogen_lines,use_core,use_bremmstrahlung]
		enables = np.int32(enables)
	try:
		n0 = plasma.composition.get(nitrogen, 0)
		n1 = plasma.composition.get(nitrogen, 1)
		n2 = plasma.composition.get(nitrogen, 2)
		n3 = plasma.composition.get(nitrogen, 3)
		n4 = plasma.composition.get(nitrogen, 4)
		n5 = plasma.composition.get(nitrogen, 5)
		n6 = plasma.composition.get(nitrogen, 6)
		n7 = plasma.composition.get(nitrogen, 7)
	except:
		use_nitrogen_lines = False
		enables = [use_deuterium_lines,use_carbon_lines,use_nitrogen_lines,use_core,use_bremmstrahlung]
		enables = np.int32(enables)


	te_samples = np.zeros((500, 500))
	ne_samples = np.zeros((500, 500))
	d0_samples = np.zeros((500, 500))
	td0_samples = np.zeros((500, 500))
	d1_samples = np.zeros((500, 500))
	c0_samples = np.zeros((500, 500))
	c1_samples = np.zeros((500, 500))
	c2_samples = np.zeros((500, 500))
	c3_samples = np.zeros((500, 500))
	c4_samples = np.zeros((500, 500))
	c5_samples = np.zeros((500, 500))
	c6_samples = np.zeros((500, 500))
	n0_samples = np.zeros((500, 500))
	n1_samples = np.zeros((500, 500))
	n2_samples = np.zeros((500, 500))
	n3_samples = np.zeros((500, 500))
	n4_samples = np.zeros((500, 500))
	n5_samples = np.zeros((500, 500))
	n6_samples = np.zeros((500, 500))
	n7_samples = np.zeros((500, 500))
	xrange = np.linspace(xl, xu, 500)
	yrange = np.linspace(yl, yu, 500)
	psi_sample = np.zeros((500, 500))



	for i, x in enumerate(xrange):
		for j, y in enumerate(yrange):
			ne_samples[j, i] = plasma.electron_distribution.density(x, 0.0, y)
			te_samples[j, i] = plasma.electron_distribution.effective_temperature(x, 0.0, y)
			d0_samples[j, i] = d0.distribution.density(x, 0.0, y)
			td0_samples[j, i] = d0.distribution.effective_temperature(x, 0.0, y)
			d1_samples[j, i] = d1.distribution.density(x, 0.0, y)
			try:
				c0_samples[j, i] = c0.distribution.density(x, 0.0, y)
				c1_samples[j, i] = c1.distribution.density(x, 0.0, y)
				c2_samples[j, i] = c2.distribution.density(x, 0.0, y)
				c3_samples[j, i] = c3.distribution.density(x, 0.0, y)
				c4_samples[j, i] = c4.distribution.density(x, 0.0, y)
				c5_samples[j, i] = c5.distribution.density(x, 0.0, y)
				c6_samples[j, i] = c6.distribution.density(x, 0.0, y)
			except:
				pass
			try:
				n0_samples[j, i] = n0.distribution.density(x, 0.0, y)
				n1_samples[j, i] = n1.distribution.density(x, 0.0, y)
				n2_samples[j, i] = n2.distribution.density(x, 0.0, y)
				n3_samples[j, i] = n3.distribution.density(x, 0.0, y)
				n4_samples[j, i] = n4.distribution.density(x, 0.0, y)
				n5_samples[j, i] = n5.distribution.density(x, 0.0, y)
				n6_samples[j, i] = n6.distribution.density(x, 0.0, y)
				n7_samples[j, i] = n7.distribution.density(x, 0.0, y)
			except:
				pass
			psi_sample[j, i] = psi_interpolator((x,y))




	plt.figure()
	plt.pcolor(xrange, yrange, psi_sample*(psi_sample>0), cmap='rainbow')
	plt.pause(0.01)

	# case = cp.deepcopy((c0_samples+c1_samples+c2_samples+c3_samples+c4_samples+c5_samples+c6_samples)/(d0_samples+d1_samples))
	case = cp.deepcopy(d0_samples)
	plt.figure()
	plt.plot(psi_sample[240:260],(case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2))
	plt.plot(psi_sample[:,240:260].T,(((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)))
	temp = np.array(((case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2)).tolist() + (((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)).tolist())
	temp[np.isnan(temp)] = 0
	print('{:.2e}'.format(temp[temp>0][np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].argmax()]))
	# fit = np.polyfit(np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0],np.log(temp[temp>0]),1)
	# plt.plot(np.linspace(-0.02,psi_sample.max()),np.exp(np.polyval(fit,np.linspace(-0.02,psi_sample.max()))),'--')
	plt.semilogy()
	# plt.ylim((temp[temp>0]).min())
	# plt.xlim(left=np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].min(),right=np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].max())
	plt.xlabel('psi')
	plt.ylabel('c1 density')
	plt.pause(0.01)

	from cherab.core import Species, Maxwellian, Plasma, Line
	from scipy.constants import electron_mass, atomic_mass
	from raysect.primitive import Cylinder
	from raysect.optical import translate, Vector3D
	from cherab.openadas import OpenADAS
	if use_core:
		plasma_core = Plasma(parent=world)
		plasma_core.atomic_data = OpenADAS(permit_extrapolation=True)
		plasma_core.geometry = Cylinder(2, 2, transform=translate(0, 0, -1))
		plasma_core.geometry_transform = translate(0, 0, -1)

		core_peak_te = 4e3	# eV
		core_peak_ne = 1e20	# #/m3


		case = cp.deepcopy(te_samples)
		temp = np.array(((case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2)).tolist() + (((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)).tolist())
		temp[np.isnan(temp)] = 0
		core_edge_te = temp[temp>0][np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].argmax()]
		case = cp.deepcopy(ne_samples)
		temp = np.array(((case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2)).tolist() + (((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)).tolist())
		temp[np.isnan(temp)] = 0
		core_edge_ne = temp[temp>0][np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].argmax()]
		case = cp.deepcopy(d0_samples + d1_samples)
		temp = np.array(((case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2)).tolist() + (((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)).tolist())
		temp[np.isnan(temp)] = 0
		core_edge_dx = temp[temp>0][np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].argmax()]
		case = cp.deepcopy(d0_samples)
		temp = np.array(((case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2)).tolist() + (((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)).tolist())
		temp[np.isnan(temp)] = 0
		core_edge_d0 = temp[temp>0][np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].argmax()]
		fit = np.polyfit(np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0],np.log(temp[temp>0]),1)
		core_peak_d0 = np.exp(np.polyval(fit,psi_sample.max()))
		case = cp.deepcopy(d1_samples)
		temp = np.array(((case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2)).tolist() + (((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)).tolist())
		temp[np.isnan(temp)] = 0
		core_edge_d1 = temp[temp>0][np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].argmax()]
		edge_psi = np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].max()
		try:
			case = cp.deepcopy((c0_samples+c1_samples+c2_samples+c3_samples+c4_samples+c5_samples+c6_samples))
			temp = np.array(((case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2)).tolist() + (((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)).tolist())
			temp[np.isnan(temp)] = 0
			core_edge_cx = temp[temp>0][np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].argmax()]
			edge_psi = np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].max()
		except:
			pass
		try:
			case = cp.deepcopy((n0_samples+n1_samples+n2_samples+n3_samples+n4_samples+n5_samples+n6_samples+n7_samples))
			temp = np.array(((case)[240:260]*np.logical_and(yrange>-1.2,yrange<1.2)).tolist() + (((case)[:,240:260]).T*np.logical_and(yrange>-1.2,yrange<1.2)).tolist())
			temp[np.isnan(temp)] = 0
			core_edge_nx = temp[temp>0][np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].argmax()]
			edge_psi = np.array(psi_sample[240:260].tolist()+psi_sample[:,240:260].T.tolist())[temp>0].max()
		except:
			pass
		mag_axis_psi = psi.max()
		core_te_interpolator = interp1d([edge_psi,mag_axis_psi],[core_edge_te,core_peak_te],bounds_error=False,fill_value="extrapolate")
		core_ne_interpolator = interp1d([edge_psi,mag_axis_psi],[core_edge_ne,core_peak_ne],bounds_error=False,fill_value="extrapolate")


		# calculating the equilibrium is infinitely faster when read_adf11 is not needed to do.
		# I build an external interpolator
		rates_path='/home/adas/adas/adf11'
		rates_year=96
		from adas import read_adf11

		element = nitrogen
		element_symbol = element.symbol.lower()
		if element_symbol=='d':
			element_symbol = 'h'
		acdfile = rates_path + '/' + 'acd' + str(rates_year) + '/' + 'acd' + str(rates_year) + '_' + element_symbol + '.dat'
		scdfile = rates_path + '/' + 'scd' + str(rates_year) + '/' + 'scd' + str(rates_year) + '_' + element_symbol + '.dat'
		ne_range = np.logspace(np.log10(5e13),np.log10(2e21),num=50)
		te_range = np.logspace(np.log10(0.2),np.log10(15000),num=50)
		acd_ = []
		scd_ = []
		for i in range(1,element.atomic_number+1):
			acd_temp = []
			scd_temp = []
			for ne in ne_range:
				ne = ne * 10 ** (0 - 6)	# from #/m3 to #/cm3
				acd_temp.append(read_adf11(acdfile, 'acd', i, 1, 1, te_range,[ne]*len(te_range)))
				scd_temp.append(read_adf11(scdfile, 'scd', i, 1, 1, te_range,[ne]*len(te_range)))
			acd_.append(acd_temp)
			scd_.append(scd_temp)
		acd_nitrogen_interpolator = RegularGridInterpolator((range(1,element.atomic_number+1),np.log(ne_range),np.log(te_range)),np.log(acd_),bounds_error=False, fill_value=None)
		scd_nitrogen_interpolator = RegularGridInterpolator((range(1,element.atomic_number+1),np.log(ne_range),np.log(te_range)),np.log(scd_),bounds_error=False, fill_value=None)

		element = carbon
		element_symbol = element.symbol.lower()
		if element_symbol=='d':
			element_symbol = 'h'
		acdfile = rates_path + '/' + 'acd' + str(rates_year) + '/' + 'acd' + str(rates_year) + '_' + element_symbol + '.dat'
		scdfile = rates_path + '/' + 'scd' + str(rates_year) + '/' + 'scd' + str(rates_year) + '_' + element_symbol + '.dat'
		ne_range = np.logspace(np.log10(5e13),np.log10(2e21),num=50)
		te_range = np.logspace(np.log10(0.2),np.log10(15000),num=50)
		acd_ = []
		scd_ = []
		for i in range(1,element.atomic_number+1):
			acd_temp = []
			scd_temp = []
			for ne in ne_range:
				ne = ne * 10 ** (0 - 6)	# from #/m3 to #/cm3
				acd_temp.append(read_adf11(acdfile, 'acd', i, 1, 1, te_range,[ne]*len(te_range)))
				scd_temp.append(read_adf11(scdfile, 'scd', i, 1, 1, te_range,[ne]*len(te_range)))
			acd_.append(acd_temp)
			scd_.append(scd_temp)
		acd_carbon_interpolator = RegularGridInterpolator((range(1,element.atomic_number+1),np.log(ne_range),np.log(te_range)),np.log(acd_),bounds_error=False, fill_value=None)
		scd_carbon_interpolator = RegularGridInterpolator((range(1,element.atomic_number+1),np.log(ne_range),np.log(te_range)),np.log(scd_),bounds_error=False, fill_value=None)

		element = deuterium
		element_symbol = element.symbol.lower()
		if element_symbol=='d':
			element_symbol = 'h'
		acdfile = rates_path + '/' + 'acd' + str(rates_year) + '/' + 'acd' + str(rates_year) + '_' + element_symbol + '.dat'
		scdfile = rates_path + '/' + 'scd' + str(rates_year) + '/' + 'scd' + str(rates_year) + '_' + element_symbol + '.dat'
		ne_range = np.logspace(np.log10(5e13),np.log10(2e21),num=50)
		te_range = np.logspace(np.log10(0.2),np.log10(15000),num=50)
		acd_ = []
		scd_ = []
		for i in range(1,element.atomic_number+1):
			acd_temp = []
			scd_temp = []
			for ne in ne_range:
				ne = ne * 10 ** (0 - 6)	# from #/m3 to #/cm3
				acd_temp.append(read_adf11(acdfile, 'acd', i, 1, 1, te_range,[ne]*len(te_range)))
				scd_temp.append(read_adf11(scdfile, 'scd', i, 1, 1, te_range,[ne]*len(te_range)))
			acd_.append(acd_temp)
			scd_.append(scd_temp)
		acd_deuterium_interpolator = RegularGridInterpolator((range(1,element.atomic_number+1),np.log(ne_range),np.log(te_range)),np.log(acd_),bounds_error=False, fill_value=None)
		scd_deuterium_interpolator = RegularGridInterpolator((range(1,element.atomic_number+1),np.log(ne_range),np.log(te_range)),np.log(scd_),bounds_error=False, fill_value=None)

		def equilibrium_calculator(element,ne,te,acd_nitrogen_interpolator=acd_nitrogen_interpolator,scd_nitrogen_interpolator=scd_nitrogen_interpolator,acd_carbon_interpolator=acd_carbon_interpolator,scd_carbon_interpolator=scd_carbon_interpolator,acd_deuterium_interpolator=acd_deuterium_interpolator,scd_deuterium_interpolator=scd_deuterium_interpolator):
			# approach taken from https://ned.ipac.caltech.edu/level5/Sept08/Kaastra/Kaastra4.html
			# from adas import read_adf11

			# load rates

			element_symbol = element.symbol.lower()
			if element_symbol=='d':
				acd_interpolator = acd_deuterium_interpolator
				scd_interpolator = scd_deuterium_interpolator
			elif element_symbol=='c':
				acd_interpolator = acd_carbon_interpolator
				scd_interpolator = scd_carbon_interpolator
			elif element_symbol=='n':
				acd_interpolator = acd_nitrogen_interpolator
				scd_interpolator = scd_nitrogen_interpolator

			# acdfile = rates_path + '/' + 'acd' + str(rates_year) + '/' + 'acd' + str(rates_year) + '_' + element_symbol + '.dat'
			# scdfile = rates_path + '/' + 'scd' + str(rates_year) + '/' + 'scd' + str(rates_year) + '_' + element_symbol + '.dat'

			# ne = ne * 10 ** (0 - 6)	# from #/m3 to #/cm3
			# acd_ = [read_adf11(acdfile, 'acd', i, 1, 1, te,ne)[0] for i in range(1,element.atomic_number+1)]
			# scd_ = [read_adf11(scdfile, 'scd', i, 1, 1, te,ne)[0] for i in range(1,element.atomic_number+1)]
			temp = np.array([range(1,element.atomic_number+1),[np.log(ne)]*element.atomic_number,[np.log(te)]*element.atomic_number]).T
			acd_ = np.exp(acd_interpolator(temp))
			scd_ = np.exp(scd_interpolator(temp))

			density = np.array([1])
			for i in range(element.atomic_number):
				temp = 0
				if i>0:
					# acd = acd_[i-1]#read_adf11(acdfile, 'acd', i, 1, 1, te,ne)	# recombination
					temp += density[i]*acd_[i-1]
				# if i>0:
					# scd = scd_[i-1]#read_adf11(scdfile, 'scd', i, 1, 1, te,ne)	# ionisation
					temp -= density[i-1]*scd_[i-1]
				# if i<element.atomic_number:
				# scd = scd_[i]#read_adf11(scdfile, 'scd', i+1, 1, 1, te,ne)	# ionisation
				temp += density[i]*scd_[i]
				# acd_1 = acd_[i]#read_adf11(acdfile, 'acd', i+1, 1, 1, te,ne)	# recombination
				density = np.append(density,temp/acd_[i])
			density = density/np.sum(density)	# the output is the ratio compared to the total density of that element
			return density

		class Linear_profile:
			"""A liner profile fron the border of the SOLPS grid to the magnetic axis."""

			def __init__(self, mag_axis_value, edge_value, out_of_bounds_value, psi_interpolator = psi_interpolator, edge_psi=edge_psi, mag_axis_psi = mag_axis_psi, max_z_for_core = 1.2, scale='linear'):

				self.mag_axis = mag_axis_value
				self.edge = edge_value
				self.psi_interpolator = psi_interpolator
				self.edge_psi = edge_psi
				self.mag_axis_psi = mag_axis_psi
				self.max_z_for_core = max_z_for_core
				self.out_of_bounds = out_of_bounds_value
				self.scale = scale

				# self.r_axis = magnetic_axis[0]
				# self.z_axis = magnetic_axis[1]

			def __call__(self, x, y, z):

				# calculate r in r-z space
				r = np.sqrt(x**2 + y**2)

				# calculate psi in that location
				psi_ = self.psi_interpolator((r,z))

				if psi_ < self.edge_psi or z<-self.max_z_for_core or z>self.max_z_for_core:
					return self.out_of_bounds
				else:
					if self.scale == 'linear':
						return (self.mag_axis - self.edge) * (psi_-self.edge_psi)/(self.mag_axis_psi-self.edge_psi) + self.edge
					elif self.scale == 'log':
						return np.exp(np.log(self.mag_axis / self.edge) * (psi_-self.edge_psi)/(self.mag_axis_psi-self.edge_psi) + np.log(self.edge))

		class Impurities_ionisation_equilibrium_profile:
			"""A liner profile fron the border of the SOLPS grid to the magnetic axis."""

			def __init__(self, total_density_edge, element, ionisation_level, out_of_bounds_value, psi_interpolator = psi_interpolator, core_te_interpolator = core_te_interpolator, core_ne_interpolator = core_ne_interpolator, edge_psi=edge_psi, mag_axis_psi = mag_axis_psi, max_z_for_core = 1.2, equilibrium_calculator=equilibrium_calculator):

				self.total_density_edge = total_density_edge
				self.psi_interpolator = psi_interpolator
				self.edge_psi = edge_psi
				self.mag_axis_psi = mag_axis_psi
				self.max_z_for_core = max_z_for_core
				self.out_of_bounds = out_of_bounds_value
				self.core_te_interpolator = core_te_interpolator
				self.core_ne_interpolator = core_ne_interpolator
				self.equilibrium_calculator = equilibrium_calculator
				self.element = element
				self.ionisation_level = ionisation_level

				# self.r_axis = magnetic_axis[0]
				# self.z_axis = magnetic_axis[1]

			def __call__(self, x, y, z):

				# calculate r in r-z space
				r = np.sqrt(x**2 + y**2)

				# calculate psi in that location
				psi_ = self.psi_interpolator((r,z))

				if psi_ < self.edge_psi or z<-self.max_z_for_core or z>self.max_z_for_core:
					return self.out_of_bounds
				else:
					ne=self.core_ne_interpolator(psi_)
					# the multiplication by ne is not really good. what I want is zeff=constant. this is true, using this formula, only when te is high enough to ionise fully everything.
					# for high z elements like carbon and expecially nitrogen this could not be valid at the edge
					# still, this is goo enough for this I think
					return (self.total_density_edge/self.core_ne_interpolator(self.edge_psi))*ne*(self.equilibrium_calculator(self.element,ne,self.core_te_interpolator(psi_))[self.ionisation_level])


		# No net velocity for any species
		zero_velocity = Vector3D(0, 0, 0)


		# define neutral species distribution
		d0_density = Linear_profile(core_peak_d0, core_edge_d0, 0, scale='log')
		d0_temperature = 0.5  # constant 0.5eV temperature for all neutrals
		d0_distribution = Maxwellian(d0_density, d0_temperature, zero_velocity,
		                             deuterium.atomic_weight * atomic_mass)
		d0_species = Species(deuterium, 0, d0_distribution)

		# define deuterium ion species distribution
		d1_density = Impurities_ionisation_equilibrium_profile(core_edge_dx, deuterium, 1, 0)
		d1_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
		d1_distribution = Maxwellian(d1_density, d1_temperature, zero_velocity,
		                             deuterium.atomic_weight * atomic_mass)
		d1_species = Species(deuterium, 1, d1_distribution)

		# define the electron distribution
		e_density = Linear_profile(core_peak_ne, core_edge_ne, 0)
		e_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
		e_distribution = Maxwellian(e_density, e_temperature, zero_velocity, electron_mass)

		try:
			# define neutral species distribution
			c0_density = Impurities_ionisation_equilibrium_profile(core_edge_cx, carbon, 0, 0)
			# c0_density = Linear_profile(0.00553, 50E6, 0, scale='log')
			c0_temperature = 0.5  # constant 0.5eV temperature for all neutrals
			c0_distribution = Maxwellian(c0_density, c0_temperature, zero_velocity,
			                             carbon.atomic_weight * atomic_mass)
			c0_species = Species(carbon, 0, c0_distribution)

			c1_density = Impurities_ionisation_equilibrium_profile(core_edge_cx, carbon, 1, 0)
			c1_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			c1_distribution = Maxwellian(c1_density, c1_temperature, zero_velocity,
			                             carbon.atomic_weight * atomic_mass)
			c1_species = Species(carbon, 1, c1_distribution)

			c2_density = Impurities_ionisation_equilibrium_profile(core_edge_cx, carbon, 2, 0)
			c2_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			c2_distribution = Maxwellian(c2_density, c2_temperature, zero_velocity,
			                             carbon.atomic_weight * atomic_mass)
			c2_species = Species(carbon, 2, c2_distribution)

			c3_density = Impurities_ionisation_equilibrium_profile(core_edge_cx, carbon, 3, 0)
			c3_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			c3_distribution = Maxwellian(c3_density, c3_temperature, zero_velocity,
			                             carbon.atomic_weight * atomic_mass)
			c3_species = Species(carbon, 3, c3_distribution)

			c4_density = Impurities_ionisation_equilibrium_profile(core_edge_cx, carbon, 4, 0)
			c4_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			c4_distribution = Maxwellian(c4_density, c4_temperature, zero_velocity,
			                             carbon.atomic_weight * atomic_mass)
			c4_species = Species(carbon, 4, c4_distribution)

			c5_density = Impurities_ionisation_equilibrium_profile(core_edge_cx, carbon, 5, 0)
			c5_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			c5_distribution = Maxwellian(c5_density, c5_temperature, zero_velocity,
			                             carbon.atomic_weight * atomic_mass)
			c5_species = Species(carbon, 5, c5_distribution)

			c6_density = Impurities_ionisation_equilibrium_profile(core_edge_cx, carbon, 6, 0)
			c6_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			c6_distribution = Maxwellian(c6_density, c6_temperature, zero_velocity,
			                             carbon.atomic_weight * atomic_mass)
			c6_species = Species(carbon, 6, c6_distribution)
		except:
			pass

		try:
			# define neutral species distribution
			n0_density = Impurities_ionisation_equilibrium_profile(core_edge_nx, nitrogen, 0, 0)
			n0_temperature = 0.5  # constant 0.5eV temperature for all neutrals
			n0_distribution = Maxwellian(n0_density, n0_temperature, zero_velocity,
			                             nitrogen.atomic_weight * atomic_mass)
			n0_species = Species(nitrogen, 0, n0_distribution)

			n1_density = Impurities_ionisation_equilibrium_profile(core_edge_nx, nitrogen, 1, 0)
			n1_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			n1_distribution = Maxwellian(n1_density, n1_temperature, zero_velocity,
			                             nitrogen.atomic_weight * atomic_mass)
			n1_species = Species(nitrogen, 1, n1_distribution)

			n2_density = Impurities_ionisation_equilibrium_profile(core_edge_nx, nitrogen, 2, 0)
			n2_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			n2_distribution = Maxwellian(n2_density, n2_temperature, zero_velocity,
			                             nitrogen.atomic_weight * atomic_mass)
			n2_species = Species(nitrogen, 2, n2_distribution)

			n3_density = Impurities_ionisation_equilibrium_profile(core_edge_nx, nitrogen, 3, 0)
			n3_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			n3_distribution = Maxwellian(n3_density, n3_temperature, zero_velocity,
			                             nitrogen.atomic_weight * atomic_mass)
			n3_species = Species(nitrogen, 3, n3_distribution)

			n4_density = Impurities_ionisation_equilibrium_profile(core_edge_nx, nitrogen, 4, 0)
			n4_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			n4_distribution = Maxwellian(n4_density, n4_temperature, zero_velocity,
			                             nitrogen.atomic_weight * atomic_mass)
			n4_species = Species(nitrogen, 4, n4_distribution)

			n5_density = Impurities_ionisation_equilibrium_profile(core_edge_nx, nitrogen, 5, 0)
			n5_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			n5_distribution = Maxwellian(n5_density, n5_temperature, zero_velocity,
			                             nitrogen.atomic_weight * atomic_mass)
			n5_species = Species(nitrogen, 5, n5_distribution)

			n6_density = Impurities_ionisation_equilibrium_profile(core_edge_nx, nitrogen, 6, 0)
			n6_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			n6_distribution = Maxwellian(n6_density, n6_temperature, zero_velocity,
			                             nitrogen.atomic_weight * atomic_mass)
			n6_species = Species(nitrogen, 6, n6_distribution)

			n7_density = Impurities_ionisation_equilibrium_profile(core_edge_nx, nitrogen, 7, 0)
			n7_temperature = Linear_profile(core_peak_te, core_edge_te, 0)
			n7_distribution = Maxwellian(n7_density, n7_temperature, zero_velocity,
			                             nitrogen.atomic_weight * atomic_mass)
			n7_species = Species(nitrogen, 7, n7_distribution)
		except:
			pass

		plasma_core.electron_distribution = e_distribution

		try:
			plasma_core.composition = [d0_species, d1_species, c0_species, c1_species, c2_species, c3_species, c4_species, c5_species, c6_species, n0_species, n1_species, n2_species, n3_species, n4_species, n5_species, n6_species, n7_species]
			print('imported densities of species: H, C, N')
		except:
			try:
				plasma_core.composition = [d0_species, d1_species, n0_species, n1_species, n2_species, n3_species, n4_species, n5_species, n6_species, n7_species]
				print('imported densities of species: H, N')
			except:
				try:
					plasma_core.composition = [d0_species, d1_species, c0_species, c1_species, c2_species, c3_species, c4_species, c5_species, c6_species]
					print('imported densities of species: H, C')
				except:
					plasma_core.composition = [d0_species, d1_species]
					print('imported densities of species: H')


		# from cherab.core.math import sample3d
		#
		# r, _, z, t_samples = sample3d(plasma_core.z_effective, (0, 1.6, 200), (0, 0, 1), (-2, 2, 200))
		# t_samples[t_samples==0] = np.nan
		# plt.figure()
		# plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[0, 1.6, -2, 2])
		# plt.colorbar()
		# plt.axis('equal')
		# plt.xlabel('r axis')
		# plt.ylabel('z axis')
		# plt.title("Ion temperature profile in r-z plane")
	else:
		pass


	from cherab.mastu.machine import MASTU_FULL_MESH
	import os
	from raysect.primitive import import_stl, Sphere, Mesh, Cylinder
	from raysect.optical.material.absorber import AbsorbingSurface


	# for cad_file in MASTU_FULL_MESH:
	# 	directory, filename = os.path.split(cad_file[0])
	# 	name, ext = filename.split('.')
	# 	print("importing {} ...".format(filename))
	# 	Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)


	# from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
	# # ray = Ray(origin=Point3D(-1.5,0,0),direction=Vector3D(1,0,0),min_wavelength=0.01,max_wavelength=1200,bins=1200*100)
	# ray = Ray(origin=Point3D(-1.5,0,-0.9),direction=Vector3D(1,0,0),min_wavelength=0.01,max_wavelength=1200,bins=1200*10)

	from cherab.openadas import OpenADAS
	# plasma = sim.create_plasma()
	plasma.models.clear()
	plasma.parent=world
	plasma.atomic_data = OpenADAS(permit_extrapolation=True)
	if use_core:
		plasma_core.models.clear()
		plasma_core.parent=world
		plasma_core.atomic_data = OpenADAS(permit_extrapolation=True)


	from cherab.core.model import ExcitationLine,RecombinationLine,Bremsstrahlung
	from cherab.core.atomic import Line

	# line = Line(carbon, 5,(2,1))	# loop on the destination state and start state for every ionisation level
	# model = ExcitationLine(line)
	# plasma.models.add(model)
	#


	from cherab.core.utility import RecursiveDict
	import os
	import json
	DEFAULT_REPOSITORY_PATH = os.path.expanduser('~/.cherab/openadas/repository')

	if use_deuterium_lines:
		for cls in ['excitation','recombination','thermalcx']:
			for element in [deuterium]:#,carbon,nitrogen]:
				if element.symbol.lower()=='d':
					charges = 1
				elif element.symbol.lower()=='c':
					charges = 6
				else:
					charges = 7
				for charge in range(charges):
					try:
						# cls, element, trash = 'excitation',carbon,0
						if element.symbol.lower()=='d':
							path = os.path.join(DEFAULT_REPOSITORY_PATH, 'pec/{}/{}/{}.json'.format(cls, 'h', charge))
						else:
							path = os.path.join(DEFAULT_REPOSITORY_PATH, 'pec/{}/{}/{}.json'.format(cls, element.symbol.lower(), charge))
						with open(path, 'r') as f:
							content = RecursiveDict.from_dict(json.load(f))
						for transition in list(content.keys()):
							line = Line(element, charge,(transition[:transition.find('->')-1],transition[transition.find('->')+3:]))	# loop on the destination state for every ionisation level
							if cls == 'excitation':
								model = ExcitationLine(line)
							elif cls == 'recombination':
								model = RecombinationLine(line)
							plasma.models.add(model)
							if use_core:
								line = Line(element, charge,(transition[:transition.find('->')-1],transition[transition.find('->')+3:]))	# loop on the destination state for every ionisation level
								if cls == 'excitation':
									model = ExcitationLine(line)
								elif cls == 'recombination':
									model = RecombinationLine(line)
								plasma_core.models.add(model)
					except Exception as e:
						print('Error '+str(e))
	# print(list(plasma.models))
	if use_carbon_lines:
		for cls in ['excitation','recombination','thermalcx']:
			for element in [carbon]:#,nitrogen]:
				if element.symbol.lower()=='d':
					charges = 1
				elif element.symbol.lower()=='c':
					charges = 6
				else:
					charges = 7
				for charge in range(charges):
					try:
						# cls, element, trash = 'excitation',carbon,0
						if element.symbol.lower()=='d':
							path = os.path.join(DEFAULT_REPOSITORY_PATH, 'pec/{}/{}/{}.json'.format(cls, 'h', charge))
						else:
							path = os.path.join(DEFAULT_REPOSITORY_PATH, 'pec/{}/{}/{}.json'.format(cls, element.symbol.lower(), charge))
						with open(path, 'r') as f:
							content = RecursiveDict.from_dict(json.load(f))
						for transition in list(content.keys()):
							line = Line(element, charge,(transition[:transition.find('->')-1],transition[transition.find('->')+3:]))	# loop on the destination state for every ionisation level
							if cls == 'excitation':
								model = ExcitationLine(line)
							elif cls == 'recombination':
								model = RecombinationLine(line)
							plasma.models.add(model)
							if use_core:
								line = Line(element, charge,(transition[:transition.find('->')-1],transition[transition.find('->')+3:]))	# loop on the destination state for every ionisation level
								if cls == 'excitation':
									model = ExcitationLine(line)
								elif cls == 'recombination':
									model = RecombinationLine(line)
								plasma_core.models.add(model)
					except Exception as e:
						print('Error '+str(e))
	# print(list(plasma.models))
	if use_nitrogen_lines:
		for cls in ['excitation','recombination','thermalcx']:
			for element in [nitrogen]:
				if element.symbol.lower()=='d':
					charges = 1
				elif element.symbol.lower()=='c':
					charges = 6
				else:
					charges = 7
				for charge in range(charges):
					try:
						# cls, element, trash = 'excitation',carbon,0
						if element.symbol.lower()=='d':
							path = os.path.join(DEFAULT_REPOSITORY_PATH, 'pec/{}/{}/{}.json'.format(cls, 'h', charge))
						else:
							path = os.path.join(DEFAULT_REPOSITORY_PATH, 'pec/{}/{}/{}.json'.format(cls, element.symbol.lower(), charge))
						with open(path, 'r') as f:
							content = RecursiveDict.from_dict(json.load(f))
						for transition in list(content.keys()):
							line = Line(element, charge,(transition[:transition.find('->')-1],transition[transition.find('->')+3:]))	# loop on the destination state for every ionisation level
							if cls == 'excitation':
								model = ExcitationLine(line)
							elif cls == 'recombination':
								model = RecombinationLine(line)
							plasma.models.add(model)
							if use_core:
								line = Line(element, charge,(transition[:transition.find('->')-1],transition[transition.find('->')+3:]))	# loop on the destination state for every ionisation level
								if cls == 'excitation':
									model = ExcitationLine(line)
								elif cls == 'recombination':
									model = RecombinationLine(line)
								plasma_core.models.add(model)
					except Exception as e:
						print('Error '+str(e))
	# print(list(plasma.models))

	if use_bremmstrahlung:
		plasma.models.add(Bremsstrahlung())
		if use_core:
			plasma_core.models.add(Bremsstrahlung())
	# print(list(plasma.models))


	# Total radiation does not depend on wavelength, therefore Spectrum can be initialised with any values.
	spectrum = Spectrum(min_wavelength=0.01, max_wavelength=1200, bins=1)
	# Calculating total radiation power density on SOLPS mesh
	direction = Vector3D(0, 0, 1)
	tot_rad_power = np.zeros((500, 500))
	for model in plasma.models:
		for i, x in enumerate(xrange):
			for j, y in enumerate(yrange):
				if ne_samples[j, i] != 0:
					point = Point3D(x, 0, y)
					tot_rad_power[j, i] += 4. * np.pi * model.emission(point, direction, spectrum.new_spectrum()).total()
	if use_core:
		for model in plasma_core.models:
			for i, x in enumerate(xrange):
				for j, y in enumerate(yrange):
					if ne_samples[j, i] == 0 and psi_sample[j, i]>=0 and y<=1.3 and y>=-1.3:
						point = Point3D(x, 0, y)
						tot_rad_power[j, i] += 4. * np.pi * model.emission(point, direction, spectrum.new_spectrum()).total()


	fig, ax = plt.subplots(figsize=(12,7))
	# grid_x.plot.line(ax=ax, x='nx_plus2')
	# grid_y.plot.line(ax=ax, x='ny_plus2')
	# plt.pcolormesh(grid_x.values, grid_y.values, impurity_radiation.values)


	plt.imshow(tot_rad_power, norm=LogNorm(vmin=min(1000,tot_rad_power.max()/100), vmax=min(total_radiation_density.values.max(),tot_rad_power.max())),cmap='rainbow',extent=[xrange.min(),xrange.max(),yrange.min(),yrange.max()])
	# plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values),cmap='rainbow')
	ax.set_ylim(top=0.1)
	ax.set_xlim(left=0.)
	plt.title('Emissivity profile from SOLPS+core')
	plt.colorbar().set_label('Emissivity [W/m^3]')
	plt.xlabel('R [m]')
	plt.ylabel('Z [m]')

	x=np.linspace(0.55-0.075,0.55+0.075,10)
	y=-1.2+np.sqrt(0.08**2-(x-0.55)**2)
	y_=-1.2-np.sqrt(0.08**2-(x-0.55)**2)
	plt.plot(x,y,'k')
	plt.plot(x,y_,'k')
	plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	# plt.show()

	# plt.plot(1.4918014 ,  -0.7198,'ro')	# pinhole
	# plt.plot(1.56467,  -0.7,'ro')	# foil centre
	# plt.plot(1.56467,  -0.7-0.045,'k+')	# foil bottom
	# plt.plot(1.4918014 + (1.4918014-1.56467)*2,  -0.7198 + (-0.7198-(-0.7-0.045))*2,'k+')	# artificial start of the LOS
	# plt.plot(1.56467,  -0.7+0.045,'k+')	# foil top
	# plt.plot(1.4918014 + (1.4918014-1.56467)*5,  -0.7198 + (-0.7198-(-0.7+0.045))*5,'k+')	# artificial start of the LOS
	# plt.plot(1.56467,  -0.7+0.015,'k+')	# foil top
	# plt.plot(1.4918014 + (1.4918014-1.56467)*4,  -0.7198 + (-0.7198-(-0.7+0.015))*4,'k+')	# artificial start of the LOS
	# plt.plot(1.56467,  -0.7-0.045-0.012,'k+')	# foil bottom new geometry
	# plt.plot(1.4918014 + (1.4918014-1.56467)*2,  -0.7198 + (-0.7198-(-0.7-0.045-0.012))*2,'k+')	# artificial start of the LOS
	ax.set_aspect(1)
	plt.savefig(mastu_path + '/spectra_'+'enables_'+str(enables)+'_'+'tot_rad_power'+'.eps', bbox_inches='tight')
	plt.close()
	# exit()



	#
	# line = Line(carbon, 2,(2,1))	# loop on the destination state for every ionisation level

	# for i in range(2,13):
	# 	for i_ in range(1,i):
	# 		line = Line(deuterium, 0,(i,i_))
	# 		model = ExcitationLine(line)
	# 		plasma.models.add(model)
	# line = Line(deuterium, 0,(2,1))	# loop on the destination state and start state for every ionisation level
	# model = ExcitationLine(line)
	# plasma.models.add(model)





	from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
	# ray = Ray(origin=Point3D(-1.15,0,-0.9),direction=Vector3D(1,0,0),min_wavelength=0.01,max_wavelength=1200,bins=1200*100)
	# ray = Ray(origin=Point3D(-0.95,0,-1.2),direction=Vector3D(1,0,0),min_wavelength=0.01,max_wavelength=1200,bins=1200*100)
	# ray = Ray(origin=Point3D(-0.8,0,-1.4),direction=Vector3D(1,0,0),min_wavelength=0.01,max_wavelength=1200,bins=1200*100)
	# plt.figure()

	if False:
		point_midplane = Point3D(-1.5,0,0)
		direction_midplane = Vector3D(1,0,0)*0.1
		type = 'midplane'
		ray = Ray(origin=point_midplane,direction=direction_midplane,min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)
		spectrum = ray.trace(world)	# samples of spectral radiance: W/m2/str/nm
		np.savez(mastu_path + '/spectra_'+'enables_'+str(enables)+'_'+type,spectrum)
	elif False:
		point_as_up_as_possible_MU04 = Point3D(1.4918014 + (1.4918014-1.56467)*2,0,-0.7198 + (-0.7198-(-0.7-0.045-0.012))*2)
		direction_as_up_as_possible_MU04 = Vector3D(1.4918014-1.56467,-0.0198-0.01,-0.7198-(-0.7-0.045-0.012))
		# ray = Ray(origin=Point3D(1.4918014 + (1.4918014-1.56467)*2,0,-0.7198 + (-0.7198-(-0.7-0.045-0.012))*2),direction=Vector3D(1.4918014-1.56467,0,-0.7198-(-0.7-0.045-0.012)),min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)	# LOS pointing as up as it can central
		type = 'as_up_as_possible_MU04'
		ray = Ray(origin=point_as_up_as_possible_MU04,direction=direction_as_up_as_possible_MU04,min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)	# LOS pointing as up as it can
		spectrum = ray.trace(world)	# samples of spectral radiance: W/m2/str/nm
		np.savez(mastu_path + '/spectra_'+'enables_'+str(enables)+'_'+type,spectrum)
	elif False:
		point_as_up_as_possible = Point3D(1.4918014 + (1.4918014-1.56467)*2,0 + (-0.0198-0.02)*2,-0.7198 + (-0.7198-(-0.7-0.045))*2)
		direction_as_up_as_possible = Vector3D(1.4918014-1.56467,-0.0198-0.02,-0.7198-(-0.7-0.045))
		# ray = Ray(origin=Point3D(1.4918014 + (1.4918014-1.56467)*2,0,-0.7198 + (-0.7198-(-0.7-0.045))*2),direction=Vector3D(1.4918014-1.56467,0,-0.7198-(-0.7-0.045)),min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)	# LOS pointing as up as it can central from the new foin shame from mu04
		ray = Ray(origin=point_as_up_as_possible,direction=direction_as_up_as_possible,min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)	# LOS pointing as up as it can from the new foin shame from mu04
		type = 'as_up_as_possible'
		spectrum = ray.trace(world)	# samples of spectral radiance: W/m2/str/nm
		np.savez(mastu_path + '/spectra_'+'enables_'+str(enables)+'_'+type,spectrum)
	elif False:
		point_as_low_as_possible = Point3D(1.4918014 + (1.4918014-1.56467)*5,0 + (-0.0198-0.017-0.003)*5,-0.7198 + (-0.7198-(-0.7+0.03-0.005))*5)
		direction_as_low_as_possible = Vector3D(1.4918014-1.56467,-0.0198-0.017-0.003,-0.7198-(-0.7+0.03-0.005))
		# # ray = Ray(origin=Point3D(1.4918014 + (1.4918014-1.56467)*5,0,-0.7198 + (-0.7198-(-0.7+0.045))*5),direction=Vector3D(1.4918014-1.56467,0,-0.7198-(-0.7+0.045)),min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)	# LOS pointing as low as it can central
		ray = Ray(origin=point_as_low_as_possible,direction=direction_as_low_as_possible,min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)	# LOS pointing in the SXD
		type = 'as_low_as_possible'
		spectrum = ray.trace(world)	# samples of spectral radiance: W/m2/str/nm
		np.savez(mastu_path + '/spectra_'+'enables_'+str(enables)+'_'+type,spectrum)
	elif True:
		point_x_point = Point3D(1.4918014 + (1.4918014-1.56467)*4,0 + (-0.0198-0.009)*4,-0.7198 + (-0.7198-(-0.7+0.007))*4)
		direction_x_point = Vector3D(1.4918014-1.56467,-0.0198-0.009,-0.7198-(-0.7+0.007))
		# ray = Ray(origin=Point3D(1.4918014 + (1.4918014-1.56467)*5,0,-0.7198 + (-0.7198-(-0.7+0.015))*4),direction=Vector3D(1.4918014-1.56467,0,-0.7198-(-0.7+0.015)),min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)	# LOS pointing at x-point central
		ray = Ray(origin=point_x_point,direction=direction_x_point,min_wavelength=0.01,max_wavelength=1200,bins=1200*1000)	# LOS pointing at x-point
		type = 'x_point'
		spectrum = ray.trace(world)	# samples of spectral radiance: W/m2/str/nm
		np.savez(mastu_path + '/spectra_'+'enables_'+str(enables)+'_'+type,spectrum)

		# # spectrum = ray.trace(world, keep_alive=True)
		# # plt.plot(spectrum.wavelengths,spectrum.samples)
		#
		#
		# np.savez('/home/ffederic/work/irvb/MAST-U/spectra_'+type+'_'+str(i),spectrum)
	print('done')
	exit()



	if False:
		ax = plt.figure().add_subplot(projection='3d')
		plt.plot(point_midplane.x+np.arange(20)*direction_midplane.x,point_midplane.y+np.arange(20)*direction_midplane.y,point_midplane.z+np.arange(20)*direction_midplane.z,label='midplane')
		plt.plot(point_as_up_as_possible_MU04.x+np.arange(20)*direction_as_up_as_possible_MU04.x,point_as_up_as_possible_MU04.y+np.arange(20)*direction_as_up_as_possible_MU04.y,point_as_up_as_possible_MU04.z+np.arange(20)*direction_as_up_as_possible_MU04.z,label='as_up_as_possible_MU04')
		plt.plot(point_as_up_as_possible.x+np.arange(20)*direction_as_up_as_possible.x,point_as_up_as_possible.y+np.arange(20)*direction_as_up_as_possible.y,point_as_up_as_possible.z+np.arange(20)*direction_as_up_as_possible.z,label='as_up_as_possible')
		plt.plot(point_as_low_as_possible.x+np.arange(20)*direction_as_low_as_possible.x,point_as_low_as_possible.y+np.arange(20)*direction_as_low_as_possible.y,point_as_low_as_possible.z+np.arange(20)*direction_as_low_as_possible.z,label='as_low_as_possible')
		plt.plot(point_x_point.x+np.arange(20)*direction_x_point.x,point_x_point.y+np.arange(20)*direction_x_point.y,point_x_point.z+np.arange(20)*direction_x_point.z,label='x_point')
		plt.plot(grid_x.values[total_radiation_density.values>0],np.zeros_like(grid_y.values)[total_radiation_density.values>0] ,grid_y.values[total_radiation_density.values>0],'.k',alpha=0.01)
		plt.plot(-grid_x.values[total_radiation_density.values>0],np.zeros_like(grid_y.values)[total_radiation_density.values>0] ,grid_y.values[total_radiation_density.values>0],'.k',alpha=0.01)
		plt.plot(np.zeros_like(grid_y.values)[total_radiation_density.values>0],grid_x.values[total_radiation_density.values>0],grid_y.values[total_radiation_density.values>0],'.k',alpha=0.01)
		plt.plot(np.zeros_like(grid_y.values)[total_radiation_density.values>0],-grid_x.values[total_radiation_density.values>0],grid_y.values[total_radiation_density.values>0],'.k',alpha=0.01)
		plt.plot(grid_x.values[total_radiation_density.values>0]/(2**0.5),grid_x.values[total_radiation_density.values>0]/(2**0.5) ,grid_y.values[total_radiation_density.values>0],'.r',alpha=0.01)
		plt.plot(-grid_x.values[total_radiation_density.values>0]/(2**0.5),grid_x.values[total_radiation_density.values>0]/(2**0.5) ,grid_y.values[total_radiation_density.values>0],'.r',alpha=0.01)
		plt.plot(grid_x.values[total_radiation_density.values>0]/(2**0.5),-grid_x.values[total_radiation_density.values>0]/(2**0.5) ,grid_y.values[total_radiation_density.values>0],'.r',alpha=0.01)
		plt.plot(-grid_x.values[total_radiation_density.values>0]/(2**0.5),-grid_x.values[total_radiation_density.values>0]/(2**0.5) ,grid_y.values[total_radiation_density.values>0],'.r',alpha=0.01)
		plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 0]*0, FULL_MASTU_CORE_GRID_POLYGON[:, 1], '-k')
		plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0]*0, FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], '-k')
		plt.plot(-FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 0]*0, FULL_MASTU_CORE_GRID_POLYGON[:, 1], '-k')
		plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0]*0, -FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], '-k')
		plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0]/(2**0.5), FULL_MASTU_CORE_GRID_POLYGON[:, 0]/(2**0.5), FULL_MASTU_CORE_GRID_POLYGON[:, 1], '-r')
		plt.plot(-FULL_MASTU_CORE_GRID_POLYGON[:, 0]/(2**0.5), FULL_MASTU_CORE_GRID_POLYGON[:, 0]/(2**0.5), FULL_MASTU_CORE_GRID_POLYGON[:, 1], '-r')
		plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0]/(2**0.5), -FULL_MASTU_CORE_GRID_POLYGON[:, 0]/(2**0.5), FULL_MASTU_CORE_GRID_POLYGON[:, 1], '-r')
		plt.plot(-FULL_MASTU_CORE_GRID_POLYGON[:, 0]/(2**0.5), -FULL_MASTU_CORE_GRID_POLYGON[:, 0]/(2**0.5), FULL_MASTU_CORE_GRID_POLYGON[:, 1], '-r')

		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')



	# SOLPS_case = 'seed_1'
	# mastu_path = "/home/ffederic/work/SOLPS/seeding/" + SOLPS_case

	# SOLPS_case = 'ramp_11'
	# mastu_path = "/home/ffederic/work/SOLPS/dscan/" + SOLPS_case

	use_deuterium_lines = False
	use_carbon_lines = False
	use_nitrogen_lines = False
	use_core = True
	use_bremmstrahlung = True
	enables = [use_deuterium_lines,use_carbon_lines,use_nitrogen_lines,use_core,use_bremmstrahlung]
	enables = np.int32(enables)


	integrated_power = []

	# for i_SOLPS_case,SOLPS_case in enumerate(['seed_1','seed_5','seed_10']):
	# 	mastu_path = "/home/ffederic/work/SOLPS/seeding/" + SOLPS_case

	for i_SOLPS_case,SOLPS_case in enumerate(['ramp_1','ramp_3.3','ramp_11']):
		mastu_path = "/home/ffederic/work/SOLPS/dscan/" + SOLPS_case

		# types = ['midplane','as_up_as_possible','as_up_as_possible_MU04','as_low_as_possible','x_point']
		types = ['as_up_as_possible_MU04','x_point']	# in reality these are the only one that matter and are usefull
		all_spectra = []
		for i in range(len(types)):
			try:
				gna = np.load(mastu_path + '/spectra_'+'enables_'+str(enables)+'_'+types[i]+'.npz')
				gna.allow_pickle=True
				spectrum = gna['arr_0'].all()	# samples of spectral radiance: W/m2/str/nm
				all_spectra.append(spectrum)
			except:
				all_spectra.append([])




		if True:
			plt.figure(2)
			plt.title('Spectra '+mastu_path[20:]+'\n enables '+str(enables))
			# plt.plot(spectrum.wavelengths,spectrum.samples)
			vmin = np.inf
			for i in range(len(all_spectra)):
				try:
					dwave = np.median(np.diff(all_spectra[i].wavelengths))
					plt.plot(all_spectra[i].wavelengths,all_spectra[i].samples,'--',label=types[i],color='C'+str(i+1))
					vmin=min(vmin,all_spectra[i].samples[all_spectra[i].wavelengths.argmax()])
					print(SOLPS_case+' enables '+str(enables)+''+types[i]+' %.3g W/m2/str' %(np.trapz(all_spectra[i].samples,all_spectra[i].wavelengths)))
					integrated_power.append(np.trapz(all_spectra[i].samples,all_spectra[i].wavelengths))

					# a=plt.plot(1239.8/all_spectra[i].wavelengths,all_spectra[i].samples,'--',label=types[i])
					# a=plt.plot(1239.8/all_spectra[i].wavelengths,all_spectra[i].samples*dwave,'--',label=types[i])
					# plt.plot(1239.8/all_spectra[i].wavelengths,all_spectra[i].samples*dwave,'|',color=a[0].get_color())
					# plt.plot(1239.8/((all_spectra[i].wavelengths[1:]+all_spectra[i].wavelengths[:-1])/2),(all_spectra[i].samples[1:]+all_spectra[i].samples[:-1])/2*dwave/np.abs(np.diff(1239.8/all_spectra[i].wavelengths)),'--',label=types[i])
				except:
					pass
			plt.ylabel('W/m2/sr/nm')
			plt.xlabel('nm')
			plt.grid()
			plt.xlim(left=0,right=2)
			plt.ylim(bottom=vmin)
			plt.semilogy()

			plt.legend(loc='best', fontsize='xx-small')
			# plt.grid()
			# plt.xlabel('eV')
			# plt.xlim(left=1,right=100000)
			# plt.ylabel('W/m2/sr')
			# plt.ylim(bottom=1E-6,top=1e3)
			# # plt.ylabel('W/m2/sr/eV')
			# # plt.ylim(bottom=1E-3,top=1e6)
			# plt.semilogx()
			# plt.semilogy()
			# plt.ylim(bottom=1E-6,top=1e5)
			# plt.pause(0.01)
		else:
			pass

		all_integral = []
		for i_ in range(len(types)):
			try:
				dwave = np.median(np.diff(all_spectra[i_].wavelengths))
				integral = [0]
				for i in range(1,len(all_spectra[i_].wavelengths)):
					integral.append(integral[-1]+np.sum(all_spectra[i_].samples[-i-1:len(all_spectra[i_].wavelengths)-i+1])*dwave/2)
				integral = np.array(integral)
			except:
				integral = [0]
			all_integral.append(integral)
			# for i in range(1,len(spectrum.wavelengths)):
			# 	integral.append(integral[-1]+np.sum(spectrum.samples[-i-1:len(spectrum.wavelengths)-i+1])*np.diff(spectrum.wavelengths[-i-1:len(spectrum.wavelengths)-i+1])[0]/2)
			# integral = np.array(integral)

		if False:
			plt.figure()
			plt.title('Integrated spectra '+mastu_path[20:]+'\n enables '+str(enables))
			# plt.plot(spectrum.wavelengths,np.flip(integral,axis=0)/np.max(integral))
			for i in range(len(all_spectra)):
				try:
					plt.plot(all_spectra[i].wavelengths,np.flip(all_integral[i],axis=0)/np.max(all_integral[i]),'--',label=types[i])
				except:
					pass
			# plt.semilogy()
			plt.legend(loc='best', fontsize='xx-small')
			plt.xlabel('nm')
			plt.grid()
			plt.ylim(bottom=0.9,top=1.02)
			plt.xlim(left=0,right=2)
			plt.pause(0.01)
			# plt.show()
			# tm.sleep(60*60)
		else:
			pass


		# now I have to look for what is the absorption level for every wavelength bin
		# foil components thicknesses
		C_thickness = 250*1e-9	# for 1 side
		C_thickness *= 2	# total
		Pt_thickness = 250*1e-9	# for 1 side
		Pt_thickness *= 2	# total
		Ti_thickness = 1000*1e-9
		Re_thickness = 0*1e-9	# for 1 side
		Re_thickness *= 2	# total
		W_thickness = 0*1e-9	# for 1 side
		W_thickness *= 2	# total
		Os_thickness = 0*1e-9	# for 1 side
		Os_thickness *= 2	# total
		Ta_thickness = 50*1e-9	# for 1 side
		Ta_thickness *= 2	# total
		Hf_thickness = 0*1e-9	# for 1 side
		Hf_thickness *= 2	# total


		fraction_absorbed_photons = 1 - \
		np.exp(-linear_coefficient_C_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*C_thickness) * \
		np.exp(-linear_coefficient_Pt_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Pt_thickness) * \
		np.exp(-linear_coefficient_Re_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Re_thickness) * \
		np.exp(-linear_coefficient_W_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*W_thickness) * \
		np.exp(-linear_coefficient_Os_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Os_thickness) * \
		np.exp(-linear_coefficient_Ti_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Ti_thickness) * \
		np.exp(-linear_coefficient_Ta_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Ta_thickness)

		if False:
			plt.figure()
			# plt.plot(1239.8/all_spectra[i_].wavelengths,fraction_absorbed_photons)
			# plt.xlabel('eV')
			# plt.semilogx()
			# plt.semilogy()
			# plt.xlim(left=1,right=100000)
			# plt.plot(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9),fraction_absorbed_photons)
			# plt.plot(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9),1/linear_coefficient_Pt_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*1e6)
			# plt.plot(all_spectra[i_].wavelengths,fraction_absorbed_photons,label='C=%.2g Pt=%.2g Ti=%.2g Re=%.2g W=%.2g Os=%.2g Ta=%.2g P=%.3g' %(C_thickness/1000*1e9,Pt_thickness/1000*1e9,Ti_thickness/1000*1e9,Re_thickness/1000*1e9,W_thickness/1000*1e9,Os_thickness/1000*1e9,Ta_thickness/1000*1e9,np.trapz(all_spectra[0].samples*fraction_absorbed_photons,all_spectra[i_].wavelengths)/np.trapz(all_spectra[0].samples,all_spectra[i_].wavelengths)))
			plt.plot(all_spectra[i_].wavelengths,fraction_absorbed_photons,label='C=%.2g Pt=%.2g Ti=%.2g Re=%.2g W=%.2g Os=%.2g Ta=%.2g P=%.3g' %(C_thickness/1000*1e9,Pt_thickness/1000*1e9,Ti_thickness/1000*1e9,Re_thickness/1000*1e9,W_thickness/1000*1e9,Os_thickness/1000*1e9,Ta_thickness/1000*1e9,np.trapz(power_density*fraction_absorbed_photons,all_spectra[i_].wavelengths)/np.trapz(power_density,all_spectra[i_].wavelengths)))
			print('C=%.2g Pt=%.2g Ti=%.2g Re=%.2g W=%.2g Os=%.2g Ta=%.2g P=%.3g' %(C_thickness/1000*1e9,Pt_thickness/1000*1e9,Ti_thickness/1000*1e9,Re_thickness/1000*1e9,W_thickness/1000*1e9,Os_thickness/1000*1e9,Ta_thickness/1000*1e9,np.trapz(power_density*fraction_absorbed_photons,all_spectra[i_].wavelengths)/np.trapz(power_density,all_spectra[i_].wavelengths)))
			plt.xlabel('nm')
			plt.xlim(left=0,right=2)
			plt.grid()
			plt.legend(loc='best', fontsize='xx-small')
			# plt.ylim(bottom=0.5,top=1.02)
			# plt.xlim(left=10,right=30000)

			plt.figure()
			plt.title('Spectra '+mastu_path[20:]+'\n enables '+str(enables))
			# plt.plot(spectrum.wavelengths,spectrum.samples)
			vmin = np.inf
			for i in range(len(all_spectra)):
				try:
					dwave = np.median(np.diff(all_spectra[i].wavelengths))
					a=plt.plot(all_spectra[i].wavelengths,all_spectra[i].samples*dwave,'--',label=types[i])
					plt.plot(all_spectra[i].wavelengths,all_spectra[i].samples*fraction_absorbed_photons*dwave,'-',color=a[0].get_color())
					vmin=min(vmin,all_spectra[i].samples[all_spectra[i].wavelengths.argmax()])
					# a=plt.plot(1239.8/all_spectra[i].wavelengths,all_spectra[i].samples*dwave,'--',label=types[i])
					# plt.plot(1239.8/all_spectra[i].wavelengths,all_spectra[i].samples*dwave,'|',color=a[0].get_color())
					# plt.plot(1239.8/all_spectra[i].wavelengths,all_spectra[i].samples*fraction_absorbed_photons*dwave,'-',color=a[0].get_color())
					# plt.plot(1239.8/((all_spectra[i].wavelengths[1:]+all_spectra[i].wavelengths[:-1])/2),(all_spectra[i].samples[1:]+all_spectra[i].samples[:-1])/2*dwave/np.abs(np.diff(1239.8/all_spectra[i].wavelengths)),'--',label=types[i])
				except:
					pass
			plt.grid()
			plt.ylabel('W/m2/sr')
			plt.xlabel('nm')
			plt.xlim(left=0,right=2)
			plt.semilogy()
			plt.legend(loc='best', fontsize='xx-small')
			plt.ylim(bottom=vmin)
			# plt.grid()
			# plt.xlabel('eV')
			# plt.xlim(left=1,right=100000)
			# plt.ylabel('W/m2/sr')
			# plt.ylim(bottom=1E-6,top=1e3)
			# plt.semilogy()
			# plt.xlim(right=20000)
			# # plt.ylabel('W/m2/sr/eV')
			# # plt.ylim(bottom=1E-3,top=1e6)
			# plt.semilogx()
			# plt.ylim(bottom=1E-6,top=1e5)
			# plt.pause(0.01)
		else:
			pass


		all_absorbed_power = []
		all_emitted_power = []
		for i_ in range(len(types)):
			try:
				dwave = np.median(np.diff(all_spectra[i_].wavelengths))
				# absorbed_power = 0
				# for i in range(1,len(all_spectra[i_].wavelengths)):
				# 	absorbed_power += np.sum(all_spectra[i_].samples[i-1:i+1] * fraction_absorbed_photons[i-1:i+1])*dwave/2
				absorbed_power = np.trapz(all_spectra[i_].samples*fraction_absorbed_photons,all_spectra[i_].wavelengths)
				emitted_power = np.trapz(all_spectra[i_].samples,all_spectra[i_].wavelengths)
			except:
				absorbed_power=0
				emitted_power=0
			all_absorbed_power.append(absorbed_power)
			all_emitted_power.append(emitted_power)
		all_absorbed_power = np.array(all_absorbed_power)
		all_emitted_power = np.array(all_emitted_power)


		print('Spectra '+mastu_path[20:]+'\n enables '+str(enables))
		print(['midplane','as_up_as_possible','as_up_as_possible_MU04','as_low_as_possible','x_point'])
		print(all_absorbed_power/all_emitted_power)



	# this works reasonably well.
	# just to make sure that i'm not doing some mistake, i just want to calculate the spectra from bremsstrahlung 0d, and see hos much it is per wavelength
	# taking it from wikipedia		https://en.wikipedia.org/wiki/Bremsstrahlung#In_plasma
	from scipy.special import exp1
	wave_range = np.linspace(0.1,1200,1200000)*10**-9
	wave_energy = scipy.constants.Planck*scipy.constants.c/wave_range*6.242e+18
	ne = 1*10**21	# #/m3
	Te = 1000	# eV
	ang_freq = 2*np.pi*scipy.constants.c/wave_range
	y = 0.5*(( scipy.constants.Planck*ang_freq/(2*np.pi*Te*1.60218e-19) )**2)
	plasma_frequency = (ne* (scipy.constants.elementary_charge**2) / (scipy.constants.epsilon_0*scipy.constants.electron_mass))**0.5
	zeta_i = 1	# let's see hydrogen
	n_i = ne*zeta_i	# #/m3
	# power_density = 8*((2**0.5)/(3*(np.pi**0.5))) * (scipy.constants.elementary_charge**2/(2*np.pi*scipy.constants.epsilon_0))**3 * (1/(scipy.constants.electron_mass* scipy.constants.c**2)**(3/2)) * (1 - (plasma_frequency / ang_freq)**2)**0.5 * zeta_i**2 * ne*n_i * exp1(y) / (Te**0.5)
	# power_density = 8*(2**0.5)/(3*(np.pi**0.5)) * (scipy.constants.elementary_charge**2/(2*np.pi*scipy.constants.epsilon_0))**3 * 1/(scipy.constants.electron_mass* scipy.constants.c**2)**(3/2) * (1 - (plasma_frequency**2) / (ang_freq**2))**0.5 * zeta_i**2 * ne*n_i * (-np.log(y) * np.exp(0.577)) / (Te**0.5)
	# power_density = (np.log(Te/wave_energy)) * np.exp(-wave_energy/(Te))
	# power_density = (np.log(2* (Te/6.242E+18)**(3/2)/(ang_freq * scipy.constants.elementary_charge**2 * scipy.constants.electron_mass**0.5))) * np.exp(-wave_energy/(Te))
	power_density = 1/(wave_range**2) * np.exp(-wave_energy/(Te))	# this is from eq 6.83 in Kunze 2009. from there it neglects the gaunt factor, that should be ~1

	temp = []
	for multiplier in np.linspace(0.1,8,20):
		C_thickness = 250*1e-9	# for 1 side
		C_thickness *= 2	# total
		Pt_thickness = multiplier*100*1e-9	# for 1 side
		Pt_thickness *= 2	# total
		Ti_thickness = 400*1e-9
		Re_thickness = multiplier*100*1e-9	# for 1 side
		Re_thickness *= 2	# total
		W_thickness = 0*1e-9	# for 1 side
		W_thickness *= 2	# total
		Os_thickness = 0*1e-9	# for 1 side
		Os_thickness *= 2	# total


		fraction_absorbed_photons = 1 - \
		np.exp(-linear_coefficient_C_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*C_thickness) * \
		np.exp(-linear_coefficient_Pt_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Pt_thickness) * \
		np.exp(-linear_coefficient_Re_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Re_thickness) * \
		np.exp(-linear_coefficient_W_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*W_thickness) * \
		np.exp(-linear_coefficient_Os_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Os_thickness) * \
		np.exp(-linear_coefficient_Ti_attenuation_interpolator(scipy.constants.c*scipy.constants.h*(scipy.constants.physical_constants['joule-electron volt relationship'][0])/(all_spectra[i_].wavelengths*1e-9))*Ti_thickness)

		absorbed_power = np.trapz(power_density*fraction_absorbed_photons,wave_range)
		radiated_power = np.trapz(power_density,wave_range)
		print(absorbed_power/radiated_power)
		temp.append(absorbed_power/radiated_power)


	# I need to add the core


	# ask omkar, what it is the G file you used for this simulations
	# create equilibrium object
	from cherab.tools.equilibrium import import_eqdisk
	eq = import_eqdisk(filename_of_g_file)

	from cherab.core import Plasma
	plasma_core = Plasma()
	plasma_core.parent=world


	temperature_3D = eq.map3d(psy,temp)	# same 3d

	from cherab.core import Maxwellian
	bulk_velocity = Vector3D(0,0,0)
	atomic_mass = deuterium.mass_number
	d1_distribution = Maxwellian(density_3D, temperature_3D,bulk_velocity, atomic_mass)
	d1_species = Species(deuterium, 1, d1_distribution)
	plasma.composition.add(d1_species)

	help(Plasma)


else:
	pass

# # 10/01/2019 for Matt, send him this data in ASCII format
#
# os.chdir("/home/ffederic/work/analysis scripts/")
# grid_R_flat = grid_x.values.flatten()
# np.savetxt('grid_R.txt', grid_R_flat)
# grid_Z_flat = grid_y.values.flatten()
# np.savetxt('grid_Z.txt', grid_Z_flat)
# total_radiation_density_flat = np.abs(total_radiation_density.values).flatten()
# np.savetxt('total_radiation_density.txt', total_radiation_density_flat)





# CHERAB section




# os.chdir("/home/ffederic/work/cherab/cherab_mastu/diagnostics/bolometry/irvb/")
# os.chdir("/home/ffederic/work/cherab/raysect/raysect")
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
IRVB_CAD_file = "IRVB_camera_no_backplate_4mm.stl"
irvb_cad = import_stl(IRVB_CAD_file, parent=world, material=AbsorbingSurface(), name="IRVB")


if False:	# (SOLPS?) radiation phantom
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


	plt.figure()
	X = np.arange(0, grid_x.values.max(), 0.001)
	Y = np.arange(grid_y.values.min(), grid_y.values.max(), 0.001)
	rad_test = np.zeros((len(X), len(Y)))
	grid_x2 = np.zeros((len(X), len(Y)))
	grid_y2 = np.zeros((len(X), len(Y)))
	tot_power = 0
	for ix, x in enumerate(X):
		for jy, y in enumerate(Y):
			rad_test[ix, jy] = radiation_function(x, 0, y)
			grid_x2[ix, jy] = x
			grid_y2[ix, jy] = y
			# if y<-1.42:
			tot_power+=radiation_function(x, 0, y)*0.001*0.001*2*np.pi*x


	plt.pcolor(grid_x2, grid_y2, rad_test, norm=LogNorm(vmin=1000, vmax=total_radiation_density.values.max()),cmap='rainbow')
	# plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values),cmap='rainbow')
	ax.set_ylim(top=-0.5)
	ax.set_xlim(left=0.)
	plt.title('Emissivity profile as imported from SOLPS using CHERAB utilities')
	plt.colorbar().set_label('Emissivity [W/m^3]')
	plt.xlabel('R [m]')
	plt.ylabel('Z [m]')

	x=np.linspace(0.55-0.075,0.55+0.075,10)
	y=-1.2+np.sqrt(0.08**2-(x-0.55)**2)
	y_=-1.2-np.sqrt(0.08**2-(x-0.55)**2)
	plt.plot(x,y,'k')
	plt.plot(x,y_,'k')

	plt.pause(0.01)


	emitter_material = RadiatedPower(radiation_function)
	outer_radius = cr_r.max() + 0.01
	plasma_height = cr_z.max() - cr_z.min()
	lower_z = cr_z.min()
	radiator = Cylinder(outer_radius, plasma_height, material=emitter_material, parent=world,transform=translate(0, 0, lower_z))
else:
	pass


# THIS IS FOR UNIFORM EMITTED POWER DENSITY
total_power=500000
center_radiator=[0.55,-1.2]
#
# TOROIDAL RADIATOR
if False:
	radiator_file = 'radiator_R0.55_Z-1.2_r0.08.stl'
	minor_radius_radiator=0.08
	volume_radiator=2*(np.pi**2)*center_radiator[0]*minor_radius_radiator**2
	power_density = total_power / volume_radiator / (4*np.pi)
elif False:
	radiator_file = '2x_radiator_R0.55_Z-1.2-1.3_r0.04.stl'
	minor_radius_radiator=0.04
	volume_radiator=2*2*(np.pi**2)*center_radiator[0]*minor_radius_radiator**2
	power_density =  total_power / volume_radiator / (4*np.pi)
else:
	radiator_file = 'radiator_all_core_and_divertor.stl'
	power_density =  50000 / (4*np.pi)

# minor_radius_radiator=0.08
# volume_radiator=2*(np.pi**2)*center_radiator[0]*minor_radius_radiator**2
# power_density = total_power / volume_radiator / (4*np.pi)
#
# # SQUARE ANULAR RADIATOR
# # side_radiator=0.14
# # volume_radiator=np.pi*((center_radiator[0]+side_radiator/2)**2-(center_radiator[0]-side_radiator/2)**2)*side_radiator
# # x_point_lower = Point2D(center_radiator[0] - side_radiator/2, center_radiator[1] - side_radiator/2)
# # x_point_upper = Point2D(center_radiator[0] + side_radiator/2, center_radiator[1] + side_radiator/2)
# power_density = total_power / volume_radiator / (4*np.pi)
#
# # FIXED EMISSIVITY IN CORE
# power_density_core=50000 / (4*np.pi)


from cherab.tools.primitives.annulus_mesh import generate_annulus_mesh_segments
# generate_annulus_mesh_segments(x_point_lower, x_point_upper, 360, world, material=UniformVolumeEmitter(ConstantSF(power_density)))
radiator = import_stl(radiator_file, material=UniformVolumeEmitter(ConstantSF(power_density)), parent=world)
# radiator = import_stl('radiator_all_closed_surface.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
# radiator = import_stl('radiator_all_super_x_divertor.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
# radiator = import_stl('radiator_all_core.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
# radiator = import_stl('radiator_all_core_and_divertor.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
# radiator = import_stl('test.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)


pinhole_centre = Point3D(1.491933, 0, -0.7198).transform(rotate_z(135-0.76004))
pinhole_target = Sphere(0.005, transform=translate(*pinhole_centre), parent=world, material=NullMaterial())


stand_off=0.060
CCD_radius=1.50467+stand_off
CCD_angle=135*(np.pi*2)/360

ccd_centre = Point3D(CCD_radius*np.cos(CCD_angle), CCD_radius*np.sin(CCD_angle), -0.699522)
ccd_normal = Vector3D(-CCD_radius*np.cos(CCD_angle), -CCD_radius*np.sin(CCD_angle),0).normalise()
ccd_y_axis = Vector3D(0,0,1).normalise()
ccd_x_axis = ccd_y_axis.cross(ccd_normal)


pixel_h=350
pixel_v=(pixel_h*9)//7

plt.ion()
power = PowerPipeline2D()
detector = TargettedCCDArray(targets=[pinhole_target], width=0.07, pixels=(pixel_h, pixel_v), targetted_path_prob=1.0,
							 parent=world, pipelines=[power],
							 transform=translate(*ccd_centre)*rotate_basis(ccd_normal, ccd_y_axis))
detector.max_wavelength = 601
detector.min_wavelength = 600
# detector.pixel_samples = 500//(1/(5*5))
detector.pixel_samples = 5000

detector.observe()


if True:
	pixel_area = 0.07*(0.07*pixel_v/pixel_h)/(pixel_h*pixel_v)
	measured_power = power.frame.mean / pixel_area
	measured_power = np.flip(np.transpose(measured_power),axis=-1)
	np.save('/home/ffederic/work/irvb/0__outputs/measured_power_'+IRVB_CAD_file[-7:-6]+'_'+str(int(stand_off*1e3))+radiator_file,measured_power)
else:
	# stand_off = 0.045	# m
	measured_power = np.load('/home/ffederic/work/irvb/0__outputs/measured_power_'+IRVB_CAD_file[-7:-6]+'_'+str(int(stand_off*1e3))+radiator_file+'.npy')

measured_power_filtered=median_filter(measured_power,size=[3,3])

pinhole_offset = np.array([-0.0198,-0.0198])	# toroidal direction parallel to the place surface, z
# pinhole_offset_extra = np.array([+0.012/(2**0.5),-0.012/(2**0.5)])
pinhole_offset_extra = np.array([0,0])
# stand_off = 0.045	# m
# Rf=1.54967	# m	radius of the centre of the foil
Rf=1.48967 + 0.01 + 0.003 + 0.002 + stand_off	# m	radius of the centre of the foil
plane_equation = np.array([1,-1,0,2**0.5 * Rf])	# plane of the foil
centre_of_foil = np.array([-Rf/(2**0.5), Rf/(2**0.5), -0.7])	# x,y,z
pinhole_offset += pinhole_offset_extra
pinhole_location = coleval.locate_pinhole(pinhole_offset=pinhole_offset)

fueling_point_location_on_foil = coleval.return_fueling_point_location_on_foil(plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
structure_point_location_on_foil = coleval.return_structure_point_location_on_foil(plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)

if True:	# plot for the paper
	plt.figure(figsize=(12,7))
	# plt.figure(figsize=(12,5))
	cmap = plt.cm.get_cmap('rainbow',20)
	cmap.set_under(color='white')
	vmin=100
	# vmin=measured_power[measured_power>0].min()
	levels = np.unique((np.ceil(np.arange(measured_power_filtered.max(),0,-(measured_power_filtered.max()-vmin)/10)).astype(int)).tolist() + [0])
	im = plt.contourf(measured_power_filtered,levels=levels,cmap=cmap, vmin=vmin,origin='lower')
	# im = plt.imshow(measured_power,origin='lower', cmap=cmap, vmin=vmin)
	cv0 = np.zeros(measured_power.shape)
	foil_size = [0.07,0.09]
	structure_alpha=0.5
	# for i in range(len(fueling_point_location_on_foil)):
	# 	plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
	# 	plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
	for i in range(len(structure_point_location_on_foil)):
		plt.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
	plt.gca().set_aspect(1)
	plt.ylim(bottom=0)
	plt.xlim(left=0)
	# plt.ylim(bottom=120,top=300)
	# plt.xlim(left=100)
	# plt.colorbar(im,fraction=0.0227, pad=0.02).set_label('Foil power density [W/m^2]\ncut-off %.3gW/m^2' %(vmin))
	plt.colorbar(im,fraction=0.0227, pad=0.02).set_label('Foil power density [W/m^2] cut-off %.3gW/m^2' %(vmin))
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	try:
		plt.title('Power density on the foil generated via CHERAB simulation\nRadiator of r=%.3gm at R=%.3gm, Z=%.3gm, %.3gMW\nstandoff=%.3gm, pinhole=%.3gmm\n' %(minor_radius_radiator,center_radiator[0],center_radiator[1],total_power*1e-6,stand_off,int(IRVB_CAD_file[-7:-6])))
	except:
		pass
	# plt.pause(0.01)
	plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/measured_power_'+IRVB_CAD_file[-7:-6]+'_'+str(int(stand_off*1e3))+radiator_file+'.eps', bbox_inches='tight')
	plt.close()
