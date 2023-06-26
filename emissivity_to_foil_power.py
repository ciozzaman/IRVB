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





if False:	# related to the SOLPS phantom
	mastu_path = "/home/ffederic/work/SOLPS/seeding/seed_1"
	# mastu_path = "/home/ffederic/work/SOLPS/dscan/ramp_10"
	if not mastu_path in sys.path:
		sys.path.append(mastu_path)
	ds_puff8 = xr.open_dataset('/home/ffederic/work/SOLPS/seeding/seed_1/balance.nc', autoclose=True).load()
	# ds_puff8 = xr.open_dataset('/home/ffederic/work/SOLPS/dscan/ramp_10/balance.nc', autoclose=True).load()

	grid_x = ds_puff8.crx.mean(dim='4')
	grid_y = ds_puff8.cry.mean(dim='4')
	impurity_radiation = ds_puff8.b2stel_she_bal.sum('ns')
	hydrogen_radiation = ds_puff8.eirene_mc_eael_she_bal.sum('nstra') - ds_puff8.eirene_mc_papl_sna_bal.isel(ns=1).sum('nstra') * 13.6 * 1.6e-19   # ds_puff8.eirene_mc_eael_she_bal.sum('nstra') is the total electron energy sink due to plasma / atoms interactions, including ionisation/excitation (dominant) and charge exchange (negligible). Here we assume all not used for ionisation goes to radiation, including the CX bit.
	total_radiation = -hydrogen_radiation + impurity_radiation
	total_radiation_density = -np.divide(hydrogen_radiation + impurity_radiation,ds_puff8.vol)

	fig, ax = plt.subplots()
	# grid_x.plot.line(ax=ax, x='nx_plus2')
	# grid_y.plot.line(ax=ax, x='ny_plus2')
	# plt.pcolormesh(grid_x.values, grid_y.values, impurity_radiation.values)


	plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values), norm=LogNorm(vmin=1000, vmax=total_radiation_density.values.max()),cmap='rainbow')
	# plt.pcolor(grid_x.values, grid_y.values, np.abs(total_radiation_density.values),cmap='rainbow')
	ax.set_ylim(top=-0.5)
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

	plt.ion()

	xl, xu = (0.0, 2.0)
	yl, yu = (-2.0, 2.0)

	print('CHERAB solps_from_balance demo')
	print('Note: code assumes presence of deuterium and carbon species in SOLPS run')
	print('Enter name of balance.nc file:')
	filename = input()

	sim = load_solps_from_balance(filename)
	plasma = sim.create_plasma()
	mesh = sim.mesh

	d0 = plasma.composition.get(deuterium, 0)
	d1 = plasma.composition.get(deuterium, 1)
	c0 = plasma.composition.get(carbon, 0)
	c1 = plasma.composition.get(carbon, 1)
	c2 = plasma.composition.get(carbon, 2)
	c3 = plasma.composition.get(carbon, 3)
	c4 = plasma.composition.get(carbon, 4)
	c5 = plasma.composition.get(carbon, 5)
	c6 = plasma.composition.get(carbon, 6)
	n0 = plasma.composition.get(nitrogen, 0)
	n1 = plasma.composition.get(nitrogen, 1)
	n2 = plasma.composition.get(nitrogen, 2)
	n3 = plasma.composition.get(nitrogen, 3)
	n4 = plasma.composition.get(nitrogen, 4)
	n5 = plasma.composition.get(nitrogen, 5)
	n6 = plasma.composition.get(nitrogen, 6)
	n7 = plasma.composition.get(nitrogen, 7)

	te_samples = np.zeros((500, 500))
	ne_samples = np.zeros((500, 500))
	d0_samples = np.zeros((500, 500))
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



	for i, x in enumerate(xrange):
	    for j, y in enumerate(yrange):
	        ne_samples[j, i] = plasma.electron_distribution.density(x, 0.0, y)
	        te_samples[j, i] = plasma.electron_distribution.effective_temperature(x, 0.0, y)
	        d0_samples[j, i] = d0.distribution.density(x, 0.0, y)
	        d1_samples[j, i] = d1.distribution.density(x, 0.0, y)
	        c0_samples[j, i] = c0.distribution.density(x, 0.0, y)
	        c1_samples[j, i] = c1.distribution.density(x, 0.0, y)
	        c2_samples[j, i] = c2.distribution.density(x, 0.0, y)
	        c3_samples[j, i] = c3.distribution.density(x, 0.0, y)
	        c4_samples[j, i] = c4.distribution.density(x, 0.0, y)
	        c5_samples[j, i] = c5.distribution.density(x, 0.0, y)
	        c6_samples[j, i] = c6.distribution.density(x, 0.0, y)
	        n0_samples[j, i] = n0.distribution.density(x, 0.0, y)
	        n1_samples[j, i] = n1.distribution.density(x, 0.0, y)
	        n2_samples[j, i] = n2.distribution.density(x, 0.0, y)
	        n3_samples[j, i] = n3.distribution.density(x, 0.0, y)
	        n4_samples[j, i] = n4.distribution.density(x, 0.0, y)
	        n5_samples[j, i] = n5.distribution.density(x, 0.0, y)
	        n6_samples[j, i] = n6.distribution.density(x, 0.0, y)
	        n7_samples[j, i] = n7.distribution.density(x, 0.0, y)


	from raysect.optical import Ray,Spectrum,World
	world = World()
	plasma.parent=world
	from cherab.mastu.machine import MASTU_FULL_MESH
	import os
	from raysect.primitive import import_stl, Sphere, Mesh, Cylinder
	from raysect.optical.material.absorber import AbsorbingSurface


	# for cad_file in MASTU_FULL_MESH:
	# 	directory, filename = os.path.split(cad_file[0])
	# 	name, ext = filename.split('.')
	# 	print("importing {} ...".format(filename))
	# 	Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)
	from cherab.mastu.machine import import_mastu_mesh
	import_mastu_mesh(world)

	from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
	ray = Ray(origin=Point3D(-1.5,0,0),direction=Vector3D(1,0,0),min_wavelength=1,max_wavelength=600,bins=600)

	from cherab.openadas import OpenADAS
	plasma.atomic_data = OpenADAS(permit_extrapolation=True)

	from cherab.core.model import ExcitationLine,RecombinationLine,Bremsstrahlung
	from cherab.core.atomic import Line

	# line = Line(carbon, 5,(2,1))	# loop on the destination state and start state for every ionisation level
	# model = ExcitationLine(line)
	# plasma.models.add(model)
	#
	#
	# line = Line(carbon, 5,(2,1))	# loop on the destination state for every ionisation level
	# model = RecombinationLine(line)
	# plasma.models.add(model)

	line = Line(deuterium, 0,(2,1))	# loop on the destination state and start state for every ionisation level
	model = ExcitationLine(line)
	plasma.models.add(model)


	plasma.models.add(Bremsstrahlung())

	spectrum = ray.trace(world)

	plt.figure()
	plt.plot(spectrum.wavelengths,spectrum.samples)



	# I need to add the core


	# ask omkar, what it is the G file you used for this simulations
	create equilibrium object
	from cherab.tools.equilibrium import import_eqdisk
	eq = import_eqdisk(filename of g file)

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
