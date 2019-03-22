# Created 10/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/preamble_indexing.py").read())







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
# plt.show()


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

plt.show()



# # THIS IS FOR UNIFORM EMITTED POWER DENSITY
# total_power=500000
# center_radiator=[0.55,-1.2]
#
# # TOROIDAL RADIATOR
# minor_radius_radiator=0.08
# volume_radiator=2*(np.pi**2)*center_radiator[0]*minor_radius_radiator**2
#
# # SQUARE ANULAR RADIATOR
# # side_radiator=0.14
# # volume_radiator=np.pi*((center_radiator[0]+side_radiator/2)**2-(center_radiator[0]-side_radiator/2)**2)*side_radiator
# # x_point_lower = Point2D(center_radiator[0] - side_radiator/2, center_radiator[1] - side_radiator/2)
# # x_point_upper = Point2D(center_radiator[0] + side_radiator/2, center_radiator[1] + side_radiator/2)
#
# power_density = total_power / volume_radiator / (4*np.pi)
# power_density_core=50000 / (4*np.pi)
#
# from cherab.tools.primitives.annulus_mesh import generate_annulus_mesh_segments
# # generate_annulus_mesh_segments(x_point_lower, x_point_upper, 360, world, material=UniformVolumeEmitter(ConstantSF(power_density)))
# # radiator = import_stl('radiator_R0.55_Z-1.2_r0.08.stl', material=UniformVolumeEmitter(ConstantSF(power_density)), parent=world)
# # radiator = import_stl('radiator_all_closed_surface.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
# # radiator = import_stl('radiator_all_super_x_divertor.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
# # radiator = import_stl('radiator_all_core.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
# radiator = import_stl('radiator_all_core_and_divertor.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)
# # radiator = import_stl('test.stl', material=UniformVolumeEmitter(ConstantSF(power_density_core)), parent=world)



emitter_material = RadiatedPower(radiation_function)
outer_radius = cr_r.max() + 0.01
plasma_height = cr_z.max() - cr_z.min()
lower_z = cr_z.min()
radiator = Cylinder(outer_radius, plasma_height, material=emitter_material, parent=world,transform=translate(0, 0, lower_z))


pinhole_centre = Point3D(1.491933, 0, -0.7198).transform(rotate_z(135-0.76004))
pinhole_target = Sphere(0.005, transform=translate(*pinhole_centre), parent=world, material=NullMaterial())


stand_off=0.045
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
detector.pixel_samples = 500

detector.observe()


pixel_area = 0.07*(0.07*pixel_v/pixel_h)/(pixel_h*pixel_v)
measured_power = power.frame.mean / pixel_area
measured_power = np.flip(np.transpose(measured_power),axis=-1)


cmap = plt.cm.rainbow
cmap.set_under(color='white')
vmin=1
plt.imshow(measured_power,origin='lower', cmap=cmap, vmin=vmin)
plt.title('Power density on the foil generated via CHERAB simulation')
plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off '+str(vmin)+'W/m^2')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.show()


np.save('measured_power',measured_power)