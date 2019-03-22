# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis scripts/preamble_indexing.py").read())





from cherab.mastu.bolometry import load_default_bolometer_config, load_standard_voxel_grid
from raysect.optical import World
world = World()
core_voxel_grid = load_standard_voxel_grid('core',parent=world)








from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z, translate, rotate_basis
from raysect.primitive import import_stl, Mesh
from raysect.optical import World
from raysect.optical.material.absorber import AbsorbingSurface

from raysect.optical.observer import TargettedCCDArray, PowerPipeline2D



os.chdir("/home/ffederic/work/cherab/cherab_mastu/")
from cherab.mastu.machine import MASTU_FULL_MESH



world = World()


for cad_file in MASTU_FULL_MESH:
	directory, filename = os.path.split(cad_file[0])
	name, ext = filename.split('.')
	print("importing {} ...".format(filename))
	Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)
#
# os.chdir("/home/ffederic/work/analysis scripts/irvb/")
# irvb_cad = import_stl("IRVB_camera_no_backplate_4mm.stl", parent=world, material=AbsorbingSurface(), name="IRVB")






















from raysect.core.math import Point2D, Point3D, Vector3D, rotate_z


pinhole_centre = Point3D(1.491933, 0, -0.7198).transform(rotate_z(135-0.76004))
pinhole_diameter = 0.004
pinhole_radius = pinhole_diameter/2
pinhole_normal = pinhole_centre.vector_to(Point3D(0, 0, -0.7198)).normalise()
basis_z = Vector3D(0, 0, 1).normalise()
basis_pol=basis_z.cross(pinhole_normal).normalise()
pinhole_fake_corner1 = pinhole_centre + pinhole_radius*(2**0.5)*basis_z
pinhole_fake_corner2 = pinhole_centre - pinhole_radius*(2**0.5)*basis_z
pinhole_fake_corner3 = pinhole_centre + pinhole_radius*(2**0.5)*basis_pol
pinhole_fake_corner4 = pinhole_centre - pinhole_radius*(2**0.5)*basis_pol



# write the slit file
# writer = csv.writer(open("/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/detectors/irvb/irvb_slits.csv", 'w'))
header = ['# ID, P1x, P1y, P1z, P2x, P2y, P2z, P3x, P3y, P3z, P4x, P4y, P4z','# Slits are numbered from top down']
to_write='1, '+str(pinhole_fake_corner1)[8:-1]+', '+str(pinhole_fake_corner2)[8:-1]+', '+str(pinhole_fake_corner3)[8:-1]+', '+str(pinhole_fake_corner4)[8:-1]

with open('/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/detectors/irvb/irvb_slits.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(header[0].split(','))
    writer.writerow([header[1]])
    writer.writerow(to_write.split(','))

f.close()








stand_off=0.045
CCD_radius=1.50467+stand_off
CCD_angle=135*(np.pi*2)/360
foil_vertical = 0.09
foil_horizontal = 0.07

ccd_centre = Point3D(CCD_radius*np.cos(CCD_angle), CCD_radius*np.sin(CCD_angle), -0.699522)
ccd_normal = Vector3D(-CCD_radius*np.cos(CCD_angle), -CCD_radius*np.sin(CCD_angle),0).normalise()
ccd_z_axis = Vector3D(0,0,1).normalise()
ccd_x_axis = ccd_z_axis.cross(ccd_normal)

# The ordering of the points must be clockwise direction

nh=2
hh=(foil_horizontal)/nh
nv=2
hv=(foil_vertical)/nv

to_write = []
index=1
for i in range(nh):
    for j in range(nv):
        centre = ccd_centre + ccd_z_axis*foil_vertical/2 - ccd_x_axis*foil_horizontal/2 -ccd_z_axis*hv*i + ccd_x_axis*hh*j
        topleft = centre +ccd_z_axis*hv/2 - ccd_x_axis*hh/2
        topright = centre +ccd_z_axis*hv/2 + ccd_x_axis*hh/2
        bottomright = centre -ccd_z_axis*hv/2 + ccd_x_axis*hh/2
        bottomleft = centre -ccd_z_axis*hv/2 - ccd_x_axis*hh/2
        to_write.append(str(index)+', 1, '+str(topleft)[8:-1]+', '+str(topright)[8:-1]+', '+str(bottomright)[8:-1]+', '+str(bottomleft)[8:-1])
        index += 1


# foil_fake_corner1 = ccd_centre + foil_vertical/2*ccd_z_axis + foil_horizontal/2*ccd_x_axis
# foil_fake_corner2 = ccd_centre - foil_vertical/2*ccd_z_axis + foil_horizontal/2*ccd_x_axis
# foil_fake_corner3 = ccd_centre - foil_vertical/2*ccd_z_axis - foil_horizontal/2*ccd_x_axis
# foil_fake_corner4 = ccd_centre + foil_vertical/2*ccd_z_axis - foil_horizontal/2*ccd_x_axis

# write the foil file
# writer = csv.writer(open("/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/detectors/irvb/irvb_foils.csv", 'w'))
header = ['# ID, slit_id, P1x, P1y, P1z, P2x, P2y, P2z, P3x, P3y, P3z, P4x, P4y, P4z','# ']
# to_write='1, 1, '+str(foil_fake_corner1)[8:-1]+', '+str(foil_fake_corner2)[8:-1]+', '+str(foil_fake_corner3)[8:-1]+', '+str(foil_fake_corner4)[8:-1]

with open('/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/detectors/irvb/irvb_foils.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(header[0].split(','))
    writer.writerow([header[1]])
    for row in to_write:
        writer.writerow(row.split(','))

f.close()






from cherab.mastu.bolometry.detectors import load_irvb_camera
from raysect.core.workflow import SerialEngine,MulticoreEngine
import time

irvb = load_irvb_camera('IRVB',parent = world)


try:
    nslots = int(os.environ['NSLOTS'])
    if nslots == 1:
        irvb.render_engine = SerialEngine()
    else:
        irvb.render_engine = MulticoreEngine(nslots)
        print('nslots ',nslots)
except KeyError:
    irvb.render_engine = SerialEngine()
    pass

print('irvb.render_engine ',irvb.render_engine)

sensitivity_matrix = np.zeros((len(irvb), core_voxel_grid.count))
for i, detector in enumerate(irvb):
    start_time = time.time()
    sensitivities = detector.calculate_sensitivity(core_voxel_grid, ray_count=10000)
    sensitivity_matrix[i, :] = sensitivities
    print("Traced detector '{}' with run time - {:.2G}mins".format(detector.name, (time.time() - start_time) / 60))

np.save('/home/ffederic/work/analysis scripts/sensitivity matrix.npy',sensitivity_matrix)


index=0
for i in range(nh):
    for j in range(nv):
        core_voxel_grid.plot(voxel_values=sensitivity_matrix[index],title='Detector '+irvb[index].name+'the pixel '+str(i)+' horizontally and '+str(j)+' vertically \n (from top left looking at the centre column)')






# # this micro bit it only to see the orientation of how the data are in the files for JET
# a='3.43811, -3.61298, 0.332294, 3.43811, -3.61298, 0.337694, 3.43564, -3.61546, 0.337694, 3.43564, -3.61546, 0.332294'
#
# a=np.fromstring( a, dtype=float, sep=',' )
#
#
# b=[a[0]-0.001,a[3],a[6],a[9]]
# c=[a[1],a[4],a[7],a[10]]
# d=[a[2],a[5],a[8],a[11]]
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(b,c,d)
# ax.legend()
# plt.show()
#
#
# a=Point2D(1.6983, 1.5510), Point2D(1.6983, 1.5061), Point2D(1.7500, 1.5061), Point2D(1.7500, 1.5510)
#
# b=[a[0][0],a[1][0],a[2][0],a[3][0]]
# c=[a[0][1],a[1][1],a[2][1],a[3][1]]
# plt.plot(b,c)
# plt.show()






############################################################################################




# print('Going to execurte example from Jack Lovell')
#
# from raysect.optical import World
# from raysect.optical.material.absorber import AbsorbingSurface
# from raysect.primitive import Mesh, import_obj, Cylinder
# from raysect.core import translate, SerialEngine, MulticoreEngine
#
# from cherab.mastu.machine import MASTU_FULL_MESH
# os.chdir("/work/ffederic/cherab/cherab_core")
# from cherab.tools.inversions.voxels import ToroidalVoxelGrid
# from cherab.mastu.bolometry import load_default_bolometer_config
# # from cherab.mastu.bolometry.grid_construction.load_voxel_grids import load_standard_voxel_grid
# from cherab.mastu.bolometry import load_default_bolometer_config, load_standard_voxel_grid
# os.chdir("/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/detectors")
#
# def load_vessel_world(mesh_parts, shift_p5=False):
#     """Load the world containing the vessel mesh parts.
#
#     <mesh_parts> is a list of filenames containing mesh files in either
#     RSM or OBJ format, which are to be loaded into the world.
#
#     If shift_p5 is True, the mesh files representing the P5 coils will
#     have a downward shift applied to them to account for the UEP sag.
#     This is described in CD/MU/04783.
#
#     Returns world, the root of the scenegraph.
#     """
#     world = World()
#     for path in mesh_parts:
#         print("importing {}  ...".format(os.path.split(path[0])[1]))
#         filename = os.path.split(path[0])[-1]
#         name, ext = filename.split('.')
#         if 'P5_' in path[0] and shift_p5:
#             p5_zshift = -0.00485  # From CD/MU/04783
#             transform = translate(0, 0, p5_zshift)
#         else:
#             transform = None
#         if ext.lower() == 'rsm':
#             Mesh.from_file(path[0], parent=world, material=AbsorbingSurface(),
#                            transform=transform, name=name)
#         elif ext.lower() == 'obj':
#             import_obj(path[0], parent=world, material=AbsorbingSurface(), name=name)
#         else:
#             raise ValueError("Only RSM and OBJ meshes are supported.")
#     # Add a solid cylinder at R=0 to prevent rays finding their way through the
#     # gaps in the centre column armour. This is only necessary for the MAST
#     # meshes, but it does no harm to add it to MAST-U meshes too
#     height = 6
#     radius = 0.1
#     Cylinder(radius=radius, height=height, parent=world,
#              transform=translate(0, 0, -height / 2),
#              material=AbsorbingSurface(), name='Blocking Cylinder')
#     return world
#
#
# def calculate_and_save_sensitivities(grid_name, camera, mesh_parts,
#                                      istart=None, iend=None, shift_p5=False,
#                                      camera_transform=None):
#     """
#     Calculate the sensitivity matrices for an entire bolometer camera,
#     and save to a Numpy save file.
#
#     The output file is named as follows:
#     <grid>_<camera>_bolo.npy
#
#     If istart and iend are not none, the output file is named:
#     <grid>_<camera>_bolo_<istart>_<iend>.npy
#
#     Parameters:
#     grid: the name of the reconstruction grid to load
#     camera: the name of the bolometer camera
#     mesh_parts: the list of mesh files to read and load into the world
#     istart: starting index for grid cells
#     iend: final index for grid cells
#
#     If istart and iend are not none, a subset [istart, iend) of the grid
#     cell sensitivities are calculated.
#     """
#     world = load_vessel_world(mesh_parts, shift_p5)
#     # Trim off metadata from camera if it exists. This metadata is only for the
#     # name of the save file.
#     cam_nometa = camera.split("-")[0]
#     if grid_name == "sxdl":
#         bolo_name = "SXDL - {}".format(cam_nometa)
#     elif grid_name in("core", "core_high_res"):
#         bolo_name = "CORE - {}".format(cam_nometa)
#     else:
#         raise ValueError("Only 'sxd, core and core_high_res' grids supported.")
#     # Using a slice object for the voxel range means that even if
#     # iend > grid.count no error will be thrown. In this case, only the voxels
#     # from istart to grid.count will be returned
#     voxel_range = slice(istart, iend)
#     grid = load_standard_voxel_grid(grid_name, parent=world, voxel_range=voxel_range)
#
#     if istart is not None and iend is not None:
#         # Check whether a full range of cells was returned, or a smaller range due
#         # to iend > grid.count
#         iend = istart + grid.count
#         file_name = "{}_{}_bolo_{}_{}.npy".format(
#             grid_name.replace("_", "-"), camera.lower(), istart, iend
#         )
#         # Load the geometry from a local file, to prevent the UDA server
#         # being overloaded by many jobs making almost-simultaneous reads
#         cached_geom_file = "tmp_bolo_geom_{}.pickle".format(camera)
#         with open(cached_geom_file, "rb") as f:
#             bolo = pickle.load(f)
#             bolo.parent = world
#     else:
#         file_name = "{}_{}_bolo.npy".format(
#             grid_name.replace("_", "-"), camera.lower()
#         )
#         bolo = load_default_bolometer_config(bolo_name, parent=world, shot=50000)
#
#     if camera_transform is not None:
#         bolo.transform = camera_transform * bolo.transform
#
#     sensitivities = np.zeros((len(bolo), grid.count))
#     for i, detector in enumerate(bolo):
#         # If running as a batch job, use only the requested number of processes
#         try:
#             nslots = int(os.environ['NSLOTS'])
#             if nslots == 1:
#                 detector.render_engine = SerialEngine()
#             else:
#                 detector.render_engine = MulticoreEngine(nslots)
#         except KeyError:
#             pass
#         print('calculating detector {}'.format(detector.name))
#         sensitivities[i] = detector.calculate_sensitivity(grid)
#     np.save(file_name, sensitivities)
#
#
# def main():
#     camera = "Tangential"
#     # try:
#     #     istart = int(sys.argv[2])
#     #     iend = int(sys.argv[3])
#     # except (IndexError, ValueError):
#     istart = None
#     iend = None
#
#     shift_p5 = False
#     camera_transform = None
#
#     if camera in ("Poloidal", "Tangential"):
#         MESH_PARTS = MASTU_FULL_MESH
#         shift_p5 = True
#         grid = "core"
#     elif camera in ("PoloidalHighRes", "TangentialHighRes"):
#         MESH_PARTS = MASTU_FULL_MESH
#         shift_p5 = True
#         grid = "core_high_res"
#         # Trim off the HighRes suffix from the camera
#         camera = camera[:-7]
#     else:
#         raise ValueError("The following cameras are supported: "
#                          "'Poloidal', 'Tangential', "
#                          "'PoloidalHighRes', 'TangentialHighRes', ")
#
#     calculate_and_save_sensitivities(
#         grid, camera, MESH_PARTS, istart, iend, shift_p5, camera_transform
#     )
#
#
# if __name__ == "__main__":
#     main()

