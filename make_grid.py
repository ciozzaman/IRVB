# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/preamble_indexing.py").read())










# 2019-01-15 I want to work on the inversion utility of CHERAB, is seems significantly simpler than using CALCAM


import pickle


exec(open("/home/ffederic/work/analysis_scripts/Jack_geometry.py").read())


# Here I try to make my own inversion grid
# rectangles must be given in anticlockwise order
from raysect.core.math import Point2D

z_min=-2.2
z_max=2.2
R_min=0.25
R_max=1.75


resolution=0.04 #[m]
nr=int(np.around((R_max-R_min)/resolution))
hr=(R_max-R_min)/nr
nz=int(np.around((z_max-z_min)/resolution))
hz=(z_max-z_min)/nz

CORE_RECTILINEAR_TEST_GRID = []
index=0
for i in range(nr):
    temp2=[]
    for j in range(nz):
        temp3=[]
        centre = Point2D(R_min+hr*(i+0.5),z_min+hz*(j+0.5))
        topleft = Point2D(centre.x-hr/2,centre.y+hz/2)
        topright = Point2D(centre.x+hr/2, centre.y+hz/2)
        bottomright = Point2D(centre.x+hr/2, centre.y-hz/2)
        bottomleft = Point2D(centre.x-hr/2, centre.y-hz/2)
        temp3.append(index)
        temp3.append(topleft)
        temp3.append(topright)
        temp3.append(bottomright)
        temp3.append(bottomleft)
        temp2.append(temp3)
        index += 1
    CORE_RECTILINEAR_TEST_GRID.append(temp2)
CORE_RECTILINEAR_REFERENCE_GRID = CORE_RECTILINEAR_TEST_GRID




import numpy as np
import matplotlib.pyplot as plt
import pyuda


client = pyuda.Client()

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


# if __name__ == '__main__':
#
plt.ion()
plt.figure()
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.axis('equal')



#######################################################################################



"""
This script demonstrates creating a Cherab ToroidalVoxelGrid from a list
of grid cells. It also calculates the grid extras, most importantly the
grid's laplacian operator which is used as a regularisation operator in
inversions.
The resulting grid is saved in the
cherab.mastu.bolometry.grid_construction directory. This makes the grid
available to use with the
cherab.mastu.bolometry.load_standard_voxel_grid function. This script
must be run before the load_standard_voxel_grid function is called, in
order to generate the necessary save file.
"""


enclosed_cells = []
grid_mask = np.empty((nr, nz), dtype=bool)
grid_index_2D_to_1D_map = {}
grid_index_1D_to_2D_map = {}


# Identify the cells that are enclosed by the polygon,
# simultaneously write out grid mask and grid map.
unwrapped_cell_index = 0
for ix in range(nr):
    for iy in range(nz):
        _, p1, p2, p3, p4 = CORE_RECTILINEAR_REFERENCE_GRID[ix][iy]

        # if any points are inside the polygon, retain this cell
        if (CORE_POLYGON_MASK(p1.x, p1.y) or CORE_POLYGON_MASK(p2.x, p2.y)
            or CORE_POLYGON_MASK(p3.x, p3.y) or CORE_POLYGON_MASK(p4.x, p4.y)):
            grid_mask[ix, iy] = True
            grid_index_2D_to_1D_map[(ix, iy)] = unwrapped_cell_index
            grid_index_1D_to_2D_map[unwrapped_cell_index] = (ix, iy)

            enclosed_cells.append((p1, p2, p3, p4))
            unwrapped_cell_index += 1
        else:
            grid_mask[ix, iy] = False


num_cells = len(enclosed_cells)


grid_data = np.empty((num_cells, 4, 2))  # (number of cells, 4 coordinates, x and y values = 2)
for i, row in enumerate(enclosed_cells):
    p1, p2, p3, p4 = row
    grid_data[i, 0, :] = p1.x, p1.y
    grid_data[i, 1, :] = p2.x, p2.y
    grid_data[i, 2, :] = p3.x, p3.y
    grid_data[i, 3, :] = p4.x, p4.y


# Try making grid laplacian matrix for spatial regularisation
grid_laplacian = np.zeros((num_cells, num_cells))

for ith_cell in range(num_cells):

    # get the 2D mesh coordinates of this cell
    ix, iy = grid_index_1D_to_2D_map[ith_cell]

    neighbours = 0

    try:
        n1 = grid_index_2D_to_1D_map[ix-1, iy]  # neighbour 1
        grid_laplacian[ith_cell, n1] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n2 = grid_index_2D_to_1D_map[ix-1, iy+1]  # neighbour 2
        grid_laplacian[ith_cell, n2] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n3 = grid_index_2D_to_1D_map[ix, iy+1]  # neighbour 3
        grid_laplacian[ith_cell, n3] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n4 = grid_index_2D_to_1D_map[ix+1, iy+1]  # neighbour 4
        grid_laplacian[ith_cell, n4] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n5 = grid_index_2D_to_1D_map[ix+1, iy]  # neighbour 5
        grid_laplacian[ith_cell, n5] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n6 = grid_index_2D_to_1D_map[ix+1, iy-1]  # neighbour 6
        grid_laplacian[ith_cell, n6] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n7 = grid_index_2D_to_1D_map[ix, iy-1]  # neighbour 7
        grid_laplacian[ith_cell, n7] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n8 = grid_index_2D_to_1D_map[ix-1, iy-1]  # neighbour 8
        grid_laplacian[ith_cell, n8] = -1
        neighbours += 1
    except KeyError:
        pass

    grid_laplacian[ith_cell, ith_cell] = neighbours


grid = {
    'voxels': grid_data,
    'index_2D_to_1D_map': grid_index_2D_to_1D_map,
    'index_1D_to_2D_map': grid_index_1D_to_2D_map,
    'mask': grid_mask,
    'laplacian': grid_laplacian,
}

# Save the files in the same directory as the loader module
directory = os.path.split(cherab.mastu.bolometry.grid_construction.__file__)[0]
file_name = 'core_res_'+str(int(resolution*100))+'cm_rectilinear_grid.pickle'
file_path = os.path.join(directory, file_name)
with open(file_path, "wb") as f:
    pickle.dump(grid, f)

print('grid file '+file_path+' created')
print("cells found = {}".format(num_cells))





