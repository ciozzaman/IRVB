
# imports



cr_r = np.transpose(ds_puff8.crx.values)
cr_z = np.transpose(ds_puff8.cry.values)

cr_r = None  # From SOLPS shape => [nx, ny, 4]
cr_z = None  # From SOLPS shape => [nx, ny, 4]


def make_solps_power_function(cr_r, cr_z, radiated_power):

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

    rad_power = np.zeros(radiated_power.shape[0]*2)
    for i in range(radiated_power.shape[0]):
        rad_power[i*2] = radiated_power[i]
        rad_power[i*2 + 1] = radiated_power[i]

    return AxisymmetricMapper(Discrete2DMesh(vertex_coords, triangles, rad_power))


class RadiatedPower(InhomogeneousVolumeEmitter):

    def __init__(self, radiation_function):
        super().__init__()
        self.radiation_function = radiation_function

    def emission_function(self, point, direction, spectrum, world, ray, primitive, to_local, to_world):

        p = point.transform(to_world)
        spectrum.samples[0] += self.radiation_function(p.x, p.y, p.z)

        return spectrum


radiation_function = make_solps_power_function(solps_cr, solps_cz, radiation_array)

rad_test = np.zeros((nx, ny))
for ix, x in enumerate(np.arange(0, 1.6, 0.01)):
    for jy, y in enumerate(np.arange(-2.0, -0.6, 0.01)):
        rad_test[ix, jy] = radiation_function(x, 0, y)

plt.imshow(rad_test)

# Step 2
# emitter_material = RadiatedPower(radiation_function)
# radiator = import_stl('radiator_all_core_and_divertor.stl', material=emitter_material, parent=world)