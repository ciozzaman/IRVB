
# position of the visible structure and the fueling location for the overlays
# data for the fueling location
fueling_r = [[0.260]]
fueling_z = [[-0.264]]
fueling_t = [[105]]
fueling_r.append([0.260])
fueling_z.append([-0.264])
fueling_t.append([195])
# tile directly below the the string of bolts on the centre column
# neighbouring points
stucture_r=[[0.333]*11]
stucture_z=[[-1.304]*11]
stucture_t=[np.linspace(60,195+15,11)]
stucture_r.append([0.539]*11)
stucture_z.append([-1.505]*11)
stucture_t.append(np.linspace(60,195+15,11))
for value in np.linspace(60,195+15,11):
	stucture_r.append([0.333,0.539])
	stucture_z.append([-1.304,-1.505])
	stucture_t.append([value]*2)
# neighbouring points
stucture_r.append([0.305]*4)
stucture_z.append([-0.853]*4)
stucture_t.append([33,93,153,213])
# neighbouring points
stucture_r.append([0.270]*4)
stucture_z.append([-0.573]*4)
stucture_t.append([33,93,153,213])
for value in [33,93,153,213]:
	stucture_r.append([0.305,0.270])
	stucture_z.append([-0.853,-0.573])
	stucture_t.append([value]*2)
# tiles around the nose
# neighbouring points
stucture_r.append([0.898]*8)
stucture_z.append([-1.300]*8)
stucture_t.append(np.linspace(0,90+15,8))
# stucture_r.append([0.895]*7)
# stucture_z.append([-1.302]*7)
# stucture_t.append(np.linspace(0,90,7))
# stucture_r.append([0.840]*7)
# stucture_z.append([-1.383]*7)
# stucture_t.append(np.linspace(0,90,7))
# stucture_r.append([0.822]*7)
# stucture_z.append([-1.514]*7)
# stucture_t.append(np.linspace(0,90,7))
stucture_r.append([0.872]*8)
stucture_z.append([-1.581]*8)
stucture_t.append(np.linspace(0,90+15,8))
for value in np.linspace(0,90+15,8):
	stucture_r.append([0.898,0.895,0.840,0.822,0.872])
	stucture_z.append([-1.3,-1.302,-1.383,-1.514,-1.581])
	stucture_t.append([value]*5)
# neighbouring points
stucture_r.append([0.898]*7)
stucture_z.append([-1.300]*7)
stucture_t.append(np.linspace(0,90,7))
stucture_r.append([1.184]*7)
stucture_z.append([-1.013]*7)
stucture_t.append(np.linspace(0,90,7))
for value in np.linspace(0,90,7):
	stucture_r.append([0.898,1.184])
	stucture_z.append([-1.3,-1.013])
	stucture_t.append([value]*2)
# neighbouring points	# a coil
stucture_r.append([1.373]*100)
stucture_z.append([-0.886]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.361]*100)
stucture_z.append([-0.878]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.312]*100)
stucture_z.append([-0.878]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.305]*100)
stucture_z.append([-0.884]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.270]*100)
stucture_z.append([-0.945]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.268]*100)
stucture_z.append([-0.954]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.269]*100)
stucture_z.append([-1.009]*100)
stucture_t.append(np.linspace(0-15,150,100))
for value in np.linspace(0-15,150,100):
	stucture_r.append([1.373,1.361,1.312,1.305,1.270,1.268,1.269])
	stucture_z.append([-0.886,-0.878,-0.878,-0.884,-0.945,-0.954,-1.009])
	stucture_t.append([value]*7)
# neighbouring points	# a coil
stucture_r.append([1.736]*100)
stucture_z.append([-0.262]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.562]*100)
stucture_z.append([-0.262]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.554]*100)
stucture_z.append([-0.270]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.554]*100)
stucture_z.append([-0.444]*100)
stucture_t.append(np.linspace(0-15,150,100))
for value in np.linspace(0-15,150,100):
	stucture_r.append([1.736,1.562,1.554,1.554])
	stucture_z.append([-0.262,-0.262,-0.270,-0.444])
	stucture_t.append([value]*4)
# silouette of the centre column
# neighbouring points
MASTU_silouette_z = [-1.881,-1.505,-1.304,-1.103,-0.853,-0.573,-0.505,-0.271,-0.147]
MASTU_silouette_r = [0.906,0.539,0.333,0.333,0.305,0.270,0.261,0.261,0.261]
from scipy.interpolate.interpolate import interp1d
R_centre_column_interpolator = interp1d(MASTU_silouette_z+(-np.flip(MASTU_silouette_z,axis=0)).tolist(),MASTU_silouette_r+np.flip(MASTU_silouette_r,axis=0).tolist(),fill_value=np.nan,bounds_error=False)
for value in [60,210]:
	stucture_r.append(MASTU_silouette_r)
	stucture_z.append(MASTU_silouette_z)
	stucture_t.append([value]*9)
# neighbouring points	# super-x divertor tiles
stucture_r.append([1.391]*5)
stucture_z.append([-2.048]*5)
stucture_t.append(np.linspace(15-30,45-30,5))
# stucture_r.append([1.763]*5)
# stucture_z.append([-1.680]*5)
# stucture_t.append(np.linspace(15-30,45-30,5))
for value in np.linspace(15-30,45-30,5):
	stucture_r.append([1.391,1.549])
	stucture_z.append([-2.048,-1.861])
	stucture_t.append([value]*2)
# neighbouring points	# super-x divertor tiles
stucture_r.append([1.073]*3)
stucture_z.append([-2.060]*3)
stucture_t.append(np.linspace(0,30,3))
stucture_r.append([1.371]*3)
stucture_z.append([-2.060]*3)
stucture_t.append(np.linspace(0,30,3))
for value in np.linspace(0,30,3):
	stucture_r.append([1.073,1.371])
	stucture_z.append([-2.060,-2.060])
	stucture_t.append([value]*2)
# neighbouring points	# coil shields
stucture_r.append([1.398]*2)
stucture_z.append([-0.823]*2)
stucture_t.append(np.linspace(-4.8,5.4,2))
stucture_r.append([1.320]*2)
stucture_z.append([-0.823]*2)
stucture_t.append(np.linspace(-4.8,5.4,2))
stucture_r.append([1.199]*2)
stucture_z.append([-1.004]*2)
stucture_t.append(np.linspace(-4.8,5.4,2))
stucture_r.append([1.398]*2)
stucture_z.append([-0.823]*2)
stucture_t.append(np.linspace(25.3,35.4,2))
stucture_r.append([1.320]*2)
stucture_z.append([-0.823]*2)
stucture_t.append(np.linspace(25.3,35.4,2))
stucture_r.append([1.199]*2)
stucture_z.append([-1.004]*2)
stucture_t.append(np.linspace(25.3,35.4,2))
stucture_r.append([1.398]*2)
stucture_z.append([-0.823]*2)
stucture_t.append(np.linspace(55.5,65.4,2))
stucture_r.append([1.320]*2)
stucture_z.append([-0.823]*2)
stucture_t.append(np.linspace(55.5,65.4,2))
stucture_r.append([1.199]*2)
stucture_z.append([-1.004]*2)
stucture_t.append(np.linspace(55.5,65.4,2))
for value in [-4.8,5.4,25.3,35.4,55.5,65.4]:
	stucture_r.append([1.398,1.320,1.199])
	stucture_z.append([-0.823,-0.823,-1.004])
	stucture_t.append([value]*3)
# neighbouring points	# beam dump SS
stucture_r.append([1.461]*5)
stucture_z.append([-0.205,-0.205,-0.465,-0.465,-0.205])
stucture_t.append([71.3,61.9,61.9,71.3,71.3])
stucture_r.append([1.461]*5)
stucture_z.append([-0.205,-0.205,-0.465,-0.465,-0.205])
stucture_t.append([51.8,61.5,61.5,51.8,51.8])
stucture_r.append([1.461]*5)
stucture_z.append([-0.205,-0.205,-0.465,-0.465,-0.205])
stucture_t.append([51.5,41.8,41.8,51.5,51.5])
# neighbouring points	# beam dump SW
stucture_r.append([1.5]*5)
stucture_z.append([0.229,0.229,0.479,0.479,0.229])
stucture_t.append([2.8,354.2,354.2,2.8,2.8])
stucture_r.append([1.5]*5)
stucture_z.append([0.229,0.229,0.479,0.479,0.229])
stucture_t.append([353.8,345.3,345.3,353.8,353.8])
stucture_r.append([1.5]*5)
stucture_z.append([0.229,0.229,0.479,0.479,0.229])
stucture_t.append([344.9,336.1,336.1,344.9,344.9])
# neighbouring points	# smaller coils
stucture_r.append([1.461,1.461,1.422,1.422,1.461])
stucture_z.append([-0.561,-0.561,-0.705,-0.705,-0.561])
stucture_t.append([36.1,17.1,17.1,36.1,36.1])
stucture_r.append([1.474,1.474,1.4,1.4,1.474])
stucture_z.append([-0.497,-0.497,-0.776,-0.776,-0.497])
stucture_t.append([38.9,14.2,14.2,38.9,38.9])


# resistive bolometer LOS

core_tangential_common_point = [-0.2165,1.734,0]	# x,y,z	17 to 28
core_tangential_arrival = []
core_tangential_arrival.append([-1.99,0.196,0])
core_tangential_arrival.append([np.nan]*3)
core_tangential_arrival.append([np.nan]*3)
core_tangential_arrival.append([-1.903,-0.615,0])
core_tangential_arrival.append([-1.806,-0.861,0])
core_tangential_arrival.append([-1.665,-1.108,0])
core_tangential_arrival.append([np.nan]*3)
core_tangential_arrival.append([np.nan]*3)
core_tangential_arrival.append([-1.064,-1.694,0])
core_tangential_arrival.append([-0.808,-1.829,0])
core_tangential_arrival.append([-0.533,-1.928,0])
core_tangential_arrival.append([-0.23,0.123,0])
core_tangential_arrival = np.array(core_tangential_arrival)

core_poloidal_common_point = [1.755,0,90]	# r,z,teta	01 to 16
core_poloidal_arrival = []
core_poloidal_arrival.append([0.417,1.424,90])
core_poloidal_arrival.append([0.335,1.311,90])
core_poloidal_arrival.append([0.335,1.093,90])
core_poloidal_arrival.append([0.311,0.921,90])
core_poloidal_arrival.append([0.285,0.703,90])
core_poloidal_arrival.append([0.261,0.493,90])
core_poloidal_arrival.append([0.261,0.285,90])
core_poloidal_arrival.append([0.261,0.09,90])
core_poloidal_arrival.append([0.261,-0.09,90])
core_poloidal_arrival.append([0.261,-0.285,90])
core_poloidal_arrival.append([0.261,-0.493,90])
core_poloidal_arrival.append([0.285,-0.703,90])
core_poloidal_arrival.append([0.311,-0.921,90])
core_poloidal_arrival.append([0.333,-1.095,90])
core_poloidal_arrival.append([0.335,-1.311,90])
core_poloidal_arrival.append([0.417,-1.424,90])
core_poloidal_arrival = np.array(core_poloidal_arrival)

divertor_poloidal_common_point = [1.846,-1.564,90]	# r,z,teta	01 to 16
divertor_poloidal_arrival = []
divertor_poloidal_arrival.append([1.32,-2.066,90])
divertor_poloidal_arrival.append([1.27,-2.066,90])
divertor_poloidal_arrival.append([1.213,-2.066,90])
divertor_poloidal_arrival.append([1.115,-2.066,90])
divertor_poloidal_arrival.append([1.09,-2.066,90])
divertor_poloidal_arrival.append([1.05,-2.03,90])
divertor_poloidal_arrival.append([1.02,-2,90])
divertor_poloidal_arrival.append([0.985,-1.965,90])
divertor_poloidal_arrival.append([0.95,-1.93,90])
divertor_poloidal_arrival.append([0.92,-1.9,90])
divertor_poloidal_arrival.append([0.88,-1.86,90])
divertor_poloidal_arrival.append([0.855,-1.82,90])
divertor_poloidal_arrival.append([0.8,-1.78,90])
divertor_poloidal_arrival.append([0.76,-1.74,90])
divertor_poloidal_arrival.append([0.715,-1.69,90])
divertor_poloidal_arrival.append([0.665,-1.645,90])
divertor_poloidal_arrival = np.array(divertor_poloidal_arrival)