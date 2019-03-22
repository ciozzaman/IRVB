# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis scripts/preamble_indexing.py").read())












type = 'simulation' # or laser
# type = 'laser'

# parameters
foilsize_hor = 9/100 #m
foilsize_ver = 7/100 #m
foil_multiplier = 5*200 #pixel/m

if type == 'simulation':
	record_length = 3 # sec
	time_multiplier = 10 # frame/sec
elif type == 'laser':
	record_length = 10 # sec
	time_multiplier = 10 # frame/sec
else:
	record_length = 3 # sec
	time_multiplier = 10 # frame/sec

nx, ny, nt = int(foilsize_hor*foil_multiplier), int(foilsize_ver*foil_multiplier), int(record_length*time_multiplier)
hx, hy, ht = foilsize_hor/(nx), foilsize_ver/(ny), record_length/(nt)
T_amb = zeroC+20

conductivity = Ptthermalconductivity * np.ones((ny, nx), float)
# thickness = resize(foilthickness,(ny,nx),order=0)
thickness = 2.5*10**-6 * np.ones((ny, nx), float)
# emissivity = resize(foilemissivity,(ny,nx),order=0)
emissivity = 1 * np.ones((ny, nx), float)
diffusivity = Ptthermaldiffusivity * np.ones((ny, nx), float)
T_reference = T_amb * np.ones((ny, nx), float)


P_init = T_amb
P_left, P_right = T_amb, T_amb
P_top, P_bottom = T_amb, T_amb




if type == 'simulation':

	# Power given by a SOLPS+CHERAB simulation

	os.chdir("/home/ffederic/work/analysis scripts/irvb/")
	Power_from_simulation = rotate(np.flip(np.flip(np.load('measured_power.npy'), -1), -2), -90)
	data_to_expand = resize(Power_from_simulation, (ny, nx), order=1)
	Power_from_simulation = np.repeat(np.expand_dims(data_to_expand, axis=0), nt, axis=0)

	Power_in = Power_from_simulation
elif type == 'laser':

	# Power given by a laser spot

	laser_freq = 0.2  # Hz
	laser_duty_cycle = 50  # %
	total_laser_power = 0.004  # W
	laser_spot_size = 0.002  # I consider the diameter at 1/e^2 of the gaussian beam
	laser_spot_location = [0.02, 0.05]

	max_laser_intensity = 2 * total_laser_power / (np.pi * ((laser_spot_size / 2) ** 2))
	P_laser = np.zeros((nt, ny, nx), float)

	data_to_expand = np.linspace(hy / 2, hy * (ny - 0.5), num=ny) - laser_spot_location[0]
	ry = np.repeat(np.expand_dims(data_to_expand, axis=-1), nx, axis=-1)

	data_to_expand = np.linspace(hx / 2, hx * (nx - 0.5), num=nx) - laser_spot_location[1]
	rx = np.repeat(np.expand_dims(data_to_expand, axis=0), ny, axis=0)

	r2 = rx ** 2 + ry ** 2
	P_laser = max_laser_intensity * np.exp(-2 * r2 / (laser_spot_size / (2)) ** 2)
	P_laser = np.repeat(np.expand_dims(P_laser, axis=0), nt, axis=0)
	# plt.imshow(P_laser[0]);plt.colorbar(),plt.show()

	data_to_expand = (np.linspace(-ht, ht * (nt - 2), num=nt) / (1 / laser_freq)) % 1 < (laser_duty_cycle / 100)
	data_to_expand = np.repeat(np.expand_dims(data_to_expand, axis=-1), ny, axis=-1)
	on_off = np.repeat(np.expand_dims(data_to_expand, axis=-1), nx, axis=-1)

	Power_in = P_laser * on_off

	# # 10/01/2019 for Matt, send him this data in ASCII format
	#
	#
	# x = np.linspace(hx / 2, hx * (nx - 0.5), num=nx)
	# y = np.linspace(hy / 2, hy * (ny - 0.5), num=ny)
	# X,Y = np.meshgrid(x,y)
	#
	# os.chdir("/home/ffederic/work/analysis scripts/")
	# grid_x_flat = X.flatten()
	# np.savetxt('grid_x.txt', grid_x_flat)
	# grid_y_flat = Y.flatten()
	# np.savetxt('grid_y.txt', grid_y_flat)
	# P_laser_flat = P_laser[0].flatten()
	# np.savetxt('P_laser.txt', P_laser_flat)

else:
	Power_in = Power_from_simulation


d2x = np.zeros_like(Power_in)
d2y = np.zeros_like(Power_in)
dt = np.zeros_like(Power_in)

def residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference,Power_in,d2x,d2y,dt):
	def residual(P):
		import numpy as np
		# d2x = np.zeros_like(P)
		# d2y = np.zeros_like(P)
		# dt = np.zeros_like(P)

		d2y[:,1:-1] = (P[:,2:]   - 2*P[:,1:-1] + P[:,:-2]) / (hy**2)
		d2y[:,0]    = (P[:,1]    - 2*P[:,0]    + P_bottom) / (hy**2)
		d2y[:,-1]   = (P_top - 2*P[:,-1]   + P[:,-2]) / (hy**2)

		d2x[:,:,1:-1] = (P[:,:,2:] - 2*P[:,:,1:-1] + P[:,:,:-2]) / (hx**2)
		d2x[:,:,0]    = (P[:,:,1]  - 2*P[:,:,0]    + P_left) / (hx**2)
		d2x[:,:,-1]   = (P_right   - 2*P[:,:,-1]   + P[:,:,-2]) / (hx**2)

		# dt[1:-1] = (P[2:] - P[:-2])/(2*ht)
		# dt[0] = (P[1] - P_init) / (2 * ht)
		# modified 2018/12/17 because it caused a "vibration" in time before the stop of the laser pulse.
		dt[1:-1] = (P[1:-1] - P[:-2]) / (ht)
		dt[0]    = (P[0]  - P_init)/(ht)
		dt[-1]   = (P[-1]   - P[-2])/ht

		BB_diff = np.add(P ** 4, - T_reference ** 4)

		return conductivity*thickness*(np.divide(dt , diffusivity)  -d2x - d2y ) + 2 * sigmaSB * np.multiply(emissivity,BB_diff)-Power_in
		# return  conductivity*thickness*(np.divide(dt , diffusivity)  -d2x - d2y ) -Power_in
	return residual

# solve
guess = T_amb*np.ones((nt ,ny, nx), float)
sol = newton_krylov(residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference,Power_in,d2x,d2y,dt), guess,method='lgmres', verbose=1,iter=30,f_tol=1e-10)
print('Residual: %g' % abs(residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference,Power_in,d2x,d2y,dt)(sol)).max())

if abs(residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference,Power_in,d2x,d2y,dt)(sol)).max() > 0.000001:
	sol = newton_krylov(residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference,Power_in,d2x,d2y,dt), sol,method='gmres', verbose=1,iter=20,f_tol=1e-16)
	# sol = newton_krylov(residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference), sol,method='bicgstab', verbose=1,f_tol=1e-10)
	print('Residual: %g' % abs(residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference,Power_in,d2x,d2y,dt)(sol)).max())

sol_C = sol-zeroC

if type == 'simulation':
	plt.imshow(sol_C[-1], origin='lower', cmap='rainbow')
	plt.title('Steady state temperature on the foil')
	plt.colorbar().set_label('Temperature [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()
elif type == 'laser':
	ani = coleval.movie_from_data(np.array([sol_C]), time_multiplier, 1000 / time_multiplier, 'Horizontal axis [pixles]','Vertical axis [pixles]', 'Temperature [°C]', extvmin=sol_C.min(), extvmax=sol_C.max())

	plt.figure()
	plt.plot(np.linspace(0, ht * (nt - 1), num=nt),sol_C[:,int(laser_spot_location[0]//hy),int(laser_spot_location[1]//hx)])
	plt.title('Temperature at the centre of the laser spot, power '+str(total_laser_power*1000)+'mW, beam '+str(laser_spot_size*1000)+'mm diam')
	plt.xlabel('Time [s]')
	plt.ylabel('Temperature [°C]')
	plt.grid()
	plt.show()
else:
	plt.imshow(sol_C[-1], origin='lower', cmap='rainbow')
	plt.title('Steady state temperature on the foil')
	plt.colorbar().set_label('Temperature [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()


