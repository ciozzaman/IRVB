# Created 01/12/2021
# Fabio Federici


from mastu_exhaust_analysis.pyLangmuirProbe import LangmuirProbe, probe_array, compare_shots
import pyuda
client=pyuda.Client()
# import matplotlib.pyplot as plt

#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

# shot=45401
shot=45371
EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+str(shot)+'.nc',pulse_ID=str(shot))

lp_data = LangmuirProbe(alp_file='/common/uda-scratch/pryan/alp0'+str(shot)+'.nc')
# lp_data.contour_plot(trange=[0,1],divertor='lower', sectors=10 ,log_data=False, quantity = 'jsat', coordinate='r',tiles=['T2','T3','T4','T5'],show=False)
# for __i in range(np.shape(efit_reconstruction.strikepointR)[1]):
# 	plt.plot(efit_reconstruction.strikepointR[:,__i],efit_reconstruction.time,'--r')
# plt.title('Jsat[A],10lower\nshot '+str(shot)+'\njsat')
# # plt.colorbar().set_label('jsat')
# plt.pause(0.01)

# lp_data.contour_plot(trange=[0,1],divertor='lower', sectors=4 ,log_data=False, quantity = 'jsat', coordinate='r',tiles=['C5','C6'],show=False)
# for __i in range(np.shape(efit_reconstruction.strikepointZ)[1]):
# 	plt.plot(efit_reconstruction.strikepointZ[:,__i],efit_reconstruction.time,'--r')
# plt.title('Jsat[A],10lower\nshot '+str(shot)+'\njsat')
# # plt.colorbar().set_label('jsat')
# plt.pause(0.01)


# lp_data.plot_lp_geometry()

# HORIZONTAL TARGET
data = lp_data.s10_lower_data.jsat
time_orig = lp_data.s10_lower_data.time
r_orig = lp_data.s10_lower_data.r
tiles_covered = lp_data.s10_lower_data.tiles_covered
time,r = np.meshgrid(time_orig,r_orig)
plt.figure(figsize=(10,10))
plt.pcolor(r,time,data.T,norm=LogNorm(),vmin=np.nanmax(data)*1e-3)
plt.colorbar().set_label('jsat')
for __i in range(np.shape(efit_reconstruction.strikepointR)[1]):
	plt.plot(efit_reconstruction.strikepointR[:,__i],efit_reconstruction.time,'--r')
plt.grid()
plt.xlim(left=0.5)
plt.ylabel('time [s]')
plt.xlabel('R [m]')
plt.title(str(shot)+'\n'+str(tiles_covered))
plt.pause(0.01)

fig, ax = plt.subplots( 3,1,figsize=(8, 12), squeeze=False,sharex=False)
# plt.figure(figsize=(10,6))
peak = np.sort(data,axis=1)
peak[np.isnan(peak)] = 0
peak = np.sort(peak,axis=1)
peak = np.mean(peak[:,10:],axis=1)
ax[0,0].plot(time_orig,peak)
ax[0,0].grid()
ax[0,0].set_xlabel('time [s]')
ax[0,0].set_ylabel('jsat max')
signal_name = '/ANE/DENSITY'
DENSITY = client.get(signal_name,shot)
ax[1,0].plot(DENSITY.time.data,DENSITY.data)
ax[1,0].grid()
ax[1,0].set_ylabel('ne [#/m3]')
ax[1,0].set_xlabel('time [s]')
DENSITY_interpolator = interp1d(DENSITY.time.data,DENSITY.data)
tend = 0.4#coleval.get_tend(shot)
for i in range(len(time_orig[time_orig<tend])):
	select = data[i]>np.nanmax(data[i])*10**-0.5
	ax[2,0].plot([DENSITY_interpolator(time_orig[i])]*len(data[i][select]), np.log10(data[i][select]),'+',color=str(0.9-i/(len(time_orig[time_orig<tend])/0.9)))
ax[2,0].grid()
ax[2,0].set_xlabel('ne [#/m3]')
ax[2,0].set_ylabel('log(high jsat)')
fig.suptitle(str(shot)+'\n'+str(tiles_covered))
plt.pause(0.01)






# VERTICAL TARGET
data = lp_data.s4_lower_data.jsat
time_orig = lp_data.s4_lower_data.time
z_orig = lp_data.s4_lower_data.z
tiles_covered = lp_data.s4_lower_data.tiles_covered
time,z = np.meshgrid(time_orig,z_orig)
plt.figure(figsize=(10,10))
plt.pcolor(z,time,data.T,norm=LogNorm())
plt.colorbar().set_label('jsat')
# for __i in range(np.shape(efit_reconstruction.strikepointZ)[1]):
# 	plt.plot(efit_reconstruction.strikepointZ[:,__i],efit_reconstruction.time,'--r')
plt.grid()
plt.xlim(left=-1.3,right=-1)
plt.ylabel('time [s]')
plt.xlabel('Z [m]')
plt.title(str(shot)+'\n'+str(tiles_covered))
plt.pause(0.01)

plt.figure(figsize=(10,6))
peak = np.nanmax(data,axis=1)
plt.plot(time_orig,peak)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel('jsat max')
plt.title(str(shot)+'\n'+str(tiles_covered))
plt.pause(0.01)




signal_name = '/ANE/DENSITY'
data = client.get(signal_name,shot)
# plt.figure()
# plt.plot(lp_data.s10_lower_data.time,np.nanmax(lp_data.s10_lower_data.jsat,axis=1))
# plt.figure()
# plt.plot(data.time.data,data.data*1e-20)
temp = []
for time in lp_data.s10_lower_data.time:
	i = np.abs(data.time.data - time).argmin()
	temp.append(data.data[i])
temp = np.array(temp)
plt.figure()
plt.plot(temp[lp_data.s10_lower_data.time<1.1],np.nanmax(lp_data.s10_lower_data.jsat,axis=1)[lp_data.s10_lower_data.time<1.1],'+')
plt.pause(0.01)

lp_data = LangmuirProbe(alp_file='/common/uda-scratch/pryan/alp0'+str(shot)+'.nc')
lp_data.contour_plot(colour_lim=[-3,0],trange=[0,1],divertor='lower', sectors=4 ,log_data=True, quantity = 'isat', coordinate='s',tiles=['C5','C6'],show=False)
# for __i in range(np.shape(efit_reconstruction.strikepointR)[1]):
# 	plt.plot(efit_reconstruction.strikepointR[:,__i],efit_reconstruction.time,'--r')
plt.title('log10 Jsat[A],10lower\nshot '+str(shot))
plt.pause(0.01)
