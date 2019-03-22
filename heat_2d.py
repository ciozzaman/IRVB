import numpy as np
import matplotlib.pyplot as plt
# import pdb

#specify the material size and deltas
depth=0.02
width=0.02
time=0.005

delta_t=1e-5
delta_x=1e-4 #in width
delta_z=5e-5 #in depth

T_0=0.
q_0=2e6

k_=49
rho=2267
cp=717
diff=k_/(rho*cp)

#make the arrays for time and depth
nz=int(depth/delta_z)
nx=int(width/delta_x)
nt=int(time/delta_t)

times=np.linspace(0,time,nt+1)
deep=np.linspace(delta_z/2,depth+(delta_z/2),nz)
radius=np.linspace(0,width,nx)

#make the heat flux array
q=np.zeros(shape=(nx,nt+1))
q[:,0:nt//2]=q_0
q[:,0]=0.

#pdb.set_trace()

#make array for the temperatures
temps=np.zeros(shape=(nx,nz,nt+1))

#set initial temperature distribution
temps[:,:,0]=T_0

#generate the upper boundary condition
#q=-kDt/dz
#=> q=-k(t(j-1)-t(j))/dz
#=> q=k(t(j)-t(j-1))/dz
#=> t(j-1)=(qdz/k)+t(j) ;replace this for t(j-1)
			#when j=0

#print((diff*delta_t/(delta_x**2*delta_z**2)))

for k in range(1,nt):
	if np.mod(k,50)==0:
		print(k,'of',nt+1)

	for j in range(0,nz-1): #depth direction
		for i in range(0,nx-1): #radial direction
			#print('i',i)
			#print('j',j)
			#print('k',k)

			if (i==0) and (j==0):
				 flux_bc=((q[j,k]*delta_z)/k_)+temps[i,j,k-1]
		#		 flux_bc=((q[j,k]*delta_t*delta_x)/ \
		#				 (delta_x*delta_z*cp*rho))+temps[i,j,k-1]
		#		 print(flux_bc, temps[i,j,k-1], temps[i,j+1,k-1], \
		#			(delta_x**2*(temps[i,j+1,k-1]+flux_bc)))

				 #pdb.set_trace()
				 temps[i,j,k]=temps[i,j,k-1]+\
					(diff*delta_t/(delta_x**2*delta_z**2))* \
					((delta_z**2*(temps[i+1,j,k-1]+temps[i,j,k-1]))- \
					(2*((delta_x**2+delta_z**2)*temps[i,j,k-1]))+ \
					(delta_x**2*(temps[i,j+1,k-1]+flux_bc)))

			if (i==0) and (j>0):
				 temps[i,j,k]=temps[i,j,k-1]+\
					(diff*delta_t/(delta_x**2*delta_z**2))* \
					((delta_z**2*(temps[i+1,j,k-1]+temps[i,j,k-1]))- \
					(2*((delta_x**2+delta_z**2)*temps[i,j,k-1]))+ \
					(delta_x**2*(temps[i,j+1,k-1]+temps[i,j-1,k-1])))

			if (i>0) and (j==0):
				 #flux_bc=((q[j,k]*delta_t*delta_x)/ \
				#		 (delta_x*delta_z*cp*rho))+temps[i,j,k-1]
				 flux_bc=((q[j,k]*delta_z)/k_)+temps[i,j,k-1]

				 temps[i,j,k]=temps[i,j,k-1]+\
					(diff*delta_t/(delta_x**2*delta_z**2))* \
					((delta_z**2*(temps[i+1,j,k-1]+temps[i-1,j,k-1]))- \
					(2*((delta_x**2+delta_z**2)*temps[i,j,k-1]))+ \
					(delta_x**2*(temps[i,j+1,k-1]+flux_bc)))

			if (i>0) and (j>0):
				 temps[i,j,k]=temps[i,j,k-1]+\
					(diff*delta_t/(delta_x**2*delta_z**2))* \
					((delta_z**2*(temps[i+1,j,k-1]+temps[i-1,j,k-1]))- \
					(2*((delta_x**2+delta_z**2)*temps[i,j,k-1]))+ \
					(delta_x**2*(temps[i,j+1,k-1]+temps[i,j-1,k-1])))

		#pdb.set_trace()

#calculate the surface temperature by interpolating to the
#z=0 point
surf_temp=np.zeros(shape=(nx,nt+1))

for l in range(0,nt+1):
		for m in range(0,nx):
				grad=(temps[m,1,l]-temps[m,0,l])/delta_z
				intcpt=temps[m,0,l]-(grad*deep[0])
				surf_temp[m,l]=intcpt
#end of for loop for surf temp

#calculate the 1d analytic
dT=(2*q[0,:]*np.sqrt(times))/np.sqrt(np.pi*k_*rho*cp)
plt.plot(times, surf_temp[0,:])
#plt.plot(temps[10,0,:])
plt.plot(times, dT)
plt.ion()
plt.show()
#pdb.set_trace()


