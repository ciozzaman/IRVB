# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 19:15:37 2021

@author: Federici
"""

# Created from code from James Harrison

import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import netCDF4
import numpy as np
import efitData

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from scipy import interpolate

class mclass:
	def __init__(self,path):


		f = netCDF4.Dataset(path)

		try:
			self.psidat = np.transpose(f['output']['profiles2D']['poloidalFlux'], (0, 2, 1))
			file_prefix = ''
		except IndexError:
			self.psidat = np.transpose(f['/epm/output']['profiles2D']['poloidalFlux'], (0, 2, 1))
			file_prefix = '/epm'

		self.q0 = f[file_prefix+'/output']['globalParameters']['q0'][:].data
		self.q95 = f[file_prefix+'/output']['globalParameters']['q95'][:].data

		try:
			self.r = f[file_prefix+'/output']['profiles2D']['r'][:].data
			self.z = f[file_prefix+'/output']['profiles2D']['z'][:].data
			self.R = self.r[0,:]
			self.Z = self.z[0,:]
			self.bVac = f[file_prefix+'/input']['bVacRadiusProduct']['values'][:].data
			self.r_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis'][:]['R']
			self.z_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis'][:]['Z']
			self.shotnumber = f.pulseNumber
		except IndexError:
			self.R = f[file_prefix+'/output']['profiles2D']['r'][:].data
			self.Z = f[file_prefix+'/output']['profiles2D']['z'][:].data
			self.bVac = f[file_prefix+'/input']['bVacRadiusProduct'][:].data
			self.r_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis']['R'][:]
			self.z_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis']['Z'][:]
			self.shotnumber = f.shot

		self.psi_bnd = f[file_prefix+'/output']['globalParameters']['psiBoundary'][:].data
		self.psi_axis = f[file_prefix+'/output']['globalParameters']['psiAxis'][:].data
		self.cpasma = f[file_prefix+'/output']['globalParameters']['plasmaCurrent'][:].data
		self.time = f[file_prefix+'/time'][:].data

		f.close()


		# Calculate the time history of some useful quantites, i.e.
		# Inner and outer LCFS positions
		# Strike point positions
		# q0 at some point?

		print('Calculating equilibrium properties...')

		self.lower_xpoint_r = np.zeros(len(self.time))
		self.lower_xpoint_z = np.zeros(len(self.time))
		self.lower_xpoint_p = np.zeros(len(self.time))
		self.upper_xpoint_r = np.zeros(len(self.time))
		self.upper_xpoint_z = np.zeros(len(self.time))
		self.upper_xpoint_p = np.zeros(len(self.time))

		self.mag_axis_r = np.zeros(len(self.time))
		self.mag_axis_z = np.zeros(len(self.time))
		self.mag_axis_p = np.zeros(len(self.time))

		self.inner_sep_r = np.zeros(len(self.time))
		self.outer_sep_r = np.zeros(len(self.time))

		rr, zz = np.meshgrid(self.R, self.Z)

		if self.psidat is not None:
			for i in np.arange(len(self.time)):
				psiarr = np.array((self.psidat[i,:,:]))
				psi_interp = self.interp2d(self.R, self.Z, psiarr)

				if np.sum(np.isfinite(psiarr)) == np.size(psiarr):
					# Find the position of the xpoints
					opoint, xpoint = self.find_critical(rr.T,zz.T, psiarr.T)

					if len(xpoint) > 0:
						xpt1r = xpoint[0][0]
						xpt1z = xpoint[0][1]
						xpt1p = xpoint[0][2]

						if len(xpoint) > 1:
							xpt2r = xpoint[1][0]
							xpt2z = xpoint[1][1]
							xpt2p = xpoint[1][2]
						else:
							xpt2r = None
							xpt2z = None
							xpt2p = None

						self.mag_axis_r[i] = opoint[0][0]
						self.mag_axis_z[i] = opoint[0][1]
						self.mag_axis_p[i] = opoint[0][2]

						if xpt1z < 0:
							self.lower_xpoint_r[i] = xpt1r
							self.lower_xpoint_z[i] = xpt1z
							self.lower_xpoint_p[i] = xpt1p

						if xpt1z > 0:
							self.upper_xpoint_r[i] = xpt1r
							self.upper_xpoint_z[i] = xpt1z
							self.upper_xpoint_p[i] = xpt1p

						if xpt2z and  xpt2z < 0:
							self.lower_xpoint_r[i] = xpt2r
							self.lower_xpoint_z[i] = xpt2z
							self.lower_xpoint_p[i] = xpt2p

						if xpt2z and xpt2z > 0:
							self.upper_xpoint_r[i] = xpt2r
							self.upper_xpoint_z[i] = xpt2z
							self.upper_xpoint_p[i] = xpt2p

					mp_r_arr = np.linspace(np.min(self.R), np.max(self.R),500)
					mp_p_arr = mp_r_arr*0.0

					for j in np.arange(len(mp_p_arr)):
							mp_p_arr[j] = psi_interp(mp_r_arr[j], self.mag_axis_z[i])

					zcr = self.zcd(mp_p_arr-self.psi_bnd[i])

					if len(zcr) > 2:
						zcr = zcr[-2:]

					self.inner_sep_r[i] = mp_r_arr[zcr[0]]
					self.outer_sep_r[i] = mp_r_arr[zcr[1]]


					# Calculate dr_sep

	def interp2d(self,R,Z,field):
		return RectBivariateSpline(R,Z,np.transpose(field))


	def find_critical(self,R, Z, psi, discard_xpoints=True):
		"""
		Find critical points

		Inputs
		------

		R - R(nr, nz) 2D array of major radii
		Z - Z(nr, nz) 2D array of heights
		psi - psi(nr, nz) 2D array of psi values

		Returns
		-------

		Two lists of critical points

		opoint, xpoint

		Each of these is a list of tuples with (R, Z, psi) points

		The first tuple is the primary O-point (magnetic axis)
		and primary X-point (separatrix)

		"""

		# Get a spline interpolation function
		f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psi)

		# Find candidate locations, based on minimising Bp^2
		Bp2 = (f(R, Z, dx=1, grid=False) ** 2 + f(R, Z, dy=1, grid=False) ** 2) / R ** 2

		# Get grid resolution, which determines a reasonable tolerance
		# for the Newton iteration search area
		dR = R[1, 0] - R[0, 0]
		dZ = Z[0, 1] - Z[0, 0]
		radius_sq = 9 * (dR ** 2 + dZ ** 2)

		# Find local minima

		J = np.zeros([2, 2])

		xpoint = []
		opoint = []

		nx, ny = Bp2.shape
		for i in range(2, nx - 2):
			for j in range(2, ny - 2):
				if (
					(Bp2[i, j] < Bp2[i + 1, j + 1])
					and (Bp2[i, j] < Bp2[i + 1, j])
					and (Bp2[i, j] < Bp2[i + 1, j - 1])
					and (Bp2[i, j] < Bp2[i - 1, j + 1])
					and (Bp2[i, j] < Bp2[i - 1, j])
					and (Bp2[i, j] < Bp2[i - 1, j - 1])
					and (Bp2[i, j] < Bp2[i, j + 1])
					and (Bp2[i, j] < Bp2[i, j - 1])
				):

					# Found local minimum

					R0 = R[i, j]
					Z0 = Z[i, j]

					# Use Newton iterations to find where
					# both Br and Bz vanish
					R1 = R0
					Z1 = Z0

					count = 0
					while True:

						Br = -f(R1, Z1, dy=1, grid=False) / R1
						Bz = f(R1, Z1, dx=1, grid=False) / R1

						if Br ** 2 + Bz ** 2 < 1e-6:
							# Found a minimum. Classify as either
							# O-point or X-point

							dR = R[1, 0] - R[0, 0]
							dZ = Z[0, 1] - Z[0, 0]
							d2dr2 = (psi[i + 2, j] - 2.0 * psi[i, j] + psi[i - 2, j]) / (
								2.0 * dR
							) ** 2
							d2dz2 = (psi[i, j + 2] - 2.0 * psi[i, j] + psi[i, j - 2]) / (
								2.0 * dZ
							) ** 2
							d2drdz = (
								(psi[i + 2, j + 2] - psi[i + 2, j - 2]) / (4.0 * dZ)
								- (psi[i - 2, j + 2] - psi[i - 2, j - 2]) / (4.0 * dZ)
							) / (4.0 * dR)
							D = d2dr2 * d2dz2 - d2drdz ** 2

							if D < 0.0:
								# Found X-point
								xpoint.append((R1, Z1, f(R1, Z1)[0][0]))
							else:
								# Found O-point
								opoint.append((R1, Z1, f(R1, Z1)[0][0]))
							break

						# Jacobian matrix
						# J = ( dBr/dR, dBr/dZ )
						#	 ( dBz/dR, dBz/dZ )

						J[0, 0] = -Br / R1 - f(R1, Z1, dy=1, dx=1)[0][0] / R1
						J[0, 1] = -f(R1, Z1, dy=2)[0][0] / R1
						J[1, 0] = -Bz / R1 + f(R1, Z1, dx=2) / R1
						J[1, 1] = f(R1, Z1, dx=1, dy=1)[0][0] / R1

						d = np.dot(np.linalg.inv(J), [Br, Bz])

						R1 = R1 - d[0]
						Z1 = Z1 - d[1]

						count += 1
						# If (R1,Z1) is too far from (R0,Z0) then discard
						# or if we've taken too many iterations
						if ((R1 - R0) ** 2 + (Z1 - Z0) ** 2 > radius_sq) or (count > 100):
							# Discard this point
							break

		# Remove duplicates
		def remove_dup(points):
			result = []
			for n, p in enumerate(points):
				dup = False
				for p2 in result:
					if (p[0] - p2[0]) ** 2 + (p[1] - p2[1]) ** 2 < 1e-5:
						dup = True  # Duplicate
						break
				if not dup:
					result.append(p)  # Add to the list
			return result

		xpoint = remove_dup(xpoint)
		opoint = remove_dup(opoint)

		if len(opoint) == 0:
			# Can't order primary O-point, X-point so return
			print("Warning: No O points found")
			return opoint, xpoint

		# Find primary O-point by sorting by distance from middle of domain
		Rmid = 0.5 * (R[-1, 0] + R[0, 0])
		Zmid = 0.5 * (Z[0, -1] + Z[0, 0])
		opoint.sort(key=lambda x: (x[0] - Rmid) ** 2 + (x[1] - Zmid) ** 2)

		# Draw a line from the O-point to each X-point. Psi should be
		# monotonic; discard those which are not

		if discard_xpoints:
			Ro, Zo, Po = opoint[0]  # The primary O-point
			xpt_keep = []
			for xpt in xpoint:
				Rx, Zx, Px = xpt

				rline = np.linspace(Ro, Rx, num=50)
				zline = np.linspace(Zo, Zx, num=50)

				pline = f(rline, zline, grid=False)

				if Px < Po:
					pline *= -1.0  # Reverse, so pline is maximum at X-point

				# Now check that pline is monotonic
				# Tried finding maximum (argmax) and testing
				# how far that is from the X-point. This can go
				# wrong because psi can be quite flat near the X-point
				# Instead here look for the difference in psi
				# rather than the distance in space

				maxp = np.amax(pline)
				if (maxp - pline[-1]) / (maxp - pline[0]) > 0.001:
					# More than 0.1% drop in psi from maximum to X-point
					# -> Discard
					continue

				ind = np.argmin(pline)  # Should be at O-point
				if (rline[ind] - Ro) ** 2 + (zline[ind] - Zo) ** 2 > 1e-4:
					# Too far, discard
					continue
				xpt_keep.append(xpt)
			xpoint = xpt_keep

		# Sort X-points by distance to primary O-point in psi space
		psi_axis = opoint[0][2]
		xpoint.sort(key=lambda x: (x[2] - psi_axis) ** 2)

		return opoint, xpoint


	def zcd(self, data):
		sign_array=np.sign(data)
		out=[]
		for i in np.arange(1,len(sign_array)):
			if sign_array[i] != sign_array[i-1]:
				out.append(i)
		return out


# path = '/home/ffederic/work/analysis_scripts/scripts/from_Lucy/epm044245.nc'
# path = '/home/jrh/EFIT++/mastu/44154/efit_jrh_01/efitOut.nc'
path = '/common/uda-scratch/lkogan/efitpp_eshed/epm044377.nc'
efit_reconstruction = coleval.mclass(path)


Rf=1.54967	# m
plane_equation = np.array([1,-1,0,2**0.5 * Rf])
pinhole_location = np.array([-1.04087,1.068856,-0.7198])
centre_of_foil = np.array([-1.095782166, 1.095782166, -0.7])
foil_size = [0.07,0.09]

def point_toroidal_to_cartesian(coords):	# r,z,teta deg	to	x,y,z
	out = np.zeros_like(coords).astype(float)
	out.T[0]=coords.T[0] * np.cos(coords.T[2]*2*np.pi/360)
	out.T[1]=coords.T[0] * np.sin(coords.T[2]*2*np.pi/360)
	out.T[2]=coords.T[1]
	return out

def find_location_on_foil(point_coord,plane_equation=plane_equation,pinhole_location=pinhole_location):
	t = (-plane_equation[-1] -np.sum(plane_equation[:-1]*point_coord,axis=-1)) / np.sum(plane_equation[:-1]*(pinhole_location-point_coord),axis=-1)
	out = point_coord + ((pinhole_location-point_coord).T*t).T
	return out

def absolute_position_on_foil_to_foil_coord(coords,centre_of_foil=centre_of_foil,foil_size=foil_size):	# out in [x,z]
	out = np.zeros((np.shape(coords)[0],np.shape(coords)[1]-1))
	out.T[1] = foil_size[1]/2 -(coords.T[2] - centre_of_foil[2])
	out.T[1][np.logical_or(out.T[1]>foil_size[1],out.T[1]<0)] = np.nan
	out.T[0] = np.sign((coords.T[0]-centre_of_foil[0]))*((coords.T[0]-centre_of_foil[0])**2 + (coords.T[1]-centre_of_foil[1])**2)**0.5 + foil_size[0]/2
	out.T[0][np.logical_or(out.T[0]>foil_size[0],out.T[0]<0)] = np.nan
	return out


from PIL import Image
path_image = "/home/ffederic/work/irvb/MAST-U/Calcam_theoretical_view.png"
MASTU_wireframe = Image.open(path_image)
MASTU_wireframe_resize = np.array(np.asarray(MASTU_wireframe)[:,:,3].tolist()).astype(np.float64)
MASTU_wireframe_resize[MASTU_wireframe_resize>0] = 1
MASTU_wireframe_resize = np.flip(MASTU_wireframe_resize,axis=0)
masked = np.ma.masked_where(MASTU_wireframe_resize == 0, MASTU_wireframe_resize)

# position of the visible structure and the fueling location for the overlays
# data for the fueling location
fueling_r = [[0.260]]
fueling_z = [[-0.264]]
fueling_t = [[105]]
fueling_r.append([0.260])
fueling_z.append([-0.264])
fueling_t.append([195])
# # neighbouring points		these are invisible!
# stucture_r.append([0.904]*10)
# stucture_z.append([-1.878]*10)
# stucture_t.append(np.linspace(60,195+15,11))
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
for value in [60,210]:
	stucture_r.append([0.906,0.539,0.333,0.333,0.305,0.270,0.261,0.261,0.261])
	stucture_z.append([-1.881,-1.505,-1.304,-1.103,-0.853,-0.573,-0.505,-0.271,-0.147])
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




# # bolts holding the tiles close to the shinethrough protection
# stucture_r.append(1.493)
# stucture_z.append(0.35)
# stucture_t.append(340.5)
# stucture_r.append(1.493)
# stucture_z.append(0.35)
# stucture_t.append(349.6)
# stucture_r.append(1.493)
# stucture_z.append(0.35)
# stucture_t.append(358.5)
# # bolts over the nose
# stucture_r.append(1.213)
# stucture_z.append(-0.97)
# stucture_t.append(0.3)
# stucture_r.append(1.213)
# stucture_z.append(-0.97)
# stucture_t.append(30.3)

structure_point_location_on_foil = []
# resolution = 1000
for time in range(len(stucture_r)):
	# x_point_location = np.array([[stucture_r[time]]*resolution,[stucture_z[time]]*resolution,np.linspace(0,360,resolution)]).T
	# x_point_location = np.array([[0]*resolution,np.linspace(-2,2,resolution),[0]*resolution]).T
	point_location = np.array([stucture_r[time],stucture_z[time],stucture_t[time]]).T
	point_location = coleval.point_toroidal_to_cartesian(point_location)
	point_location = coleval.find_location_on_foil(point_location)
	structure_point_location_on_foil.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))
# all_time_x_point_location = np.array(all_time_x_point_location)
fueling_point_location_on_foil = []
# resolution = 1000
for time in range(len(fueling_r)):
	# x_point_location = np.array([[stucture_r[time]]*resolution,[stucture_z[time]]*resolution,np.linspace(0,360,resolution)]).T
	# x_point_location = np.array([[0]*resolution,np.linspace(-2,2,resolution),[0]*resolution]).T
	point_location = np.array([fueling_r[time],fueling_z[time],fueling_t[time]]).T
	point_location = coleval.point_toroidal_to_cartesian(point_location)
	point_location = coleval.find_location_on_foil(point_location)
	fueling_point_location_on_foil.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))
# all_time_x_point_location = np.array(all_time_x_point_location)

plt.figure(figsize=(20, 10))
# plt.title(path_image)
# plt.imshow(MASTU_wireframe_resize,'rainbow',origin='lower',extent = [0,foil_size[0],0,foil_size[1]])
for i in range(len(fueling_point_location_on_foil)):
	plt.plot(fueling_point_location_on_foil[i][:,0],fueling_point_location_on_foil[i][:,1],'+k',markersize=40,alpha=0.5)
	plt.plot(fueling_point_location_on_foil[i][:,0],fueling_point_location_on_foil[i][:,1],'ok',markersize=5,alpha=0.5)
for i in range(len(structure_point_location_on_foil)):
	plt.plot(structure_point_location_on_foil[i][:,0],structure_point_location_on_foil[i][:,1],'--k',alpha=0.5)
plt.xlim(left=0,right=foil_size[0])
plt.ylim(bottom=0,top=foil_size[1])
ax = plt.gca()
ax.set_aspect(1)
plt.pause(0.01)

R_centre_column = 0.261	# m

MASTU_silouette_z = [-1.881,-1.505,-1.304,-1.103,-0.853,-0.573,-0.505,-0.271,-0.147]
MASTU_silouette_r = [0.906,0.539,0.333,0.333,0.305,0.270,0.261,0.261,0.261]
R_centre_column_interpolator = interp1d(MASTU_silouette_z+(-np.flip(MASTU_silouette_z,axis=0)).tolist(),MASTU_silouette_r+np.flip(MASTU_silouette_r,axis=0).tolist(),fill_value=np.nan,bounds_error=False)


from scipy.signal import find_peaks, peak_prominences as get_proms
from scipy.interpolate import interp2d
def efit_reconstruction_to_separatrix_on_foil(efit_reconstruction,ref_angle=60):
	all_time_separatrix = []
	for time in range(len(efit_reconstruction.time)):
		gna = efit_reconstruction.psidat[time]
		sep_up = efit_reconstruction.upper_xpoint_p[time]
		sep_low = efit_reconstruction.lower_xpoint_p[time]
		x_point_z_proximity = np.abs(np.nanmin([efit_reconstruction.upper_xpoint_z[time],efit_reconstruction.lower_xpoint_z[time],-0.573-0.2]))-0.2	# -0.573 is an arbitrary treshold in case both are nan

		psi_interpolator = interp2d(efit_reconstruction.R,efit_reconstruction.Z,gna,kind='linear')
		refinement = 1000
		r_fine = np.unique(np.linspace(efit_reconstruction.R.min(),efit_reconstruction.R.max(),refinement).tolist() + np.linspace(R_centre_column-0.01,R_centre_column+0.08,refinement).tolist())
		z_fine = np.linspace(efit_reconstruction.Z.min(),efit_reconstruction.Z.max(),refinement)
		psi = psi_interpolator(r_fine,z_fine)
		psi_up = -np.abs(psi-sep_up)
		psi_low = -np.abs(psi-sep_low)
		all_peaks_up = []
		all_z_up = []
		all_peaks_low = []
		all_z_low = []
		for i_z,z in enumerate(z_fine):
			# psi_z = psi[i_z]
			peaks = find_peaks(psi_up[i_z])[0]
			all_peaks_up.append(peaks)
			all_z_up.append([i_z]*len(peaks))
			peaks = find_peaks(psi_low[i_z])[0]
			all_peaks_low.append(peaks)
			all_z_low.append([i_z]*len(peaks))
		all_peaks_up = np.concatenate(all_peaks_up).astype(int)
		all_z_up = np.concatenate(all_z_up).astype(int)
		found_psi_up = np.abs(psi_up[all_z_up,all_peaks_up])
		all_peaks_low = np.concatenate(all_peaks_low).astype(int)
		all_z_low = np.concatenate(all_z_low).astype(int)
		found_psi_low = np.abs(psi_low[all_z_low,all_peaks_low])

		# plt.figure()
		# plt.plot(all_z[found_points<(gna.max()-gna.min())/500],all_peaks[found_points<(gna.max()-gna.min())/500],'+b')
		all_peaks_up = all_peaks_up[found_psi_up<(gna.max()-gna.min())/100]
		all_z_up = all_z_up[found_psi_up<(gna.max()-gna.min())/100]
		all_peaks_low = all_peaks_low[found_psi_low<(gna.max()-gna.min())/100]
		all_z_low = all_z_low[found_psi_low<(gna.max()-gna.min())/100]
		plt.figure()
		plt.plot(r_fine[all_peaks_up],z_fine[all_z_up],'+r')
		plt.plot(r_fine[all_peaks_low],z_fine[all_z_low],'+b')
		plt.pause(0.01)

		left_up = []
		right_up = []
		left_up_z = []
		right_up_z = []
		left_low = []
		right_low = []
		left_low_z = []
		right_low_z = []
		for i_z,z in enumerate(z_fine):
			if i_z in all_z_up:
				temp = all_peaks_up[all_z_up==i_z]
				if len(temp) == 1:
					right_up.append(temp[0])
					right_up_z.append(i_z)
				elif len(temp) == 2:
					# # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
					# if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
					left_up.append(temp.min())
					left_up_z.append(i_z)
					right_up.append(temp.max())
					right_up_z.append(i_z)
				elif len(temp) == 3:
					left_up.append(np.sort(temp)[1])
					left_up_z.append(i_z)
					right_up.append(temp.max())
					right_up_z.append(i_z)
				elif len(temp) == 4:
					left_up.append(np.sort(temp)[1])
					left_up_z.append(i_z)
					right_up.append(np.sort(temp)[2])
					right_up_z.append(i_z)
			if i_z in all_z_low:
				temp = all_peaks_low[all_z_low==i_z]
				if len(temp) == 1:
					right_low.append(temp[0])
					right_low_z.append(i_z)
				elif len(temp) == 2:
					# # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
					# if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
					left_low.append(temp.min())
					left_low_z.append(i_z)
					right_low.append(temp.max())
					right_low_z.append(i_z)
				elif len(temp) == 3:
					left_low.append(np.sort(temp)[1])
					left_low_z.append(i_z)
					right_low.append(temp.max())
					right_low_z.append(i_z)
				elif len(temp) == 4:
					left_low.append(np.sort(temp)[1])
					left_low_z.append(i_z)
					right_low.append(np.sort(temp)[2])
					right_low_z.append(i_z)
		sep_r = [left_up,right_up,left_low,right_low]
		sep_z = [left_up_z,right_up_z,left_low_z,right_low_z]

		plt.figure()
		plt.plot(r_fine[sep_r[0]],z_fine[sep_z[0]],'r')
		plt.plot(r_fine[sep_r[1]],z_fine[sep_z[1]],'b')
		plt.plot(r_fine[sep_r[2]],z_fine[sep_z[2]],'k')
		plt.plot(r_fine[sep_r[3]],z_fine[sep_z[3]],'g')
		plt.grid()
		plt.pause(0.01)

		separatrix = []
		for i in range(len(sep_r)):
			point_location = np.array([r_fine[sep_r[i]],z_fine[sep_z[i]],[ref_angle]*len(sep_z[i])]).T
			point_location = coleval.point_toroidal_to_cartesian(point_location)
			point_location = coleval.find_location_on_foil(point_location)
			separatrix.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))
		all_time_separatrix.append(separatrix)
	return all_time_separatrix
# left_up = np.concatenate(left_up)
# left_up_z = np.concatenate(left_up_z)
# left_low = np.concatenate(left_low)
# right_low_z = np.concatenate(right_low_z)

a=np.concatenate([r_fine[sep_r[0]],r_fine[sep_r[2]]])
b=np.concatenate([z_fine[sep_z[0]],z_fine[sep_z[2]]])
c=np.abs(a-R_centre_column)
peaks = find_peaks(-c)[0]
peaks = peaks[c[peaks]<1e-3]
plt.plot(a[peaks],b[peaks],'xr')

plt.figure()
plt.plot(r_fine[left_up],z_fine[left_up_z],'r')
plt.plot(r_fine[right_up],z_fine[right_up_z],'--r')
plt.plot(r_fine[left_low],z_fine[left_low_z],'b')
plt.plot(r_fine[right_low],z_fine[right_low_z],'--b')



plt.figure(figsize=(20, 10))
# plt.title(path_image)
# plt.imshow(MASTU_wireframe_resize,'rainbow',origin='lower',extent = [0,foil_size[0],0,foil_size[1]])
for i in range(len(fueling_point_location_on_foil)):
	plt.plot(fueling_point_location_on_foil[i][:,0],fueling_point_location_on_foil[i][:,1],'+k',markersize=40,alpha=0.5)
	plt.plot(fueling_point_location_on_foil[i][:,0],fueling_point_location_on_foil[i][:,1],'ok',markersize=5,alpha=0.5)
for i in range(len(structure_point_location_on_foil)):
	plt.plot(structure_point_location_on_foil[i][:,0],structure_point_location_on_foil[i][:,1],'--k',alpha=0.5)
for i in range(len(separatrix)):
	plt.plot(separatrix[i][:,0],separatrix[i][:,1],'--b',alpha=0.5)
plt.plot(strike_poits[:,0],strike_poits[:,1],'xr',alpha=0.5,markersize=10)
for i in range(len(strike_point_location_rot)):
	plt.plot(strike_point_location_rot[i][:,0],strike_point_location_rot[i][:,1],'--r',alpha=0.5)
plt.xlim(left=0,right=foil_size[0])
plt.ylim(bottom=0,top=foil_size[1])
ax = plt.gca()
ax.set_aspect(1)
plt.pause(0.01)

f = netCDF4.Dataset(path)
dndXpoint1InnerStrikepointR = f['/epm/output']['separatrixGeometry']['dndXpoint1InnerStrikepointR'][:].data
dndXpoint1OuterStrikepointR = f['/epm/output']['separatrixGeometry']['dndXpoint1OuterStrikepointR'][:].data
dndXpoint2InnerStrikepointR = f['/epm/output']['separatrixGeometry']['dndXpoint2InnerStrikepointR'][:].data
dndXpoint2OuterStrikepointR = f['/epm/output']['separatrixGeometry']['dndXpoint2OuterStrikepointR'][:].data
dndXpoint1InnerStrikepointZ = f['/epm/output']['separatrixGeometry']['dndXpoint1InnerStrikepointZ'][:].data
dndXpoint1OuterStrikepointZ = f['/epm/output']['separatrixGeometry']['dndXpoint1OuterStrikepointZ'][:].data
dndXpoint2InnerStrikepointZ = f['/epm/output']['separatrixGeometry']['dndXpoint2InnerStrikepointZ'][:].data
dndXpoint2OuterStrikepointZ = f['/epm/output']['separatrixGeometry']['dndXpoint2OuterStrikepointZ'][:].data


strikepointR = f['/epm/output']['separatrixGeometry']['strikepointR'][:].data
strikepointZ = f['/epm/output']['separatrixGeometry']['strikepointZ'][:].data
point_location = np.array([strikepointR[42],-np.abs(strikepointZ[42]),[60]*len(strikepointZ[42])]).T
point_location = coleval.point_toroidal_to_cartesian(point_location)
point_location = coleval.find_location_on_foil(point_location)
strike_poits = coleval.absolute_position_on_foil_to_foil_coord(point_location)
strike_point_location_rot = []
resolution = 1000
for time in range(len(strikepointR[42])):
	point_location = np.array([[strikepointR[42][time]]*resolution,[-np.abs(strikepointZ[42][time])]*resolution,np.linspace(0,360,resolution)]).T
	# x_point_location = np.array([[0]*resolution,np.linspace(-2,2,resolution),[0]*resolution]).T
	# point_location = np.array([fueling_r[time],fueling_z[time],fueling_t[time]]).T
	point_location = coleval.point_toroidal_to_cartesian(point_location)
	point_location = coleval.find_location_on_foil(point_location)
	strike_point_location_rot.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))
# all_time_x_point_location = np.array(all_time_x_point_location)


drsepIn = f['/epm/output']['separatrixGeometry']['drsepIn'][:].data
drsepOut = f['/epm/output']['separatrixGeometry']['drsepOut'][:].data
rmidplaneIn = f['/epm/output']['separatrixGeometry']['rmidplaneIn'][:].data



#
