
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 19:15:37 2021

@author: Federici
"""

# Created from code from James Harrison
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import netCDF4
from . import efitData

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
			self.strikepointR = f[file_prefix+'/output']['separatrixGeometry']['strikepointCoords'][:]['R']
			self.strikepointZ = f[file_prefix+'/output']['separatrixGeometry']['strikepointCoords'][:]['Z']
		except IndexError:
			self.R = f[file_prefix+'/output']['profiles2D']['r'][:].data
			self.Z = f[file_prefix+'/output']['profiles2D']['z'][:].data
			self.bVac = f[file_prefix+'/input']['bVacRadiusProduct'][:].data
			self.r_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis']['R'][:]
			self.z_axis = f[file_prefix+'/output']['globalParameters']['magneticAxis']['Z'][:]
			self.shotnumber = f.shot
			self.strikepointR = f[file_prefix+'/output']['separatrixGeometry']['strikepointR'][:].data
			self.strikepointZ = f[file_prefix+'/output']['separatrixGeometry']['strikepointZ'][:].data

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

					# if len(zcr) > 2:	# it should literally have no effect
					# 	zcr = zcr[-2:]
					if len(zcr) < 2:	# made to prevent the error when there is only one zero in (mp_p_arr-self.psi_bnd[i])
						zcr.append(zcr[0])

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

