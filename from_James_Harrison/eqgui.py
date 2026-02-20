# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:34:04 2021

@author: James
"""


from tkinter import simpledialog
import tkinter as tk
import tkinter.filedialog

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
from numpy import zeros
from numpy.linalg import inv
from numpy import (
    dot,
    linspace,
    argmax,
    argmin,
    abs,
    clip,
    sin,
    cos,
    pi,
    amax,
    arctan2,
    sqrt,
    sum,
)

class mclass:
    def __init__(self,  window):
        self.window = window
        self.toolbar = tk.Frame(self.window)

        self.open_button = tk.Button(self.toolbar, text="Open", command=self.open)
        self.save_button = tk.Button(self.toolbar, text="Save to eqdsk", command=self.save_eqdsk)
        self.save_animation = tk.Button(self.toolbar, text="Save to gif", command=self.save_gif)
        self.launch_fieldlines = tk.Button(self.toolbar, text="Launch Fieldlines", command=self.launch_fieldlines)
        self.label1 = tk.Label(self.toolbar, text = ' Time index: ')
        self.label2 = tk.Label(self.toolbar, text = ' ')
        self.time_slider = tk.Spinbox(self.toolbar, command = self.plot, from_ = 0, width=4)

        self.fig = Figure(figsize=(4,4))
        self.a1 = self.fig.add_subplot(131)
        self.a2 = self.fig.add_subplot(332)
        self.a3 = self.fig.add_subplot(333)
        self.a4 = self.fig.add_subplot(335)
        self.a5 = self.fig.add_subplot(336)
        self.a6 = self.fig.add_subplot(338)
        self.a7 = self.fig.add_subplot(339)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)

        # Set the window size
        self.window.geometry("500x500")

        self.toolbar.pack(side=tk.TOP)
        self.open_button.pack(side=tk.LEFT)
        self.save_button.pack(side=tk.LEFT)
        self.launch_fieldlines.pack(side=tk.RIGHT)
        self.label1.pack(side=tk.LEFT)
        self.time_slider.pack(side=tk.LEFT)
        self.label2.pack(side=tk.LEFT)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

        rwall = [1.6,1.2503,1.3483,1.47,1.47,1.45,1.45,1.3214,1.1904,0.89296,
                 0.86938,0.83981,0.82229,0.81974,0.81974,0.82734,0.8548,
                 0.89017,0.91974,0.94066,1.555,1.85,2,2,2,2,1.3188,1.7689,
                 1.7301,1.35,1.09,1.09,0.90576,0.87889,0.87889,0.90717,
                 0.53948,0.5112,0.50553,0.53594,0.5074,0.4974,0.5074,0.4788,
                 0.4688,0.4788,0.333,0.333,0.275,0.334,0.261,0.261,0.244,
                 0.261,0.261,0.244,0.261,0.261]

        zwall = [1,1,0.86,0.86,0.81,0.81,0.82,0.82,1.007,1.304,1.3312,1.3826,
                 1.4451,1.4812,1.4936,1.5318,1.5696,1.5891,1.5936,1.5936,
                 1.567,1.08,1.08,1.7,2.035,2.169,2.169,1.7189,1.68,2.06,
                 2.06,2.06,1.8786,1.9055,1.9055,1.8772,1.5095,1.5378,1.5321,
                 1.5017,1.4738,1.4838,1.4738,1.4458,1.4558,1.4458,1.303,
                 1.1,1.1,1.1,0.502,0.348,0.348,0.348,0.146,0.146,0.146,0]

        # Concatenate with mirror image in Z,
        # with points in reverse order
        self.rwall = np.array(rwall + rwall[::-1])
        self.zwall = np.array(zwall + [-z for z in zwall[::-1]])

        self.psidat = None
        self.shotnumber = None


    def plot (self):
        # Clear all of the plots
        self.a1.clear()
        self.a2.clear()
        self.a3.clear()
        self.a4.clear()
        self.a5.clear()
        self.a6.clear()
        self.a7.clear()

        tindx = int(self.time_slider.get())

        # Plot the equilibrium
        if np.sum(np.isfinite(self.psidat[tindx,:,:])) == np.size(self.psidat[tindx,:,:]):

            self.a1.contour(self.R, self.Z, self.psidat[tindx,:,:], levels = 30, colors = '0.8', linestyles = 'solid')

            self.a1.contour(self.R, self.Z, self.psidat[tindx,:,:], levels = [self.lower_xpoint_p[tindx]], colors = 'b', linestyles = 'solid')
            self.a1.contour(self.R, self.Z, self.psidat[tindx,:,:], levels = [self.upper_xpoint_p[tindx]], colors = 'b', linestyles = 'solid')
            self.a1.plot(self.mag_axis_r[tindx], self.mag_axis_z[tindx],'.r')
            self.a1.plot(self.lower_xpoint_r[tindx], self.lower_xpoint_z[tindx],'xr')
            self.a1.plot(self.upper_xpoint_r[tindx], self.upper_xpoint_z[tindx],'xr')

            self.a1.plot(self.rwall, self.zwall, 'k')
            self.a1.set_aspect(1.0)
            self.a1.set_title('Shot '+str(self.shotnumber)+' '+str(self.time[tindx])+'s')

            self.a2.plot(self.time, self.cpasma*1.0E-3,'k')
            self.a2.plot(self.time[tindx], self.cpasma[tindx]*1.0E-3, '+r')
            self.a2.text(self.time[tindx], self.cpasma[tindx]*1.0E-3,'{0:.3f}'.format(self.cpasma[tindx]*1.0E-3))
            self.a2.set_title('Ip (kA)')

            self.a3.plot(self.time, self.q0,'k',label='q0')
            self.a3.plot(self.time, self.q95,'b',label='q95')
            self.a3.plot(self.time[tindx], self.q0[tindx], '+r')
            self.a3.plot(self.time[tindx], self.q95[tindx], '+r')
            self.a3.text(self.time[tindx], self.q0[tindx],'{0:.3f}'.format(self.q0[tindx]))
            self.a3.text(self.time[tindx], self.q95[tindx],'{0:.3f}'.format(self.q95[tindx]))
            self.a3.legend()
            self.a3.set_title('q profile')

            self.a4.plot(self.time, self.inner_sep_r, 'k')
            self.a4.set_ylim(0.2,0.5)
            self.a4.set_title('Inner LCFS R')

            self.a5.plot(self.time, self.outer_sep_r, 'k')
            self.a5.set_ylim(1.0,1.5)
            self.a5.set_title('Outer LCFS R')

            self.a6.set_title('Connection Length (m)')
            self.a7.set_title('Poloidal Flux Expansion')

        self.canvas.draw()

    def open (self):
        self.efit_filename = tk.filedialog.askopenfilename()

        f = netCDF4.Dataset(self.efit_filename)

        self.psidat = np.transpose(f['output']['profiles2D']['poloidalFlux'], (0, 2, 1))
        self.q0 = f['output']['globalParameters']['q0'][:].data
        self.q95 = f['output']['globalParameters']['q95'][:].data
        self.r = f['output']['profiles2D']['r'][:].data
        self.z = f['output']['profiles2D']['z'][:].data
        self.R = self.r[0,:]
        self.Z = self.z[0,:]
        self.bVac = f['input']['bVacRadiusProduct']['values'][:].data
        self.psi_bnd = f['output']['globalParameters']['psiBoundary'][:].data
        self.psi_axis = f['output']['globalParameters']['psiAxis'][:].data
        self.cpasma = f['output']['globalParameters']['plasmaCurrent'][:].data
        self.r_axis = f['output']['globalParameters']['magneticAxis'][:]['R']
        self.z_axis = f['output']['globalParameters']['magneticAxis'][:]['Z']
        self.time = f['time'][:].data

        self.shotnumber = f.__dict__['pulseNumber']

        f.close()

        self.time_slider.config(to=len(self.time)-1)

        self.calc_equil_properties()

    def save_eqdsk(self):
        out_dir = tk.filedialog.askdirectory()

        tmp = efitData.EfitData(filename=self.efit_filename)

        tindx = int(self.time_slider.get())

        tmp.save_eqdsk(tindx, out_dir)

    def save_gif(self):
        out_file = tk.filddialog.askopenfilename()

        # Inlcude code that plots all of teh equilibria to a GIF

    def interp2d(self,R,Z,field):
        return RectBivariateSpline(R,Z,np.transpose(field))

    def calc_bfield(self,psi, r, z, rbt):

        R = np.linspace(np.min(r),np.max(r),len(r)*2)
        Z = np.linspace(np.min(z),np.max(z),len(r)*2)
        Rgrid,Zgrid = R,Z

        psi_interp = self.interp2d(r, z, psi)

        psi  = psi_interp(Rgrid,Zgrid)
        deriv = np.gradient(psi)    #gradient wrt index

        #Note np.gradient gives y derivative first, then x derivative
        ddR = deriv[1]
        #ddR = self.psi(Rgrid,Zgrid,dx=1)
        ddZ = deriv[0]
        #ddZ = self.psi(Rgrid,Zgrid,dy=1)
        dRdi = 1.0/np.gradient(R)
        dZdi = 1.0/np.gradient(Z)
        dpsidR = ddR*dRdi[np.newaxis,:] #Ensure broadcasting is handled correctly
        dpsidZ = ddZ*dZdi[:,np.newaxis]

        BR = -1.0*dpsidZ/R[np.newaxis,:]
        BZ = dpsidR/R[np.newaxis,:]

        # Calculate the toroidal field
        Bphi = np.zeros(np.shape(BR))

        for i in np.arange(len(R)):
            Bphi[:,i] = rbt/R[i]

        return R, Z, BR, BZ, Bphi


    def ccw(self, Ax, Ay, Bx, By, Cx, Cy):
        """ check if points are counterclockwise """

        if (Cy - Ay)*(Bx-Ax) > (By-Ay)*(Cx-Ax) : result = 1
        else: result = 0
        return result


    def wall_intersection(self, Ax, Ay, Bx, By):
        """
        Check for an intersection between the line (x1,y1),(x2,y2) and the wall
        """

        wallR = self.rwall
        wallZ = self.zwall

        intr = np.zeros(2,dtype=np.double)

        for i in range(wallR.shape[0]-1):
            Cx = wallR[i]
            Cy = wallZ[i]
            Dx = wallR[i+1]
            Dy = wallZ[i+1]
            if self.ccw(Ax,Ay,Cx,Cy,Dx,Dy) != self.ccw(Bx,By,Cx,Cy,Dx,Dy) and self.ccw(Ax,Ay,Bx,By,Cx,Cy) != self.ccw(Ax,Ay,Bx,By,Dx,Dy):

                #Find gradient of lines
                if Ax == Bx:
                    Mab = 1e10*(By-Ay)/np.abs(By-Ay)
                else:
                    Mab = (By - Ay)/(Bx - Ax)
                if Cx == Dx:
                    Mcd = 1e10*(Dy-Cy)/np.abs(Dy-Cy)
                else:
                    Mcd = (Dy - Cy)/(Dx - Cx)

                #Find axis intercepts
                Cab = By - Mab*Bx
                Ccd = Cy - Mcd*Cx

                #Find line intersection point
                intr[0] = (Ccd - Cab)/(Mab - Mcd)
                intr[1] =  Cab + Mab*(Ccd - Cab)/(Mab - Mcd)


                return intr

        return intr

    def follow_fieldline(self, start_r, start_z, Br_interp, Bz_interp, Bphi_interp, ds):

        # Maximium number of steps along a field line before returning, if it doesn't collide
        # with a wall segment first
        maxstep = 100000

        R = start_r
        Z = start_z
        phi = 0.0

        wall_r = self.rwall
        wall_z = self.zwall

        # Set up arrays to store the position along a field line
        rarr = np.zeros(maxstep+1)
        zarr = np.zeros(maxstep+1)
        phiarr = np.zeros(maxstep+1)
        lpar = 0.0

        rarr[0] = R
        zarr[0] = Z

        for i in np.arange(maxstep):
            # Step along the field line using a 4th order Runge-Kutta integrator

            dR1 = ds * Br_interp(R, Z)
            dZ1 = ds * Bz_interp(R, Z)
            dphi1 = ds * Bphi_interp(R, Z) / R

            dR2 = ds * Br_interp(R + 0.5 * dR1, Z + 0.5 * dZ1)
            dZ2 = ds * Bz_interp(R + 0.5 * dR1, Z + 0.5 * dZ1)
            dphi2 = ds * Bphi_interp(R + 0.5 * dR1, Z + 0.5 * dZ1) / R

            dR3 = ds * Br_interp(R + 0.5 * dR2, Z + 0.5 * dZ2)
            dZ3 = ds * Bz_interp(R + 0.5 * dR2, Z + 0.5 * dZ2)
            dphi3 = ds * Bphi_interp(R+0.5*dR2, Z + 0.5 * dZ2) / R

            dR4 = ds * Br_interp(R + dR3, Z + dZ3)
            dZ4 = ds * Bz_interp(R + dR3, Z + dZ3)
            dphi4 = ds * Bphi_interp(R + dR3, Z + dZ3) / R

            dR = (1. / 6.)*(dR1 + 2.0*dR2 + 2.0*dR3 + dR4)
            dZ = (1. / 6.)*(dZ1 + 2.0*dZ2 + 2.0*dZ3 + dZ4)
            dphi = (1. / 6.)*(dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4)

            # Check for a collision with the wall
            intr = self.wall_intersection(R, Z, R + dR, Z + dZ)

            if intr[0]:
                # If there is a collision with the wall, only increment the field
                # line up to the hit point

                dR = intr[0] - R
                dZ = intr[1] - Z
                dphi = dphi*(intr[0]-R)/dR

            rarr[i+1] = R + dR
            zarr[i+1] = Z + dZ
            phiarr[i+1] = phi + dphi
            lpar = lpar + np.sqrt(dR * dR + dZ * dZ + R * R * dphi * dphi)

            if intr[0]:
                # If there is a collision, break out of the loop
                return rarr[0:i+2], zarr[0:i+2], lpar[0][0]

            else:
                # If there is not a collision, update the current position and
                # return to the top of the loop
                R = R + dR
                Z = Z + dZ
                phi = phi + dphi

        return rarr, zarr, lpar[0][0]

    def launch_fieldlines(self):

        tindx = int(self.time_slider.get())

        num_fl = 5

        start_r = self.outer_sep_r[tindx]+np.linspace(5.0E-3,2.0E-2,num_fl)
        start_z = self.mag_axis_z[tindx]

        # Calculate the magnetic field components
        R, Z, BR, BZ, Bphi = self.calc_bfield(self.psidat[tindx,:,:].T, self.r[0,:], self.z[0,:], self.bVac[tindx])
        Br_interp = self.interp2d(R, Z, BR)
        Bz_interp = self.interp2d(R, Z, BZ)
        Bphi_interp = self.interp2d(R, Z, Bphi)
        ds = 2.0E-2

        all_fl_r = []
        all_fl_z = []
        all_lpar = []
        dr_upstream = []
        fexp = []
        dr_upstream_fexp = []

        for i in np.arange(num_fl):
            fl_r, fl_z, lpar = self.follow_fieldline(start_r[i], start_z, Br_interp, Bz_interp, Bphi_interp, ds)
            all_fl_r.append(fl_r)
            all_fl_z.append(fl_z)
            all_lpar.append(lpar)
            dr_upstream.append(fl_r[0]-self.outer_sep_r[tindx])

            self.a1.plot(fl_r, fl_z, ':b')

        for i in np.arange(num_fl-1):
            dr_upstream_fexp.append(0.5*(all_fl_r[i][0]+all_fl_r[i+1][0])-self.outer_sep_r[tindx])

            dr_us = all_fl_r[i+1][0]-all_fl_r[i][0]

            dr = all_fl_r[i+1][-1]-all_fl_r[i][-1]
            dz = all_fl_z[i+1][-1]-all_fl_z[i][-1]
            dpol = np.sqrt(dr*dr+dz*dz)

            fexp.append(dpol/dr_us)


        self.a6.plot(dr_upstream, all_lpar, 'k')
        self.a7.plot(dr_upstream_fexp, fexp, 'k')
        self.canvas.draw()


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

        J = zeros([2, 2])

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
                        #     ( dBz/dR, dBz/dZ )

                        J[0, 0] = -Br / R1 - f(R1, Z1, dy=1, dx=1)[0][0] / R1
                        J[0, 1] = -f(R1, Z1, dy=2)[0][0] / R1
                        J[1, 0] = -Bz / R1 + f(R1, Z1, dx=2) / R1
                        J[1, 1] = f(R1, Z1, dx=1, dy=1)[0][0] / R1

                        d = dot(inv(J), [Br, Bz])

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

                rline = linspace(Ro, Rx, num=50)
                zline = linspace(Zo, Zx, num=50)

                pline = f(rline, zline, grid=False)

                if Px < Po:
                    pline *= -1.0  # Reverse, so pline is maximum at X-point

                # Now check that pline is monotonic
                # Tried finding maximum (argmax) and testing
                # how far that is from the X-point. This can go
                # wrong because psi can be quite flat near the X-point
                # Instead here look for the difference in psi
                # rather than the distance in space

                maxp = amax(pline)
                if (maxp - pline[-1]) / (maxp - pline[0]) > 0.001:
                    # More than 0.1% drop in psi from maximum to X-point
                    # -> Discard
                    continue

                ind = argmin(pline)  # Should be at O-point
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

    def calc_equil_properties(self):
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

        rr, zz = np.meshgrid(self.r[0,:], self.z[0,:])

        if self.psidat is not None:
            for i in np.arange(len(self.time)):
                psiarr = np.array((self.psidat[i,:,:]))
                psi_interp = self.interp2d(self.r[0,:], self.z[0,:],psiarr)

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

                    mp_r_arr = np.linspace(np.min(self.r[0,:]), np.max(self.r[0,:]),500)
                    mp_p_arr = mp_r_arr*0.0

                    for j in np.arange(len(mp_p_arr)):
                            mp_p_arr[j] = psi_interp(mp_r_arr[j], self.mag_axis_z[i])

                    zcr = self.zcd(mp_p_arr-self.psi_bnd[i])

                    if len(zcr) > 2:
                        zcr = zcr[-2:]

                    self.inner_sep_r[i] = mp_r_arr[zcr[0]]
                    self.outer_sep_r[i] = mp_r_arr[zcr[1]]


                    # Calculate dr_sep



            print('Done calculating equilibrium properties.')

window= tk.Tk()
start= mclass (window)
window.mainloop()
