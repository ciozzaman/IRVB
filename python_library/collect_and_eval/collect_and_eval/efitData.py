
import os
import time
import importlib

import numpy as np
import xarray as xr
import netCDF4 as nc
import fortranformat as ff

from .efitUtils import chi2calc, errorsCalc, writeColumns, chi2totalcalc, get_boundary_classification


class EfitData():
    def __init__(self, include_converged_times=True, include_vacuum_times=True,
                 include_non_converged_times=True, include_fatal_times=False,
                 subsample=1, sample_count=None, time_min=None, time_max=None,
                 filename="efitOut.nc",
                 calc_chi2=True, calc_errors=True,
                 geqdsk_indexfile=None, n_pf=0,
                 shot=None, tokamak='', debug=0):

        self.debug = debug
        self.tokamak = tokamak

        # Dictionary of xarray objects mirroring hierarchy of the NetCDF file.
        # Initially setup empty dictionary structure
        self.data = {"top": None,
                     "constraints": {"plasmaCurrent": None,
                                     "pfCircuits" : None,
                                     "fluxLoops" : None,
                                     "magneticProbes" : None,
                                     "mse" : None,
                                     "lineIntegratedDensities" : None,
                                     "discreteDensities" : None,
                                     "pressures" : None,
                                     "faradayRotationChannels" : None,
                                     "diamagneticFlux" : None,
                                     "boundaries" : None},
                     "bVacRadiusProduct": None,
                     "ironModel": None,
                     "regularGrid": None,
                     "geometry": {"limiter": None, "pfSystem": None},
                     "profiles2D": None,
                     "globalParameters": None,
                     "radialProfiles": None,
                     "fluxFunctionProfiles": None,
                     "separatrixGeometry": None,
                     "numerics": {"numericalControls": None,
                                  "pp": None,
                                  "ff": None,
                                  "ww": None,
                                  "ne": None,
                                  "numericalDetails": None}}

        self.time_resolution = 0

        read_archive = False
        if shot is not None and tokamak is not None:
            if tokamak.lower() != "":
                read_archive = True

        read_eqdsk = False
        if geqdsk_indexfile is not None:
            read_eqdsk = True

        if not read_archive and not read_eqdsk:
            self._read_data(filename, include_converged_times=include_converged_times, include_vacuum_times=include_vacuum_times,
                            include_non_converged_times=include_non_converged_times, include_fatal_times=include_fatal_times,
                            subsample=subsample, sample_count=sample_count, time_min=time_min, time_max=time_max)
            if calc_chi2:
                self.calc_chi2()

            if calc_errors:
                self.calc_errors()
        elif read_archive:
            tok_module = importlib.import_module('efitdata_{}'.format(tokamak.lower()))
            all_data = tok_module.read_tok_data(shot, include_converged_times=include_converged_times, include_vacuum_times=include_vacuum_times,
                                                include_non_converged_times=include_non_converged_times, include_fatal_times=include_fatal_times,
                                                subsample=subsample, sample_count=sample_count, time_min=time_min, time_max=time_max)
            self.data = all_data
        elif read_eqdsk:
            self._read_eqdsk(geqdsk_indexfile, subsample=subsample, sample_count=sample_count, time_min=time_min, time_max=time_max, n_pf=n_pf)


    def _read_group(self, filename, group, itimes=None):
        try:
            xrdata = xr.open_dataset(filename, group=group, decode_times=False)

            if itimes is not None:
                xrdata = xrdata.isel(time=itimes)

            # This is needed since when reading in at group level the coordinate values are not picked up from higher up levels
            if "time" in xrdata.dims:
                xrdata = xrdata.assign(time = self.data["top"].time.values)


        except OSError:
            if self.debug > 0:
                print("<<WARNING>> Could not find group {} in netcdf file {}".format(group, filename))

            xrdata = None

        return xrdata

    def _read_data(self, filename, include_converged_times=True, include_vacuum_times=True,
                   include_non_converged_times=True, include_fatal_times=False,
                   subsample=1, sample_count=None, time_min=None, time_max=None):

        # --------------------------
        # Read in top level status and time
        # And select time slices that were requested
        # --------------------------
        top = xr.open_dataset(filename, decode_times=False)

        # Change time to floats, in seconds

        equilibrium_status = top.equilibriumStatusInteger.values

        time_only_flag = np.zeros(equilibrium_status.size, dtype="bool") + 1

        flag = np.zeros(equilibrium_status.size, dtype="bool")
        if include_converged_times:
            flag = np.logical_or(flag, equilibrium_status == 1)
        if include_vacuum_times:
            flag = np.logical_or(flag, equilibrium_status == 2)
        if include_non_converged_times:
            flag = np.logical_or(flag, equilibrium_status == -1)
        if include_fatal_times:
            flag = np.logical_or(flag, equilibrium_status == -2)
        if time_min is not None:
            flag = np.logical_and(flag, top.time.values.astype(np.float64) >= time_min)
            time_only_flag = np.logical_and(time_only_flag, top.time.values.astype(np.float64) >= time_min)
        if time_max is not None:
            flag = np.logical_and(flag, top.time.values.astype(np.float64) <= time_max)
            time_only_flag = np.logical_and(time_only_flag, top.time.values.astype(np.float64) <= time_max)

        # Record number of each type of timeslice in time range requested
        time_range_indices = np.arange(top.time.size)[(time_only_flag == 1)]
        self.n_converged = np.sum(equilibrium_status[time_range_indices] == 1)
        self.n_vacuum = np.sum(equilibrium_status[time_range_indices] == 1)
        self.n_unconverged = np.sum(equilibrium_status[time_range_indices] == -1)
        self.n_fatal = np.sum(equilibrium_status[time_range_indices] == -2)
        self.n_all = self.n_converged + self.n_vacuum + self.n_unconverged + self.n_fatal

        efit_time_indices = np.arange(top.time.size)[flag]

        if efit_time_indices.size == 0:
            if self.debug > 0:
                print("<<ERROR>> No timeslices were found in the file")
                print("          You requested to include converged? {} vacuum? {} non-converged? {} fatal? {}".format(include_converged_times,
                                                                                                                       include_vacuum_times,
                                                                                                                       include_non_converged_times,
                                                                                                                       include_fatal_times))
                if time_min is not None and time_max is not None:
                    print("          For times between {} and {} seconds.".format(time_min, time_max))
                elif time_min is not None:
                    print("          For times >= {} seconds.".format(time_min))
                elif time_max is not None:
                    print("          For times <= {} seconds.".format(time_max))
            return

        # Subsample the time-slices
        efit_time_indices = efit_time_indices[::subsample]

        # specify the number of time-slices (optional)
        if sample_count is not None:
            efit_time_indices = efit_time_indices[:sample_count]

        top = top.isel(time=efit_time_indices)

        if top.time.size > 1:
            self.time_resolution = (top.time.values[1] - top.time.values[0])

        mintime = top.time.min().values
        maxtime = top.time.max().values

        if len(efit_time_indices) == 1:
            print("<<INFO>> EFIT++ data available for a single time slice: {} seconds".format(mintime))
        else:
            efitTimeResolution = np.diff(top.time.values)
            averageEfitTimeResolution = np.mean(efitTimeResolution)
            print("<<INFO>> EFIT++ data available from {} to {} seconds (av. tiime resol. {} ms)".format(mintime,
                                                                                                         maxtime,
                                                                                                         averageEfitTimeResolution * 1000))

        print("<<INFO>> efitplot will process: {}  time-slices in the time range: {}<t<{}".format(len(efit_time_indices), mintime, maxtime) )

        self.data["top"] = top

        # --------------------------
        # Constraints
        # --------------------------
        self.data["constraints"] = {}
        self.data["constraints"]["plasmaCurrent"] = self._read_group(filename, "/input/constraints/plasmaCurrent", itimes = efit_time_indices)
        self.data["constraints"]["pfCircuits"] = self._read_group(filename, "/input/constraints/pfCircuits", itimes = efit_time_indices)
        self.data["constraints"]["fluxLoops"] = self._read_group(filename, "/input/constraints/fluxLoops", itimes = efit_time_indices)
        self.data["constraints"]["magneticProbes"] = self._read_group(filename, "/input/constraints/magneticProbes", itimes = efit_time_indices)
        self.data["constraints"]["mse"] = self._read_group(filename, "/input/constraints/mse", itimes = efit_time_indices)
        self.data["constraints"]["lineIntegratedDensities"] = self._read_group(filename, "/input/constraints/lineIntegratedDensities", itimes = efit_time_indices)
        self.data["constraints"]["discreteDensities"] = self._read_group(filename, "/input/constraints/discreteDensities", itimes = efit_time_indices)
        self.data["constraints"]["pressures"] = self._read_group(filename, "/input/constraints/pressures", itimes = efit_time_indices)
        self.data["constraints"]["faradayRotationChannels"] = self._read_group(filename, "/input/constraints/faradayRotationChannels", itimes = efit_time_indices)
        self.data["constraints"]["diamagneticFlux"] = self._read_group(filename, "/input/constraints/diamagneticFlux", itimes = efit_time_indices)
        self.data["constraints"]["boundaries"] = self._read_group(filename, "/input/constraints/boundaries", itimes = efit_time_indices)
        self.data["constraints"]["q0"] = self._read_group(filename, "/input/constraints/q0", itimes = efit_time_indices)

        # --------------------------
        # bvacRadiusProduct
        # --------------------------
        # Can't read this in directly using xarray because the variable is called values
        self.data["bVacRadiusProduct"] = self._read_bvacradiusproduct(filename, efit_time_indices)
        
        # --------------------------
        # Iron Model
        # --------------------------
        self.data["ironModel"] = self._read_group(filename, "/input/constraints/ironModel/geometry")

        # --------------------------
        # Grid
        # --------------------------
        self.data["regularGrid"] = self._read_group(filename, "/input/regularGrid")

        # --------------------------
        # Geometry
        # --------------------------
        self.data["geometry"] = {}
        self.data["geometry"]["limiter"] = self._read_group(filename, "/input/limiter")
        self.data["geometry"]["pfSystem"] = self._read_group(filename, "/input/pfSystem")

        # --------------------------
        # Variables
        # --------------------------
        self.data["profiles2D"] =  self._read_group(filename, "/output/profiles2D", itimes = efit_time_indices)
        self.data["globalParameters"] = self._read_group(filename, "/output/globalParameters", itimes = efit_time_indices)
        self.data["radialProfiles"] = self._read_group(filename, "/output/radialProfiles", itimes = efit_time_indices)
        self.data["fluxFunctionProfiles"] = self._read_group(filename, "/output/fluxFunctionProfiles", itimes=efit_time_indices)

        # Can't read this in directly using xarray, since pandas can't cope with nan's for arrays of objects (ie. compound types...)
        self.data["separatrixGeometry"] = self._read_separatrix(filename, efit_time_indices)

        # --------------------------
        # Numerics
        # --------------------------
        self.data["numerics"] = {}
        self.data["numerics"]["numericalControls"] = self._read_group(filename, "/input/numericalControls", itimes=efit_time_indices)
        self.data["numerics"]["pp"] = self._read_group(filename, "/input/numericalControls/pp", itimes=efit_time_indices)
        self.data["numerics"]["ff"] = self._read_group(filename, "/input/numericalControls/ff", itimes=efit_time_indices)
        self.data["numerics"]["ww"] = self._read_group(filename, "/input/numericalControls/ww", itimes=efit_time_indices)
        self.data["numerics"]["ne"] = self._read_group(filename, "/input/numericalControls/ne", itimes=efit_time_indices)
        self.data["numerics"]["numericalDetails"] = self._read_group(filename, "/output/numericalDetails", itimes=efit_time_indices)

        # Computed boundary at midplane
        if self.data["constraints"]["boundaries"] is not None:
            bound_computed = self.data["separatrixGeometry"].rmidplaneOut
            self.data["constraints"]["boundaries"] = self.data["constraints"]["boundaries"].assign(computedSep = (['time', 'boundariesDim'], bound_computed.values.reshape((self.data["top"].time.size, 1))))

        # --------------------------
        # Build info
        # --------------------------
        self.data["buildInformation"] = self._read_group(filename, "buildInformation")


    def _read_bvacradiusproduct(self, filename, itimes):
        # This is problematic because the variable is called values which seems to mess with xarray NC reading
        if self.data["top"] is None:
            if self.debug > 0:
                print("<<ERROR>> Read main data before bvacradiusproduct.")
            return None

        rootgrp = nc.Dataset(filename)

        try:
            bvacradiusproduct = rootgrp.groups['input']['bVacRadiusProduct'].variables['values'][itimes]
        except KeyError as err:
            if self.debug > 0:
                print("<<WARNING>> Could not find output group or variable in output group: {} in netcdf file {}".format(err, filename))

            return None


        bvacradiusproduct_ds = xr.Dataset({'dataValues': (['time'], bvacradiusproduct)},
                                          coords={'time': self.data["top"].time.values})

        return bvacradiusproduct_ds


    def _read_separatrix(self, filename, itimes):
        if self.data["top"] is None:
            if self.debug > 0:
                print("<<ERROR>> Read main data before separatrix data.")
            return None

        rootgrp = nc.Dataset(filename)

        try:
            group = rootgrp.groups['output'].groups['separatrixGeometry']
            boundclassint = group.variables['boundaryClassification'][itimes]
            rbound = group.variables['boundaryCoords'][itimes]['R']
            zbound = group.variables['boundaryCoords'][itimes]['Z']
            xpointCount = group.variables['xpointCount'][itimes]
            xpointR = group.variables['xpointCoords'][itimes]['R']
            xpointZ = group.variables['xpointCoords'][itimes]['Z']
            strikepointR = group.variables['strikepointCoords'][itimes]['R']
            strikepointZ = group.variables['strikepointCoords'][itimes]['Z']
            limiterCoordsR = group.variables['limiterCoords'][itimes]['R']
            limiterCoordsZ = group.variables['limiterCoords'][itimes]['Z']
            rGeom = group.variables['geometricAxis'][itimes]['R']
            zGeom = group.variables['geometricAxis'][itimes]['Z']

            minorRadius = group.variables['minorRadius'][itimes]
            elongation = group.variables['elongation'][itimes]
            upperTriangularity = group.variables['upperTriangularity'][itimes]
            lowerTriangularity = group.variables['lowerTriangularity'][itimes]
            rmidplaneIn = group.variables['rmidplaneIn'][itimes]
            rmidplaneOut = group.variables['rmidplaneOut'][itimes]
            drsepIn = group.variables['drsepIn'][itimes]
            drsepOut = group.variables['drsepOut'][itimes]

        except KeyError as err:
            if self.debug > 0:
                print("<<WARNING>> Could not find output group or variable in output group: {} in netcdf file {}".format(err, filename))

            return None


        separatrix_ds = xr.Dataset({'boundaryClassificationInteger': (['time'], boundclassint),
                                    'rBoundary': (['time', 'boundaryCoordsDim'], rbound),
                                    'zBoundary': (['time', 'boundaryCoordsDim'], zbound),
                                    'minorRadius': (['time'], minorRadius),
                                    'elongation': (['time'], elongation),
                                    'upperTriangularity': (['time'], upperTriangularity),
                                    'lowerTriangularity': (['time'], lowerTriangularity),
                                    'rmidplaneIn': (['time'], rmidplaneIn),
                                    'rmidplaneOut': (['time'], rmidplaneOut),
                                    'rGeom': (['time'], rGeom),
                                    'zGeom': (['time'], zGeom),
                                    'xpointCount': (['time'], xpointCount),
                                    'xpointR': (['time', 'xpointDim'], xpointR),
                                    'xpointZ': (['time', 'xpointDim'], xpointZ),
                                    'strikepointR': (['time', 'strikepointDim'], strikepointR),
                                    'strikepointZ': (['time', 'strikepointDim'], strikepointZ),
                                    'limiterR': (['time'], limiterCoordsR),
                                    'limiterZ': (['time'], limiterCoordsZ),
                                    'drsepIn': (['time'], drsepIn),
                                    'drsepOut': (['time'], drsepOut)},
                                    coords={'time': self.data["top"].time.values})

        # Add x-points as special case if they exist in the file
        try:
            separatrix_ds = separatrix_ds.assign(dndXpointCount = (['time'], group.variables['dndXpointCount'][itimes]))

            dndXpoints = group.variables['dndXpointCoords'][itimes, :]
            ind_nan = np.where(np.absolute(dndXpoints['R']) < 1e-5)
            dndXpoints['R'][ind_nan] = np.nan
            dndXpoints['Z'][ind_nan] = np.nan

            separatrix_ds = separatrix_ds.assign(dndXpoint1R = (['time'], dndXpoints['R'][:,0]))
            separatrix_ds = separatrix_ds.assign(dndXpoint1Z = (['time'], dndXpoints['Z'][:,0]))
            separatrix_ds = separatrix_ds.assign(dndXpoint2R = (['time'], dndXpoints['R'][:,1]))
            separatrix_ds = separatrix_ds.assign(dndXpoint2Z = (['time'], dndXpoints['Z'][:,1]))

            dndstrikepoints = group.variables['dndStrikepointCoords'][itimes, :]

            ind_nan = np.where(np.absolute(dndstrikepoints['R']) < 1e-5)
            dndstrikepoints['R'][ind_nan] = np.nan
            dndstrikepoints['Z'][ind_nan] = np.nan

            separatrix_ds = separatrix_ds.assign(dndXpoint1InnerStrikepointR = (['time'], dndstrikepoints['R'][:,0]))
            separatrix_ds = separatrix_ds.assign(dndXpoint1InnerStrikepointZ = (['time'], dndstrikepoints['Z'][:,0]))
            separatrix_ds = separatrix_ds.assign(dndXpoint1OuterStrikepointR = (['time'], dndstrikepoints['R'][:,1]))
            separatrix_ds = separatrix_ds.assign(dndXpoint1OuterStrikepointZ = (['time'], dndstrikepoints['Z'][:,1]))
            separatrix_ds = separatrix_ds.assign(dndXpoint2InnerStrikepointR = (['time'], dndstrikepoints['R'][:,2]))
            separatrix_ds = separatrix_ds.assign(dndXpoint2InnerStrikepointZ = (['time'], dndstrikepoints['Z'][:,2]))
            separatrix_ds = separatrix_ds.assign(dndXpoint2OuterStrikepointR = (['time'], dndstrikepoints['R'][:,3]))
            separatrix_ds = separatrix_ds.assign(dndXpoint2OuterStrikepointZ = (['time'], dndstrikepoints['Z'][:,3]))

        except KeyError:
            pass

        return separatrix_ds

    def _read_eqdsk(self, geqdsk_indexfile, subsample=1, sample_count=None, time_min=None, time_max=None, n_pf=0):
        try:
            from pyEquilibrium.geqdsk import Geqdsk
            from pyEquilibrium.aeqdsk import Aeqdsk
        except ImportError:
            print("<<ERROR>> Reading from eqdsk files requires pyEquilibrium")
            print("          See https://git.ccfe.ac.uk/SOL_Transport/pyEquilibrium for installation instructions.")
            return

        all_times = np.array([])
        all_geqdsk = np.array([])
        all_aeqdsk = np.array([])

        # Read index.dat file
        with open(geqdsk_indexfile, 'r') as f_in:
            ntimes = f_in.readline()
            file_dir_line = f_in.readline()
            file_dir = file_dir_line[file_dir_line.find("path=")+len("path="):]
            file_dir = file_dir.replace("\"", "")
            file_dir = file_dir.replace("\n", "")

            for ind, row in enumerate(f_in):
                row_split = [r for r in row.split(" ") if r != ""]

                time = None
                geqdsk = None
                aeqdsk = None

                if row_split[0].startswith("time="):
                    time = float(row_split[0][row_split[0].find("time=")+len("time="):])
                else:
                    print("Could not detemine time for row {}".format(row))
                    continue

                if row_split[1].startswith("filename="):
                    geqdsk = row_split[1][row_split[1].find("filename=")+len("filename="):]
                    geqdsk = geqdsk.replace('\n', '')
                    geqdsk = file_dir+'/'+geqdsk
                else:
                    print("Could not detemine geqdsk filename fo row {}".format(row))
                    continue

                if len(row_split) > 2:
                    if row_split[2].startswith("afile="):
                        aeqdsk = row_split[2][row_split[2].find("afile=")+len("afile="):]
                        aeqdsk = aeqdsk.replace('\n', '')
                        aeqdsk = file_dir+'/'+aeqdsk

                all_times = np.append(all_times, time)
                all_geqdsk = np.append(all_geqdsk, geqdsk)
                all_aeqdsk = np.append(all_aeqdsk, aeqdsk)

        # Make sure we are in time order
        ind_time_sort = np.argsort(all_times)
        all_times = all_times[ind_time_sort]
        all_geqdsk = all_geqdsk[ind_time_sort]
        all_aeqdsk = all_aeqdsk[ind_time_sort]

        # Filter times
        flag = np.zeros(all_times.size, dtype="bool")
        flag[:] = True
        if time_min is not None:
            flag = np.logical_and(flag, all_times >= time_min)
        if time_max is not None:
            flag = np.logical_and(flag, all_times <= time_max)

        all_times = all_times[flag]
        all_geqdsk  = all_geqdsk[flag]
        all_aeqdsk = all_aeqdsk[flag]

        if all_times.size == 0:
            print("<<ERROR>> No timeslices found")
            if time_min is not None and time_max is not None:
                print("          For times between {} and {} seconds.".format(time_min, time_max))
            elif time_min is not None:
                print("          For times >= {} seconds.".format(time_min))
            elif time_max is not None:
                print("          For times <= {} seconds.".format(time_max))
            return

        all_times = all_times[::subsample]
        all_geqdsk = all_geqdsk[::subsample]
        all_aeqdsk = all_aeqdsk[::subsample]

        if sample_count is not None:
            all_times = all_times[:sample_count]
            all_geqdsk = all_geqdsk[:sample_count]
            all_aeqdsk = all_aeqdsk[:sample_count]

        n_times = len(all_times)

        nan_array = np.zeros(n_times)
        nan_array[:] = np.nan
        coord_nan_array = np.zeros(n_times, dtype=np.dtype([('R', '<f8'), ('Z', '<f8')]))
        coord_nan_array[:] = np.nan

        # Initialize time and xarray datasets with time dependent arrays
        self.data["top"] = xr.Dataset({'equilibriumStatusInteger': (['time'], np.copy(nan_array))},
                            coords={'time': all_times})

        # First, read to find maximum nbbbs
        nbbbs = 0
        for geqdsk in all_geqdsk:
            geqdsk_data = Geqdsk(filename=geqdsk)
            if geqdsk_data.data['nbbbs'] > nbbbs:
                nbbbs = geqdsk_data.data['nbbbs']

        # Read in rest of data
        for ind_time, (time, geqdsk, aeqdsk) in enumerate(zip(all_times, all_geqdsk, all_aeqdsk)):
            geqdsk_data = Geqdsk(filename=geqdsk)

            # Time independent data: will only be recorded once
            if self.data["regularGrid"] is None:
                self.data["regularGrid"] = xr.Dataset({'rMin': (['unityDim'], [geqdsk_data.data['rleft']]),
                                                       'rMax': (['unityDim'], [geqdsk_data.data['rleft'] + geqdsk_data.data['rdim']]),
                                                       'zMin': (['unityDim'], [geqdsk_data.data['zmid'] + geqdsk_data.data['zdim']/2.0]),
                                                       'zMax': (['unityDim'], [geqdsk_data.data['zmid'] - geqdsk_data.data['zdim']/2.0]),
                                                       'nR': (['unityDim'], [geqdsk_data.data['nw']]),
                                                       'nZ': (['unityDim'], [geqdsk_data.data['nh']])})

            if self.data["geometry"]["limiter"] is None:
                self.data["geometry"]["limiter"] = xr.Dataset({'rValues': (['limiterCoord'], geqdsk_data.data['rlim']),
                                                               'zValues': (['limiterCoord'], geqdsk_data.data['zlim'])})

            # Time dependent data: initialize xarray datasets the first time with nan arrays
            if self.data["constraints"]["plasmaCurrent"] is None:
                self.data["constraints"]["plasmaCurrent"] = xr.Dataset({'target': (['time'], np.copy(nan_array)),
                                                                        'computed': (['time'], np.copy(nan_array))},
                                                                       coords={'time': all_times})

            if self.data["globalParameters"] is None:
                self.data["globalParameters"] = xr.Dataset({'magneticAxis': (['time'], np.copy(coord_nan_array)),
                                                            'psiAxis': (['time'], np.copy(nan_array)),
                                                            'psiBoundary': (['time'], np.copy(nan_array)),
                                                            'bvacRmag': (['time'], np.copy(nan_array)),
                                                            'plasmaCurrent': (['time'], np.copy(nan_array))},
                                                            coords={'time': all_times})

            # Profiles 2D
            if self.data["profiles2D"] is None:
                rgrid = np.arange(geqdsk_data.data['nw']) * geqdsk_data.data['rdim'] / float(geqdsk_data.data['nw'] - 1) + geqdsk_data.data['rleft']
                zgrid = np.arange(geqdsk_data.data['nh']) * geqdsk_data.data['zdim'] / float(geqdsk_data.data['nh'] - 1) + geqdsk_data.data['zmid'] - 0.5 * geqdsk_data.data['zdim']
                nan_array_2d = np.zeros((len(all_times), geqdsk_data.data['nw']))
                nan_array_2d[:] = np.nan
                nan_array_3d = np.zeros((len(all_times), geqdsk_data.data['nw'], geqdsk_data.data['nh']))
                nan_array_3d[:] = np.nan

                self.data["profiles2D"] = xr.Dataset({'r': (['time', 'rgrid'], np.copy(nan_array_2d)),
                                                          'z': (['time', 'zgrid'], np.copy(nan_array_2d)),
                                                          'poloidalFlux': (['time', 'rgrid', 'zgrid'], np.copy(nan_array_3d)),
                                                          'jphi': (['time', 'rgrid', 'zgrid'], np.copy(nan_array_3d))},
                                                         coords={'time': all_times, 'rgrid': rgrid, 'zgrid': zgrid})

            rValues = rgrid
            self.data["profiles2D"].r.values[ind_time, :] = rValues
            zValues = zgrid
            self.data["profiles2D"].z.values[ind_time, :] = zValues


            if geqdsk_data.data['rmaxis'] > 0.0:

                if self.data["fluxFunctionProfiles"] is None:
                    normalized_pol_flux = np.arange(len(geqdsk_data.data['fpol'])) / (len(geqdsk_data.data['fpol'])-1)
                    nan_array_2d = np.zeros((len(all_times), len(normalized_pol_flux)))
                    nan_array_2d[:] = np.nan

                    self.data["fluxFunctionProfiles"] = xr.Dataset({'rBphi': (['time', 'normalizedPoloidalFlux'], np.copy(nan_array_2d)),
                                                                    'staticPressure': (['time', 'normalizedPoloidalFlux'], np.copy(nan_array_2d)),
                                                                    'ffPrime': (['time', 'normalizedPoloidalFlux'], np.copy(nan_array_2d)),
                                                                    'staticPPrime': (['time', 'normalizedPoloidalFlux'], np.copy(nan_array_2d)),
                                                                    'q': (['time', 'normalizedPoloidalFlux'], np.copy(nan_array_2d))},
                                                                      coords={'time': all_times, 'normalizedPoloidalFlux': normalized_pol_flux})



                # Separatrix
                if self.data["separatrixGeometry"] is None and nbbbs > 0:
                    nan_array_2d = np.zeros((len(all_times), nbbbs))
                    nan_array_2d[:] = np.nan
                    self.data["separatrixGeometry"] = xr.Dataset({'rBoundary': (['time', 'boundaryDims'], np.copy(nan_array_2d)),
                                                                  'zBoundary': (['time', 'boundaryDims'], np.copy(nan_array_2d)),
                                                                  'rmidplaneOut': (['time'], np.copy(nan_array)),
                                                                  'rmidplaneIn': (['time'], np.copy(nan_array))},
                                                                 coords={'time': all_times})


                # "DIII-D convention" in eqdsk : psi must be increasing, EFIT++ output is decreasing for positive ip so switch to EFIT++ conventions
                flip_psi_factor = 1.0
                if geqdsk_data.data['current'] > 0.0:
                     flip_psi_factor = -1.0

                self.data["profiles2D"].poloidalFlux.values[ind_time,:,:] = flip_psi_factor * np.transpose(geqdsk_data.data['psirz'])

                self.data["globalParameters"].magneticAxis.values[ind_time]['R'] = geqdsk_data.data['rmaxis']
                self.data["globalParameters"].magneticAxis.values[ind_time]['Z'] = geqdsk_data.data['zmaxis']
                self.data["globalParameters"].psiAxis.values[ind_time] = flip_psi_factor * geqdsk_data.data['simag']
                self.data["globalParameters"].psiBoundary.values[ind_time] = flip_psi_factor * geqdsk_data.data['sibry']

                self.data["globalParameters"].bvacRmag.values[ind_time] = geqdsk_data.data['rmaxis'] * geqdsk_data.data['bcentr'] / geqdsk_data.data['rcentr']
                self.data["globalParameters"].plasmaCurrent.values[ind_time] = geqdsk_data.data['current']
                self.data["constraints"]["plasmaCurrent"].computed.values[ind_time] = geqdsk_data.data['current']

                self.data["fluxFunctionProfiles"].rBphi.values[ind_time,:] = geqdsk_data.data['fpol']
                self.data["fluxFunctionProfiles"].staticPressure.values[ind_time,:] = geqdsk_data.data['pres']

                self.data["fluxFunctionProfiles"].ffPrime.values[ind_time,:] = flip_psi_factor * geqdsk_data.data['ffprime']

                self.data["fluxFunctionProfiles"].staticPPrime.values[ind_time,:] = flip_psi_factor * geqdsk_data.data['pprime']
                self.data["fluxFunctionProfiles"].q.values[ind_time,:] = flip_psi_factor *  geqdsk_data.data['qpsi']

                if geqdsk_data.data['nbbbs'] > 0:
                    self.data["separatrixGeometry"].rBoundary.values[ind_time, 0:geqdsk_data.data['nbbbs']] = geqdsk_data.data['rbbbs']
                    self.data["separatrixGeometry"].zBoundary.values[ind_time, 0:geqdsk_data.data['nbbbs']] = geqdsk_data.data['zbbbs']

            if aeqdsk is not None:
                aeqdsk_data = Aeqdsk(filename=aeqdsk)

                # Shot number
                if 'pulseNumber' not in self.data["top"].keys():
                    self.data["top"].attrs['pulseNumber'] = int(aeqdsk_data.data["shot"])

                # Equilibrium Status Integer
                if aeqdsk_data.data["jflag"] > 0:
                    equilbriumStatus = aeqdsk_data.data["jflag"]
                else:
                    equilbriumStatus = -1.0 * aeqdsk_data.data["lflag"]

                self.data["top"].equilibriumStatusInteger.values[ind_time] = equilbriumStatus

                # Assign variables if they don't exist already (check for rGeom, assign them all if it isn't already assigned)
                if self.data["separatrixGeometry"] is not None and 'rGeom' not in self.data["separatrixGeometry"]:
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(rGeom=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(zGeom=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(minorRadius=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(elongation=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(upperTriangularity=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(lowerTriangularity=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(dndXpointCount=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(dndXpoint1R=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(dndXpoint1Z=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(dndXpoint2R=(['time'], np.copy(nan_array)))
                    self.data["separatrixGeometry"] = self.data["separatrixGeometry"].assign(dndXpoint2Z=(['time'], np.copy(nan_array)))

                    self.data["globalParameters"] = self.data["globalParameters"].assign(plasmaVolume=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(currentCentroid=(['time'], np.copy(coord_nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(betat=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(betap=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(li=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(q95=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(s1=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(s2=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(s3=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(poloidalArea=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(plasmaEnergy=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(q0=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(diamagneticFlux=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(alpha=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(rt=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(bphiRmag=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(q1Radius=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(q2Radius=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(q3Radius=(['time'], np.copy(nan_array)))
                    self.data["globalParameters"] = self.data["globalParameters"].assign(betapd=(['time'], np.copy(nan_array)))

                # Plasma variables
                if aeqdsk_data.data['rout'] > 0.0:
                    self.data["constraints"]["plasmaCurrent"].target.values[ind_time] = aeqdsk_data.data['pasmat']
                    self.data["separatrixGeometry"].rGeom.values[ind_time] = aeqdsk_data.data['rout'] / 100.0
                    self.data["separatrixGeometry"].zGeom.values[ind_time] = aeqdsk_data.data['zout'] / 100.0
                    self.data["separatrixGeometry"].minorRadius.values[ind_time] = aeqdsk_data.data['aout'] / 100.0
                    self.data["separatrixGeometry"].elongation.values[ind_time] = aeqdsk_data.data['eout']
                    self.data["separatrixGeometry"].upperTriangularity.values[ind_time] = aeqdsk_data.data['doutu']
                    self.data["separatrixGeometry"].lowerTriangularity.values[ind_time] = aeqdsk_data.data['doutl']
                    self.data["separatrixGeometry"].rmidplaneOut.values[ind_time] = aeqdsk_data.data['rmidout']
                    self.data["separatrixGeometry"].rmidplaneIn.values[ind_time] = aeqdsk_data.data['rmidin']


                    if aeqdsk_data.data['rseps0'] > 0 and aeqdsk_data.data['rseps1'] > 0:
                        self.data["separatrixGeometry"].dndXpointCount.values[ind_time] = 2
                    elif aeqdsk_data.data['rseps0'] > 0:
                        self.data["separatrixGeometry"].dndXpointCount.values[ind_time] = 1
                    else:
                        self.data["separatrixGeometry"].dndXpointCount.values[ind_time] = 0

                    if self.data["separatrixGeometry"].dndXpointCount.values[ind_time] > 0:
                        self.data["separatrixGeometry"].dndXpoint1R.values[ind_time] = aeqdsk_data.data['rseps0'] / 100.0
                        self.data["separatrixGeometry"].dndXpoint1Z.values[ind_time] = aeqdsk_data.data['zseps0'] / 100.0

                    if self.data["separatrixGeometry"].dndXpointCount.values[ind_time] > 1:
                        self.data["separatrixGeometry"].dndXpoint2R.values[ind_time] = aeqdsk_data.data['rseps1'] / 100.0
                        self.data["separatrixGeometry"].dndXpoint2Z.values[ind_time] = aeqdsk_data.data['zseps1'] / 100.0

                    self.data["globalParameters"].currentCentroid.values[ind_time]['R'] = aeqdsk_data.data['rcurrt'] / 100.0
                    self.data["globalParameters"].currentCentroid.values[ind_time]['Z'] = aeqdsk_data.data['zcurrt'] / 100.0
                    self.data["globalParameters"].betat.values[ind_time] = aeqdsk_data.data['betat'] / 100.0
                    self.data["globalParameters"].betap.values[ind_time] = aeqdsk_data.data['betap']
                    self.data["globalParameters"].li.values[ind_time] = aeqdsk_data.data['ali']
                    self.data["globalParameters"].q95.values[ind_time] = aeqdsk_data.data['qpsi95']
                    self.data["globalParameters"].s1.values[ind_time] = aeqdsk_data.data['s1']
                    self.data["globalParameters"].s2.values[ind_time] = aeqdsk_data.data['s2']
                    self.data["globalParameters"].s3.values[ind_time] = aeqdsk_data.data['s3']
                    self.data["globalParameters"].poloidalArea.values[ind_time] = aeqdsk_data.data['areao']
                    self.data["globalParameters"].plasmaEnergy.values[ind_time] = aeqdsk_data.data['wplasm']
                    self.data["globalParameters"].q0.values[ind_time] = aeqdsk_data.data['qqmagx']
                    self.data["globalParameters"].diamagneticFlux.values[ind_time] = aeqdsk_data.data['cdflux']
                    self.data["globalParameters"].alpha.values[ind_time] = aeqdsk_data.data['alpha']
                    self.data["globalParameters"].rt.values[ind_time] = aeqdsk_data.data['rttt']
                    self.data["globalParameters"].q1Radius.values[ind_time] = aeqdsk_data.data['aaq1']
                    self.data["globalParameters"].q2Radius.values[ind_time] = aeqdsk_data.data['aaq2']
                    self.data["globalParameters"].q3Radius.values[ind_time] = aeqdsk_data.data['aaq3']
                    self.data["globalParameters"].betapd.values[ind_time] = aeqdsk_data.data['betapd']

                # Things that exist without plasma
                nsilop = aeqdsk_data.data['nsilop0']
                nmagpr = aeqdsk_data.data['magpri0']
                nfcoil = aeqdsk_data.data['nfcoil0']
                necurrt = aeqdsk_data.data['nesum0']

                if self.data["constraints"]["fluxLoops"] is None and nsilop > 0:
                    nan_array_2d = np.zeros((len(all_times), nsilop))
                    nan_array_2d[:] = np.nan
                    id_array_fl = np.arange(nsilop) + 1
                    self.data["constraints"]["fluxLoops"] = xr.Dataset({'computed': (['time', 'fluxLoopDim'], np.copy(nan_array_2d)),
                                                                        'target': (['time', 'fluxLoopDim'], np.copy(nan_array_2d)),
                                                                        'id': (['fluxLoopDim'], np.copy(id_array_fl))},
                                                                        coords={'time': all_times})

                if self.data["constraints"]["magneticProbes"] is None and nmagpr > 0:
                    nan_array_2d = np.zeros((len(all_times), nmagpr))
                    nan_array_2d[:] = np.nan
                    id_array_mp = np.arange(nmagpr) + 1
                    self.data["constraints"]["magneticProbes"] = xr.Dataset({'computed': (['time', 'magneticProbeDim'], np.copy(nan_array_2d)),
                                                                             'target': (['time', 'magneticProbeDim'], np.copy(nan_array_2d)),
                                                                             'id': (['magneticProbeDim'], np.copy(id_array_mp))},
                                                                             coords={'time': all_times})

                if self.data["constraints"]["pfCircuits"] is None and nfcoil > 0:
                    nan_array_2d = np.zeros((len(all_times), nfcoil+necurrt))
                    nan_array_2d[:] = np.nan
                    id_array_pf = np.zeros(nfcoil+necurrt)
                    timeslicesource = np.zeros(nfcoil+necurrt)

                    if n_pf == 0:
                        id_array_pf[:] = np.arange(nfcoil+necurrt) + 1
                        timeslicesource[:] = 2
                    else:
                        # Set coils 0:n_pf as active PF circuits, n_pf: as passive PF circuits
                        id_array_pf[0:n_pf] = np.arange(n_pf) + 1
                        id_array_pf[n_pf:] = np.arange(nfcoil+necurrt-n_pf) + 1

                        timeslicesource[0:n_pf] = 2
                        timeslicesource[n_pf:] = 3

                    self.data["constraints"]["pfCircuits"] = xr.Dataset({'computed': (['time', 'pfCircuitsDim'], np.copy(nan_array_2d)),
                                                                         'target': (['time', 'pfCircuitsDim'], np.copy(nan_array_2d)),
                                                                         'timeSliceSource':(['pfCircuitsDim'], timeslicesource),
                                                                         'id': (['pfCircuitsDim'], np.copy(id_array_pf))},
                                                                          coords={'time': all_times})

                if nsilop > 0:
                    self.data["constraints"]["fluxLoops"].computed.values[ind_time, :] = 2 * np.pi * aeqdsk_data.data['csilop']
                if nmagpr > 0:
                    self.data["constraints"]["magneticProbes"].computed.values[ind_time, :] = aeqdsk_data.data['cmpr2']
                if necurrt > 0:
                    self.data["constraints"]["pfCircuits"].computed.values[ind_time, :necurrt] = aeqdsk_data.data['eccurt']
                if nfcoil > 0:
                    self.data["constraints"]["pfCircuits"].computed.values[ind_time, necurrt:] = aeqdsk_data.data['ccbrsp']


    def calc_chi2(self):
        for constraint in self.data["constraints"].keys():
            if self.data["constraints"][constraint] is None:
                continue

            if constraint != "boundaries":
                self.data["constraints"][constraint] = chi2calc(self.data["constraints"][constraint])
            else:
                self.data["constraints"][constraint] = chi2calc(self.data["constraints"][constraint], computedname='computedSep', targetname='rCoords', sigmaname='rSigmas')

    def calc_errors(self):
        for constraint in self.data["constraints"].keys():
            if self.data["constraints"][constraint] is None:
                continue

            if constraint != "boundaries":
                self.data["constraints"][constraint] = errorsCalc(self.data["constraints"][constraint])
            else:
                self.data["constraints"][constraint] = errorsCalc(self.data["constraints"][constraint], computedname='computedSep', targetname='rCoords')

    def calc_bfield(self):
        if self.data["profiles2D"] is None:		
            return

        # Assuming grid doesn't change with time
        Refit = self.data["profiles2D"].r.values[0, :]
        Zefit = self.data["profiles2D"].z.values[0, :]

        # Calculate psiNorm
        psi = np.transpose(self.data["profiles2D"].poloidalFlux.values,
                           axes=[0, 2, 1])

        psiNorm = ((self.data["profiles2D"].poloidalFlux - self.data["globalParameters"].psiAxis)
               / (self.data["globalParameters"].psiBoundary - self.data["globalParameters"].psiAxis))
        if (psiNorm.values.any() < 0):
            if self.debug > 0:
                print("<<WARNING>> Negative values for psinorm!!!")
            np.place(psiNorm.values, psiNorm.values < 0, 0.0)

        psiNorm.values = (psiNorm - psiNorm.min(axis=(2,1))).values

        # Calculate gradients
        dR = Refit[1] - Refit[0]
        dZ = Zefit[1] - Zefit[0]
        try:
            psiDz, psiDr = np.gradient(psi, dZ, dR, edge_order=2, axis=(1,2))
        except ValueError:
            psiDz, psiDr = np.gradient(psi, dZ, dR, axis=(1, 2))

        # Bz, Br 2D
        Bz2d = 1.0 * psiDr / Refit
        Br2d = -1.0 * psiDz / Refit

        self.data["profiles2D"] = self.data["profiles2D"].assign(psiNorm = (['time', 'rGrid', 'zGrid'], psiNorm))
        self.data["profiles2D"] = self.data["profiles2D"].assign(Bz = (['time', 'rGrid', 'zGrid'], Bz2d))
        self.data["profiles2D"] = self.data["profiles2D"].assign(Br = (['time', 'rGrid', 'zGrid'], Br2d))

        # B poloidal
        ip_sign = 1
        if np.sum(np.isnan(self.data["globalParameters"].plasmaCurrent.values)) != len(self.data["globalParameters"].plasmaCurrent.values):
            ip_sign = np.sign(self.data["globalParameters"].plasmaCurrent)

        Bpol2d =  ip_sign * (self.data["profiles2D"].Bz ** 2 + self.data["profiles2D"].Br ** 2) ** 0.5

        self.data["profiles2D"] = self.data["profiles2D"].assign(Bpol = Bpol2d)


    def save_eqdsk(self, time_index, savedir, chease=False, rCentr=1.0, limloc=None, snd_psin_thresh=0.03, afile=True):
        """
        Save A- and G-eqdsk files
        A-eqdsk writer in particular still under development.
        :param time_index: Time index to save
        :param savedir: Directory in which to save files
        :param chease: G-eqdsk compatible with Chease (ie. rCentr = rGeom)
        :param rCentr: Reference R-value for toroidal field
        :param limloc: Boundary classification, IN, OUT, TOP or BOT for limited plasmas, SNT, SNB or DN for single null and double null. If not set will try to determine from x-points
        :param snd_psin_thresh: Psin difference between primary and secondary x-points above which plasma is considered double-null. Has no effect if limloc is set.
        :return:
        """

        if self.data is None:
            if self.debug > 0:
                print("<<ERROR>> EfitData: No data available. Read data first.")
                return

        dirpath=os.path.abspath(savedir)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        pulseNumber = self.data["top"].pulseNumber
        time_value = self.data["top"].time.values[time_index]
        geqdsk_filename = "g_p{}_t{:.5f}".format(pulseNumber, time_value)
        aeqdsk_filename = "a_p{}_t{:.5f}".format(pulseNumber, time_value)

        # Create or update index.dat file
        idexline =  'time={:.5f}   filename={}'.format(time_value, geqdsk_filename)
        if afile:
            idexline += '   afile={}'.format(aeqdsk_filename)
        idexline += '\n'
            
        if time_index == 0:
            idexfile=open(dirpath+'/index.dat','w')
            idexfile.write('ntimes=1\n')
            idexfile.write('path="{}"\n'.format(dirpath))
        else:
            idexfile=open(dirpath+'/index.dat','a')

        idexfile.write(idexline)

        # Close index file for eqdsk
        idexfile.close()

        if self.debug > 0:
            print('----------------------------------------------------------')
            print('Writing GEQDSK file')
            print('----------------------------------------------------------')

        with open("{}/{}".format(dirpath, geqdsk_filename), 'w') as fp:
            title='EFIT++   '+str(time.strftime("%x"))+'#'+str(pulseNumber)+'-'+str(int(time_value*1000))+'ms'

            # First line: comment, nr, nz
            print("%40s%4s%8s%4d%4d"%(title,'    ',0,
                                      self.data["regularGrid"].nr.values[0],
                                      self.data["regularGrid"].nz.values[0]),
                                      file=fp)

            # "DIII-D convention" : psi must be increasing, EFIT++ output is decreasing for positive ip
            flip_psi_factor = 1.0
            if self.data["constraints"]["plasmaCurrent"].target.values[time_index] > 0.0:
                flip_psi_factor = -1.0

            rMax = self.data["regularGrid"].rMax.values[0]
            rMin = self.data["regularGrid"].rMin.values[0]
            zMax = self.data["regularGrid"].zMax.values[0]
            zMin = self.data["regularGrid"].zMin.values[0]
            rGeom = self.data["separatrixGeometry"].rGeom.values[time_index]

            if chease:
                rCentr = rGeom  # Chease wants the geometric axis

            magneticAxis = self.data["globalParameters"].magneticAxis.values[time_index]

            # Width grid r, width grid z, rCentr,
            print("% .9e% .9e% .9e% .9e% .9e"% ( rMax-rMin, zMax-zMin,
                                                 rCentr, rMin, 0.5*(zMax+zMin)), file=fp)
            print("% .9e% .9e% .9e% .9e% .9e"% ( magneticAxis[0], magneticAxis[1],
                                                 flip_psi_factor * self.data["globalParameters"].psiAxis.fillna(0.0).values[time_index],
                                                 flip_psi_factor *  self.data["globalParameters"].psiBoundary.fillna(0.0).values[time_index],
                                                 rGeom * self.data["globalParameters"].bvacRgeom.fillna(0.0).values[time_index] / rCentr),
                                               file=fp)
            print("% .9e% .9e% .9e% .9e% .9e"% ( self.data["globalParameters"].plasmaCurrent.fillna(0.0).values[time_index],
                                                 flip_psi_factor * self.data["globalParameters"].psiAxis.fillna(0.0).values[time_index],
                                                 0.0,  magneticAxis[0], 0.0 ), file=fp)
            print("% .9e% .9e% .9e% .9e% .9e"% ( magneticAxis[1], 0.0,
                                                 flip_psi_factor * self.data["globalParameters"].psiBoundary.fillna(0.0).values[time_index],
                                                 0.0, 0.0), file=fp)
            writeColumns(fp,self.data["fluxFunctionProfiles"].rBphi.fillna(0.0).values[time_index][:],5,"% .9e")
            writeColumns(fp,self.data["fluxFunctionProfiles"].staticPressure.fillna(0.0).values[time_index][:],5,"% .9e")
            writeColumns(fp, flip_psi_factor * self.data["fluxFunctionProfiles"].ffPrime.fillna(0.0).values[time_index][:], 5, "% .9e")
            writeColumns(fp, flip_psi_factor * self.data["fluxFunctionProfiles"].staticPPrime.fillna(0.0).values[time_index][:], 5, "% .9e")

            psiefit_t=flip_psi_factor * np.transpose(self.data["profiles2D"].poloidalFlux.fillna(0.0).values[time_index][:])
            writeColumns(fp,psiefit_t,5,"% .9e")
            writeColumns(fp, np.absolute(self.data["fluxFunctionProfiles"].q.fillna(0.0)).values[time_index][:], 5, "% .9e")

            rlimsize = self.data["geometry"]["limiter"].rValues.size
            print("%5d%5d"% ( self.data["separatrixGeometry"].boundaryCoordsDim.size,
                              rlimsize ), file=fp)

            rBoundary = self.data["separatrixGeometry"].rBoundary.fillna(0.0).values[time_index][:]
            zBoundary = self.data["separatrixGeometry"].zBoundary.fillna(0.0).values[time_index][:]
            boundaryCoordsDim = self.data["separatrixGeometry"].zBoundary.boundaryCoordsDim.size
            # boundaryCoords.resize(boundaryCoordsDim,1)
            i=0
            bc = np.array( np.zeros(2*boundaryCoordsDim) )
            for v0, v1 in zip(rBoundary, zBoundary):
                 (bc[i], bc[i+1]) = (v0, v1)
                 i += 2
            writeColumns(fp, bc,5,"% .9e")
            i=0
            lm = np.array( np.zeros(2*rlimsize) )
            for v in range(rlimsize):
                 lm[i] = self.data["geometry"]["limiter"].rValues.fillna(0.0).values[v]
                 lm[i+1] = self.data["geometry"]["limiter"].zValues.fillna(0.0).values[v]
                 i += 2
            writeColumns(fp,lm,5,"% .9e")

        # Update ntimes
        if time_index > 0:
            with open('{}/index.dat'.format(dirpath), 'r') as f_in:
                all_lines = f_in.readlines()
            new_n_times = int(all_lines[0][all_lines[0].find('=')+1:].replace('\n', '')) + 1
            all_lines[0] = "ntimes={}\n".format(new_n_times)
        
            with open('{}/index.dat'.format(dirpath), 'w') as f_out:
                f_out.write("".join(all_lines))

        if not afile:
            return

        if self.debug > 0:
            print('----------------------------------------------------------')
            print('Writing AEQDSK file')
            print('----------------------------------------------------------')

        # Fortran formatters
        ff_str2 = ff.FortranRecordWriter('(1x,i5,11x,i5)')
        if pulseNumber > 99999:
            ff_str2 = ff.FortranRecordWriter('(1x,i6,11x,i5)')
        ff_str3 = ff.FortranRecordWriter('(1x,4e16.9)')
        ff_str4 = ff.FortranRecordWriter('(a1,f7.2,10x,i5,11x,i5,1x,a3,1x,i3,1x,i3,1x,a3)')

        if limloc is None:
            get_boundary_classification(self, time_index, snd_psin_thresh=snd_psin_thresh)

        mco2v = 0  # Number of vertical CO2 density chords
        mco2r = 0  # Number of radial CO2 density chords
        qmflag = 'CLC'  # Axial q(0) flag, FIX if constrainted and CLC for float
        if self.data["constraints"]["q0"].weights.max() > 0:
            qmflag = 'FIX'
        rcencm = rCentr
        qstar = 0.0  # EFM used to have this... But not in EFIT output
        otop = 0.0
        obott = 0.0
        vertn = 0.0
        shearb = 0.0
        bpolav = 0.0
        olefs = 0.0
        orighs = 0.0
        otops = 0.0
        psiref = 0.0
        xndnt = 0.0
        sepexp = 0.0
        obots = 0.0
        seplim = 0.0
        taumhd = 0.0
        betatd = 0.0  # We could calculate this...
        wplasmd = 0.0  # We could calculate this...
        vloopt = 0.0  # We could calculate this...
        taudia = 0.0
        qmerci = 0.0
        tavem = 0.0
        nesum0 = 0.0

        with open("{}/{}".format(dirpath, aeqdsk_filename), 'w') as fp:
            title='EFIT++   '+str(time.strftime("%x"))+'#'+str(pulseNumber)+'-'+str(int(time_value*1000))+'ms'
            print("%40s"%(title), file=fp)
            fp.write(ff_str2.write([pulseNumber, 1])+'\n')

            # jflag 0 for error, lflag > 0 for error
            if self.data["top"].equilibriumStatusInteger.values[time_index] < 0:
                jflag = 0 # Zero for error
                lflag = np.absolute(self.data["top"].equilibriumStatusInteger.values[time_index]) # > 0 for error
                terror = np.absolute(self.data["top"].equilibriumStatusInteger.values[time_index])
            else:
                jflag = self.data["top"].equilibriumStatusInteger.values[time_index]
                lflag = 0
                terror = 0

            # *,time,jflag,lflag,limloc,mco2v,mco2r,qmflag
            fp.write(ff_str4.write(['*', int(time_value*1000), jflag, lflag, limloc, mco2v, mco2r, qmflag])+'\n')
            pfCircuits = self.data["constraints"]["pfCircuits"]
            active_circuits = pfCircuits.where(pfCircuits.timeSliceSource < 3, drop=True)
            chi2_magnetics = (chi2totalcalc(self.data["constraints"]["magneticProbes"], norm=True, weighted=True)
                              +chi2totalcalc(self.data["constraints"]["fluxLoops"], norm=True, weighted=True)
                              +chi2totalcalc(active_circuits, norm=True, weighted=True))
            # tsaisq,rcencm,bcentr,pasmat
            bcentr = rcencm * self.data["globalParameters"].bvacRgeom.fillna(0.0).values[time_index] / rGeom
            fp.write(ff_str3.write([chi2_magnetics.values[time_index],
                                    rcencm, bcentr,
                                    self.data["constraints"]["plasmaCurrent"].target.values[time_index]])+'\n')
            # cpasma,rout,zout,aout
            rgeom = self.data["separatrixGeometry"].rGeom.values[time_index] * 100.0
            zgeom = self.data["separatrixGeometry"].zGeom.values[time_index] * 100.0

            if np.isclose(rgeom, 0.0):
                rgeom = 0.0
                zgeom = 0.0

            fp.write(ff_str3.write([self.data["constraints"]["plasmaCurrent"].computed.values[time_index],
                                    rgeom, zgeom,
                                    100.0 * self.data["separatrixGeometry"].minorRadius.fillna(0.0).values[time_index]])+'\n')
            # eout,doutu,doutl,vout
            fp.write(ff_str3.write([self.data["separatrixGeometry"].elongation.fillna(0.0).values[time_index],
                                    self.data["separatrixGeometry"].upperTriangularity.fillna(0.0).values[time_index],
                                    self.data["separatrixGeometry"].lowerTriangularity.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].plasmaVolume.fillna(0.0).values[time_index]])+'\n')
            # rcurrt,zcurrt,qsta,betat
            fp.write(ff_str3.write([self.data["globalParameters"].currentCentroid.values[time_index][0] * 100.0,
                                    self.data["globalParameters"].currentCentroid.values[time_index][1] * 100.0,
                                    qstar, self.data["globalParameters"].betat.fillna(0.0).values[time_index] * 100.0])+'\n')

            if np.min(self.data["separatrixGeometry"].rBoundary.values[0]) > 0.1:
                plasma_inner_gap = np.nanmin(self.data["separatrixGeometry"].rBoundary.values[time_index][:]
                                         - np.min(self.data["geometry"]["limiter"].rValues.values)) * 100.0
                plasma_outer_gap = np.nanmax(np.max(self.data["geometry"]["limiter"].rValues.values)
                                        - self.data["separatrixGeometry"].rBoundary.values[time_index][:]) * 100.0
            else:
                plasma_inner_gap = 0.0
                plasma_outer_gap = 0.0

            # betap,ali,oleft,oright
            fp.write(ff_str3.write([self.data["globalParameters"].betap.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].li.fillna(0.0).values[time_index],
                                    plasma_inner_gap, plasma_outer_gap])+'\n')
            # otop,obott,qpsi95,vertn
            fp.write(ff_str3.write([otop, obott,
                                    self.data["globalParameters"].q95.fillna(0.0).values[time_index], vertn])+'\n')

            # Do we need dummy values for these where they don't exist?
            # rco2v
            # dco2v
            # rco2r
            # dco2r

            # shearb,bpolav,s1,s2
            fp.write(ff_str3.write([shearb, bpolav,
                                    self.data["globalParameters"].s1.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].s2.fillna(0.0).values[time_index]])+'\n')
            # s3,qout,olefs,orighs
            ind_bound, = np.where(np.isclose(self.data["fluxFunctionProfiles"].normalizedPoloidalFlux.values, 1.0) == True)
            if len(ind_bound) > 0:
                qout = self.data["fluxFunctionProfiles"].q.fillna(0.0).values[time_index, ind_bound]
            else:
                qout = 0.0
            fp.write(ff_str3.write([self.data["globalParameters"].s3.fillna(0.0).values[time_index],
                                    qout, olefs, orighs])+'\n')
            # otops,sibdry,areao,wplasm
            fp.write(ff_str3.write([otops, self.data["globalParameters"].psiBoundary.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].poloidalArea.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].plasmaEnergy.fillna(0.0).values[time_index]])+'\n')
            # terror,elongm,qqmagx,cdflux
            ind_mag, = np.where(np.isclose(self.data["fluxFunctionProfiles"].normalizedPoloidalFlux.values, 0.0))
            if len(ind_mag) > 0:
                elongm = self.data["fluxFunctionProfiles"].elongation.fillna(0.0).values[time_index, ind_mag]
            else:
                elongm = 0
            fp.write(ff_str3.write([terror, elongm,
                                    self.data["globalParameters"].q0.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].diamagneticFlux.fillna(0.0).values[time_index]])+'\n')
            # alpha,rttt,psiref,xndnt
            fp.write(ff_str3.write([self.data["globalParameters"].alpha.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].rt.fillna(0.0).values[time_index],
                                    psiref, xndnt])+'\n')
            # rseps[0],zseps[0],rseps[1],zseps[1]
            xp1_r = 0.0
            xp2_r = 0.0
            xp1_z = 0.0
            xp2_z = 0.0
            try:
                n_xpoint = self.data["separatrixGeometry"].dndXpointCount.values[time_index]
                if n_xpoint > 0:
                    xp1_r = self.data["separatrixGeometry"].dndXpoint1R.values[time_index]
                    xp1_z = self.data["separatrixGeometry"].dndXpoint1Z.values[time_index]
                if n_xpoint > 1:
                    xp2_r = self.data["separatrixGeometry"].dndXpoint2R.values[time_index]
                    xp2_z = self.data["separatrixGeometry"].dndXpoint2Z.values[time_index]
            except AttributeError:
                n_xpoint = self.data["separatrixGeometry"].xpointCount.values[time_index]
                if n_xpoint > 0:
                    xp1_r = self.data["separatrixGeometry"].xpointR.values[time_index, 0]
                    xp1_z = self.data["separatrixGeometry"].xpointZ.values[time_index, 0]
                if n_xpoint > 1:
                    xp2_r = self.data["separatrixGeometry"].xpointR.values[time_index, 1]
                    xp2_z = self.data["separatrixGeometry"].xpointZ.values[time_index, 1]
            fp.write(ff_str3.write([xp1_r, xp1_z, xp2_r, xp2_z])+'\n')

            # sepexp,obots,btaxp,btaxv
            fp.write(ff_str3.write([sepexp, obots,
                                    self.data["globalParameters"].bphiRmag.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].bphiRmag.fillna(0.0).values[time_index]])+'\n')
            # aaq1,aaq2,aaq3,seplim
            fp.write(ff_str3.write([self.data["globalParameters"].q1Radius.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].q2Radius.fillna(0.0).values[time_index],
                                    self.data["globalParameters"].q3Radius.fillna(0.0).values[time_index],
                                   seplim])+'\n')
            # rmagx,zmagx,simagx,taumhd
            fp.write(ff_str3.write([self.data["globalParameters"].magneticAxis.values[time_index][0],
                                    self.data["globalParameters"].magneticAxis.values[time_index][1],
                                    self.data["globalParameters"].psiAxis.fillna(0.0).values[time_index], taumhd])+'\n')
            # betapd,betatd,wplasmd,fluxx
            fp.write(ff_str3.write([self.data["globalParameters"].betapd.fillna(0.0).values[time_index],
                                   betatd, wplasmd,
                                    self.data["globalParameters"].diamagneticFlux.fillna(0.0).values[time_index]])+'\n')
            # vloopt,taudia,qmerci,tavem
            fp.write(ff_str3.write([vloopt, taudia, qmerci, tavem])+'\n')
            # nsilop0, magpri0, nfcoilk0, nesum0
            nsilop = len(self.data["constraints"]["fluxLoops"].fluxLoopDim)
            nmagpr = len(self.data["constraints"]["magneticProbes"].magneticProbeDim)
            nfcoil = len(self.data["constraints"]["pfCircuits"].pfCircuitsDim)
            fp.write(ff_str3.write([nsilop, nmagpr, nfcoil, nesum0])+'\n')
            # csilop(*),cmpr2(*)
            fl_mp_computed = np.hstack((self.data["constraints"]["fluxLoops"].computed.values[time_index, :],
                                        self.data["constraints"]["magneticProbes"].computed.values[time_index, :]))
            for pind in range(int(np.ceil((nsilop + nmagpr)/4.0))):
                fp.write(ff_str3.write(fl_mp_computed[pind*4:(pind+1)*4])+'\n')

            # ccbrsp(*)
            for pind in range(int(np.ceil(nfcoil/4.0))):
                fp.write(ff_str3.write(self.data["constraints"]["pfCircuits"].computed.values[time_index,pind*4:(pind+1)*4])+'\n')
            # eccurt(*)

            # pbinj,rvsin,zvsin,rvsout
            # zvsout,vsurfa,wpdot,wbdot
            # slantu,slantl,zuperts,chipre
            # cjor95,pp95,ssep,yyy2
            # xnnc,cprof,oring,cjor0
            # fexpan,qqmin,chigamt,ssi01
            # fexpvs,sepnose,ssi95,rqqmin
            # cjor99,cj1ave,rmidin,rmidout
            # psurfa,xdum,xdum,xdum


