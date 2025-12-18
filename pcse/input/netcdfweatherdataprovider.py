# -*- coding: utf-8 -*-
# Copyright (c) 2004-2025 Wageningen Environmental Research, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), Steven Hoek (October 2025)

# A weather data provider reading its data from Netcdf files.
import os
from sys import _getframe
import glob
import xarray as xr
from math import exp, pow
import numpy as np
from datetime import datetime as dt, date
import time
import pandas as pd
from calendar import isleap
from warnings import warn, simplefilter

from lmgeo import raster, asciigrid as ag, gridenvelope2d as ge

from pcse.base import WeatherDataContainer, WeatherDataProvider
from pcse.util import reference_ET, angstrom, check_angstromAB
from pcse.exceptions import PCSEError
from pcse.settings import settings
from typing import List, Tuple

# Conversion functions
no_conv = lambda x: x
W_to_J_per_day = lambda x: x*86400
Kelvin2Celsius = lambda x: x-273.16
kJ_to_J = lambda x: x*1000.
kPa_to_hPa = lambda x: x*10.
mm_to_cm = lambda x: x/10.
Kg_M2_Sec_to_cm_day = lambda x: 86400*x/10. 

DEBUG_LEVEL = 1

def vap_from_sh(sh, alt):
    """ Vapour pressure in hPa, from specific humidity 
    Arguments:
    sh  - specific humidity 
    alt - altitude above sea level
    """
    # Correct pressure for altitude
    p = 101.325 * pow(1 - 2.25577e-5 * alt, 5.2588)
    
    # Specific humidity is the mass of water vapour per mass of air mixture; therefore:
    mr = sh / (1 - sh); # mixing ratio, mass of water vapour per mass of dry air
    vap = p * mr / (0.622 + mr); # constant is based on ratio of mole weights for dry air and water
    return vap

# Saturated Vapour pressure [kPa] at temperature temp [C]
SatVapourPressure = lambda temp: 0.6108 * exp((17.27 * temp) / (237.3 + temp))

class NetcdfWeatherDataProvider(WeatherDataProvider, ge.GridEnvelope2D):
    """Reading weather data from an Netcdf file (.nc only).
    For reading weather data from file, initially only the CABOWeatherDataProvider
    was available that reads its data from a text file in the CABO Weather format.
    Nevertheless, building CABO weather files is tedious as for each year a new
    file must constructed. Moreover it is rather error prone and formatting
    mistakes are easily leading to errors.

    To simplify providing weather data to PCSE models, a new data provider
    was written that reads its data from Netcdf files

    The NetcdfWeatherDataProvider assumes that records are complete and does
    not make an effort to interpolate data. Only SNOW_DEPTH is allowed to be missing
    as this parameter is usually not provided outside the winter season.
    """
    netcdf_variables = ["rsds", "tasmin", "tasmax", "tas", "sfcWind", "pr"]
        
    # Variable names in dataset files - derive vap from "hur" or "hus" at a later stage
    pcse_variables = [("IRRAD", "rsds", W_to_J_per_day, "J/m2/day"), 
                      ("TMIN", "tasmin", Kelvin2Celsius, "Celsius"),
                      ("TMAX", "tasmax", Kelvin2Celsius, "Celsius"), 
                      ("TEMP", "tas", Kelvin2Celsius, "Celsius"),
                      ("WIND", "sfcWind", no_conv, "m/sec"),
                      ("RAIN", "pr", Kg_M2_Sec_to_cm_day, "mm/day")]
    
    # other constants
    angstA = 0.25
    angstB = 0.5
    _ETmodel = "PM"
    _missing_snow_depth = None
    __date_patterns = ["19??0101-19??1231", "19??0101-20??1231", "20??0101-20??1231"]

    # Dictionaries, lists etc. to store relevant stuff
    __file_pattern = ""
    __fp_txt_fname = ""
    NetcdfDataset = None
    available_years = []

    # Latitude, longitude and elevation - initialise for Tema, Ghana
    _lat = 5.65
    _lon = 0.0
    elevation = 1
    
    # First and last year - just initialise for the zeros
    _firstyear = 2000
    _lastyear = 2010

    def __init__(self, nc_path, fn_pattern, longitude, latitude, missing_snow_depth=None, force_reload=False):
        # Pattern example: "_EUR-11_IPSL-IPSL-CM5A-MR_historical_r1i1p1_KNMI-RACMO22E_v1_day_"
        WeatherDataProvider.__init__(self)

        # Process input
        self._lat = float(latitude)
        self._lon = float(longitude)
        self._missing_snow_depth = missing_snow_depth
        self.__file_pattern = fn_pattern

        # Do some logging
        msg = "About to retrieve weather data from Netcdf files for lat/lon: (%f, %f)."
        self.logger.info(msg % (self._lat, self._lon))

        # Construct search path, search the files and store the names in a text file
        search_path = self._construct_search_path(nc_path)
        nc_files = self.__scan_nc_files(search_path)
        self.__fp_txt_fname = os.path.join(search_path, "vars" + fn_pattern + "txt")
        if not os.path.exists(self.__fp_txt_fname):
            # Create the text file anew
            self.__write_filenames(nc_files)
        else:
            # Check at least whether the number of files is still the same
            tmplist = self.__read_filenames()
            if len(tmplist) != len(nc_files):
                # Otherwise update the file!
                self.__write_filenames(nc_files)    

        # Get the relevant dates
        datestrings = self._get_date_strings(nc_files, fn_pattern)
        cache_fname = os.path.join(fn_pattern[1:] + "-".join(datestrings) + ".cache")
        if force_reload or not self._load_cache_file(cache_fname):
            # No cache file found, so start loading. Of the required files, find out which are available 
            self.NetcdfDataset, self.available_years = self.__open_nc_files(nc_files)
                
            # Read attributes, assign them to the envelope and calculate the nearest point
            self.__read_attributes()
            ge.GridEnvelope2D.__init__(self, self.ncols, self.nrows, self.xll, self.yll, self._dx, self._dy)
            self.longitude, self.latitude = self.getNearestCenterPoint(longitude, latitude)
            
            # Retrieve the records for this location and store them
            self.elevation = self._get_elevation(search_path, longitude, latitude)
            self._process_netcdf(search_path)
            
            # Make sure a cache file is available next time
            cache_filename = self._get_cache_filename(cache_fname)
            self._dump(cache_filename)
    
    def __write_filenames(self, nc_files):
        with open(self.__fp_txt_fname, 'w') as f:
            f.write(str(self._lon) + "," + str(self._lat) + "\n")
            for fname in nc_files:
                f.write(os.path.basename(fname) + "\n")
    
    def __read_filenames(self):
        # Initialise
        result = []
        
        # Loop over the lines in the file object
        with open(self.__fp_txt_fname, 'r') as f:
            for line in f:
                fname = line.strip('\n')
                result.append(fname)
                
        # Return the names but not the first line
        return result[1:]

    def _get_date_strings(self, nc_files, fn_pattern) -> str:
        # Initialise
        start_dates, end_dates = [], []
        result = "19800101-20251231"
        
        try:
            # Loop over the filenames and find dates
            for fname in nc_files:
                nc_stem = os.path.splitext(os.path.basename(fname))[0]
                pos = str.find(nc_stem, "_")
                if pos != -1:
                    start_date, end_date = nc_stem[pos + len(fn_pattern):].split("-")
                    if not start_date in start_dates: start_dates.append(start_date)
                    if not end_date in end_dates: end_dates.append(end_date)
                    
            # Get the earliest start date and the latest end date
            if len(start_dates) == 0 or len(end_dates) == 0:
                errmsg = "No Netcdf4 files found with dates in the name in folder %s"
                raise PCSEError(errmsg % os.path.dirname(nc_files[0]))
            start_dates = [dt.strptime(ds, '%Y%m%d').date() for ds in start_dates]
            end_dates = [dt.strptime(ds, '%Y%m%d').date() for ds in end_dates]
            result = (min(start_dates).strftime("%Y%m%d"), max(end_dates).strftime("%Y%m%d"))
        except Exception as e:
            warn(e)
            warn("Unable to derive start and end date from the file names!")
        finally:
            return result
    
    def _get_elevation(self, search_path, longitude, latitude):
        # Open the elevation grid - it probably has a different extent than the NetCDF rasters
        result = None
        try:
            # Assume a folder "geodata" with the same parent as the folder with meteo data
            path2geodata = os.path.dirname(os.path.dirname(search_path))
            path2gridfile = os.path.join(path2geodata, "geodata", "Europe", "elev_pt1_deg_grid.asc")
            elevation_grid = ag.AsciiGrid(path2gridfile)
            assert elevation_grid.open('r'), "Unable to open the elevation grid!"
            lon, lat = elevation_grid.getNearestCenterPoint(longitude, latitude)
            k, i = elevation_grid.getColAndRowIndex(lon, lat)
            k, i = self.check_bounds(k, i, elevation_grid)
            result = elevation_grid.get_value(i, k)
            elevation_grid.close()
        finally:
            return result
        
    def check_bounds(self, k, i, elevation_grid):
        assert isinstance(elevation_grid, raster.Raster), "Given feature class is not a raster!"
        if (k >= elevation_grid.ncols) or (k < 0) or (i >= elevation_grid.nrows) or (i < 0):
            warn("Unable to derive exact elevation from the elevation grid!")
        if k >= elevation_grid.ncols: k = elevation_grid.ncols - 1
        elif k < 0: k = 0
        if i >= elevation_grid.nrows or i < 0: i = elevation_grid.nrows - 1
        elif i < 0: i = 0
        return (k, i)

    def _construct_search_path(self, fpath):
        """Construct the path where to look for files"""
        if fpath is None:
            # assume NC4 files in current folder
            p = os.getcwd()
        elif os.path.isabs(fpath):
            # absolute path specified
            p = fpath
        else:
            # assume path relative to current folder
            p = os.path.join(os.getcwd(), fpath)
        return os.path.normpath(p)
    
    def __scan_nc_files(self, search_path):
        # Initialise
        result = []
        
        # Find all Netcdf files on given path with given pattern
        # Also sorts the list, checks for missing years and sets self.lastyear
        # Assume this pattern: [variable_name]_[fn_pattern]_[????0101-????1231].[nc]
        if not search_path.endswith(os.path.sep): search_path += os.path.sep
        for date_pattern in self.__date_patterns:
            result += glob.glob(search_path + "*" + self.__file_pattern + date_pattern + ".nc")
        result = sorted(result)
        if len(result) == 0:
            raise PCSEError("No Netcdf4 files found when searching at %s" % search_path)
        return result

    def __open_nc_files(self, nc_files) -> xr.Dataset:
        # Loop over the files in the list nc_files
        mfds = None
        tmplist = []
        for nc_file in nc_files:
            # Check that we need this ncfile in the first place
            fn = os.path.basename(nc_file)
            pos = fn.find('_'); # it is assumed that the first part indicates the variable
            varname = fn[0:pos]
            if (varname in self.netcdf_variables):
                # Ok, the ncfile contains data wrt. one of the relevant variables
                tmplist.append(nc_file)
                
            # Assume that we can retrieve the first and last year from the variable nc_file 
            lenext = len(os.path.splitext(fn)[1])
            (dtstr_0, dtstr_n) = nc_file[(-17 - lenext):-1 * lenext].split("-")
            (dt_0, dt_n) = (dt.strptime(dtstr_0, "%Y%m%d"), dt.strptime(dtstr_n, "%Y%m%d"))
            self.available_years = list(range(dt_0.year, dt_n.year + 1))
                
        try:
            mfds = xr.open_mfdataset(
                nc_files, 
                data_vars=None, 
                coords='all', 
                compat='override',
                combine='by_coords' 
            )
        except Exception as e:
            print(e)
            raise PCSEError("An error occurred while opening the Netcdf files.")

        return mfds, self.available_years

    def __get_lon_values(self, ds):
        return ds.lon.values
    
    def __get_lat_values(self, ds):
        return ds.lat.values

    def __read_attributes(self):
        result = False
        try:
            # Get the name and instance of the first xarray DataSet
            keylist = list(self.NetcdfDataset.keys())
            lastds = self.NetcdfDataset[keylist[-1]]
            
            # Assume they have used a so-called rotated pole
            infomsg = "Problem: the NetCDF file does not contain a rotated pole as was assumed"
            assert "rotated_pole" in keylist, infomsg
            
            # Get the attributes wrt. the georeference
            shp = lastds.shape
            assert len(shp) == 3, "Unexpected shape found for DataArray " + lastds.name
            self.ncols = shp[2]
            self.nrows = shp[1]
            lon_values = self.__get_lon_values(lastds)
            lat_values = self.__get_lat_values(lastds)
            self.xll = lon_values[0, 0, 0]
            self.yll = lat_values[0, 0, 0]
            self._dx = lon_values[0, 0, 1] - self.xll
            self._dy = lat_values[0, 1, 0] - self.yll
            self.NODATA_value = np.nan
            
            # Get the attributes wrt. temporal reference
            # Suppress the warnings generated by the call to method to_datetimeindex
            curlineno = _getframe().f_lineno
            simplefilter('ignore', lineno=curlineno+2)
            datetimeindex = lastds.indexes['time'].to_datetimeindex()
            self._firstyear = datetimeindex[0].date().year
            self._lastyear  = datetimeindex[-1].date().year
            self._timesteps = len(datetimeindex)
            result = True
        finally:
            return result
    
    # Todo: adapt !!!   
    def _process_netcdf(self, search_path):
        # At this point netcdf_variables is not yet including huss, rotated pole and time_bnds
        if len(self.NetcdfDataset.data_vars) - 3 == len(self.netcdf_variables):
            # Prepare to time the data extraction
            t1 = time.time()
            
            # Find out more about the target point
            keylist = list(self.NetcdfDataset.keys())
            lastds = self.NetcdfDataset[keylist[-1]]
            lon_vals = self.__get_lon_values(lastds)
            lat_vals = self.__get_lat_values(lastds)
            dist = np.sqrt((lon_vals[0] - self.longitude)**2 + (lat_vals[0] - self.latitude)**2)
            i, k = np.unravel_index(np.argmin(dist), dist.shape)
            if DEBUG_LEVEL > 0:
                print("About to extract values for (%.2f, %.2f)" % (lon_vals[0,i,k], lat_vals[0,i,k])) 

            # Loop over the variables
            df = pd.DataFrame()
            for j, varname in enumerate(self.netcdf_variables):
                # Get the DataArray relevant for the variable
                ncda = self.NetcdfDataset[varname]
                
                # Retrieve the value of the pixel nearest to the given latitude and longitude
                series = ncda[:, i, k].to_series()
                assert len(series) == self._timesteps, "Time steps for variable %s are not right!" % varname
                
                # Assign the found values to the dataframe
                if j == 0: df = df.assign(time=pd.to_datetime(series.index, format="ISO8601")) 
                df = df.assign(**{str(varname): series.values})

            # Assign some extra attributes - what if value == nodata_value?
            varname = "huss"
            tmpnc_files = []
            if not search_path.endswith(os.path.sep): search_path += os.path.sep
            for date_pattern in self.__date_patterns:
                tmplist = sorted(glob.glob(search_path + varname + self.__file_pattern + date_pattern +".nc"))
                tmpnc_files = tmpnc_files + tmplist
            assert len(tmpnc_files) >= 1, "Netcdf4 file not found when searching at %s" % search_path
            
            # With mfdataset, we can add extra datasets by using assign
            try:                  
                xtrds = xr.open_mfdataset(tmpnc_files, 
                    data_vars=None, 
                    coords='all', 
                    compat='override',
                    combine='by_coords' 
                )
                self.NetcdfDataset = xr.combine_by_coords(
                    data_objects = [self.NetcdfDataset, xtrds],
                    data_vars = 'all',
                    compat='override',
                    coords = "all", 
                    join = 'override'
                )
                
            except Exception as e:
                print(e)
                raise PCSEError("An error occurred while opening file " + tmpnc_files[0])

            ncda = self.NetcdfDataset[varname]
            series = ncda[:, i, k].to_series()
            assert len(series) == self._timesteps, "Time steps for variable %s are not right!" % varname
            vap_values = vap_from_sh(series.values, self.elevation)
            df = df.assign(vap=vap_values)

            #self.description = "Meteo data from HDF5 file " + f.filename
            t2 = time.time()

            # Now store them
            self._make_WeatherDataContainers(df)
            t3 = time.time()
            self.logger.debug("Reading Netcdf took %7.4f seconds" % (t2-t1))
            self.logger.debug("Processing rows took %7.4f seconds" % (t3-t2))

    def _make_WeatherDataContainers(self, df):
        # Prepare to loop over all the rows derived from the table
        for i, row in df.iterrows():
            t = {"LAT": self.latitude, "LON": self.longitude, "ELEV": self.elevation}
            t["DAY"] = row["time"].to_pydatetime().date()
            mapping = {v: (k, f, u) for k, v, f, u in (self.pcse_variables + [("VAP", "vap", kPa_to_hPa, "hPa")])}
            varlist = [mapping[var] for var in list(row.index)[1:]]
            values = row.values[1:]
            for i, (ucname, conversion, unit) in enumerate(varlist):
                if conversion is not None: t[ucname] = conversion(values[i])
                else: t[ucname] = values[i]

            # Add calculation of E0, ES0 and ET0. Reference ET returns values in mm/day - convert to cm/day
            args = (t["DAY"], t["LAT"], t["ELEV"], t["TMIN"], t["TMAX"], t["IRRAD"], t["VAP"], t["WIND"])
            e0, es0, et0 = reference_ET(*args, ANGSTA=self.angstA, ANGSTB=self.angstB)
            t.update({"E0": e0 / 10., "ES0": es0 / 10., "ET0": et0 / 10.})

            # Build weather data container from dict 't'
            wdc = WeatherDataContainer(**t)

            # add wdc to dictionary for this date
            self._store_WeatherDataContainer(wdc, wdc.DAY)
            
        # Check for leap years whether February 29 exists. If not, copy the data for February 28
        leap_years = [yr for yr in self.available_years if isleap(yr)]
        for yr in leap_years:
            # The data are stored with a tuple as key: (date, 0)
            if not (date(yr, 2, 29),0) in self.store:
                # Copy the data for the 28th to the 29th
                data = self.store[(date(yr, 2, 28), 0)]
                self.store[(date(yr, 2, 29), 0)] = data

    def _load_cache_file(self, cache_fname):
        cache_filename = self._find_cache_file(cache_fname)
        if cache_filename is None:
            return False
        else:
            try:
                self._load(cache_filename)
                return True
            except:
                return False

    def _find_cache_file(self, cache_fname):
        """Try to find a cache file for file name
        Returns None if the cache file does not exist, else it returns the full path
        to the cache file.
        """
        # Initialise
        result = None
        eps = 0.001
        
        # Get hold of the cache file now
        cache_filename = self._get_cache_filename(cache_fname)
        if os.path.exists(cache_filename):
            cache_date = os.stat(cache_filename).st_mtime
        else:
            return result

        try:
            if os.path.exists(self.__fp_txt_fname):
                search_path = os.path.dirname(self.__fp_txt_fname)
                with open(self.__fp_txt_fname, 'r') as f:
                    newer_file_found = False
                    other_coords_found = False
                    
                    # Get the first line
                    firstline = f.readline().strip('\n')
                    latlon = [float(coord) for coord in firstline.split(",")]
                    if (abs(latlon[0] - self._lon) > eps) or (abs(latlon[1] - self._lat) > eps):
                        other_coords_found = True
                    else:
                        # Continue wit the rest of the file    
                        for fname in f:
                            nc_file = os.path.join(search_path, fname.strip())
                            if os.path.exists(nc_file):
                                nc_date = os.stat(nc_file).st_mtime
                                if cache_date < nc_date:
                                    # cache is less recent than NetCDF file
                                    newer_file_found = True
                                    break
                            else:
                                raise IOError("File %s was not found!" % fname)
                    
                    # Okay, we looped over all the NetCDF files
                    if (not newer_file_found) or (not other_coords_found):
                        result = cache_filename
        finally:
            return result

    def _get_cache_filename(self, dataset_fname):
        """Constructs the filename used for cache files given dataset_name
        """
        basename = os.path.basename(dataset_fname)
        filename, ext = os.path.splitext(basename)

        tmp = "%s_%s.cache" % (self.__class__.__name__, filename)
        cache_filename = os.path.join(settings.METEO_CACHE_DIR, tmp)
        return cache_filename

    def _write_cache_file(self, dataset_fname):

        cache_filename = self._get_cache_filename(dataset_fname)
        try:
            self._dump(cache_filename)
        except (IOError, EnvironmentError) as e:
            msg = "Failed to write cache to file '%s' due to: %s" % (cache_filename, e)
            self.logger.warning(msg)
