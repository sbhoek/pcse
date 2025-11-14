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
from datetime import datetime, date
import time
import pandas as pd
from calendar import isleap
import warnings

from lmgeo import asciigrid as ag, gridenvelope2d as ge

from pcse.base import WeatherDataContainer, WeatherDataProvider
from pcse.util import reference_ET, angstrom, check_angstromAB
from pcse.exceptions import PCSEError
from pcse.settings import settings
from pickle import NONE

# Conversion functions
no_conv = lambda x: x
W_to_J_per_day = lambda x: x*86400
Kelvin2Celsius = lambda x: x-273.16
kJ_to_J = lambda x: x*1000.
kPa_to_hPa = lambda x: x*10.
mm_to_cm = lambda x: x/10.
Kg_M2_Sec_to_cm_day = lambda x: 86400*x/10. 

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
    __date_pattern = "20??0101-20??1231"

    # Dictionaries, lists etc. to store relevant stuff
    __file_pattern = ""
    __fp_txt_fname = ""
    NetcdfDatasets = {} 
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
            with open(self.__fp_txt_fname, 'w') as f:
                for fname in nc_files:
                    f.write(os.path.basename(fname) + "\n")
        
        # Get the relevant dates
        datestrings = self._get_date_strings(nc_files[0], fn_pattern)
        cache_fname = os.path.join(fn_pattern[1:] + "-".join(datestrings) + ".cache")
        if force_reload or not self._load_cache_file(cache_fname):
            # No cache file found, so start loading. Of the required files, find out which are available 
            self.NetcdfDatasets, self.available_years = self.__open_nc_files(nc_files)
                
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

    def _get_date_strings(self, nc_file, fn_pattern):
        result  = []
        nc_stem = os.path.splitext(os.path.basename(nc_file))[0]
        pos = str.find(nc_stem, "_")
        if pos != -1:
            datepart = nc_stem[pos + len(fn_pattern):]
            result = datepart.split("-")
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
            result = elevation_grid.get_value(i, k)
            elevation_grid.close()
        finally:
            return result

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
        # Find all Netcdf files on given path with given pattern
        # Also sorts the list, checks for missing years and sets self.lastyear
        # Assume this pattern: [variable_name]_[fn_pattern]_[20??]0101-20??1231.[nc]
        if not search_path.endswith(os.path.sep): search_path += os.path.sep
        result = sorted(glob.glob(search_path + "*" + self.__file_pattern + self.__date_pattern + ".nc"))
        if len(result) == 0:
            raise PCSEError("No Netcdf4 files found when searching at %s" % search_path)
        return result

    def __open_nc_files(self, nc_files):
        # Loop over the files in the list nc_files, open them and keep a refrence
        key_ds_dict = {}
        for ncfile in nc_files:
            # Check that we need this ncfile in the first place
            fn = os.path.basename(ncfile)
            pos = fn.find('_'); # it is assumed that the first part indicates the variable
            varname = fn[0:pos]
            if (varname in self.netcdf_variables):
                # Ok, the ncfile contains data wrt. one of the relevant variables
                try:
                    ncds = xr.open_dataset(
                        ncfile, 
                        engine="h5netcdf", 
                        decode_cf=True,       # CF-compliant decoding (default True)
                        mask_and_scale=True,  # Apply _FillValue and scale factors
                        decode_times=True,    # Convert time units to datetime objects
                        chunks={"time": 365}
                    )
                except Exception as e:
                    print(e)
                    raise PCSEError("An error occurred while opening file " + fn)
                key_ds_dict[varname] = ncds
        
        # Assume that we can retrieve the first and last year from the variable ncfile 
        timestr = ncfile[-20:-3]
        dt_0, dt_n = timestr.split("-")
        yr_0, yr_n = datetime.strptime(dt_0, "%Y%m%d").year, datetime.strptime(dt_n, "%Y%m%d").year 
        self.available_years = list(range(yr_0, yr_n + 1))
        return key_ds_dict, self.available_years

    def __get_lon_values(self, ds):
        return ds.lon.values
    
    def __get_lat_values(self, ds):
        return ds.lat.values

    def __read_attributes(self):
        result = False
        try:
            # Get the name and instance of the first xarray DataSet
            varname = next(iter(self.NetcdfDatasets.keys()))
            firstds = self.NetcdfDatasets[varname]
            
            # Get the attributes wrt. the georeference
            shp = firstds.data_vars[varname].shape
            self.ncols = shp[2]
            self.nrows = shp[1]
            
            # Assume they have used a so-called rotated pole
            infomsg = "Problem: the NetCDF file does not contain a rotated pole as was assumed"
            assert "rotated_pole" in firstds.variables, infomsg
            lon_values = self.__get_lon_values(firstds)
            lat_values = self.__get_lat_values(firstds)
            self.xll = lon_values[0, 0]
            self.yll = lat_values[0, 0]
            self._dx = lon_values[0, 1] - self.xll
            self._dy = lat_values[1, 0] - self.yll
            self.NODATA_value = np.nan
            
            # Get the attributes wrt. temporal reference
            # Suppress the warnings generated by the call to method to_datetimeindex
            curlineno = _getframe().f_lineno
            warnings.simplefilter('ignore', lineno=curlineno+2)
            datetimeindex = firstds.indexes['time'].to_datetimeindex()
            self._firstyear = datetimeindex[0].date().year
            self._lastyear  = datetimeindex[-1].date().year
            self._timesteps = len(datetimeindex)
            result = True
        finally:
            return result
    
    # Todo: adapt !!!   
    def _process_netcdf(self, search_path):
        if len(self.NetcdfDatasets) == len(self.netcdf_variables):
            # Prepare to time the data extraction
            t1 = time.time()
            
            # Find out more about the target point
            varname = next(iter(self.NetcdfDatasets.keys()))
            firstds = self.NetcdfDatasets[varname]
            lon_vals = self.__get_lon_values(firstds)
            lat_vals = self.__get_lat_values(firstds)
            dist = np.sqrt((lon_vals - self.longitude)**2 + (lat_vals - self.latitude)**2)
            i, k = np.unravel_index(np.argmin(dist), dist.shape)

            # Loop over the variables
            df = pd.DataFrame()
            for j, varname in enumerate(self.netcdf_variables):
                # Get the DataArray relevant for the variable
                ncds = self.NetcdfDatasets[varname]
                ncda = ncds.data_vars[varname]              
                
                # Retrieve the value of the pixel nearest to the given latitude and longitude
                series = ncda.isel(rlat=i, rlon=k).to_series()
                assert len(series) == self._timesteps, "Time steps for variable %s are not right!" % varname
                
                # Assign the foudn values to the dataframe
                if j == 0: df = df.assign(time=pd.to_datetime(series.index, format="ISO8601")) 
                df = df.assign(**{str(varname): series.values})

            # Assign some extra attributes - what if value == nodata_value?
            varname = "huss"
            if not search_path.endswith(os.path.sep): search_path += os.path.sep
            tmpnc_files = sorted(glob.glob(search_path + varname + self.__file_pattern + self.__date_pattern +".nc"))
            assert len(tmpnc_files) == 1, "Netcdf4 file not found when searching at %s" % search_path
            try:
                ncds = xr.open_dataset(tmpnc_files[0], engine="h5netcdf")
            except Exception as e:
                print(e)
                raise PCSEError("An error occurred while opening file " + tmpnc_files[0])
            ncda = ncds.data_vars[varname]
            series = ncda.isel(rlat=i, rlon=k).to_series()
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
        result = None
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
                    if not newer_file_found:
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
