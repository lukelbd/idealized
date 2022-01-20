#!/usr/bin/env python3
"""
Load the experiment data.
"""
import glob
import os

import metpy.interpolate as interp
import numpy as np
import xarray as xr
from climopy import vreg
from icecream import ic  # noqa: F401

from . import _make_stopwatch, _warn_simple, definitions  # noqa: F401

ATTRS_OVERRIDES = {
    'u': {'long_name': 'zonal wind'},
    'v': {'long_name': 'meridional wind'},
}
SEASON_IDXS = {
    'djf': (11, 0, 1),
    'mma': (2, 3, 4),
    'jja': (5, 6, 7),
    'son': (8, 9, 10),
    **{i: (i - 1,) for i in range(1, 13)},  # e.g. 'month 1' is index 0 in month array
}


def combine_cam_kernels(
    path='~/data/kernels-cam5/cam5_plev.nc',
    # ref='~/data/timescales_constants/hs1_t85l60e.nc',
    month=None,
    season=None,
    shortwave=True,
):
    """
    Create averages of

    Parameters
    ----------
    path : path-like, optional
        CAM data location.
    month : month-spec, optional
        Month(s) to be averaged.
    season : season-spec, optional
        Season(s) to be averaged.
    shortwave : bool, optional
        Whether to include shortwave components.
    """
    # Load kernel data, fixing sign convention so both LW and SW are +ve toward
    # atmosphere (default is SW positive down, LW positive up, understandably).
    # Also get cloudy-sky and net atmospheric components.
    # NOTE: See load.py:load_cam for details
    # NOTE: Xarray caches already-loaded datasets so this is fast after one run
    print('Calculating averages...')
    ds_cam5 = xr.open_dataset(path)
    keep = ('ps', 'bnd', 'fln') + (('fsn',) if shortwave else ())
    drop = tuple(var for var in ds_cam5 if not any(s in var for s in keep))
    ds_cam5 = ds_cam5.drop_vars(drop)  # drop surface kernels, and maybe shortwave
    ds_cam5 = ds_cam5.climo.sel_time(season=season, month=month)
    ds_cam5 = ds_cam5.climo.mean('time')
    ds_cam5['ps'].attrs['standard_name'] = 'surface_air_pressure'
    with xr.set_options(keep_attrs=True):
        for var in ds_cam5:
            if 'flnt' in var or 'fsns' in var:
                ds_cam5[var] *= -1
        for var in ('t_fln', 'q_fln', 'q_fsn', 'ts_fln', 'alb_fsn'):
            for s in ('c', ''):
                if var + 't' + s in ds_cam5 and var + 's' + s in ds_cam5:
                    ds_cam5[var + 'a' + s] = ds_cam5[var + 't' + s] + ds_cam5[var + 's' + s]  # noqa: E501
            for s in ('t', 's', 'a'):
                if var + s + 'c' in ds_cam5 and var + s in ds_cam5:
                    ds_cam5[var + s + 'l'] = ds_cam5[var + s] - ds_cam5[var + s + 'c']
    ds_cam5 = ds_cam5.climo.standardize_coords()
    ds_cam5 = ds_cam5.climo.add_cell_measures(verbose=True)
    ds_cam5 = ds_cam5.climo.quantify()
    return ds_cam5


def load_cam_kernels(
    path='~/data/kernels-cam5/',
    folders=('forcing', 'kernels'),
    ref='~/data/timescales_constants/hs1_t85l60e.nc',
    timer=False,
):
    """
    Load the CAM forcing and feedback kernel data.

    Parameters
    ----------
    base : path-like
        The base directory in which `folders` is indexed.
    ref : path-like
        The path used for destination parameters.
    """
    # Get list of files
    stopwatch = _make_stopwatch(timer=timer)  # noqa: F841
    ps = set()
    files = [
        file
        for folder in folders
        for file in glob.glob(os.path.expanduser(f'{path}/{folder}/*.nc'))
    ]
    files = [  # remove surface pressure file and record it separately
        file for file in files
        if os.path.basename(file) != 'PS.nc' or ps.add(file)  # set.add returns None
    ]
    if not ps:
        raise RuntimeError('Surface pressure data not found.')

    # Load data using PS.nc for time coords (times are messed up on other datasets)
    print('Loading data...')
    ignore = ('gw', 'nlon', 'ntrk', 'ntrm', 'ntrn', 'w_stag', 'wnummax')
    ds_mlev = xr.Dataset()  # will add variables one-by-one
    for file in (*ps, *files):  # put PS.nc first
        data = xr.open_dataset(file, use_cftime=True)
        time_dim = data.time.dims[0]
        if 'ncl' in time_dim:  # fix issue where 'time' coord dim is an NCL dummy name
            data = data.swap_dims({time_dim: 'time'})
        if os.path.basename(file) != 'PS.nc':  # ignore bad time indices
            data = data.reset_index('time', drop=True)
        var, _ = os.path.splitext(os.path.basename(file))
        var, *_ = var.split('.')
        # if 'time' in ds_mlev.data:
        for name, da in data.items():  # iterates through variables, not coordinates
            if name in ignore:
                continue
            name = name.lower()
            if name[:2] in ('fl', 'fs'):  # longwave or shortwave radiative flux
                name = var + '_' + name
            ds_mlev[name] = da

    # Normalize by pressure thickness and standardize units
    print('Normalizing data...')
    pi = ds_mlev.hyai * ds_mlev.p0 + ds_mlev.hybi * ds_mlev.ps
    pi = pi.transpose('time', 'ilev', 'lat', 'lon')
    pm = 0.5 * (pi.data[:, 1:, ...] + pi.data[:, :-1, ...])
    dp = pi.data[:, 1:, ...] - pi.data[:, :-1, ...]
    for name, da in ds_mlev.items():
        if 'lev' in da.dims and da.ndim > 1:
            da.data /= dp
            da.attrs['units'] = 'W m^-2 K^-1 Pa^-1'
        elif name[:2] == 'ts':
            da.attrs['units'] = 'W m^-2 K^-1'
        elif name[:3] == 'alb':
            da.attrs['units'] = 'W m^-2 %^-1'
        elif da.attrs.get('units', None) == 'W/m2':
            da.attrs['units'] = 'W m^-2'
    dims = ('time', 'lev', 'lat', 'lon')
    attrs = {'units': 'Pa', 'long_name': 'approximate pressure thickness'}
    ds_mlev['lev_delta'] = (dims, dp, attrs)
    ds_mlev.lon.attrs['axis'] = 'X'
    ds_mlev.lat.attrs['axis'] = 'Y'
    ds_mlev.lev.attrs['axis'] = 'Z'
    ds_mlev.time.attrs['axis'] = 'T'

    # Interpolate onto standard pressure levels using metpy
    # NOTE: Also normalize by pressure level thickness, giving units W m^-2 100hPa^-1
    # TODO: Are copies of data arrays necessary?
    ds_hs94 = xr.open_dataset(ref, decode_times=False)
    if 'plev' in ds_hs94.coords:
        ds_hs94 = ds_hs94.rename(plev='lev', plev_bnds='lev_bnds')
    ds_plev = xr.Dataset(
        coords={
            'lon': ds_mlev.lon,
            'lat': ds_mlev.lat,
            'lev': ds_hs94.lev,
            'time': ds_mlev.time,
        }
    )
    ds_plev['lev'].attrs['bounds'] = 'lev_bnds'
    ds_plev['lev_bnds'] = ds_hs94.lev_bnds
    for name, da in ds_mlev.items():
        if 'lev' in da.dims and da.ndim > 1 and name != 'lev_delta':
            print(f'Interpolating {name!r}...')
            dims = list(da.dims)
            shape = list(da.shape)
            dims[1] = 'lev'
            shape[1] = ds_plev.lev.size
            data = np.empty(shape)
            for i in range(shape[0]):  # iterate through times
                for j in range(shape[1]):  # iterate through levels
                    data[i, j, ...] = interp.interpolate_to_isosurface(
                        pm[i, ...], da.data[i, ...], 100 * ds_plev.lev.data[j]
                    )
            ds_plev[name] = (dims, data, da.attrs)
        elif not any(dim in da.dims for dim in ('lev', 'ilev')):
            print(f'Copying {name!r}.')
            ds_plev[name] = da

    # Save and return
    try:
        ds_mlev.to_netcdf(f'{path}/cam5_mlev.nc')
        ds_plev.to_netcdf(f'{path}/cam5_plev.nc')
    except Exception:
        _warn_simple('Failed to save datasets. Possible permissions error.')
    return ds_mlev, ds_plev


def load_gfdl_file(
    *files, hemi='ave', lat=None, plev=None, add=None, days=None, timer=False, variables=None,  # noqa: E501
):
    """
    Load and standardize the dataset and apply `pint` units.

    Parameters
    ----------
    *files : str
        The file name(s).
    hemi : {'globe', 'ave', 'nh', 'sh'}
        The region used for combining hemispheric data.
    lat, plev : str, optional
        Selections to reduce size of spectral data.
    add : str, optional
        Add across these dimensions to reduce size of spectral data.
    days : int, optional
        Take averages over this many consecutive days.
    timer : bool, optional
        Time operations.
    variables : list of str, optional
        Use this to filter variables in the returned dataset.
    """
    # Load data from scratch or storage, may raise FileNotFoundError
    # TODO: Add date bounds for sepctral data filenames.
    # NOTE: This replaces yzload and xyload functions, since it searches
    # scratch along with storage
    stopwatch = _make_stopwatch(timer=timer)
    if len(files) == 1:
        dataset = xr.open_dataset(*files, decode_times=False)
    else:
        dataset = xr.open_mfdataset(files, combine='by_coords', decode_times=False)
    if isinstance(variables, str):
        variables = [variables]
    if variables:  # truncate variables
        dataset = dataset[variables]

    # Standardize coordinates
    # NOTE: This also renames old vertical coordinate conventions to 'lev'
    dataset = dataset.load()  # otherwise add_scalar_coords fails
    stopwatch('Load')
    dataset = dataset.climo.standardize_coords()
    stopwatch('Standardize')
    dataset = dataset.climo.add_scalar_coords()
    stopwatch('Scalar')

    # Select hemisphere.
    # Do this first in case it reduces amount of data for below operations
    # NOTE: Don't do enforce_global here to boost speed
    if 'lat' in dataset.dims:  # select hemisphere
        invert = ('v', 'vdt', 'emf', 'ehf', 'egf', 'pv', 'slope')
        dataset = dataset.climo.sel_hemisphere(hemi, invert=invert)
        stopwatch(f'Hemisphere {hemi}')

    # Sum over dimensions or make selections for spectral data
    # NOTE: Critical to drop extra dimensions when we make selections, or will have
    # issues combining spectral data with pressure data.
    # TODO: Have the Variable naming class detect selected pressure levels and
    # latitudes here... or just remove this block and take the efficiency hit,
    # probably better that way.
    if add is not None:
        dataset = dataset.sum(dim=add, keep_attrs=True)
    if lat is not None:
        dataset = dataset.sel(lat=lat, method='nearest', drop=True)
    if plev is not None:
        dataset = dataset.sel(plev=plev, method='nearest', drop=True)

    # Average over blocks of time to reduce resolution
    if dataset.sizes.get('time', 1) > 1:
        dataset = dataset.sortby('time')  # sometimes is messed up
    if 'time' in dataset.dims and days is not None:
        dataset = dataset.climo.coarsen(time=days).mean()

    # Update cfvariable-relevent attributes
    for name, attrs in ATTRS_OVERRIDES.items():
        if name in dataset:
            dataset[name].attrs.update(attrs)

    # Repair Lorenz energy terms and stopwatch dividing by g. Also fix missing units.
    # Q: Why do this? Shouldn't vertical integral always be per 100hPa?
    # A: Makes more sense when comparing with e.g. thermodynamic budget. Put
    # 100hPa in denominator only when displaying these terms in isolation.
    if 'k' in dataset.coords:
        k = dataset.coords['k']
        if np.all(k.data < 1):
            k = k / (k[1] - k[0])  # not sure how this happened
            dataset = dataset.climo.replace_coords(k=k)
    for name, da in dataset.items():
        if 'units' not in da.attrs:
            if name == 'z':
                da.attrs['units'] = 'm'
            elif name == 'egf':
                da.attrs['units'] = 'm2 / s'
            elif not da.climo._is_bounds:
                raise RuntimeError(f'Missing units for variable {name!r}.')
        if name in vreg.lorenz and da.attrs['units'] in ('J/m2 Pa', 'W/m2 Pa'):
            da *= 9.80665
            da.attrs['units'] = da.attrs['units'].replace('m2 Pa', 'kg')
    stopwatch('Fixes')

    # Upsample to float64 for computational accuracy
    # NOTE: This was only recently fixed. Requires xarray dev branch.
    dataset = dataset.astype(np.float64, keep_attrs=True)
    stopwatch('Upsample')
    dataset.attrs['filename'] = ','.join(files)
    if 'history' in dataset.attrs:
        del dataset.attrs['history']  # not useful
    return dataset
