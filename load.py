#!/usr/bin/env python3
"""
Classes for loading experiment series with the dry dynamical
core experiments. Interprets and understands folder hiearchies and
experiment naming conventions.

Todo
----
Support "fake" experiments representing ultra-slow damping (dynamical
equilibrium atmosphere) and ultra-fast damping (thermal equlibrium atmosphere).
Set zonal wind to thermal wind and adjust the forcing terms accordingly.

Consider switching from current approach of specifying 'forcing scheme'
to simply specifying all variable changes (including boundary layer)
in the experiment name. Much simpler that way.
"""
import copy
import functools
import glob
import itertools
import numbers
import os
import re
import time
import warnings

import metpy.interpolate as interp
import numpy as np
import xarray as xr
from climopy import vreg
from icecream import ic  # noqa: F401

import definitions  # noqa: F401

_LoadWarning = type('LoadWarning', (UserWarning,), {})
_warn_simple = functools.partial(warnings.warn, category=_LoadWarning, stacklevel=2)

SCRATCH = '/mdata1/ldavis'  # TODO: iterate through possible scratch dirs
STORAGE = os.path.expanduser('~/data/timescales')

REGEX_RESO = re.compile(r'\At[0-9]+l[0-9]+[spe]\Z')
REGEX_SCHEME = re.compile(r'\A(hs|pk|pkmod)[12][cln]*\Z')

SEASON_IDXS = {
    'djf': (11, 0, 1),
    'mma': (2, 3, 4),
    'jja': (5, 6, 7),
    'son': (8, 9, 10),
    **{i: (i - 1,) for i in range(1, 13)},  # e.g. 'month 1' is index 0 in month array
}

ATTRS_OVERRIDES = {
    'u': {'long_name': 'zonal wind'},
    'v': {'long_name': 'meridional wind'},
}


def _filter_args(*args):
    """
    Filter arguments into schemes, resos, and parameter names.
    """
    # Separate the keys into groups of schemes and resolutions
    schemes = []
    resos = []
    names = []
    args = [_ for arg in args for _ in arg.split('_')]
    for arg in args:
        if REGEX_SCHEME.match(arg):
            schemes.append(arg)
        elif REGEX_RESO.match(arg):
            resos.append(arg)
        else:
            names.append(arg)
    return tuple(schemes), tuple(resos), tuple(names)


def _joint_name(*args):
    """
    Return suitable string representation for the input experiments.
    """
    exps = tuple(
        exp for exps in args
        for exp in ((exps,) if isinstance(exps, Experiment) else exps)
    )
    return '_'.join((
        '-'.join(sorted({scheme.item() for exp in exps for scheme in exp.schemes})),
        '-'.join(sorted({reso.item() for exp in exps for reso in exp.resos})),
        '-'.join(sorted({name for exp in exps for name in exp.names})),
    )).rstrip('_')


def _make_stopwatch(timer=False, fixedwidth=20):
    """
    Make a stopwatch using scary globals() fuckery.
    """
    if 't' not in globals():
        global t
        t = time.time()
    def _stopwatch(message):  # noqa: E306
        global t
        if timer:
            print(
                message + ':' + ' ' * (fixedwidth - len(message)),
                -t + (t := time.time())
            )
    return _stopwatch


def load_cam5(
    base='~/data/cam5/',
    hs94='~/data/timescales_constants/hs1_t85l60e.nc',
    subdirs=('forcing', 'kernels'),
    timer=False,
):
    """
    Load the CAM5 forcing and feedback kernel data.
    """
    # Get list of files
    stopwatch = _make_stopwatch(timer=timer)  # noqa: F841
    ps = set()
    files = [
        file
        for subdir in subdirs
        for file in glob.glob(os.path.expanduser(f'{base}/{subdir}/*.nc'))
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
    ds_hs94 = xr.open_dataset(hs94, decode_times=False)
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
        ds_mlev.to_netcdf(f'{base}/cam5_mlev.nc')
        ds_plev.to_netcdf(f'{base}/cam5_plev.nc')
    except Exception:
        _warn_simple('Failed to save datasets. Possible permissions error.')
    return ds_mlev, ds_plev


def load_gfdl(
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


class Experiment(object):
    r"""
    Class describing the results of an experiment or set of experiments with *deferred*
    loading of related `~xarray.Dataset`\ s. This is more-or-less a thin wrapper around
    the *current* `~xarray.Dataset`, which is set with `~Experiment.load`.
    """
    def __str__(self):
        return _joint_name(self)

    def __repr__(self):
        pairs = {
            'scheme': tuple(_ for _ in self.schemes),
            'reso': tuple(_ for _ in self.resos),
            **{da.name: tuple(da.values) for da in self.parameters_all}
        }
        string = ', '.join(f'{key}={value!r}' for key, value in pairs.items())
        return f'Experiment({string})'

    def __init__(self, *args, verbose=True, **params):
        """
        Parameters
        ----------
        *args
            Forcing specs, resolution specs, specific parameter values, and
            parameter names whose values we should search for. Pass multiple
            values to build datasets concatenated along that dimension.
        verbose : bool, optional
            Whether to print message.
        **params
            Parameters sampled at particular values.
        """
        # Helper function
        # NOTE: Important that parameter unquantified for now
        def _param_array(name, values):
            if not np.iterable(values):
                values = (values,)
            return xr.DataArray(
                np.array([np.nan if _ is None else _ for _ in values], dtype='d'),
                name=name,
                dims=name,
                attrs={'units': vreg[name].standard_units},
            )

        # Filter and get defaults
        # NOTE: Param placeholders can be passed as e.g. ntau=np.nan
        schemes, resos, names = _filter_args(*args)
        params = {
            **{name: _param_array(name, None) for name in names},
            **{name: _param_array(name, value) for name, value in params.items()}
        }
        schemes = schemes or ('hs1',)
        resos = resos or ('t42l20s',)
        if any(da.size > 1 and np.any(~np.isfinite(da)) for da in params.values()):
            raise ValueError('Invalid use of param placeholders in param spec.')

        # For each forcing scheme, reso combination, and fixed param value, search
        # for available values of unfixed params.
        # NOTE: Globs have to end in '*' to capture the 'c'
        regex = r'_(?:([a-z]+)([0-9.]+)(?=_|c?\Z))'  # match e.g. (param)(0100.000)c
        number = '[0-9][0-9][0-9][0-9].[0-9][0-9][0-9]'  # glob e.g. 0100.00
        params_found = []
        for scheme, reso, *param in itertools.product(schemes, resos, *params.values()):
            # Find existing paths including 'reference' experiments
            globs = tuple(
                (
                    f'{STORAGE}/{scheme}_{reso}_' + '_'.join(
                        da.name + (number if ~np.isfinite(da) else f'{da.item():08.3f}')
                        for da in param
                        if da != da.climo.cfvariable.reference
                        and (i == 1 or np.isfinite(da))
                    )
                ).rstrip('_') for i in range(2)  # i == 0 is for 'reference' experiments
            )
            paths = sorted(  # will filter from cold-warm start versions when loading
                path for iglob in globs for suffix in ('', 'c')
                for path in glob.glob(iglob + suffix)
            )
            # Get available parameters from paths
            iparams_found = {k: [] for k in params}
            for path in paths:
                jparams_found = {k: v for k, v in re.findall(regex, path)}
                for k in params:
                    if k in jparams_found:
                        iparams_found[k].append(float(jparams_found[k]))
                    else:  # reference params are not explicitly written in path name
                        iparams_found[k].append(params[k].climo.cfvariable.reference)
            params_found.append({k: sorted(set(v)) for k, v in iparams_found.items()})

        # Filter to inner combo of unfixed param values and outer combo of fixed
        # param values, and build coordinate arrays....
        schemes = xr.DataArray(np.array(schemes, dtype='U'), dims='scheme')
        resos = xr.DataArray(np.array(resos, dtype='U'), dims='reso')
        inner = tuple(da.name for da in params.values() if np.any(~np.isfinite(da)))
        params = {
            k: _param_array(
                k,
                sorted(
                    functools.reduce(
                        lambda a, b: set(a) & set(b) if k in inner else set(a) | set(b),
                        (_[k] for _ in params_found)
                    ),
                    reverse=vreg[k].axis_reverse,
                )
            )
            for k in params
        }
        for k, v in params.items():
            if v.size == 0:
                raise RuntimeError(f'Failed to find any {k!r} values.')

        # Fill up matrix of experiment names
        shape = tuple(map(len, (schemes, resos, *params.values())))
        expnames = xr.DataArray(
            np.empty(shape, dtype='U50'),  # allow up to 50 characters
            dims=('scheme', 'reso', *params),
            coords={'scheme': schemes, 'reso': resos, **params},
        )
        for scheme, reso, *param in itertools.product(schemes, resos, *params.values()):
            expname = (
                f'{scheme.item()}_{reso.item()}_' + '_'.join(
                    f'{da.name}{da.item():08.3f}' for da in param
                    if da != da.climo.cfvariable.reference
                )
            ).rstrip('_')
            loc = (scheme.item(), reso.item(), *(da.item() for da in param))
            expnames.loc[loc] = expname

        # Store experiment names and print message
        self._dataset = None
        self._datasets = {}
        self._expnames = expnames
        if verbose:
            print(f'Experiment {self}')
            for dim in expnames.dims[2:]:
                print(f'  {dim}: ' + ', '.join(map(repr, expnames.coords[dim].values)))

    def __len__(self):
        # Return size of major parameter axis. Account for situation where major
        # parameter axis was reduced during load() due to scalar selection.
        data = self.data
        name = self.name
        if not data:
            _warn_simple('Data not yet loaded.')
            return 0
        elif name in data.dims:
            return data.sizes[name]
        else:
            return 1

    def __iter__(self):
        # Iterate over major parameter axis. Account for situation where major
        # parameter axis was reduced during load() due to scalar selection.
        data = self.data
        name = self.name
        for i in range(len(self)):
            if name in data.dims:
                yield data.isel({name: i})
            else:
                yield data  # length will have been one

    def __getitem__(self, key):
        # Retrieve from parameter axis ncks-style, using integers for isel-selection
        # and floats for sel-selection. Account for situation where major parameter axis
        # was reduced during load() due to scalar selection.
        data = self.data
        name = self.name
        if not data:
            raise RuntimeError('Data not yet loaded.')
        if np.iterable(key):
            test = key = list(key)  # select explicit values
        elif isinstance(key, slice):
            test = (key.start, key.stop, key.step)
        else:
            test = (key,)
        if all(isinstance(val, numbers.Integral) for val in test if val is not None):
            if name in data.dims:
                data = data.isel({name: key})
            elif name in data.coords and key == 0:  # scalar coordinate
                pass
            else:
                raise KeyError(f'Invalid selection {name}={key!r}.')
        elif all(isinstance(val, numbers.Real) for val in test if val is not None):
            if name in data.dims:
                data = data.climo.sel({name: key})  # climo.sel supports units
            elif name in data.coords and key == data.coords[name]:
                pass
            else:
                raise KeyError(f'Invalid selection {name}={key!r}.')
        else:
            raise KeyError(f'Invalid selection {name}={key!r}.')
        return data

    def __getattr__(self, attr):
        # Redirect to major parameter accessor. Permits e.g. exp.long_name
        if attr[:1] == '_':
            return super().__getattribute__(attr)  # trigger builtin AttributeError
        if hasattr(self.parameters.climo, attr):
            return getattr(self.parameters.climo, attr)
        else:
            return super().__getattribute__(attr)  # trigger builtin AttributeError

    def load(
        self, file='2xdaily_inst_climate', file_base='2xdaily_inst_climate',
        *, day1='*', day2='*', hemi='ave', teq=False, iso=False,
        squeeze=True, reload=False, verbose=True, timer=False,
        **kwargs,
    ):
        """
        Set the current dataset file. The `Experiment` class calls this and
        filters to `Experiment`\\ s for which a FileNotFoundError is not raised.

        Parameters
        ----------
        file : str
            The file prefix.
        file_base : str, optional
            The base file prefix. Added to "special" files containing e.g. EOF data.
        day1, day2 : int, optional
            The starting and ending days or glob patterns for the file.
        hami : {'ave', 'nh', 'sh', 'globe'}
            The hemisphere specification.
        teq : bool, optional
            Add a "dummy" experiment where temp is at equilibrium everywhere.
        iso : bool, optional
            Add a "dummy" experiment where temp is horizontally isothermal.
        squeeze : bool, optional
            Whether to squeeze singleton forcing dimensions.
        reload : bool, optional
            Whether to reload if the data has already been loaded.
        **params : list of float, optional
            Lists of param values to explicitly select.
        **params_drop : list of float, optional
            Lists of param values to explicitly ignore. Should have suffix '_drop'.
        **kwargs
            Passed to `load_gfdl`.
        """
        # Bail if selection has not changed since last call
        # This prevents unnecessary loads and calculations
        stopwatch = _make_stopwatch(timer=timer)
        day1 = format(day1, '05d') if not isinstance(day1, str) else day1
        day2 = format(day2, '05d') if not isinstance(day2, str) else day2
        pattern = f'{file}.d{day1}-d{day2}.nc'
        pattern_base = f'{file_base}.d{day1}-d{day2}.nc'
        kwargs_compare = {'day1': day1, 'day2': day2, 'hemi': hemi, 'teq': teq, 'iso': iso, 'squeeze': squeeze, **kwargs}  # noqa: E501
        datasets = self._datasets.get(file, None)
        if teq or iso:
            raise NotImplementedError
        if not reload and datasets and datasets[1] == kwargs_compare:
            self._dataset = datasets[0]  # set currently "active" dataset
            return

        # Filter experiments to be loaded
        params = {
            key: kwargs.pop(key) for key in tuple(kwargs)
            if key in self.names and kwargs[key]  # i.e. if key is not None
        }
        params_drop = {
            key[:-5]: kwargs.pop(key) for key in tuple(kwargs)
            if key[-5:] == '_drop' and key[:-5] in self.names and kwargs[key]
        }
        expnames = self._expnames.sel(**params).drop_sel(**params_drop, errors='ignore')

        # Iterate through schemes, resos, and params
        # NOTE: You can fill object-type numpy arrays with literally anything, including
        # xarray datasets! Use this to arange data then merge with combine_nested.
        datasets = np.full(expnames.shape, None)
        for idx, expname in np.ndenumerate(expnames):
            # Locate data, preferring non-continuation versions
            for parts in itertools.product((STORAGE, SCRATCH), (expname, expname + 'c'), ('', 'netcdf')):  # noqa: E501
                dir = os.path.join(*parts)
                if files := sorted(glob.glob(os.path.join(dir, pattern))):
                    break
            if not files:
                raise FileNotFoundError(f'File(s) {pattern!r} not found for experiment {expname!r}.')  # noqa: E501

            # Load data
            if parts[0] == STORAGE:  # only use dataset with longest end time
                files = files[-1:]
            dataset = load_gfdl(*files, timer=timer, **kwargs)
            if verbose:
                days = tuple(int(d) for f in files for d in re.findall(r'\bd([0-9]+)', f))  # noqa: E501
                print(f'Loaded {expname} (days {min(days)}-{max(days)})')

            # Fix variables
            # WARNING: Ugly kludge. Should return to adding these suffixes to var names
            if 'timescale' in file or 'autocorr' in file:
                method = 'timescale' if 'timescale' in file else 'autocorr'
                for name, da in dataset.items():
                    if method in da.attrs.get('long_name', '') and method not in name:
                        new = re.sub(r'\A(.*?)(_err)?\Z', rf'\1_{method}\2', name)
                        dataset = dataset.rename({name: new})

            # Add base data to datasets without 'basic' quantites
            files = sorted(glob.glob(os.path.join(dir, pattern_base)))
            if 'u' in dataset:
                pass
            elif not files:
                _warn_simple(f'File(s) {pattern_base!r} not found for experiment {self}.')  # noqa: E501
            else:
                if parts[0] == STORAGE:  # only use dataset with longest end time
                    files = files[-1:]
                dataset_base = load_gfdl(*files, **kwargs)
                for var in dataset_base:
                    if var not in dataset:  # e.g. spectral also has 'tvar'
                        dataset[var] = dataset_base[var]

            # Add forcing data to dataset
            # TODO: Add ndamp_mean, ndamp_anom, etc. Problem is just some older
            # experiments do not have this data.
            file_constants = os.path.join(dir, 'constants.nc')
            if file_constants not in files:
                dataset_constants = load_gfdl(file_constants, **kwargs)
                for var in ('teq', 'forcing', 'ndamp', 'rdamp'):
                    if var in dataset_constants:
                        dataset[var] = dataset_constants[var]
                    elif var == 'forcing':  # i.e. no forcing
                        da = 0.0 * dataset_constants['teq']
                        da.attrs.update({'long_name': 'forcing', 'units': 'K/s'})
                        dataset[var] = da

            # Replace time coordinate with NaN so it doesn't show up in cfvariable
            # WARNING: This is kind of ugly kludge
            if 'time' not in dataset.sizes:
                dataset = dataset.climo.replace_coords(time=np.nan)
            datasets[idx] = dataset

        # Merge with combine_nested
        dataset = xr.combine_nested(
            datasets.tolist(),  # convert object array containing datasets
            concat_dim=tuple(expnames.dims),
            coords='minimal',
            compat='override',  # compatibility of dimensions
            combine_attrs='override',
            join='exact',
        )
        stopwatch('Concatenate')

        # Fix various issues: push new dimensions to LHS of array instead of RHS,
        # stop combining coordinate bound variables, add missing coordinate bounds,
        # and squeeze singleton param dims
        dataset = dataset.transpose(*expnames.dims[::-1], ...)
        dataset = dataset.assign_coords(expnames.coords)
        dataset.attrs.clear()
        for var, da in dataset.items():  # fix bounds
            if da.climo._is_bounds:
                isel = {dim: 0 for dim in expnames.dims}
                dataset[var] = dataset[var].isel(isel)
        for da in dataset.values():  # add methods
            for method in ('timescale', 'autocorr'):
                if method in da.attrs.get('long_name', ''):
                    da.climo.update_cell_methods(time=method)
        if squeeze:
            dims = tuple(dim for dim in expnames.dims if dataset.sizes[dim] == 1)
            dataset = dataset.squeeze(dims)
        stopwatch('Cleanup')

        # Perform intensive calculations
        # WARNING: This step is bottleneck so do it to merged dataset rather
        # than individual GFDL simulation datasets.
        zero = (
            'u', 'udt', 'v', 'vdt', 'emf', 'ehf', 'egf', 'eqf',
            'uvar', 'vvar', 'ke', 'km', 'dke', 'dkm', 'cpeke', 'cpmkm',
            'ke_tropic', 'ke_clinic', 'km_tropic', 'km_clinic',
        )
        dataset = dataset.climo.enforce_global(zero=zero)
        stopwatch('Global')  # too slow! wait until merge!
        dataset = dataset.climo.quantify()
        stopwatch('Quantify')  # too slow! wait until merge!
        dataset = dataset.climo.add_cell_measures()
        stopwatch('Cell measures')  # too slow! wait until merge!

        # Save dataset with quantities and weights added
        # NOTE: Save add_cell_measures to the end to keep it short
        self._dataset = dataset  # currently active dataset
        self._datasets[file] = (dataset, kwargs_compare)  # repository

    @property
    def data(self):
        return self._dataset

    @property
    def schemes(self):
        return tuple(self._expnames.coords['scheme'].values)

    @property
    def scheme(self):
        return self._expnames.coords['scheme'].values[0]

    @property
    def resos(self):
        return tuple(self._expnames.coords['reso'].values)

    @property
    def reso(self):
        return self._expnames.coords['reso'].values[0]

    @property
    def names(self):
        return tuple(self._expnames.dims[2:])

    @property
    def name(self):
        return self._expnames.dims[2]

    @property
    def reference(self):
        return self.parameters.climo.cfvariable.reference

    @property
    def ireference(self):
        return self.parameters.values.tolist().index(self.reference)

    @property
    def ireference_loaded(self):
        return self.parameters_loaded.values.tolist().index(self.reference)

    @property
    def parameters(self):  # DataArray for first param
        expnames = self._expnames
        return expnames.climo.coords[expnames.dims[2]].climo.dequantify()

    @property
    def parameters_all(self):  # DataArray for first param
        expnames = self._expnames
        return tuple(expnames.climo.coords[dim].climo.dequantify() for dim in expnames.dims[2:])  # noqa: E501

    @property
    def parameters_loaded(self):
        return self.data.climo.coords[self.name].climo.dequantify()


class ExperimentCollection(object):
    """
    Collection of `Experiment` objects. Helps create simple 'parameter sweep'
    Experiments with a single parameter varied and others fixed.
    """
    def __str__(self):
        return _joint_name(*(_ for _ in self._experiments))

    def __repr__(self):
        pairs = {
            'scheme': self._schemes, 'reso': self._resos,
            **{name: (None,) for name in self._unfixed if name is not None},
            **{name: value for name, value in self._fixed.items() if name is not None},
        }
        string = ', '.join(f'{key}={value!r}' for key, value in pairs.items())
        return f'ExperimentCollection({string})'

    def __init__(self, *args, **params):
        """
        Parameters
        ----------
        *args
            The forcing scheme, resolution, and unfixed params.
        **params
            The fixed params later passed to `Experiment`.
        """
        schemes, resos, names = _filter_args(*args)
        exps = []
        self._schemes = schemes or ('hs1',)
        self._resos = resos or ('t42l20s',)
        self._fixed = params or {None: None}  # placeholder when using itertools.product
        self._unfixed = names or (None,)  # placeholder when using itertoold.product
        for scheme, reso, unfixed, (fixed, value) in itertools.product(
            self._schemes, self._resos, self._unfixed, self._fixed.items(),
        ):
            args = (unfixed,) if unfixed is not None else ()
            kwargs = {fixed: value} if fixed is not None else {}
            try:
                exps.append(Experiment(scheme, reso, *args, **kwargs))
            except RuntimeError:
                pass
        if not exps:
            raise RuntimeError('Failed to find any Experiments.')
        self._experiments = exps

    def __len__(self):
        # Get number of experiments
        return len(self._experiments)

    def __iter__(self):
        # Iterate over experiments
        yield from self._experiments

    def __getitem__(self, key):
        # Return nth item
        if isinstance(key, numbers.Integral):
            return self._experiments[key]

        # Separate the keys into groups of schemes and resolutions
        keys = key if isinstance(key, tuple) else (key,)
        if not all(isinstance(key, str) for key in keys):
            raise ValueError('Keys must be string.')
        exps = []
        schemes, resos, names = _filter_args(*keys)
        for scheme, reso, name in itertools.product(
            schemes or (None,), resos or (None,), names or (None,),
        ):
            exps.extend(
                exp for exp in self
                if (not scheme or scheme in exp.schemes)
                and (not reso or reso in exp.resos)
                and (not name or name in exp.names)
            )

        # Return the subgroup
        # NOTE: Important not to create new Experiment so do not lose data
        c = copy.copy(self)
        c._experiments = exps
        c._schemes = tuple(_ for _ in c._schemes if any(_ in exp.schemes for exp in exps))  # noqa: E501
        c._resos = tuple(_ for _ in c._resos if any(_ in exp.resos for exp in exps))
        c._unfixed = tuple(_ for _ in c._unfixed if any(_ in exp.names for exp in exps))
        c._fixed = {k: v for k, v in c._fixed.items() if any(k in exp.names for exp in exps)}  # noqa: E501
        return c

    def load(self, *args, **kwargs):
        """
        Call `Experiment.load` for all experiments in the collection.
        """
        names = tuple(exp.name for exp in self)
        for exp in self:
            ikwargs = {
                key: value for key, value in kwargs.items()
                if key.split('_')[0] not in names or key.split('_')[0] == exp.name
            }
            exp.load(*args, **ikwargs)
