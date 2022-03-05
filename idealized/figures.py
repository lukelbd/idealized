#!/usr/bin/env python3
"""
Templates for figures detailing arbitrary experiment results.
"""
import functools
import itertools
import os
import re
import warnings

import climopy as climo
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import numpy as np
import proplot as pplt
from icecream import ic  # noqa: F401

from . import experiment
from climopy import vreg, ureg, const  # noqa: F401

pplt.rc.autoformat = False
pplt.rc['subplots.refwidth'] = 1.8
pplt.rc['subplots.panelpad'] = 0.8


class _dict_of_lists(dict):
    """
    Simple dictionary for storing groups of lists.
    """
    def add(self, key, item):
        # Add to list and initialize if list is not present
        if key not in self:
            self[key] = []
        self[key].append(item)


def _double_colorbar_patch(cax):
    """
    Monkey patch that ensures double colorbar axes positions are locked.
    """
    # TODO: Why does this example require monkey patch, but simple
    # example in proplot does not?
    # TODO: Support GridSpec updating even *twins* of *child* axes.
    # Think this was the problem.
    def _reposition(self, *args, **kwargs):
        self.set_position(cax.get_position())
        type(self).draw(self, *args, **kwargs)
    return _reposition


def _compare_experiments(*args):
    """
    Return dictionary of various properties that can be used to style figure
    components and compare the experiments in a legend.

    Parameters
    ----------
    *args : `experiment.Experiment`
        The experiments.

    Returns
    -------
    dict
        Dictionary containing the keys:

        * leader : Boolean list, indicates whether each experiment is group 'leader'.
        * labels : Labels used to differentiate experiment with same name.
        * colors : Colors used to differentiate different experiment.
        * linestyles : Line styles used to differentiate experiment with same name.
    """
    # Simple settings
    unique = lambda x: (seen := set()) or tuple(_ for _ in x if _ not in seen and not seen.add(_))  # noqa: E501
    linestyles_multi = ['-', (0, (1, 0.5)), (0, (2, 0.5)), (0, (3, 0.5))]
    linestyles = ['-'] * len(args)  # default
    colors = [exp.color for exp in args]
    labels = [None for exp in args]  # no labels by default
    leader = [True] * len(args)  # is this the "first" entry in a subgroup?
    if len(args) > len(linestyles_multi):
        raise ValueError('Too many experiments.')

    # Get labels by iterating through *subgroups*, for now just groups
    # where the parameter is the same (e.g. 'ntau', 'tgrad', etc.)
    # NOTE: For parameters without any peers, set the label to None.
    names = [re.sub('ntau(mean|anom)', 'ntau', exp.name) for exp in args]
    for name in set(names):
        # Get subgroups
        idxs = [i for i, iname in enumerate(names) if iname == name]
        iargs = [args[idx] for idx in idxs]
        if len(idxs) <= 1:
            continue

        # Get label names for this subgroup
        ilabels = [exp.category for exp in iargs]  # default labels
        schemes = unique('-'.join(unique(exp.schemes)) for exp in iargs)
        names = unique('-'.join(unique(exp.names)) for exp in iargs)
        resos = unique('-'.join(unique(exp.resos)) for exp in iargs)
        # Multiple schemes
        if len(schemes) > 1:
            modes = tuple(re.sub(r'[0-9]\Z', '', scheme) for scheme in schemes)
            blopts = tuple(re.sub(r'\A[a-zA-Z]*', '', scheme) for scheme in schemes)
            ilabels = [scheme.upper() for scheme in schemes]  # default
            if set(blopts) == {'1', '2'}:
                ilabels = [
                    'scaled boundary layer damping' if blopt == '1'
                    else 'fixed boundary layer damping' for blopt in blopts
                ]
            elif set(modes) == {'hs', 'pk'}:
                ilabels = [
                    'inactive stratosphere' if mode == 'hs'
                    else 'active stratosphere' for mode in modes
                ]
        # Multiple experiment names
        if len(names) > 1:
            iparams = set(names)
            ilabels = [exp.long_label for exp in iargs]  # default
            if iparams == {'ntau', 'ntaumean'}:
                ilabels = [
                    'damping full field' if name == 'ntau'
                    else 'damping zonal mean component' for name in names
                ]
        # Multiple resolutions
        if len(resos) > 1:
            ilabels = [reso[:-1].upper() for reso in resos]

        # Add colors and linestyles for legend entries
        for i, idx in enumerate(idxs):
            leader[idx] = i == 0
            labels[idx] = ilabels[i]
            colors[idx] = iargs[idxs[0]].color
            linestyles[idx] = linestyles_multi[i]

    # Return settings
    return {
        'leader': leader,
        'labels': labels,
        'colors': colors,
        'linestyles': linestyles,
    }


def _get_item(name, specs, idx=2):
    """
    Retrieve unique arguments from one of the spec dictionaries. This is
    used to retrieve the color cycle from `kwopt` for each axes.
    """
    items = tuple(spec[idx][name] for spec in specs if name in spec[idx])
    ids = set(map(id, items))
    if not ids:
        return None
    elif len(ids) == 1:
        return items[0]
    else:
        raise ValueError(f'Ambiguous or conflicting {name!r} items {items!r}.')


def _get_lims(lim, coords):
    """
    Sanitize axis limits.
    """
    # Sanitize input limits
    if lim is None:
        return lim
    lo, hi = lim
    if lo is None:
        lo = -np.inf
    if hi is None:
        hi = np.inf
    reverse = lo > hi
    if reverse:
        lo, hi = hi, lo  # coordinates of *data* are always stores increasing
    override = coords.dims[0] == 'lat' and np.sum(coords < 0) == 1
    if override:
        lo = np.min(coords) - 1e-10  # do not remove this
    lim_new = [coords.min().item(), coords.max().item()]
    if override:
        lim_new[0] = 0
    if reverse:
        lim_new = (lim_new[1], lim_new[0])
    lim[:] = lim_new  # fix old limits
    return slice(lo, hi)


def _get_scalar_data(exp, name, standardize=True, **kwargs):
    """
    Return `xarray.DataArray` with 0 spatial dimensions.
    """
    # NOTE: Line tracking across e.g. a forcing parameter can be
    # easily done with e.g. get('ehf', lev='int', lat='absmax', dim_track='ntau')
    x = exp.data.climo.get(name, standardize=standardize, quantify=False, **kwargs)  # noqa: E501
    x = x.squeeze()
    if 'track' in x.dims:
        x = x.transpose(..., 'track')
    if x.ndim > 1 and x.dims[1] != 'track':
        raise ValueError(
            f'Expected 1D or 0D variable, but param {name!r} with {kwargs} '
            f'returned variable with dimensions {x.sizes}.'
        )
    return x


def _get_1d_data(data, name, **kwargs):
    """
    Return 1D `xarray.DataArray` and 1D x 2 `xarray.DataArray` of error bounds, with
    *axis* bounds filtered based on limits.
    """
    y = data.climo.get(name, standardize=True, quantify=False, **kwargs)
    y = y.squeeze().climo.dequantify()  # NOTE: dequantify no longer needed?
    yerr = None
    if 'track' in y.dims:
        y = y.transpose(..., 'track')
    if y.dims[-1] == 'bounds':  # error bounds
        y = y.isel(bounds=1, drop=True)
        yerr = y.isel(bounds=(0, 2))
    if y.ndim > 1 and y.dims[1] != 'track':
        raise ValueError(
            f'Expected 1D variable, but param {name!r} with {kwargs} '
            f'returned variable with dimensions {y.dims}.'
        )
    return y, yerr


def _get_2d_data(data, name, **kwargs):
    """
    Return 2D `xarray.DataArray` with *axis* bounds filtered based on limits.
    """
    z = data.climo.get(name, standardize=True, quantify=False, **kwargs).squeeze()  # noqa: E501
    if z.ndim != 2:
        raise ValueError(
            f'Expected 2D variable, but param {name!r} with {kwargs} '
            f'returned variable with dimensions {z.dims}.'
        )
    z = z.drop({'scheme', 'reso'} & set(z.coords))
    return z


def _get_xyprops(ds, mode='yz', **kwargs):
    """
    Return dictionary of default `proplot.axes.Axes.format` arguments for variety of
    cross-section types. Optionally override them with input args.
    """
    # Latitude coordinate properties
    # Detect when plotting single hemisphere
    # TODO: sine scale has issues with MultipleLocator minor locator and proplot
    # format function has bug due to numpy array 'xminorlocator' argument.
    lat = ds['lat'].values
    hemi = np.sum(lat < 0) <= 1
    maxlat = 89.9  # avoid sine latitude issues with limits at exactly 90
    xlats = {
        'xlim': (0, maxlat) if hemi else (-maxlat, maxlat),
        'xlocator': list(range(-80, 81, 20)),
        'xminorlocator': list(range(-80, 81, 10)),
        'xformatter': 'deg',
    }
    ylats = {
        'y' + key[1:]: value
        for key, value in xlats.items()
    }

    # Latitude-pressure cross-sections
    mode = mode.lower()
    if mode == 'yz':
        x, y = 'lat', 'lev'
        props = xlats
        if ds.climo.cf.vertical_type == 'temperature':
            props.update(
                {
                    'ylim': (260, 400),  # or more restricted?
                    'yscale': 'linear',
                    'ylocator': np.arange(260, 401, 20),
                    'yminorlocator': np.arange(260, 401, 10),
                }
            )
        elif kwargs.get('yscale', None) == 'log':
            props.update(
                {
                    'yreverse': True,
                    'yscale': 'log',
                    'ylocator': [800, 400, 200, 100, 50, 20, 10, 5, 2, 1],
                    'yminorlocator': 'null',
                }
            )
        else:
            props.update(
                {
                    'yreverse': True,
                    'yscale': 'linear',
                    'ylocator': 200,
                    'yminorlocator': 100,
                }
            )

    # Hovmoller of spindown
    # TODO: Consider changing the xlim
    elif mode == 'ty':
        x, y = 'time', 'lat'
        time = ds['time'].values
        t0, t1 = time.min(), time.max()
        props = {
            'xlocator': 2 * ((t1 - t0) // 20),
            'xminorlocator': (t1 - t0) // 20,
            **ylats,
        }

    # Spectral stuff
    # TODO: Why not implement this in param2d_spectral? Probably slower
    # to perform these operations on the entire dataset!
    elif mode == 'yk':
        x, y = 'lat', 'k'
        props = {
            'yminorlocator': None,
            'ylocator': 2,
            'ylim': (1, 20),
            **xlats,
        }

    elif mode == 'cy':
        x, y = 'c', 'lat'
        props = {  # (cycles/s) / (cycles/m)
            'xlim': (-50, 50),
            'xlocator': 10,
            'xminorlocator': 5,
            **ylats,
        }

    elif mode == 'ck':
        x, y = 'c', 'k'
        props = {
            'xminorlocator': 2.5,
            'xlocator': 5,
            'xlim': (-50, 50),
            'yminorlocator': None,
            'ylocator': 1,
            'ylim': (0, 20),
        }

    else:
        raise ValueError(f'Unknown cross-section mode {mode!r}.')

    # Return properties and make limits *mutable* (see _get_2d_data)
    dims = (y, x)
    props.update({key: kwargs.pop(key) for key in tuple(kwargs) if key[0] in 'xy'})
    props = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in props.items()
    }
    props.update(kwargs)  # remaining args
    return dims, props


def _normalize_contourf(data, *, N=None, lo=1, hi=99, sigfigs=2, extend='both'):
    """
    Normalize data to be used for filled contours.

    Parameters
    ----------
    data : `DataArray`
        The data.
    N : int
        The number of levels.
    lo, hi : float
        The low and high percentiles.

    Returns
    -------
    data : `DataArray`
        The normalized data.
    levels : `ndarray`
        The suitable level boundaries.
    label : str
        A string label for indicating the normalized range.
    """
    # Negative or positive numbers
    formatter = pplt.SigFigFormatter(sigfigs)
    N = N or 10
    if extend == 'min':
        levels = np.linspace(-1, 0, N + 1)
        mmin = np.nanpercentile(data.climo.magnitude, lo)
        data /= abs(mmin)
        label = f'min: {formatter(mmin)}'
    elif extend == 'max':
        levels = np.linspace(0, 1, N + 1)
        mmax = np.nanpercentile(data.climo.magnitude, hi)
        data /= abs(mmax)
        label = f'max: {formatter(mmax)}'
    else:
        # Symmetric
        levels = np.linspace(-1, 1, N + 1)
        mmax = np.max(np.abs([
            np.nanpercentile(data.climo.magnitude, lo),
            np.nanpercentile(data.climo.magnitude, hi)
        ]))
        data /= abs(mmax)
        label = rf'range: $\pm$ {formatter(mmax)}'

    return data, levels, label


def _parse_varspec(spec, **kwargs):
    """
    Used to parse variable specs in `_parse_speclists`. I *know* this
    doesn't look pretty but it's the best choice we got.

    Parameters
    ----------
    spec
        The variable specification.
    **kwargs
        Default group properties.

    Returns
    -------
    spec : 5-tuple
        The standardized specification. The first entry is the name and the
        second 4 entries are the following dictionaries:

        * `name`: The variable name.
        * `kwvar`: Options passed to the variable retrieval command.
        * `kwopts`: Special non-standard options.
        * `kwplot`: Options passed to the plotting command.
        * `kwaxes`: Options passed to `~matplotlib.axes.Axes.format`.
    """
    # Get total dictionary
    # raise ValueError(f'Invalid parameter specification {spec!r}.')
    if isinstance(spec, tuple) and len(spec) == 1 and isinstance(spec[0], str):
        name = spec[0]
    elif isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str) and isinstance(spec[1], dict):  # noqa: E501
        name, ikwargs = spec
        kwargs.update(ikwargs)
    else:
        name = spec
    if not isinstance(name, str):
        raise ValueError(f'Bad variable spec {name!r}.')

    # Non-standard params; usage depends on the function
    # TODO: Perhaps find cleaner way to do this
    kwopts = {}
    for key in (
        'alt',  # whether to use alternate axes
        'cycle',  # custom color cycle
        'base',  # for quiver key
        'bpanel', 'tpanel', 'lpanel', 'rpanel',  # for cross sections
        'bpanel_kw', 'tpanel_kw', 'lpanel_kw', 'rpanel_kw',  # for cross sections
        'oneone', 'zeroline', 'equality',  # draw zero line or x == y line
        'loc', 'ncol', 'order', 'space',  # legend settings
    ):
        if key in kwargs:
            kwopts[key] = kwargs.pop(key)

    # Matplotlib artist params
    kwplot = {}
    for key in (
        'lw', 'linewidth', 'linewidths', 'ls', 'linestyle', 'linestyles', 'alpha',
        'ec', 'edgecolor', 'edgecolors', 'fc', 'facecolor', 'facecolors',
        'marker', 'markersize', 'markeredgewidth', 'markeredgecolor', 'markerfacecolor',
        'color', 'colors', 'cmap', 'extend', 'levels', 'values', 'locator',  # colormaps
        'label', 'labels', 'zorder', 'norm', 'norm_kw',
        'symmetric', 'positive', 'negative', 'nozero',
        'creverse', 'clocator', 'cformatter', 'cminorlocator',  # colorbar settings
    ):
        if key in kwargs:
            kwplot[key] = kwargs.pop(key)

    # Axes format params
    kwaxes = {}  # passed to Axes.format()
    for key in (
        'title', 'titleweight', 'titleloc', 'abc', 'abcweight', 'abcloc',
        'xlim', 'ylim', 'xmargin', 'ymargin',
        'xmin', 'ymin', 'xmax', 'ymax', 'xreverse', 'yreverse',
        'xticks', 'yticks', 'xminorticks', 'yminorticks',
        'xlocator', 'xminorlocator', 'xscale', 'xtickminor', 'xformatter',
        'ylocator', 'yminorlocator', 'yscale', 'ytickminor', 'yformatter',
        'xlabel', 'ylabel',
    ):
        if key in kwargs:
            kwaxes[key] = kwargs.pop(key)

    # Keyword args for retrieving params are *remaining* keywords
    kwvar = kwargs.copy()

    return name, kwvar, kwopts, kwplot, kwaxes


def _parse_speclists(*specs, prefer_single_subplot=False):
    """
    Parse lists (of lists) of variable names.

    Parameters
    ----------
    *specs : param-specs
        The variable specification(s). Options are as follows:

        * `name`: Variable name.
        * `(name, kw)`: Variable name with keyword args.
        * `[spec1, spec2, ...]`: List of specifications.
        * `[[spec1A, ...], [spec2A, ...]]`: List of lists of specifications.
        * `[[spec1A, ..., kw], [spec2A, ..., kw]]`: list of lists of
          specifications, with a keyword dictionary indicating properties
          that apply identically to every variable in the group.

    Returns
    -------
    *specs : [[spec1A, spec1B, ...], [spec2A, spec2B, ...]]
        The variable specifications for different subplots (outer list)
        and the same subplot (inner lists). See `_parse_varspec` for details.
    """
    # Interpret param specs
    specs_final = []
    for spec in specs:
        # Get specs for each subplot
        spec = spec or []
        spec = spec.copy() if isinstance(spec, list) else [spec]
        kwglobal = {}
        if isinstance((spec or [None])[-1], dict):
            kwglobal = spec.pop().copy()  # props for all vars in all subplots
        if not prefer_single_subplot:
            spec = [_ if isinstance(_, list) else [_] for _ in spec]
        elif isinstance(spec, list) and not any(isinstance(_, list) for _ in spec):
            spec = [spec]  # inside one subplot by default

        # Standardize each spec
        spec = [ispec.copy() for ispec in spec]
        for ispec in spec:
            kwgroup = kwglobal.copy()
            if ispec and isinstance(ispec[-1], dict):
                kwgroup.update(ispec.pop())  # props for all vars in subplot
            ispec[:] = [_parse_varspec(_, **kwgroup) for _ in ispec if _ is not None]

        specs_final.append(spec)

    # Ensure specs have compatible lengths (so far just used for xsections)
    nspecs = max(map(len, specs_final))
    for spec in specs_final:
        if not spec:
            spec[:] = [[]] * nspecs  # list of empty sublists
        elif len(spec) == 1:
            spec[:] = spec * nspecs  # same for each group
    if not all(not spec or len(spec) == len(specs_final[0]) for spec in specs_final):
        raise ValueError('Variable specification mismatch.')

    return specs_final[0] if len(specs_final) == 1 else specs_final


def _parse_explists(args):
    """
    Parse list (of lists) of experiments.
    """
    if not any(isinstance(arg, list) for arg in args):
        args = [[arg] for arg in args]  # default to single series per subplot
    if not all(isinstance(_, experiment.Experiment) for arg in args for _ in arg):
        raise ValueError('Input data must be Experiment.')
    return args


def _plot_spectral(ax):
    """
    Add special "spectral" lines showing meaningful reference quantities.
    """
    # Upscale energy cascade line
    x = np.array([7.0, 15.0])
    ax.plot(x, 10 * x**(-5 / 3), color='k', ls='--')
    ax.text(10, 0.25, '$k^{-5/3}$', transform='data', weight='bold')

    # Downscale enstrophy cascade line
    x = np.array([15.0, 40.0])
    ax.plot(x, 100 * x**(-3.0), color='k', ls='--')
    ax.text(25, 0.007, '$k^{-3}$', transform='data', weight='bold')


def _plot_equality(ax, params, line, color='gray5', markercolor='gray5'):
    """
    Plot the line where *x-axis* equals this special variable.
    Also try to plot the intersection point.
    """
    # NOTE: keady should already be inverted by params()
    # NOTE: Remember x axis is in relative units
    ax.plot(params, params, color=color, lw=1, dashes=(5, 5), zorder=0.5)
    ax.fill_between(params, params, 1e3, color=color, alpha=0.5, lw=0, zorder=0)
    try:
        px, py = climo.intersection(params, params, line, xlog=False)
    except (ValueError, TypeError):
        pass
    else:
        ax.scatter(px, py, 200, markercolor, marker='+', markeredgewidth=1, zorder=3)


def _plot_explines(ax, params, ibase):
    """
    Plot lines at the specified variables.
    """
    props = pplt.rc.fill(
        {'color': 'grid.color', 'alpha': 'grid.alpha', 'linewidth': 'grid.linewidth'}
    )
    phs = ax.axvline(
        params[ibase],
        label='Held and Suarez (1994) value',
        **{**props, 'linewidth': 2},
    )
    for param in params:
        pm = ax.axvline(param, label='Modeled values', **props)
    return (phs, pm)


def _plot_reflines(ax, vlines=None, hlines=None):
    """
    Plot zero lines with consistent style.
    """
    for (lines, func) in ((vlines, ax.axvline), (hlines, ax.axhline)):
        if lines is not None:
            for line in np.atleast_1d(lines):
                func(line, ls='-', lw=1, color='k', alpha=0.3, zorder=4)


def _plot_refdiag(ax, origin):
    """
    Plot a reference line between the origin and the point ``origin``.
    """
    extent = 1e3
    x, y = origin
    return ax.plot(
        [-extent * x, extent * x], [-extent * y, extent * y],
        ls='--', lw=1, color='k',
        scalex=False, scaley=False,
        zorder=1,
    )


def _savefig(func):
    """
    Decorator that adds a save option to plotting functions.

    Parameters
    ----------
    filename : str, optional
        The figure name. Final name will be ``f'{experiment}-{filename}.pdf'``.
    dir : str, optional
        The figure directory. Default is ``'figures'``.
    """
    @functools.wraps(func)
    def wrapper(*args, prefix=True, filename=None, dir='../figures', **kwargs):
        fig, axs = func(*args, **kwargs)
        if filename:
            prefix = experiment._joint_name(*args) + '_' if prefix else ''
            path = f'./{dir}/{prefix}{filename}.pdf'
            print(f'Saving figure: {os.path.basename(path)}')
            fig.save(path)
        return fig, axs
    return wrapper


def _subplot_per_simulation(
    args, nvars=1,
    fig=None, axs=None, as_param=None, to_ntau0=False, show_hs94=True,
    share=None, sharex=None, sharey=None,
    array=None, nrows=None, ncols=None,
    ref=1, aspect=1, refwidth=None, figwidth=None, journal=None, outerpad=None,
    abc='A.', abcloc='l', hspace=None, wspace=None,
    vertical=False, refhighlight=False, refcolor='red',
    labels=None, titles=None, title=None, **kwargs,
):
    """
    Generate the figure used for `xsections` and `curves`.

    Parameters
    ----------
    exp : `experiment.Experiment` or list thereof
        The experiment(s).
    nvars : int
        The number of variables to plot.
    fig, axs : `proplot.subplots.Figure`, `proplot.axes.Axes`
        Existing figure and axes used for animations.
    nrows, ncols : int
        Rows and columns for the figure.
    vertical : bool
        If ``True``, experiment members are shown along *columns*, not rows.
    refhighlight : bool
        Whether to highlight the base experiment.
    as_param : str, optional
        Whether to show alternative parameter in the title.
    to_ntau0 : bool, optional
        Whether to convert ntau to ntau0.
    show_hs94 : bool, optional
        Whether to indicate the HS94 configuration in the title.
    title : str
        The overall figure title.
    titles : list of str, optional
        Axes titles to manually override the default titles.
    labels : list of str, optional
        Row or column labels for the variables. Length must equal `nvars`.

    Returns
    -------
    **kwargs
        Remaining keyword arguments. These can be passed on to other
        functions like `~proplot.axes.Axes.format`.
    """
    # Bail out for existing figures and axes
    if fig is not None and axs is not None:
        return fig, axs, kwargs

    # Make figure
    # NOTE: Always go A...B...C along experiments first
    order = 'F' if vertical else 'C'
    nsims = len(args)
    if nsims == 1 or nvars == 1:
        if ncols is None:
            ncols = min(3, max(nsims, nvars))
        if nrows is None:
            nrows = (max(nvars, nsims) - 1) // ncols + 1
    elif vertical:
        ncols, nrows = nvars, nsims
    else:
        ncols, nrows = nsims, nvars
    if sharex is None:
        sharex = 3 if vertical or nvars == 1 or nsims == 1 else 0
    if sharey is None:
        sharey = 3 if not vertical or nvars == 1 or nsims == 1 else 0
    fig, axs = pplt.subplots(
        array=array, ncols=ncols, nrows=nrows, span=False,
        share=share, sharex=sharex, sharey=sharey,
        hspace=hspace, wspace=wspace,
        refwidth=refwidth, figwidth=figwidth, journal=journal,
        ref=ref, refaspect=aspect,
        order=order, outerpad=outerpad,
    )
    for ax in axs[nvars * nsims + 1:]:
        ax.set_visible(False)
    axs.format(suptitle=title, abc=abc, abcloc=abcloc, titleabove=True)

    # Put experiment labels above every plot, even if redundant
    # NOTE: For now only support 'as_param' sensitivity parameters
    # NOTE: Plots were confusing when using these as row or column labels
    titles = titles or (None,) * len(args)
    for i in range(nvars):
        iaxs = axs if nsims == 1 or nvars == 1 else axs[:, i] if vertical else axs[i, :]
        for ax, data, title in zip(iaxs, args, titles):
            if title is None:
                ntau0 = None
                param = data.climo.parameter
                suffix = ''
                if show_hs94 and param == param.climo.reference:
                    suffix = ' (HS94)'
                if param.name == 'ntau':
                    ntau0 = param.climo.to_variable('ntau0', scheme=data.scheme, standardize=True)  # noqa: E501
                    param = ntau0 if to_ntau0 else param
                title = param.climo.scalar_label + suffix
                if as_param and ntau0 is not None:
                    param = ntau0.climo.to_variable(as_param, standardize=True)
                    title += '\n' + param.climo.scalar_label
                if param.scheme == 'hs2':  # WARNING: kludge for uniform damping scheme
                    title = re.sub(r'_(\{max\}|\{min\}|\{0\}|0)', '', title)
            ax.title.set_text(title)

    # Add variable labels as row or column labels
    if labels:
        iaxs = axs[0, :] if vertical else axs[:, 0]
        for ax, label in zip(iaxs, labels):
            obj = ax._top_label if vertical else ax._left_label
            obj.set_text(label)
            obj.set_rotation(0 if vertical else 90)

    # Add reference highlighting
    # NOTE: Guaranteed nsims == len(axs) here
    for data in args:
        if refhighlight and data.climo.parameter == data.climo.parameter.reference:
            ax.format(color=refcolor, linewidth=pplt.rc.linewidth * 2)

    # Return figure
    return fig, axs, kwargs


def _subplot_per_experiment(
    nvars=1, fig=None, axs=None,
    array=None, nrows=None, ncols=None, ref=1, aspect=1, order='C',
    left=None, right=None, top=None, bottom=None,
    refwidth=None, figwidth=None, journal=None, outerpad=None,
    hspace=None, wspace=None, hratios=None, wratios=None,
    span=None, spanx=None, spany=None, share=None, sharex=None, sharey=None,
    title=None, tight=None, abc='A.', abcloc='l',
    **kwargs,
):
    """
    Generate the figure used for `stacks`, `series`, and `parametric`.
    This standardizes the format across different figure types.

    Parameters
    ----------
    nvars : int, optional
        Number of subplots.
    fig, axs : `proplot.supropsbplots.Figure`, `proplot.axes.Axes`
        Existing figure and axes used for animations.
    nrows, ncols : int
        Rows and columns for the figure.
    """
    if ncols is None and nrows is None:
        ncols = nvars
    if ncols is None:
        ncols = (nvars - 1) // nrows + 1
    if nrows is None:
        nrows = (nvars - 1) // ncols + 1  # e.g. 2 cols, 3-4 vars = 2 rows
    fig, axs = pplt.subplots(
        array=array,
        left=left, right=right, top=top, bottom=bottom,
        ncols=ncols, nrows=nrows, order=order,
        hspace=hspace, wspace=wspace,
        hratios=hratios, wratios=wratios,
        refwidth=refwidth, figwidth=figwidth, journal=journal,
        ref=ref, refaspect=aspect,
        spanx=spanx, spany=spany,
        sharex=sharex, sharey=sharey,
        share=share, span=span,
        tight=tight, outerpad=outerpad,
    )
    axs.format(suptitle=title, abc=abc, abcloc=abcloc, titleabove=False)
    return fig, axs, kwargs


@_savefig
def curves(
    *args,
    spec=None,
    transpose=False,
    rightlegend=False,
    ncol=1,
    hlines=None,
    vlines=None,
    **kwargs,
):
    """
    Draw line plots as a function of e.g. latitude for each experiment. This is similar
    to `xsections`.

    Parameters
    ----------
    *args : `xarray.Dataset`
        The dataset(s). Often this is an expanded `experiment.Experiment`.
    spec : str, (str, dict), or list thereof
        The variable spec(s) to be plotted. Parsed by `_parse_speclists`.
    transpose : bool, optional
        If ``True`` the dependent variable is plotted on the *y* axis.
    ncol : int, optional
        The number of legend columns.

    Other parameters
    ----------------
    **kwargs
        Passed to `~proplot.subplots.subplots` or  `~proplot.axes.Axes.format`.

    Todo
    ----
    Add helper function that generates figure and subplots for plotting stuff
    on *each subplot*. Then use this for heat budget stuff.
    """
    # Get figure and axes
    # NOTE: For xsections, wusewant default behavior that each list item
    # represents different subplot. Here, should represent same subplot.
    if spec is None:
        raise ValueError('Must specify variable for plotting.')
    vertical = kwargs.get('vertical', False)
    cycle = kwargs.pop('cycle', None) or '538'
    specs = _parse_speclists(spec, prefer_single_subplot=True)
    nvars = len(specs)
    nsims = len(args)
    fig, axs, kwargs = _subplot_per_simulation(args, nvars=nvars, **kwargs)

    # Iterate through variables and experiments
    for i, (j, k) in enumerate(itertools.product(range(nvars), range(nsims))):
        iax = ax = axs[i]  # order is F or C depending on 'vertical' setting
        data = args[k]
        ispecs = specs[j]

        # Draw lines on the plot
        idx = 0
        idxs = []
        lines = []
        units = set()
        icycle = None
        for l, (name, kwvar, kwopts, kwplot, kwaxes) in enumerate(ispecs):
            kwplot = kwplot.copy()
            kwplot['zorder'] = 2 - 0.5 * (l + 1) / len(ispecs)
            lim = kwaxes.get('ylim' if transpose else 'xlim', None)
            y, err = _get_1d_data(data, name, lim=lim, **kwvar)
            x = y.coords[y.dims[0]]
            xy = (y, x) if transpose else (x, y)
            lines.append((*xy, err, kwaxes, kwplot, kwopts))
            if units and y.climo.units_label not in units:
                idx += 1
            idxs.append(idx)
            units.add(y.climo.units_label)
            icycle = icycle or kwopts.get('cycle', None) or cycle

        # Use colored property cycler if requested parameters have different units
        # TODO: Make this work for multiple series, not just variables
        if max(idxs) > 1:
            raise ValueError('No more than two different units.')
        alt = max(idxs) > 0
        icycle = pplt.Cycle(icycle)
        icycle = itertools.cycle(icycle)

        # Iterate through lines
        hs = []
        for idx, (x, y, err, kwaxes, kwplot, kwopts) in zip(idxs, lines):
            # Plot the line data, possibly on alternate axes
            # TODO: Standardize method for determining whether we use alternate y-axes
            # between curves() and series(). Below is simpler than series() method.
            kw = next(icycle).copy()
            kw.update({'label': y.climo.long_label, **kwplot})
            color = None
            if alt:
                color = kw.get('color', 'k')
                if idx > 0:
                    if hasattr(ax, '_alt_child'):
                        iax = ax._alt_child
                    else:
                        iax = ax._alt_child = ax.altx() if transpose else ax.alty()
            ihs = iax.plot(x, y, **kw)
            hs.append(ihs[0])
            if err is not None:
                iax.errorbar(
                    x, y, fmt='none', ecolor='k',
                    capsize=4, capthick=1, elinewidth=1, clip_on=False,
                    **{'xerr' if transpose else 'yerr': err},
                )

            # Format axes
            xs, ys = 'yx' if transpose else 'xy'
            kw = {
                # xs + 'label': x.climo.long_label,
                xs + 'label': x.climo.short_label,
                xs + 'scale': x.climo.axis_scale,
                xs + 'formatter': x.climo.axis_formatter,
                xs + 'margin': 0,
                # ys + 'label': y.climo.units_label,
                ys + 'label': y.climo.short_label,
                ys + 'scale': y.climo.axis_scale,
                ys + 'formatter': y.climo.axis_formatter,
                ys + 'margin': 0.05,
                ys + 'color': color,
                # ys + 'color': kw.get('color'),
                **kwaxes,
                **kwargs,
            }
            iax.format(**kw)

        # Add extra lines
        for ax in {ax, iax}:
            if name == 'spectral':
                _plot_spectral(ax)
            _plot_reflines(ax, vlines=vlines, hlines=hlines)

        # Add legend
        # TODO: Support both multiple variables with alternate axes
        # and multiple variables with legends like in `series`.
        kw = {'ncol': ncol, 'loc': 'r' if rightlegend else 'b'}
        if nvars == 1:
            if i == 0:
                fig.legend(hs, **kw)
            else:
                pass
        elif nsims == 1:
            ax.legend(hs, **kw)
        elif i % nsims == (nsims - 1 if vertical ^ rightlegend else nsims // 2):
            ax.legend(hs, **kw)

    return fig, axs


@_savefig
def lorenz(
    *args,
    rednoise=False,
    parametric=True,
    sum=True,
    tdiff=False,  # use tdiff for x-axis, or diabatic cooling?
    lw=2,
    boxalpha=0.1,
    margin=0.05,
    boxcolor='yellow6',
    color='k',  # constant color
    suptitle='Eddy-mean Lorenz energy cycle for sweep experiments',
    redcolor='red7',
    blackcolor='gray7',
    timescale=False,
    nlag=100,
    **kwargs
):  # kwargs go to rednoisefit function
    """
    Compare the Lorenz energy budgets of multiple experiments and (optionally)
    experiments.

    Parameters
    ----------
    *args : `experiment.Experiment`
        The experiment(s). The "current data" should be NetCDF files
        containing time series of globally averaged energy budget terms.
    parametric : bool, optional
        Whether to use "parametric" plots.
    rednoise : bool, optional
        Whether to plot the *autocorrelation timescale* of the parameter
        rather than the parameter itself.
    """
    # Standardize input
    # TODO: Clean this function up.
    if not all(isinstance(_, experiment.Experiment) for _ in args):
        raise ValueError('Input args must be Experiment.')
    kwargs.update({'xmargin': 0, 'ymargin': 0.05})

    # Variables to be plotted. Prefer ckmpm to cpmkm
    terms = [d.name for d in vreg.lorenz]
    if not sum:
        terms = [term for term in terms if len(term) > 1]

    # Create figure plot
    if sum:
        hratios = [1, 1, 1, 0.2, 1]
        array = [
            [1, 2, 3, 4, 5],
            [0, 6, 0, 7, 0],
            [8, 9, 10, 11, 12],
            [0, 0, 0, 0, 0],
            [13, 14, 15, 16, 17]
        ]
    else:
        hratios = None
        array = [
            [1, 2, 3, 4, 5],
            [0, 6, 0, 7, 0],
            [8, 9, 10, 11, 12]
        ]
    fig, axs = pplt.subplots(
        array, hratios=hratios,
        figwidth=7.5, sharex=3, sharey=0, span=False
    )

    # Iterate over datasets
    ylims = {}
    handles = {}  # dictionary
    for exp in args:
        for ax, term in zip(axs, terms):
            # Retrieve data
            x, y = [], []  # indices
            ikwargs = {
                'lat': 'avg',
                'lev': 'int',
                'time': 'timescale' if timescale else 'mean',
                'nlag': nlag,  # only relevant if rednoise is True
                **kwargs,
            }
            for param, data in zip(exp.parameters_loaded, exp):
                iy = data.climo.get(term, standardize=True, **ikwargs)
                if not parametric:
                    ix = param.item()  # use the *parameter* for x-axis
                else:
                    ix = data.climo['cooling']
                x.append(ix)
                y.append(iy)
            x = np.asarray(x)
            y = np.asarray(y)

            # Axis limits
            # this won't change per axes! only y-axis values change!
            xlim = (np.nanmin(x), np.nanmax(x))
            if not tdiff:
                xlim = (-1.5, 0)  # necssary for some reasion
            else:
                xlim = (10, 80)
            if term not in ylims:
                ylims[term] = (np.nanmin(y), np.nanmax(y))
            else:
                ylims[term] = (
                    np.nanmin([ylims[term][0], *y]),
                    np.nanmax([ylims[term][1], *y])
                )
            ylim = ylims[term]  # use this one
            diff = np.diff(ylim)[0]
            ylim = (ylim[0] - margin * diff, ylim[1] + margin * diff)
            # TODO: Remove this this
            yticks = None
            if term == 'gpe':
                ylim = (-1.1, 0.05)
                yticks = 1

            # Parametric plot, legend entries are experiment descriptions
            if parametric:
                h = ax.plot(
                    x, y, cmap=exp.cmap, values=exp.parameters_loaded,
                    lw=lw, cmap_kw={'left': 0.2}
                )
                handles[exp.long_label] = h
            # Line plots, legend entries are energy descriptions
            else:
                h, = ax.plot(
                    x, y, lw=lw, color=color, alpha=1, label=exp.long_label
                )
                handles[color] = h  # add to labels dictionary

            # Format the plots, just once
            ikw = {}
            if (sum and term != 'c') or (not sum and term != 'cpeke'):
                xlabel, title = '', ''
            else:
                if sum:
                    title = 'Sum of eddy and zonal-mean components'
                else:
                    title = ''
                if parametric:  # variable label
                    xlabel = y.climo.short_label
                else:  # use experiment label
                    xlabel = exp.long_label

            # Make the *axes border colors* correspond to energy flow
            if parametric:
                ikw.update({'color': color})
            else:
                ikw.update({
                    'xscale': exp.scale,
                    'xlocator': x[::2]
                })
            ikw.update(kwargs)  # optional override
            ax.format(
                abc=False, title=title, suptitle=suptitle,
                xlim=xlim, ylim=ylim, yticks=yticks,
                xlabel=xlabel, **ikw
            )
            ax.text(
                0.9, 0.9, y.climo.symbol, ha='right', va='top',
                weight='bold', transform='axes', border=True
            )
            ax.patch.set_alpha(1)

            # The y-axis labels -- one each for energy content and flow label
            # TODO: Make sure autocorrelation stuff works! And is worth doing
            # in this case we ***do*** need to label axes
            if parametric and term in ('gpe', 'gpm', 'g'):
                prefix = ''
                if rednoise:
                    prefix = 'autocorrelation of '
                ax.yaxis.label.update({
                    'text': prefix + 'energy flow (W/m$^2$)',
                    'color': blackcolor
                })
                t = ax.text(0, 0, '')
                t.update({
                    'color': redcolor,
                    'text': prefix + 'energy content (MJ/m$^2$)'
                })
                t.update({
                    'transform': ax.transAxes, 'position': (-0.38, 0.5),
                    'ha': 'center', 'va': 'bottom',
                    'rotation': 90, 'rotation_mode': 'anchor'
                })

    # Put shading/box behind the bottom row
    # Avoid weird problem where shading appears above the axes patch
    if sum:
        if parametric:
            corner = (0, 0.185)
            extend = (1, 0.24)
        else:
            corner = (0, 0.05)
            extend = (1, 0.28)
        axs[-5].add_patch(mpatches.Rectangle(
            corner, *extend, transform=fig.transFigure,
            facecolor=boxcolor,
            zorder=0, clip_on=False, alpha=boxalpha,
            edgecolor='k', linewidth=0)
        )
        axs[-5].fill_between(
            [0, 1], 0, 1,
            transform=axs[-5].transAxes, color='w',
            edgecolor='none', zorder=1
        )

    # Arrows pointing to energy flow
    akwargs = {
        'width': 0.02, 'head_width': 0.1,
        'facecolor': 'k', 'edgecolor': 'none', 'clip_on': False
    }
    for ax in (axs[0], axs[4]):
        ax.arrow(
            0.15, -0.1, 0.6, 0, shape='left',
            transform=ax.transAxes, **akwargs
        )
    for ax in axs[2], :
        ax.arrow(
            1 - 0.15, -0.1, -0.6, 0, shape='right',
            transform=ax.transAxes, **akwargs
        )
    for ax in (axs[7], axs[9], axs[11]):
        ax.arrow(
            0.15, 1.1, 0.6, 0, shape='right',
            transform=ax.transAxes, **akwargs
        )
    axs[5].arrow(
        1.2, 1 - 0.15, 0, -0.6, shape='right',
        transform=axs[5].transAxes, **akwargs
    )
    axs[6].arrow(
        -0.2, 0.15, 0, 0.6, shape='right',
        transform=axs[6].transAxes, **akwargs
    )

    # Legend and/or colorbar
    if parametric:
        for i, (label, handle) in enumerate(handles.items()):
            fig.colorbar(
                handle, label=label, loc='b', length=0.4
            )
    else:
        fig.legend(
            handles=list(handles.values()),
            loc='b', center=True, ncol=2, columnspacing=3
        )
    return fig, axs


@_savefig
def parametric(
    *args,
    xspec=None,
    yspec=None,
    as_param=None,
    as_param_bottom=False,
    to_ntau0=False,
    show_hs94=True,
    loc='b',
    span=None,
    clength=0.5,
    cwidth='1.5em',
    lw=None,
    linewidth=None,
    **kwargs
):
    """
    Make "parametric" plots of experiments, comparing different scalar quantities on
    each axes. The variables can be specified as string names or as 2-tuples that
    indicate the variable name and keyword arguments to be passed to the `param`
    function.

    Parameters
    ----------
    *args : `experiment.Experiment`
        The experiments.
    xspec, yspec : str, (str, dict), or list thereof
        Variable specs passed to `param`. These are parsed with `_parse_speclists`.
    as_param : str, optional
        Whether to add secondary parameter on the other side of the colorbar.
    as_param_bottom : bool, optional
        Whether to put the secondary parameter on the bottom. Default is ``False``.
    loc, span, clength, cwidth
        Various colorbar properties.
    lw, linewidth : float or list of float, optional
        Line widths for each experiment.

    Other parameters
    ----------------
    **kwargs
        Passed to `~proplot.subplots.subplots` or  `~proplot.axes.Axes.format`.
    """
    sprops = _compare_experiments(*args)  # props for comparing experiments
    xspecs, yspecs = _parse_speclists(xspec, yspec, prefer_single_subplot=False)
    lw = lw or linewidth or 5
    lws = [lw] * len(args) if np.isscalar(lw) or lw is None else list(lw)

    # Plot stuff
    fig, axs, kwargs = _subplot_per_experiment(nvars=len(xspecs), share=False, **kwargs)
    for i, (ax, xspec, yspec) in enumerate(zip(axs, xspecs, yspecs)):
        hs = []
        xspec = xspec[0]  # strictly one variable per plot; see _parse_speclists
        yspec = yspec[0]
        refpts = []  # reference points
        kwopts = {}
        for j, exp in enumerate(args):
            # Get x and y coordinates
            if not isinstance(exp, experiment.Experiment):
                raise ValueError(f'Invalid experiment {exp}.')
            lines = []
            kwaxes = {}
            kwplot = {}
            for x, (name, kwvar, kwop, kwpl, kwax) in zip('xy', (xspec, yspec)):
                kwpl = kwpl.copy()
                line = _get_scalar_data(exp, name, **kwvar)
                if 'label' in kwpl:
                    line.attrs['long_name'] = kwpl.pop('label')
                kwopts.update(kwop)
                kwplot.update(kwpl)
                kwaxes.update(kwax)
                lines.append(line)
            xline, yline = lines
            coords = xline.climo.coords[xline.dims[0]]
            x0 = xline.sel({xline.dims[0]: coords.climo.reference})
            y0 = yline.sel({yline.dims[0]: coords.climo.reference})
            if show_hs94:
                ax.scatter(x0, y0, s=0.9 * lws[j] ** 2, m='D', c='k', z=3)
                # ax.scatter(x0, y0, s=2 * lws[j] ** 2, m='D', c='k', z=3)
            if coords.name == 'ntau' and to_ntau0:
                coords = coords.climo.to_variable('ntau0')
            refpts.append((1, 1) if xline.climo.units == yline.climo.units else (x0, y0))  # noqa: E501

            # Plot parametric line
            # TODO: Add back support for different *forcing schemes*
            # or *resolutions* with different line styles?
            values = coords.values
            kw = {
                'values': values,
                'cmap': exp.colormap_line,
                'joinstyle': 'round',
                'capstyle': 'round',
                'ls': sprops['linestyles'][j],
                'lw': lws[j],
                **kwplot,
            }
            h = ax.parametric(xline.data, yline.data, **kw)
            if sprops['leader'][j]:  # the "leader" in a subgroup
                hs.append((h, exp))

            # Format the axes with optional overrides
            kwaxes = {
                'xreverse': exp.axis_reverse,  # TODO: remove?
                'xlabel': xline.climo.long_label,
                'ylabel': yline.climo.long_label,
                'xscale': xline.climo.axis_scale,
                'yscale': yline.climo.axis_scale,
                'xformatter': xline.climo.axis_formatter,
                'yformatter': yline.climo.axis_formatter,
                'margin': 0.05,
                **kwargs,
                **kwaxes,
            }
            ax.format(**kwaxes)

        # Add the 1:1 line
        # TODO: Check if all pionts are all equivalent?
        if kwopts.get('oneone', False):
            _plot_refdiag(ax, refpts[0])

    del kwargs
    for i, (h, exp) in enumerate(hs):
        # Get colorbar params and settings
        ntau0 = None
        params = exp.parameters_loaded
        if params.name == 'ntau':
            ntau0 = params.climo.to_variable('ntau0', scheme=exp.scheme, standardize=1)
            params = ntau0 if to_ntau0 else params
        params_alt = None
        if as_param and ntau0 is not None:
            params_alt = ntau0.climo.to_variable(as_param, standardize=True)
        kwargs = {
            'locator': params.climo.magnitude,  # tick on exact locations
            'minorlocator': 'null',
            'width': cwidth,
            'length': clength,
            'span': span,
            'extend': 'neither',
            'tickminor': False,
            'reverse': not params.climo.axis_reverse,
        }

        # Add primary colorbar
        # NOTE: Have to manually add space or else figure looks fine when shown in
        # notebook but gets messed up when saved to file.
        fmt = params.climo.scalar_formatter
        cb = fig.colorbar(
            h,
            loc=loc,
            label=params.climo.long_label,
            tickloc='top' if params_alt is not None and as_param_bottom else 'bottom',
            ticklabels=[fmt(param.item()) for param in params],
            space=7 if params_alt is not None and i == 0 else 4,
            **kwargs,
        )

        # Add secondary colorbar
        cbalt = None
        if params_alt is not None:
            def _reposition(self, *args, cax=cb.ax, **kwargs):
                self.set_position(cax.get_position())
                type(self).draw(self, *args, **kwargs)
            fmt = params_alt.climo.scalar_formatter
            xax = cb.ax.altx()
            cbalt = xax.colorbar(
                h,
                loc='fill',
                label=params_alt.climo.long_label,
                tickloc='bottom' if as_param_bottom else 'top',
                ticklabels=[fmt(param.item()) for param in params_alt],
                **kwargs,
            )
            cb.minorticks_off()  # necessary for some reason
            cb.ax.xaxis.set_inverted(kwargs['reverse'])  # necessary for some reason
            for attr in (cbalt.outline, cbalt.solids, cbalt.patch):
                attr.set_visible(False)  # only want axis to be visible
            if h.norm._descending:  # cbar function inverts *twice*... so must re-invert
                cbalt.ax.xaxis.set_inverted(True)
            cbalt.ax.draw = _reposition.__get__(cbalt.ax)
        for cb in (cb, cbalt):
            if cb and '^' in cb.ax.get_xlabel() and cb.ax.xaxis.get_label_position() == 'bottom':  # noqa: E501
                cb.ax.xaxis.labelpad = 0  # avoid excessive padding due to exponent

    # Add legend entries for comparing experiment
    hs = []
    for j, exp in enumerate(args):
        label = sprops['labels'][j]
        if label is None:
            continue
        kwargs = {
            'label': label,
            'color': exp.color,
            'linestyle': sprops['linestyles'][j],
            'linewidth': lws[j],
            'solid_capstyle': 'butt',
        }
        hs.extend(ax.plot([], [], **kwargs))
    if hs:
        fig.legend(hs, loc=loc, ncol=1, frame=False, span=span, handlelength=3)

    return fig, ax


@_savefig
def series(
    *args,
    spec=None,
    colors=None,
    reference=False,
    as_param=None,
    as_param_bottom=False,
    title=None,
    titles=None,
    refhighlight=True,
    loc='bottom',
    ncol=1,
    ordr='C',
    space=None,
    frame=False,
    precision=2,
    **kwargs,
):
    """
    Plot lines of scalar values against the parameter space on the x-axis. Can be used
    to plot multiple series or multiple variables in single or separate subplots.
    Currently, more than one series and variable are not allowed.

    Parameters
    ----------
    *args : `experiment.Experiment`
        The experiment(s) passed as positional arguments. Pass a list of lists to
        put different experiments in a single subplot.
    spec : str, (str, dict), or list thereof
        The variable spec(s). Parsed by `_parse_speclists`.
    reference : bool, optional
        Whether to translate damping timescales to the "reference" value.
    as_param : str, optional
        Whether to add secondary parameter on the other side of the colorbar.
    as_param_bottom : bool, optional
        Whether to put the secondary parameter on the bottom. Default is ``False``.
    title : str, optional
        The overall plot super title.
    titles : list of str, optional
        Individual subplot titles.
    refhighlight : bool, optional
        Whether to draw reference gridlines where experiments were held.
    loc, ncol, ordr, space, frame : optional
        Passed to the legend.

    Other parameters
    ----------------
    **kwargs
        Passed to `~proplot.subplots.subplots` or  `~proplot.axes.Axes.format`.
    """
    # Initial stuff
    # TODO: Leverage _compare_experiments to permit compare more than 2 series
    # on this plot, e.g. mean damping, normal damping, and temperature grad.
    # TODO: Generalized methods for drawing arbitrary plots of various
    # things on different (a) subplots or different (b) twin axes. Maybe
    # can use some simpler helper class e.g. MultiPlot.
    args = _parse_explists(args)
    specs = _parse_speclists(spec, prefer_single_subplot=True)
    suptitle = title
    handlelength = 2  # em units
    handles = {}
    c, ls = zip(*itertools.product(['k', 'gray6', 'gray3'], ['-', ':', '--', '-.']))
    blacks = pplt.Cycle(c, ls=ls)
    colors = pplt.Cycle(colors or pplt.rc.cycle)
    kwargs.setdefault('xmargin', 0)
    kwargs.setdefault('ymargin', 0.05)
    xscale = kwargs.pop('xscale', 'linear')
    xformatter = kwargs.pop('xformatter', ('sigfig', precision))
    xlocator = kwargs.pop('xlocator', None)
    xminorlocator = kwargs.pop('xminorlocator', None)
    xlocator_alt = kwargs.pop('xlocator_alt', None)
    xminorlocator_alt = kwargs.pop('xminorlocator_alt', None)
    xformatter_alt = kwargs.pop('xformatter_alt', ('sigfig', precision))
    # xformatter = kwargs.pop('xformatter', pplt.SimpleFormatter(precision))

    # Draw the figure
    kwargs.setdefault('share', 0)
    nplots = len(args) * len(specs)
    fig, axs, kwargs = _subplot_per_experiment(nplots, **kwargs)
    for i, (exps, ispecs) in enumerate(itertools.product(args, specs)):
        ps = []
        ax = axs[i]
        sprops = _compare_experiments(*exps)  # noqa: F841 TODO: use this!
        if len(ispecs) > 1 and len(exps) > 1:
            raise ValueError('Cannot plot multiple specs and series in same subplot.')
        scheme = exps[0].scheme
        params = exps[0].parameters_loaded
        if reference and params.name == 'ntau':
            params = params.climo.to_variable('ntau0', standardize=True, scheme=scheme)
        xax = None
        if as_param and params.name in ('ntau', 'ntau0'):
            kw = {'scheme': scheme} if params.name == 'ntau' else {}
            params_alt = params.climo.to_variable(as_param, standardize=True, **kw)
            xax = ax.altx(loc='bottom' if as_param_bottom else 'top')  # noqa: E501

        for j, exp in enumerate(exps):
            # Get line composed of scalar values for each experiment, and
            # allocate each line to the original axes or a twin axes ('idx')
            # TODO: Encode this scheme into some general helper function
            # NOTE: We use the kwopts retrieved here for legend location
            idx = 0  # "axes number", i.e. 0 for original and 1 for twin
            idxs = []  # axes numbers
            units = set()
            cycle = None
            lines = []
            for name, kwvar, kwopts, kwplot, kwaxes in ispecs:
                line = _get_scalar_data(exp, name, **kwvar)
                lines.append((line, kwaxes, kwplot, kwopts))
                if units and line.climo.units_label not in units:
                    idx += 1
                idxs.append(idx)
                units.add(line.climo.units_label)
                cycle = cycle or kwopts.get('cycle', None)

            # Use colored property cycler if requested parameters have different units
            # TODO: Make this work for multiple series, not just variables
            if max(idxs) > 1:
                raise ValueError('No more than two different units.')
            alty = max(idxs) > 0
            if cycle is None:
                cycle = colors if alty else blacks
            cycle = pplt.Cycle(cycle)
            cycle = itertools.cycle(cycle)

            # Plot the line data, possibly on alternate axes
            for idx, (line, kwaxes, kwplot, kwopts) in zip(idxs, lines):
                if not line.size:
                    warnings.warn(f'Empty line {line!r}.')
                    continue

                # Multiple series or variables on the same plot
                ips = []
                yax = ax
                props = next(cycle)
                ycolor = None
                if alty:
                    ycolor = props['color']
                    if idx > 0:
                        if hasattr(ax, '_alt_child'):
                            yax = ax._alt_child
                        else:
                            yax = ax._alt_child = ax.alty()

                # Plot the requested line(s)
                # TODO: Handle this more intelligently, build arbitrary grids of plots
                label = str(exp) if len(exps) > 1 else getattr(line.climo, 'long_name', '')  # noqa: E501
                title = titles[i] if titles else str(exp) if len(args) > 1 else None
                hs = yax.plot(params, line, label=label, **props)
                for h in hs:
                    h.update(kwplot)
                ips.append(hs[0])

                # Plot line where x-axis equals this variable
                if kwopts.get('equality', None):
                    p = _plot_equality(yax, params, line)
                    ips.append(p)
                ps.extend(ips)

                # Format axes with potentially experiment-specific settings
                kwformat = {
                    'ycolor': ycolor,
                    'ylabel': line.climo.short_label,
                    'yformatter': line.climo.axis_formatter,
                    'title': title,
                    **kwaxes
                }
                yax.format(**kwformat)

        # Add vertical lines indicating experiment locations
        if refhighlight:
            _plot_explines(ax, params, exp.ireference_loaded)
        if kwopts.get('zeroline', None):
            _plot_reflines(ax, hlines=0)

        # Draw legend for different series and variables
        # WARNING: Important to draw legend on *original* axes because 1) that way
        # inner legend is placed above twin axes lines and 2) otherwise for outer
        # panels the 'panelpad' argument is ignored.
        iloc = kwopts.get('loc', loc)
        incol = kwopts.get('ncol', ncol)
        iorder = kwopts.get('order', ordr)
        ispace = kwopts.get('space', space)
        if iloc is not None:
            if isinstance(iloc, str):
                b, iloc = re.match(r'\A(fig(?:ure)?_)?(.*)\Z', iloc).groups()
            else:
                b = False
            if b:
                ps = ps[::2] + ps[1::2]  # WARNING: kludge for extratropics plot
                handles[iloc] = handles.get(iloc, []) + ps
            else:
                ax.legend(
                    ps, ncol=incol, loc=iloc, order=iorder, space=ispace,
                    frame=frame, handlelength=handlelength,
                )

        # Format axes
        xlabel = params.climo.long_label
        xlocator = getattr(xlocator, 'tolist', lambda: xlocator)()
        xminorlocator = getattr(xminorlocator, 'tolist', lambda: xminorlocator)()
        if xscale == 'log':
            xlocator = xlocator or 'log'
            xminorlocator = xminorlocator or 'logminor'
        if params.scheme == 'hs2':  # WARNING: kludge for uniform damping scheme
            xlabel = re.sub('reference ', '', xlabel)
        kw = {
            'xgrid': False,
            'xlim': (params.min().item(), params.max().item()),
            'xscale': xscale,
            'xlabel': xlabel,
            'xlocator': xlocator or params.data,
            'xminorlocator': xminorlocator,
            'xformatter': xformatter,
            'suptitle': suptitle,
            **kwargs,
        }
        ax.format(**kw)

        # Format as_param axes
        if xax is not None:
            xlabel = params_alt.climo.long_label
            if params.scheme == 'hs2':  # WARNING: kludge for uniform damping scheme
                xlabel = re.sub('reference ', '', xlabel)
            kw = {
                'xgrid': False,
                'xlim': (params_alt.min().item(), params_alt.max().item()),
                'xscale': xscale,
                'xlabel': xlabel,
                'xlocator': xlocator_alt or xlocator or params_alt.data,
                'xminorlocator': xminorlocator_alt or xminorlocator,
                'xformatter': xformatter_alt,
                **kwargs,
            }
            xax.format(**kw)

        # Hide duplicate *x* labels when subplot is in top/bottom of column
        nrow, _, _, _ = ax.get_subplotspec()._get_subplot_geometry()
        row1, row2, _, _ = ax.get_subplotspec()._get_subplot_rows_columns()
        for iax in (ax, xax):
            if not iax:
                continue
            axis = iax.xaxis
            if (
                nrow > 1 and row1 == 0 and axis.get_label_position() == 'bottom'
                or nrow > 1 and row2 >= nrow - 1 and axis.get_label_position() == 'top'
            ):
                axis.set_major_locator(pplt.Locator('null'))
                axis.set_minor_locator(pplt.Locator('null'))
                axis.label.set_text('')

    # Draw global figure legend
    for loc, ps in handles.items():
        fig.legend(
            ps, loc=loc, ncol=ncol, order=ordr, space=space,
            frame=frame, handlelength=handlelength,
        )

    # Return
    return fig, axs


@_savefig
def stacks(
    *args,
    spec=None,
    hlines=None,
    vlines=None,  # plot zero lines
    fill=False,  # do filled plots, or just lines?
    ncol=1,
    loc='b',
    frame=False,
    base=0,
    sep=0,
    tick=None,
    tickminor=None,
    transpose=False,
    length=None,
    **kwargs
):
    """
    Plot "stacks" showing line plots of all experiments on the same subplot.
    Separate variables and experiments can be shown on separate subplots.

    Parameters
    ----------
    *args : `experiment.Experiment`
        The experiments.
    spec : str, (str, dict), or list thereof, optional
        The variable spec(s) to be plotted. Parsed by `_parse_speclists`.
    transpose : bool, optional
        If ``True`` the dependent variable is plotted on the *y* axis.

    Other parameters
    ----------------
    **kwargs
        Passed to `~proplot.subplots.subplots` or  `~proplot.axes.Axes.format`.

    Todo
    ----
    Add helper function that generates figure and subplots for plotting stuff
    on *each subplot*. Then use this for heat budget stuff.
    """
    if spec is None:
        raise ValueError('Must specify variable for plotting.')
    if not all(isinstance(_, experiment.Experiment) for _ in args):
        raise ValueError('Input must be Experiment instances.')
    cmaps = {}  # name, experiment pairs
    specs = _parse_speclists(spec, prefer_single_subplot=False)

    # Properties
    # nsims = max(map(len, args))
    nvars = len(specs)
    nseries = len(args)
    nplots = nseries * nvars
    fig, axs, kwargs = _subplot_per_experiment(nplots, **kwargs)
    if sep > 0:
        tick = tick or sep
        tickminor = tickminor or tick / 4

    # Iterate through variables (first priority) and exp (second priority)
    for i, (j, k) in enumerate(itertools.product(range(nseries), range(nvars))):
        ax = axs[i]
        exp = args[j]
        ispecs = specs[k]
        nsims = len(exp)

        # Get offsets for lines and make sure y-coordinates for 'base' experiments match
        ibase = base + sep * ((nsims - nseries) // 2)  # keep centering
        zorder = 10 + nseries  # max zorder
        cmap = _get_item('cmap', ispecs, idx=3) or exp.colormap_line
        cmap = pplt.Colormap(cmap)
        colors = pplt.Colors(cmap, N=nsims)
        cmaps[cmap.name] = (exp, colors)  # record for drawing colorbar
        # cmaps[cmap.name] = (exp, cmap)  # record for drawing colorbar

        # Loop through experiments and variables
        hs = []
        for spec, m in itertools.product(ispecs, range(nsims)):
            # Get the line(s)
            name, kwvar, kwopts, kwplot, kwaxes = spec
            kwplot = kwplot.copy()
            kwplot.pop('cmap', None)
            data = exp[m]
            lim = kwaxes.get('ylim' if transpose else 'xlim', None)
            y, err = _get_1d_data(data, name, lim=lim, **kwvar)
            x = y.coords[y.dims[0]]

            # Plot experiments from top to bottom
            c = colors[m]
            kw = {'label': y.climo.long_label, 'zorder': zorder + m}
            offset = ibase + (nsims - m - 1) * sep
            if not fill:
                xy = (y + offset, x) if transpose else (x, y + offset)
                kw = {'color': c, 'markerfacecolor': c, 'markeredgecolor': c, **kw, **kwplot}  # noqa: E501
                h, *_ = ax.plot(*xy, **kw)
            elif y.ndim == 1:
                attr = 'fill_betweenx' if transpose else 'fill_between'
                kw = {'facecolor': c, 'edgecolor': 'k', **kw, **kwplot}
                h = getattr(ax, attr)(offset, y + offset, **kwplot)
            else:
                raise ValueError(f'For {fill=}, multiple lines not allowed.')
            if err is not None:
                ax.errorbar(  # TODO: use proplot features
                    y + offset, fmt='none',
                    ecolor='k', elinewidth=1, zorder=100,
                    capsize=4, capthick=1, clip_on=False,
                    **{'xerr' if transpose else 'yerr': err},
                )
            if m == nsims // 2:
                hs.append(h)

        # Add "special" spectral lines
        _plot_reflines(ax, vlines=vlines, hlines=hlines)
        if name == 'spectral':
            _plot_spectral(ax)

        # Draw legend or title
        if len(hs) == 1:
            label = hs[0].get_label()
            ax.set_title(label[:1].upper() + label[1:])
        else:
            ax.legend(hs, ncol=ncol, loc=loc, frame=frame)

        # Set up tick labels and stuff
        # NOTE: This generally won't make sense unless variables drawn
        # simultaneously in same subplot have similar properties.
        if sep == 0:
            ylocator = tick
            yminorlocator = tickminor
        else:
            sep = max([sep, tick])
            lim = (-sep * nsims * 2 + ibase, ibase + sep * nsims * 5)
            ylocator = pplt.arange(*lim, tick)
            yminorlocator = pplt.arange(*lim, tickminor)
        xs, ys = 'yx' if transpose else 'xy'
        kw = {
            # 'xlabel': x.climo.long_label,
            # 'ylabel': y.climo.long_label,
            xs + 'scale': x.climo.axis_scale,
            xs + 'label': x.climo.short_label,
            xs + 'formatter': x.climo.axis_formatter,
            ys + 'scale': y.climo.axis_scale,
            ys + 'label': y.climo.short_label,
            ys + 'locator': ylocator,
            ys + 'minorlocator': yminorlocator,
            ys + 'formatter': y.climo.axis_formatter,
            **kwaxes,
            **kwargs,
        }
        ax.format(**kw)

    # Colorbar
    # TODO: Fix issue where LinearSegmentedNorm is not applied to colors
    for exp, colors in cmaps.values():
        ticklabels = [
            f'{_.item():.3f}'.rstrip('0').rstrip('.') for _ in exp.parameters_loaded
        ]
        fig.colorbar(
            colors,
            loc='b',
            length=length,
            label=exp.long_label,
            ticklabels=ticklabels,
            values=np.arange(len(ticklabels)),
            # values=exp.parameters_loaded,
        )

    return fig, axs


@_savefig
def xsections(
    *args,
    mode='yz',
    contourf=None,
    contour=None,
    quiver1=None,
    quiver2=None,
    normalize=False,
    tropopause=False,
    curvature=False,
    surf=False,
    ubar=False,
    precision=3,
    lo=0.5,
    hi=99.5,
    ncol=1,
    loc=None,
    labelloc='ll',
    length=1,
    **kwargs
):
    """
    Generate grids of latitude-height or time-latitude cross-sections.
    It can draw filled contours, line contours, tropopause lines, quiver
    arrows, and stippling or hatching to show significance.

    Parameters
    ----------
    *args : `xarray.Dataset`
        The dataset(s). Often this is an expanded `experiment.Experiment`.
    mode : {'yz', 'ty', 'cy', 'ky', 'ck'}
        The plotting cross-section mode. Each character stands for a dimension.
    contourf, contour : str, optional
        Variables for filled and line contours. Parsed by `_parse_speclists`.
    quiver1, quiver2 : (str, str), optional
        Variables to use for quiver arrows. Parsed by `_parse_speclists`.
    normalize : bool, optional
        How to handle automatic level generation. If ``True`` filled contours
        are scaled to within -1 to 1 depending on `extend` with text boxes
        in the corner of each subplot indicating the range. Otherwise levels
        are generated from the data range from all experiments.
    tropopause, curvature : bool, optional
        Whether to draw the AMS or curvature tropoapuse, respectively.
    surf : bool, optional
        Whether to draw the surface contours for isentropic plots.
    precision : int, optional
        Precision for number annotations.
    ncol : int, optional
        Number of legend columns.
    loc : str, optional
        Legend location.
    length : float, optional
        Colorbar location.
    **kwargs
        Passed to `_get_xyprops` and `_subplot_per_simulation`.

    Returns
    -------
    fig, axs
        The figure and axes.
    """
    contourfs, contours, quiver1s, quiver2s = _parse_speclists(
        contourf, contour, quiver1, quiver2, prefer_single_subplot=False,
    )
    formatter = pplt.Formatter('simple', precision=precision)
    line_kw = {'ls': '-', 'color': 'red9', 'lw': 2, 'zorder': 2.5}
    surf_kw = {'ls': ':', 'color': 'k', 'lw': 1, 'zorder': 2.5}
    trop1_kw = {'lw': 2, 'color': 'gray5', 'zorder': 2.5, 'dashes': (1, 1)}
    trop2_kw = {'lw': 2, 'color': 'gray7', 'zorder': 2.5, 'dashes': (1, 1)}
    legend_kw = {'ncol': ncol, 'frame': False, 'center': ncol == 1, 'order': 'FC'[ncol == 1]}  # noqa: E501
    colorbar_kw = {'length': length, 'pad': 0.8}
    quiver_kw = {
        'width': 0.003, 'angles': 'uv', 'scale': 10, 'scale_units': 'height',
        'minlength': 0, 'headwidth': 3, 'headlength': 5, 'headaxislength': 3,
    }

    # Get the figure
    # TODO: Titles for comparing pressure-latitude levels across experiments
    nsims = len(args)
    nvars = len(contourfs)
    video = kwargs.get('fig', None) and kwargs.get('axs', None)
    vertical = kwargs.get('vertical', False)
    fig, axs, kwargs = _subplot_per_simulation(args, nvars=nvars, **kwargs)

    # Draw stuff
    plots = _dict_of_lists()
    for i, (j, k) in enumerate(itertools.product(range(nvars), range(nsims))):
        # Axes properties and coordinates
        ax = axs[i]
        data = args[k]
        icontourfs = contourfs[j]
        icontours = contours[j]
        iquiver1s = quiver2s[j]
        iquiver2s = quiver1s[j]
        dims, kwfmt = _get_xyprops(data, mode=mode, **kwargs)
        xlim = kwfmt.pop('xlim', None)
        ylim = kwfmt.pop('ylim', None)
        if xlim is not None:
            ax.format(xlim=xlim)  # then auto-inbounds
        if ylim is not None:
            ax.format(ylim=ylim)  # then auto-inbounds

        # Normalized contours when levels is an integer
        # NOTE: For now only draw colorbar for *first* contourf in
        # sublist. Use subsequent ones for things like forcing shading.
        for contourf, kwvar, kwopts, kwplot, kwaxes in icontourfs:
            # Get contour data
            if contourf is None:
                continue
            kwplot = kwplot.copy()
            levels = kwplot.pop('levels', 20)
            extend = kwplot.pop('extend', 'both')
            dcontourf = _get_2d_data(data, contourf, **kwvar)
            if dcontourf.dims != dims:
                raise RuntimeError(f'Dims {dcontourf.dims=} and {dims=} disagree.')

            # Normalize data
            if normalize and not np.iterable(levels):
                dcontourf, levels, label = _normalize_contourf(dcontourf, N=levels, lo=lo, hi=hi, extend=extend)  # noqa: E501
                label = label + ' ' + dcontourf.climo.units_label
                ax.format(**{labelloc + 'title': label})

            # Plot filled contours
            # TODO: Remove kludge where we assume first contourf is mappable,
            # subsequent are legend entries.
            dcontourf.attrs['long_name'] = kwplot.get('label', dcontourf.climo.long_name)  # noqa: E501
            kwplot['label'] = dcontourf.climo.long_label
            kwplot['levels'] = levels
            kwplot['extend'] = extend
            kwplot.setdefault('corner_mask', True)
            plots.add((j, contourf, 'contourf'), (ax, dcontourf, kwplot))

            # Add panels showing means or slices along side!
            for side in 'lrbt':
                # Make panel
                panel = kwopts.get(side + 'panel', None)
                panel_kw = (kwopts.get(side + 'panel_kw', None) or {}).copy()
                if panel is None:
                    continue
                visible = panel_kw.pop('visible', True)
                width = panel_kw.pop('width', None)
                share = panel_kw.pop('share', None)
                pax = ax.panel_axes(side, space=0, share=share, width=width)
                if not visible:  # account for colorbars
                    pax.set_visible(False)
                    continue
                # Get line data
                kw = {dcontourf.dims[int(side in 'lr')]: panel, **kwvar}
                dplot, _ = _get_1d_data(data, contourf, **kw)
                coords = dplot.coords[dplot.dims[0]]
                if panel_kw.pop('normalize', None):
                    dplot /= np.max(np.abs(dplot.data))
                # Plot data
                # TODO: Remove kludges
                # TODO: Add proplot 'plotx' method similar to 'fill_betweenx'
                # color = (kwplot.get('cmap', pplt.rc.cmap), 0.8)
                color = 'red9'
                cmd = pax.plotx if side in 'lr' else pax.plot
                cmd(coords, dplot, color=color, linewidth=0.9)
                cmd = pax.fill_betweenx if side in 'lr' else pax.fill_between
                cmd(coords, dplot, color=color, alpha=0.3, zorder=0.5)
                x, y = 'yx' if side in 'lr' else 'xy'
                kw = {
                    'grid': False,
                    x + 'lim': ylim if side in 'lr' else xlim,  # yreverse fails
                    x + 'scale': kwfmt.get(x + 'scale', 'linear'),
                }
                kw.update(panel_kw)
                kw.setdefault('xformatter', 'null')
                kw.setdefault('yformatter', 'null')
                pax.format(**kw)

        # Line contours, optionally more than one
        # Draw contour2 *first* so contour1 has higher zorder
        for contour, kwvar, kwaxes, kwplot, kwopts in icontours:
            # Get contour data
            if contour is None:
                continue
            kwplot = kwplot.copy()
            levels = kwplot.pop('levels', None)
            dcontour = _get_2d_data(data, contour, **kwvar)
            if dcontour.dims != dims:
                raise RuntimeError(f'Dims {dcontour.dims=} and {dims=} disagree.')

            # Plot contours
            dcontour.attrs['long_name'] = kwplot.get('label', dcontour.climo.long_name)
            kwplot['label'] = dcontour.climo.long_label
            kwplot['levels'] = levels
            kwplot.setdefault('labels', True)
            kwplot.setdefault('colors', 'k')
            kwplot.setdefault('linewidths', 0.7)
            plots.add((j, contour, 'contour'), (ax, dcontour, kwplot))

        # Quiver arrows
        for quiver1, quiver2 in zip(iquiver1s, iquiver2s):
            # Retrieve data
            if mode != 'yz':
                raise ValueError('Invalid plot mode for quiver arrows.')
            qdatas = []
            bases = []
            for quiver, kwvar, kwopts, kwplot, kwaxes in (quiver1, quiver2):
                qkey = kwopts.get('key', 1)  # whether to toggle quiver key
                base = kwopts.get('base', 1)  # default to unit
                qdata = _get_2d_data(data, quiver, **kwvar)
                if qdata.dims != dims:
                    raise RuntimeError(f'Dims {qdata.dims=} and {dims=} disagree.')
                xstride = kwopts.get('xstride', 1)
                ystride = kwopts.get('ystride', 1)
                qdata = qdata.climo.dequantify() / base
                qdata = qdata[::ystride, ::xstride]
                qdata[np.isnan(qdata)] = 0
                qdatas.append(qdata)
                bases.append(base)

            # Draw arrows
            kwplot = quiver_kw.copy()
            kwplot['hlabel'] = f'{formatter(bases[0])} {qdatas[0].climo.units_label}'
            kwplot['vlabel'] = f'{formatter(bases[1])} {qdatas[1].climo.units_label}'
            plots.add((j, quiver1, quiver2, 'quiver'), (ax, *qdatas, kwplot))

        # Tropopause
        # NOTE: The 'second' tropopause should show up darker than first
        if tropopause or curvature:
            if mode != 'yz':
                raise ValueError('Invalid plot mode for tropopause.')
            suffixes = '12' if 'pair' in data.dims else ('',)
            for suffix, trop_kw in zip(suffixes, (trop2_kw, trop1_kw)):
                name = 'ctrop' if curvature else 'trop'
                dtrop, _ = _get_1d_data(data, name + suffix)
                kwplot = {'label': dtrop.climo.long_label, **trop_kw}
                plots.add((j, name + suffix, 'plot'), (ax, dtrop, kwplot))

        # Draw surface contours
        if surf:
            if data.climo.cf.vertical_type != 'temperature':
                raise ValueError('Invalid plot mode for surface contours.')
            nx = data['x'].values.size
            lat = data['lat'].values
            spctiles = [5, 20, 50, 80, 95]
            for pctile in spctiles:
                idx = np.round(pctile * nx / 100).astype(int)
                dtheta = data['slth'].isel(x=idx)
                kwplot = {'label': 'surface isopleths', **surf_kw}
                plots.add((j, 'slth', ax.plot), (ax, dtheta, kwplot))

        # Draw zonal wind line
        if ubar:
            # nlat = data['lat'].size
            # nlev = data['lev'].size
            if mode != 'cy':
                raise ValueError('Invalid mode for zonal wind line.')
            u = data['u'].sel(lev=kwvar['lev']).values.squeeze()
            lat = data['lat']
            if u.ndim != 1:
                raise ValueError('Zonal wind climate should be 1-dimensional.')
            ax.plot(u, lat, zorder=2.5, **line_kw)
            ax.axvline(0, color='gray8', zorder=2.5, ls=':', lw=2)

    # Execute queued plotting commands
    # NOTE: This is done so we can determine default levels based on *multiple* datasets
    # using proplot's auto-level generator. Also simplifies colorbar and legend
    # generation. Unique dictionary keys correspond to unique plot elements.
    hs = _dict_of_lists()
    mappables = _dict_of_lists()
    for key, commands in plots.items():
        # Get default levels for contour plots from *all* experiment data
        idx, varname, funcname = key
        if not normalize and funcname in ('contour', 'contourf'):
            ax = commands[0][0]  # to sample the axis limits
            args = [arg for _, *args, _ in commands for arg in args]
            kwargs = commands[0][2].copy()  # identical for all experiments
            if all(arg.shape == args[0].shape for arg in args):
                x, y = (args[0].coords[dim] for dim in args[0].dims)
            else:
                x = y = None  # not restricted to in-bounds
            vmin, vmax, _ = ax._parse_level_lim(x, y, *args)
        else:
            vmin = vmax = None

        # Iterate through each experiment for this given variable spec
        for ax, *args, kwargs in commands:
            # Call plotting function after attenuating special keys
            kwargs = kwargs.copy()
            label = kwargs.pop('label', None)
            hlabel = kwargs.pop('hlabel', None)  # used for quiverkey
            vlabel = kwargs.pop('vlabel', None)
            cbar_kw = colorbar_kw.copy()
            for key in ('creverse', 'clocator', 'cminorlocator', 'cformatter'):
                cbar_kw.setdefault(key[1:], kwargs.pop(key, None))
            if vmin is not None:
                kwargs.setdefault('vmin', vmin)
            if vmax is not None:
                kwargs.setdefault('vmax', vmax)
            h = getattr(ax, funcname)(*args, **kwargs)

            # Get colorbar or legend handles and draw quiver keys
            # TODO: Allow quiver key placement in one of standard title locations
            # and add optional outline or background just like inset titles.
            if funcname == 'plot':
                # Line handle
                hs.add(key, (ax, h[0]))
            elif funcname == 'contour' or funcname == 'contourf' and 'colors' in kwargs:
                # Contour handle
                h = h.legend_elements()[0][-1]
                h.set_label(label)
                hs.add(ax, h)
            elif funcname == 'contourf':
                # Mappable object
                cbar_kw['label'] = label
                if cbar_kw['locator'] is None and isinstance(h.norm._norm, pplt.DivergingNorm):  # noqa: E501
                    cbar_kw['locator'] = pplt.Locator('maxn', symmetric=True)
                mappables.add(ax, (h, cbar_kw))  # will be added to panel
            elif funcname == 'quiver' and qkey:
                # Quiver key. Position of each arrow is its *central* position, so left
                # point of the horizontal arrow is 0.5 * scale axes units to left
                kw = {'lw': 2, 'color': 'w', 'zorder': 5}
                hkw = {'labelpos': 'E', 'angle': 0, 'coordinates': 'axes'}
                vkw = {'labelpos': 'N', 'angle': 90, 'coordinates': 'axes'}
                fontsize = 8
                wscale = pplt.units(1, 'em', 'ax', ax=ax, fontsize=fontsize, width=True)
                hscale = pplt.units(1, 'em', 'ax', ax=ax, fontsize=fontsize, width=False)   # noqa: E501
                scale = kwargs.get('scale', 1)
                width, height = ax.get_size_inches()
                xpos = 1 - scale - 0.3 * wscale * len(hlabel.strip('$'))
                ypos = 1 - scale - 1.8 * hscale
                hargs = (h, xpos, ypos, 1)
                vargs = (h, xpos - scale * 0.5, ypos + scale * 0.5, 1)
                ax.quiverkey(*hargs, '', **kw, **hkw)
                ax.quiverkey(*vargs, '', **kw, **vkw)
                hk = ax.quiverkey(*hargs, hlabel, zorder=6, **hkw)
                vk = ax.quiverkey(*vargs, vlabel, zorder=6, **vkw)
                for ik in (hk, vk):
                    ik.text.update({
                        'color': 'k',
                        'zorder': 5,
                        'path_effects': [
                            mpatheffects.Stroke(linewidth=kw['lw'], foreground='w'),
                            mpatheffects.Normal()
                        ]
                    })

            # Format axes
            # NOTE: Here we get coordinates from lat plotting command
            if not video:
                data = args[0]
                y = data.coords[data.dims[0]]
                x = data.coords[data.dims[1]]
                ax.format(
                    xlabel=x.climo.long_label,
                    ylabel=y.climo.long_label,
                    xformatter=x.climo.axis_formatter,
                    yformatter=y.climo.axis_formatter,
                    xtickminor=True, ytickminor=True,
                )

    # Colorbar for contourf elements
    if video or not mappables:
        pass
    elif nvars == 1:  # single variable group, single variable name
        for mappable, cbar_kw in tuple(mappables.values())[1]:
            fig.colorbar(mappable, loc='b', **cbar_kw)
    elif mappables:
        axs_cbar = axs[-1, :] if vertical else axs[:, -1]
        loc_cbar = loc or ('b' if vertical else 'r')
        for ax, imappables in mappables.items():
            if ax not in axs_cbar:
                continue
            for mappable, cbar_kw in imappables:
                ax.colorbar(mappable, loc=loc_cbar, **cbar_kw)

    # Legend for contour elements (always put on bottom, because has horizontal extent)
    # From: https://github.com/matplotlib/matplotlib/issues/11134
    if video or not hs:
        pass
    elif nvars == 1:  # single variable group, single variable name
        ihs = tuple(hs.values())[1]
        fig.legend(ihs, loc='b', **legend_kw)
    elif hs:
        center = axs.shape[1] // 2
        axs_leg = axs[-1, :] if vertical else axs[:, center]
        for ax, ihs in hs.items():
            if ax not in axs_leg:
                continue
            ax.legend(ihs, loc='b', **legend_kw)

    # Reapply settings
    axs.format(xlim=xlim, ylim=ylim, **kwfmt)
    for ax in axs:  # WARNING: kludge
        ylim = ax.get_ylim()
        if ylim[1] > ylim[0] and kwfmt.get('yreverse'):
            ax.format(ylim=ylim[::-1])

    return fig, axs


# Public utilities go down here
def forcing_patch(exp):
    """
    Return a suitable forcing patch for the experiment.
    """
    if len(exp.names) == 1:
        return None
    mode = exp.names[1]
    if mode in ('vortex',):
        return ('forcing2', {
            'levels': np.asarray([-1e10, -0.2, 1e10]),
            'colors': ['#00000044', '#00000000'],
            'label': '-0.2K/day forcing threshold',
        })
    elif mode in ('tropical', 'arctic',):
        return ('forcing2', {
            'levels': np.asarray([-1e10, 0.2, 1e10]),
            'colors': ['#00000000', '#00000044'],
            'label': '0.2K/day forcing threshold',
        })
    else:
        return None


def mirror_levels(*args, zero=False):
    """
    Mirror and concatenate arbitrary lists of values.

    Parameters
    ----------
    *args : iterable
        Arbitrary iterables.
    zero : bool, optional
        Whether to include the zero.
    """
    levels = np.concatenate(args, axis=0).tolist()
    if not zero:
        try:
            levels.remove(0)
        except ValueError:
            pass
    levels = np.array(levels)
    values = np.unique(np.array([*(-levels[::-1]), *levels])).tolist()
    return values
