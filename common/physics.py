#!/usr/bin/env python3
"""
Register CF variables, transformations, and derivations in climopy.
Lightly leverages a few pre-defined climopy variables.
"""
# WARNING: keep_attrs fails for unary operations -data, +data, and abs(data)
# NOTE: Important question: Why not just add attributes in the conversions
# below and forego the separate CFVariable interface? Because do not want to
# embed metadata definitions in a *derivation*, e.g. if we are loading the
# data externally instead of calculating it. Extremely convenient to have a
# consolidated, separate database from which "attributes" can be retrieved.
import re
import warnings

import metpy  # noqa: F401
import numpy as np
from icecream import ic  # noqa: F401

import climopy as climo
from climopy import const, ureg, vreg
from climopy.internals.warnings import ClimoPyWarning

# Ranges for bulk gradients and fluxes
SURFACE = 900  # for climate sensitivity
LAT_LIM = (20, 70) * ureg.deg
LEV_LIM = (300, 900) * ureg.hPa
LEV_LIM = (300, 1013.25) * ureg.hPa
LEV_LIM = (200, 1013.25) * ureg.hPa
LEV_LIM = (200 * ureg.hPa, None)

# Custom constants (see paper for citations)
Q_2XCO2 = 3.4 * ureg.watts / ureg.meter ** 2
Q_4XCO2 = 6.8 * ureg.watts / ureg.meter ** 2
C_DELTA = (const.cp / const.g).to('J m^-2 K^-1 Pa^-1')
C_COLUMN = (const.cp * const.p0 / const.g).to('J m^-2 K^-1')

# Components for different variables
PARTS_LORENZ = {
    'g': ('gpe', 'gpm'),
    'p': ('pe', 'pm'),
    'c': ('cpmkm', 'cpeke'),
    'k': ('km', 'ke'),
    'd': ('dkm', 'dke')
}
PARTS_FLUX = {  # used in yzparam
    'mf': ('ucos', 'v'),
    'hf': ('t', 'v'),
    'gf': ('z', 'v'),
    'dsef': ('dse', 'v'),
    'pvf': ('pv', 'v'),  # isentropic only
    'qgpvf': ('qgpv', 'v'),  # isobaric only
    'vhf': ('pt', 'omega'),
    'vdsef': ('dse', 'omega'),  # should be idential to vhf?
}


with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=ClimoPyWarning)

    # Coordinate variables and aliases
    vreg.alias('longitude', 'lon')
    vreg.alias('latitude', 'lat')
    vreg.alias('meridional_coordinate', 'y')
    vreg.alias('cosine_latitude', 'cos')
    vreg.alias('coriolis_parameter', 'cor')
    vreg.alias('beta_parameter', 'beta')
    vreg.alias('reference_height', 'z0')
    vreg.alias('reference_density', 'rho0')
    vreg.alias('reference_potential_temperature', 'pt0')
    vreg.define('lev', 'vertical level')  # 'appropriate' units are ambiguous
    vreg.define('plev', 'pressure', 'hPa', aliases='pres')
    vreg.define('zlev', 'height', 'km')
    vreg.define('time', 'integration time', 'days', 'time')  # for equinoctial runs
    vreg.define('period', 'initial time', 'days', 'time')
    vreg.define('lag', 'lag', 'days')  # stored in days?
    vreg.define('eof', 'EOF number', '')
    vreg.define('phase', 'phase speed', 'm / s')
    vreg.define('count', 'count', '')
    vreg.define('rate', 'rate', 'days^-1')
    vreg.define('timescale', 'timescale', 'days')
    vreg.define('scale', 'length scale', '1000km')
    vreg.define('corr', 'correlation coefficient', '', 'correlation')
    # vreg.define('k', 'zonal wavenumber', '')
    # vreg.define('f', 'frequency', 'days')  # i.e. temporal wavenumber

    # Basic state quantities
    vreg.define('u', 'zonal wind', 'm / s', parents='momentum')  # noqa: E501
    vreg.define('v', 'meridional wind', 'm / s', parents='momentum')  # noqa: E501
    vreg.define('t', 'air temperature', 'K', parents='energy')
    vreg.define('z', 'geopotential height', 'km', parents='energy')  # noqa: E501
    vreg.define('w', 'vertical wind', 'cm / s', parents='momentum')
    vreg.define('q', 'specific humidity', 'g / kg')
    vreg.define('rh', 'relative humidity', '%')
    vreg.define('alb', 'surface albedo', '%')
    vreg.define('ts', 'surface temperature', 'K', parents='t')
    vreg.define('pt', 'potential temperature', 'K', parents='t')
    vreg.define('pv', 'potential vorticity', 'PVU')
    vreg.define('rho', 'density', 'kg / m3')
    vreg.define('exner', 'Exner function', '')
    vreg.define('dse', 'static energy', symbol='DSE', parents='energy')  # parent units
    vreg.define('mse', 'moist static energy', symbol='MSE', parents='energy')
    vreg.define('qgpv', 'quasi-geostrophic potential vorticity', 'VU')
    vreg.define('omega', 'omega vertical wind', 'Pa / s')
    vreg.define('sigma', 'isentropic density', 'kg K^-1 m^-2')  # = dp / dtheta * g
    vreg.define('ps', 'surface pressure', parents='plev', standard_name='surface_air_pressure')  # noqa: E501
    vreg.define('slp', 'sea-level pressure', parents='plev', standard_name='air_pressure_at_mean_sea_level')  # noqa: E501
    vreg.define('zsurf', 'surface geopotential height', 'km')
    vreg.define('deltat', 'deviation from $T^e$', 'K', symbol=r'\Delta T')
    vreg.define('teq', 'equilibrium temperature', 'K')
    vreg.define('pteq', 'equilibrium potential temperature', 'K')

    # Mass budget terms
    vreg.define('mpsi', 'mass streamfunction', 'Tg / s')
    vreg.define('mpsie', long_prefix='eddy-induced', parents='mpsi')
    vreg.define('mpsiresid', long_prefix='residual', parents='mpsi')
    vreg.define('ve', long_prefix='eddy-induced', parents='v')
    vreg.define('vresid', long_prefix='residual', parents='v')
    vreg.define('omegae', long_prefix='eddy-induced', parents='omega')
    vreg.define('omegaresid', long_prefix='residual', parents='omega')

    # Thermodynamic budget terms
    # Very similar but these are all thermal so unintegrated units are K / s
    vreg.define('heating', 'heating', 'K / day', 'heating', parents='energy_flux')
    vreg.define('cooling', 'cooling', 'K / day', 'cooling', parents='energy_flux')
    vreg.define('adiabatic', long_prefix='adiabatic', parents='heating')
    vreg.define('iadiabatic', long_prefix='adiabatic', parents='cooling')
    vreg.define('eadiabatic', long_prefix='eddy', parents='adiabatic')
    vreg.define('ieadiabatic', long_prefix='eddy', parents='iadiabatic')
    vreg.define('madiabatic', long_prefix='mean', parents='adiabatic')
    vreg.define('eheating', long_prefix='eddy', parents='heating')
    vreg.define('mheating', long_prefix='overturning circulation', parents='heating')
    vreg.define('tdt', long_prefix='diabatic', parents='heating')
    vreg.define('itdt', long_prefix='diabatic', parents='cooling')
    # vreg.define('tdt', long_prefix='diabatic', parents='heating')
    # vreg.define('itdt', long_prefix='diabatic', parents='cooling')
    vreg.define('rheating', long_prefix='residual', parents='heating')
    vreg.define('irheating', long_prefix='residual', parents='cooling')
    vreg.define('diabatic', long_prefix='total diabatic', parents='heating')
    vreg.define('idiabatic', long_prefix='total diabatic', parents='cooling')
    vreg.define('aheating', long_prefix='all', parents='heating')
    vreg.define('iaheating', long_prefix='all', parents='cooling')
    vreg.define('dheating', long_prefix='differential', parents='heating')
    vreg.define('diss', long_prefix='dissipative', short_name='dissipation', parents='heating')  # noqa: E501
    vreg.define('forcing', 'forcing perturbation', reference=0, symbol='Q', parents='heating')  # noqa: E501
    vreg.define('iforcing', 'negative forcing perturbation', reference=0, symbol='Q', parents='cooling')  # noqa: E501
    vreg.define('rforcing', 'net diabatic heating', parents='heating')

    # Momentum budget terms
    # NOTE: acceleration already defined
    vreg.define('a', 'net acceleration', parents='acceleration')
    vreg.define('torque', 'eastward torque', parents='acceleration')
    vreg.define('torquee', long_prefix='eddy-induced', parents='torque')
    vreg.define('torqueresid', long_prefix='residual', parents='torque')
    vreg.define('drag', 'drag', short_name='drag', parents='acceleration')
    vreg.define('idrag', 'inverse drag', short_name='drag', parents='acceleration')
    vreg.define('vdt', long_prefix='meridional', parents='drag')
    vreg.define('udt', long_prefix='zonal', parents='drag')
    vreg.define('iudt', long_prefix='inverse zonal', parents='drag')

    # Lorenz energy storage terms
    vreg.define('lorenz', 'Lorenz energy budget')
    vreg.define('ke', 'eddy kinetic energy', symbol='K_E', parents=('energy', 'lorenz'))  # noqa: E501
    vreg.define('km', 'mean kinetic energy', symbol='K_M', parents=('energy', 'lorenz'))  # noqa: E501
    vreg.define('k', 'kinetic energy', symbol='K_M + K_E', parents=('energy', 'lorenz'))  # noqa: E501
    vreg.define('pe', 'eddy potential energy', symbol='P_E', parents=('energy', 'lorenz'))  # noqa: E501
    vreg.define('pm', 'mean potential energy', symbol='P_M', parents=('energy', 'lorenz'))  # noqa: E501
    vreg.define('p', 'potential energy', symbol='P_M + P_E', parents=('energy', 'lorenz'))  # noqa: E501
    vreg.define('km_tropic', long_prefix='barotropic', parents='km')
    vreg.define('km_clinic', long_prefix='baroclinic', parents='km')
    vreg.define('ke_tropic', long_prefix='barotropic', parents='ke')
    vreg.define('ke_clinic', long_prefix='baroclinic', parents='ke')

    # Lorenz energy flux terms
    vreg.define('c', 'potential-kinetic energy conversion', symbol='C(P_M + P_E, K_M + K_E)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('cpmkm', 'mean potential-kinetic energy conversion', symbol='C(P_M, K_M)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('ckmpm', 'mean kinetic-potential energy conversion', symbol='C(K_M, P_M)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('cpeke', 'eddy potential-kinetic energy conversion', symbol='C(P_E, K_E)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('ckepe', 'eddy kinetic-potential energy conversion', symbol='C(K_E, P_E)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('cpmpe', 'potential mean-eddy energy conversion', symbol='C(P_M, P_E)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('ckekm', 'kinetic eddy-mean energy conversion', symbol='C(K_E, K_M)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('gpm', 'mean potential energy generation', symbol='G(P_m)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('gpe', 'eddy potential energy generation', symbol='G(P_E)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('g', 'potential energy generation', symbol='G(P_M + P_E)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('dkm', 'mean kinetic energy dissipation', symbol='D(K_M)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('dke', 'eddy kinetic energy dissipation', symbol='D(K_E)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('d', 'kinetic energy dissipation', symbol='D(K_M + K_E)', parents=('energy_flux', 'lorenz'))  # noqa: E501
    vreg.define('cpe_gpe', 'EPE dissipation ratio', '')
    vreg.define('pe_gpe', 'EPE dissipation timescale', '')

    # Static energy fluxes
    # NOTE: Units for convergence reduce to m / s
    # dsef = climo.Descriptor('dry static energy flux', 'W 100hPa^-1 m^-1')
    # msef = climo.Descriptor('moist static energy flux', 'W 100hPa^-1 m^-1')
    vreg.alias('meridional_energy_flux', 'ef')
    vreg.define('dsef', 'static energy flux', parents='meridional_energy_flux')
    vreg.define('mdsef', long_prefix='mean', parents='dsef')
    vreg.define('edsef', long_prefix='eddy', parents='dsef')
    vreg.define('edsef_bulk', long_prefix='bulk', parents='edsef')
    vreg.define('cdsef', short_suffix='convergence', standard_units='W 100hPa^-1 m^-2', parents=('dsef', 'energy_flux'))  # noqa: E501
    vreg.define('cmdsef', long_prefix='mean', parents='cdsef')
    vreg.define('cedsef', long_prefix='eddy', parents='cdsef')

    # Heat fluxes
    # NOTE: Eddy flux equals zonal covariance. Spectral breakdowns are included here.
    vreg.define('hf', 'heat flux', 'K m / s', 'heat flux', parents='meridional_energy_flux')  # noqa: E501
    vreg.define('mhf', long_prefix='mean', parents='hf')
    vreg.define('ehf', long_prefix='eddy', parents='hf')
    vreg.define('chf', short_name='heating rate', long_suffix='convergence', standard_units='K / day', parents=('hf', 'energy_flux'))  # noqa: E501
    vreg.define('cmhf', long_prefix='mean', parents='chf')
    vreg.define('cehf', long_prefix='eddy', parents='chf')
    vreg.define('ehf_corr', long_prefix='eddy heat', parents='corr')
    vreg.define('ehf_bulk', long_prefix='bulk', parents='ehf')

    # Geopotential fluxes
    vreg.define('gf', 'geopotential flux', 'm^2 / s', 'geopotential flux', parents='meridional_energy_flux')  # noqa: E501
    vreg.define('mgf', long_prefix='mean', parents='gf')
    vreg.define('egf', long_prefix='eddy', parents='gf')
    vreg.define('cgf', short_name='expansion rate', long_suffix='convergence', standard_units='m / day', parents=('gf', 'energy_flux'))  # noqa: E501
    vreg.define('cmgf', long_prefix='mean', parents='cgf')
    vreg.define('cegf', long_prefix='eddy', parents='cgf')

    # Vertical heat flux
    vreg.define('vhf', 'vertical heat flux', 'K Pa / s')
    vreg.define('mvhf', long_prefix='mean', parents='vhf')
    vreg.define('evhf', long_prefix='eddy', parents='vhf')
    vreg.define('cvhf', short_name='heating rate', long_suffix='convergence', standard_units='K / day', parents=('vhf', 'energy_flux'))  # noqa: E501
    vreg.define('cmvhf', long_prefix='mean', parents='cvhf')
    vreg.define('cevhf', long_prefix='eddy', parents='cvhf')

    # Momentum fluxes
    vreg.alias('meridional_momentum_flux', 'mf')
    vreg.define('mmf', long_prefix='mean', parents='mf')
    vreg.define('emf', long_prefix='eddy', parents='mf')
    vreg.define('emf_cos', long_prefix='scaled', parents='emf')  # scaled by cosine
    vreg.define('emf_corr', parents='corr', long_prefix='eddy momentum')
    vreg.define('cmf', short_name='eastward torque', long_suffix='convergence', standard_units='m s^-1 day^-1', parents=('mf', 'acceleration'))  # noqa: E501
    vreg.define('cmmf', long_prefix='mean', parents='cmf')
    vreg.define('cemf', long_prefix='eddy', parents='cmf')

    # Eliassen Palm terms
    vreg.define('epd', 'EP flux divergence', parents='acceleration')
    vreg.define('epc', 'EP flux convergence', parents='acceleration')
    vreg.define('qgepd', 'QG EP flux divergence', parents='epd')
    vreg.define('qgepc', 'QG EP flux convergence', parents='epc')
    vreg.define('epy', 'meridional EP flux', parents='mf')
    vreg.define('epz', 'vertical EP flux', parents='mf')
    vreg.define('qgepy', 'meridional QG EP flux', parents='epy')
    vreg.define('qgepz', 'vertical QG EP flux', parents='epz')

    # Ertel and QG potential vorticity fluxes
    vreg.define('pvf', 'potential vorticity flux', 'PVU m / s')
    vreg.define('mpvf', long_prefix='mean', parents='pvf')
    vreg.define('epvf', long_prefix='eddy', parents='pvf')
    vreg.define('cpvf', short_suffix='convergence', standard_units='VU / day', parents='pvf')  # noqa: E501
    vreg.define('cmpvf', long_prefix='mean', parents='cpvf')
    vreg.define('cepvf', long_prefix='eddy', parents='cpvf')
    vreg.define('qgpvf', 'quasi-geostrophic potential vorticity flux', 'VU m / s', parents='acceleration')  # noqa: E501
    vreg.define('mqgpvf', long_prefix='mean', parents='qgpvf')
    vreg.define('eqgpvf', long_prefix='eddy', parents='qgpvf')
    vreg.define('cqgpvf', short_suffix='convergence', standard_units='VU / day', parents='qgpvf')  # noqa: E501
    vreg.define('cmqgpvf', long_prefix='mean', parents='cqgpvf')
    vreg.define('ceqgpvf', long_prefix='eddy', parents='cqgpvf')

    # Variance and covariance terms
    # NOTE: This includes parameters that may have spectral decomposition
    vreg.define('var', 'zonal variance')
    vreg.define('std', 'zonal standard deviation')
    vreg.define('tvar', 'eddy temperature variance', 'K^2', parents='var')
    vreg.define('uvar', 'eddy zonal wind variance', 'm^2 / s^2', parents='var')
    vreg.define('vvar', 'eddy meridional wind variance', 'm^2 / s^2', parents='var')
    vreg.define('zvar', 'eddy geopotential height variance', 'm^2', parents='var')
    vreg.define('pvvar', 'eddy potential vorticity variance', 'PVU^2', parents='var')
    vreg.define('tstd', 'RMS eddy temperature', 'K', parents='std')
    vreg.define('ustd', 'RMS eddy zonal wind', 'm / s', parents='std')
    vreg.define('vstd', 'RMS eddy meridional wind', 'm / s', parents='std')
    vreg.define('pvstd', 'RMS eddy potential vorticity', 'PVU', parents='std')
    vreg.define('bvstd', 'Barry RMS eddy meridional wind', 'm / s', parents='std')

    # Stability terms
    # NOTE: Potential temperature *gradients* are different from temperature gradients
    # because they get multiplied by height-dependent Exner function.
    vreg.define('dtdy', 'meridional temp gradient', 'K / 1000km')
    vreg.define('dtdy_bulk', long_prefix='bulk', parents='dtdy')
    vreg.define('dptdy', 'meridional potential temp gradient', 'K / 1000km')
    vreg.define('dptdy_bulk', long_prefix='bulk', parents='dptdy')
    vreg.define('dpvdy', 'meridional PV gradient', 'PVU / 1000km')
    vreg.define('dpvdy_bulk', long_prefix='bulk', parents='dpvdy')
    vreg.define('ddsedy', 'static energy gradient', 'J kg^-1 km^-1')
    vreg.define('ddsedy_bulk', long_prefix='bulk', parents='ddsedy')
    vreg.define('curvature', 'lapse rate curvature', 'K / km^2')
    vreg.define('dcurvature', 'derivative of curvature', 'K / km^3')
    vreg.define('dtdz', 'lapse rate', 'K / km')
    vreg.define('dtdz_bulk', long_prefix='bulk', parents='dtdz')
    vreg.define('dptdz', 'isentropic lapse rate', 'K / km')
    vreg.define('dptdz_bulk', long_prefix='bulk', parents='dptdz')
    vreg.define('dptdp', 'isentropic pressure lapse rate', 'K / hPa')
    vreg.define('b', 'buoyancy frequency', 'hr^-1')
    vreg.define('baro', 'Lorenz baroclinicity', 'day^-1')
    vreg.define('slope', 'isentropic slope', 'm / km')
    vreg.define('pslope', 'isentropic pressure slope', 'hPa / km')
    vreg.define('slope_bulk', long_prefix='bulk', parents='slope')
    vreg.define('slope_diff', long_prefix='dimensionless', standard_units='K / K', parents='slope', symbol=r'\Delta\Theta_h / \Delta\Theta_v')  # noqa: E501
    vreg.define('ratio', 'equilibrium temperature gradient ratio', symbol=r'\Delta\Theta_h / \Delta\Theta_{h,eq}')  # noqa: E501
    vreg.define('trop', 'WMO tropopause', parents='plev')
    vreg.define('ztrop', 'WMO tropopause height', parents='zlev')
    vreg.define('ctrop', 'thermal tropopause', parents='plev')
    vreg.define('zctrop', 'thermal tropopause height', parents='zlev')
    vreg.define('ttrop', 'tropopause temperature', parents='t')
    vreg.define('pttrop', 'tropopause potential temperature', parents='pt')
    vreg.define('pvtrop', '2PVU tropopause', parents='plev')
    vreg.define('mtrop', 'mass-streamfunction tropopause', parents='plev')

    # Momentum terms
    vreg.define('ucos', long_prefix='cosine-weighted', parents='u')
    vreg.define('uweight', long_prefix='friction-weighted', parents='u')
    vreg.define('shear', 'zonal shear', 'm s^-1 km^-1', 'shear')
    vreg.define('shear_bulk', 'bulk zonal shear', 'm / s', 'shear')
    vreg.define('uthermal', 'thermal wind', 'm / s')
    vreg.define('uthermal_integrand', 'thermal wind integrand', 'm s^-1 100hPa^-1')
    vreg.define('L', 'angular momentum', 'm^2 s^-1')

    # Length scales, timescales, and growth rates
    vreg.define('erate', long_prefix='Eady growth', parents='rate')
    vreg.define('etimescale', long_prefix='Eady growth', parents='timescale')
    vreg.define('rtimescale', long_prefix='forcing response', parents='timescale')
    vreg.define('ld', 'deformation radius', symbol='L_d', parents='scale')
    vreg.define('lr', 'Rhines scale', symbol='L_R', parents='scale')
    vreg.define('ldisp', 'displacement scale', symbol='L_{disp}', parents='scale')
    vreg.define('ls', 'stationary wavelength', symbol='L_s', parents='scale')

    # Diffusivity parameters
    vreg.define('diffusivity', 'diffusivity', 'km^2 / s')
    vreg.define('t_diffusivity', 'thermal diffusivity', parents='diffusivity')
    vreg.define('pv_diffusivity', 'PV diffusivity', parents='diffusivity')
    vreg.define('dse_diffusivity', 'static energy diffusivity', parents='diffusivity')
    vreg.define('diffusivity_bulk', long_prefix='bulk', parents='diffusivity')
    vreg.define('t_diffusivity_bulk', long_prefix='bulk', parents='t_diffusivity')
    vreg.define('pv_diffusivity_bulk', long_prefix='bulk', parents='pv_diffusivity')
    vreg.define('dse_diffusivity_bulk', long_prefix='bulk', parents='dse_diffusivity')
    vreg.define('ehf_diffusive', 'diffusivity-predicted heat flux', parents='ehf')
    vreg.define('epvf_diffusive', 'diffusivity-predicted potential vorticity flux', parents='epvf')  # noqa: E501
    vreg.define('edsef_diffusive', 'diffusivity-predicted intensity', parents='edsef')
    vreg.define('itdt_numerator', '$T - T^e$ contribution', parents='itdt')
    vreg.define('itdt_denominator', r'$\tau$ contribution', parents='itdt')
    # vreg.define('itdt_denominator', r'$\tau_t$ contribution', parents='itdt')

    # Dynamical core parameters
    # NOTE: Need to round to nearest 5 for the sake of the parametric plot.
    # Otherwise would have really weird values. Not important at all.
    # NOTE: Want to compare everything against increasing timescale or sensitivity. In
    # general this means experiments with 'strong' circulations on left and with 'weak'
    # circulations on right. If parameter order is opposite must set reverse to True.
    fmt = ('sigfig', 2, True, 5)
    # fmt = ('sigfig', 2)
    vreg.define('scheme', 'forcing scheme', '')  # string value
    vreg.define('reso', 'resolution', '')  # string value
    vreg.define('tgrad', 'equilibrium temperature difference', 'K', colormap='stellar', symbol=r'\Delta_y', reference=60, axis_reverse=True, parents='t')  # noqa: E501
    vreg.define('tmean', 'average equilibrium surface temperature', 'K', symbol=r'\overline{T}', reference=300, parents='t')  # noqa: E501
    vreg.define('tshift', 'equilibrium temperature gradient shift', '', symbol=r'\gamma', reference=1)  # noqa: E501
    vreg.define('damp', 'damping rate', standard_units='day^-1', scalar_formatter=fmt)
    vreg.define('ndamp', 'thermal damping coefficient', parents='damp')
    vreg.define('ndamp_mean', long_prefix='mean', parents='ndamp')
    vreg.define('ndamp_anom', long_prefix='eddy', parents='ndamp')
    vreg.define('rdamp', 'mechanical damping coefficient', parents='damp')
    vreg.define('rdamp_mean', long_prefix='mean', parents='rdamp')
    vreg.define('rdamp_anom', long_prefix='eddy', parents='rdamp')
    vreg.define('sdamp', 'sponge damping coefficient', parents='damp')
    # vreg.define('tau', 'damping timescale', 'days', colormap='dusk', axis_scale='log', scalar_formatter=fmt)  # noqa: E501
    vreg.define('tau', 'relaxation timescale', 'days', colormap='dusk', axis_scale='log', scalar_formatter=fmt)  # noqa: E501
    vreg.define('rtau', long_prefix='mechanical', reference=1, symbol=r'\tau_{min}', parents='tau')  # noqa: E501
    vreg.define('ntau', long_prefix='maximum', reference=40, symbol=r'\tau_{max}', parents='tau')  # noqa: E501
    vreg.define('ntau0', long_prefix='reference', symbol=r'\tau_{0}', parents='tau')  # noqa: E501
    vreg.define('ntaumean', long_prefix='zonal-mean', symbol=r'\overline{\tau}_{max}', parents='ntau')  # noqa: E501
    vreg.define('ntauanom', long_prefix='zonal-anomaly', symbol=r'\tau^*_{max}', parents='ntau')  # noqa: E501
    # vreg.define('rtau', long_prefix='minimum mechanical', reference=1, symbol=r'\tau_{max}', parents='tau')  # noqa: E501
    # vreg.define('ntau', long_prefix='maximum thermal', reference=40, symbol=r'\tau_{max}', parents='tau')  # noqa: E501
    # vreg.define('ntau0', long_prefix='reference thermal', symbol=r'\tau_{0}', parents='tau')  # noqa: E501
    # vreg.define('ntaumean', long_prefix='zonal-mean thermal', symbol=r'\overline{\tau}_{max}', parents='ntau')  # noqa: E501
    # vreg.define('ntauanom', long_prefix='zonal-anomaly thermal', symbol=r'\tau^*_{max}', parents='ntau')  # noqa: E501
    vreg.define('qglobal', 'globally uniform forcing', parents='forcing')
    vreg.define('qrealistic', 'realistic forcing', parents='forcing')
    vreg.define('qsurface', 'boundary layer forcing', parents='forcing')
    vreg.define('qtropical', 'tropical forcing', parents='forcing')
    vreg.define('qvortex', 'polar vortex forcing', parents='forcing')
    vreg.define('qarctic', 'polar surface forcing', parents='forcing')

    # Climate feedbacks and sensitivity
    # NOTE: Wikipedia climate sensitivity page says there is not yet a consensus on
    # whether 'climate sensitivity parameter' can be called 'climate sensitivity'.
    # However IPCC and Hartmann's book both use 'parameter', so be conservative.
    vreg.define('feedback', 'feedback parameter', 'W m^-2 / K', colormap='lajolla', symbol=r'\lambda')  # noqa: E501
    vreg.define('cfp', long_prefix='climate', parents='feedback')  # empriical
    vreg.define('rfp', long_prefix='relaxation climate', standard_units='W m-2 100hPa-1 / K', symbol=r'\lambda_{\tau}', parents='feedback')  # noqa: E501
    # vreg.define('rfp', long_prefix='local relaxation', standard_units='W m-2 100hPa-1 / K', symbol=r'\lambda(\tau)', parents='feedback')  # noqa: E501
    vreg.define('sensitivity', 'sensitivity', colormap='lajolla', axis_scale='log', scalar_formatter=fmt)  # noqa: E501
    vreg.define('cs', 'climate sensitivity', 'K', 'temperature', symbol=r'\Delta T', parents='sensitivity')  # noqa: E501
    vreg.define('css', long_prefix='near-surface', parents='cs')
    vreg.define('csp', 'climate sensitivity parameter', 'K / W m^-2', symbol='s', parents='sensitivity')  # noqa: E501
    vreg.define('rs', long_prefix='relaxation', symbol=r'\Delta T_{\tau}', parents='cs')  # noqa: E501
    vreg.define('rsp', 'relaxation sensitivity parameter', symbol=r's_{\tau}', parents='csp')  # noqa: E501
    # vreg.define('rsp', long_prefix='relaxation', symbol=r's_{\tau}', parents='csp')  # noqa: E501
    vreg.define('rss', long_prefix='near-surface', parents='rs')
    vreg.define('rs2xco2', long_prefix=r'2$\times$CO$_2$', parents='rs')
    vreg.define('rs4xco2', long_prefix=r'4$\times$CO$_2$', parents='rs')
    # vreg.define('rs', long_prefix='relaxation', symbol=r'\Delta T_{\tau_0}', parents='cs')  # noqa: E501
    # vreg.define('rsp', 'relaxation sensitivity parameter', symbol=r's_{\tau_0}', parents='csp')  # noqa: E501

    # ### Simple variable transformations
    def _reference_timescale(tau_max, /, *, tau_ratio=10.0, sigma_bl=0.7, scheme=None):
        """
        Convert maximum damping timescale to "reference" damping timescale (inverse
        atmosphere-mean inverse). Account for different configurations.
        """
        # Auto-fill arguments based on preset "forcing scheme" strings
        if scheme == 'hs1':
            tau_ratio = 10
            sigma_bl = 0.7
        elif scheme == 'hs2':
            tau_ratio = 1
            sigma_bl = 0.7
        elif scheme:
            raise ValueError(f'Unknown forcing scheme preset {scheme!r}.')
        # Atmosphere average
        # NOTE: Boundary layer scale is cos(x)^4 plus a cosine for area weighting:
        # integral_0^pi/2 cos(x)^5 = 8 / 15 via wolfram alpha. Vertical ave over the
        # boundary layer is 0.5. Then average the boundary layer and trosophere.
        tau_min = tau_max / tau_ratio
        k_bl = 1.0 / tau_min
        k_trop = 1.0 / tau_max
        k_ave = (
            sigma_bl * k_trop
            + (1.0 - sigma_bl) * (k_trop + 0.5 * (k_bl - k_trop) * 8.0 / 15.0)
        )
        return 1.0 / k_ave

    # Damping timescale transformations (use the HS94 damping timescale)
    climo.register_transformation('ndamp', 'ntau')(lambda _: 1 / _)
    climo.register_transformation('rdamp', 'rtau')(lambda _: 1 / _)
    climo.register_transformation('ntau', 'ntau0')(_reference_timescale)
    climo.register_transformation('ntaumean', 'ntau')(lambda _: _)  # dummy
    climo.register_transformation('ntauanom', 'ntau')(lambda _: _)  # dummy

    # Relaxation sensitivity parameter transformations
    climo.register_transformation('ntau', 'rfp')(lambda _: -C_DELTA / _)
    climo.register_transformation('ntau0', 'rsp')(lambda _: _ / C_COLUMN)
    climo.register_transformation('ntau0', 'rs2xco2')(lambda _: Q_2XCO2 * _ / C_COLUMN)
    climo.register_transformation('ntau0', 'rs4xco2')(lambda _: Q_4XCO2 * _ / C_COLUMN)
    climo.register_transformation('ntau0', 'rfp0')(lambda _: -C_COLUMN / _)

    # ### Basic thermodynamic quantities
    @climo.register_derivation(re.compile(r'\A[ptz]lev\Z'))
    def vertical_level(self, name):
        # Vertical level. This lets us use the same name for the vertical
        # coordinate but have more precise variables for different level types.
        lev = self.lev
        if name[0] == 'p':
            assert lev.climo.units.is_compatible_with('Pa')
        elif name[0] == 'z':
            assert lev.climo.units.is_compatible_with('m')
        else:
            assert lev.climo.units.is_compatible_with('K')
        return lev

    @climo.register_derivation(('pres', 'pt', 't'))
    def temperature_pressure(self, name):
        # Retrieve temperature and pressure accounting for situations
        # where these represent the *vertical coordinate*.
        # WARNING: Cannot call pressure 'p' due to name conflict with potential energy
        vert = self.cf.vertical_type
        if name == 'pres':
            if vert == 'pressure':
                return self.lev
            else:
                raise NotImplementedError('Can only get pres from isobaric levels.')
        elif name == 't':
            if vert == 'temperature':
                return self.lev * self.exner
            else:
                raise NotImplementedError('Can only get temp from isentropic levels.')
        else:
            if vert == 'temperature':
                return self.lev
            else:
                return self.t / self.exner

    @climo.register_derivation('w')
    def vertical_wind(self):
        # Vertical wind in m/s converted from omega
        return (-1 / (const.g * self.rho)) * self.omega

    @climo.register_derivation('rho')
    def density(self):
        # Calculate the density in units kg/m^3
        return self.pres / (const.Rd * self.t)

    @climo.register_derivation('dse')
    def dry_static_energy(self):
        # Dry static energy
        units = 'J kg^-1'
        return sum(self[name].climo.to_units(units, 'climo') for name in ('t', 'z'))

    @climo.register_derivation('deltat')
    def teq_difference(self):
        # Difference between temp and equlibrium temp
        return self.t - self.teq

    @climo.register_derivation(('pt', 'pteq'))
    def potential_temperature(self, name):
        # Potential temperature
        return self[name[1:]] / self.exner

    @climo.register_derivation('exner')
    def exner_function(self):
        # Exner function T/theta == (p / p0) ** kappa
        return (self.pres / const.p0).climo.to_units('') ** const.kappa

    @climo.register_derivation(('tdt', 'itdt'))
    def thermal_damping(self, name):
        # Thermal damping
        data = self.tdt
        if name == 'itdt':
            data = -1 * data
        return data

    @climo.register_derivation(('damping', 'idamping'))
    def thermal_damping_manual_calculation(self, name):
        # Manually calculated thermal damping (used for verification)
        data = -1 * self.ndamp * (self.t - self.teq)
        if name == 'idamping':
            data = -data
        return data

    @climo.register_derivation('pv')
    def potential_vorticity(self):
        # Potential vorticity with PV below 95th percentil surface level to NaN
        nx = self.x
        idx = np.round(0.05 * nx).astype(int)
        surf = self.slth.isel(x=idx)
        data = self.pv.copy(deep=True)  # copy because we filter!
        lev = self.lev
        data[lev <= surf] = np.nan
        return data

    @climo.register_derivation(('dtdy', 'dptdy'))
    def temperature_gradient(self, name):
        # Temperature gradient
        name = re.sub(r'\Ad(.*)dy\Z', r'\1', name)
        return -1 * self[name].climo.derivative(y=1)

    @climo.register_derivation('dpvdy')
    def potential_vorticity_gradient(self):
        # PV gradient
        return self.pv.climo.derivative(y=1)

    @climo.register_derivation('ddsedy')
    def dry_static_energy_gradient(self):
        # Eddy static energy gradient
        return -1 * self.dse.climo.derivative(y=1)

    @climo.register_derivation('dtdp')
    def lapse_rate_pressure(self):
        # Temperature lapse rate in pressure coords
        return self.t.climo.derivative(lev=1)

    @climo.register_derivation('dtdz')
    def lapse_rate_height(self):
        # Temperature lapse rate in height coords.
        if self.cf.vertical_type == 'temperature':
            return climo.deriv_uneven(self.z, self.t, keepedges=True)
        else:
            return -self.rho * const.g * self.t.climo.derivative(lev=1)

    @climo.register_derivation('dptdp')
    def potential_lapse_rate_pressure(self):
        # Potential temp lapse rate in pressure coords
        return self.pt.climo.derivative(lev=1)

    @climo.register_derivation('dptdz')
    def potential_lapse_rate_height(self):
        # Potential temp lapse rate in height coords
        pt = self.pt
        rho = self.rho  # first get density
        data = -rho * const.g * pt.climo.derivative(lev=1)
        return data.climo.to_units('K / m')

    @climo.register_derivation(('slope', 'pslope'))
    def isentropic_slope(self, name):
        # Isentropic slope. Would be negative everywhere since potential temp
        # *increases* with increasing z and *decreases* with increasing y, and pt
        # gradient negative with latitude. But correctd by dtdy returning negative.
        if self.cf.vertical_type == 'temperature':
            data = self.z.climo.derivative(y=1)
        elif name == 'slope':
            data = self.dptdy / self.dptdz
        else:
            data = self.dptdy / self.dptdp
        return data

    @climo.register_derivation('baro')
    def baroclinicity(self):
        # Baroclinicity from Lorenz and Hartmann, 2000, JC
        # Final units are (m/s2)*(K/m)/(K*1/s) = (1/s2)/(1/s) = (1/s)
        return const.g * self.dtdy / (self.b * 300 * ureg.K)

    @climo.register_derivation('b')
    def buoyancy_frequency(self):
        # Brunt-Vaisala frequency in units 1 / s.
        # This is N^2 = (g / theta) dtheta / dz = -(rho * g^2 / theta) dtheta / dp
        # = -(p * g^2 / R * T * theta) dtheta / dp for pressure coordinates
        # data = np.sqrt((-lev[:,None] * const.g**2 * dptdp) / (const.Rd * t * pt))
        pt = self.pt
        rho = self.rho
        dptdp = self.dptdp
        return np.sqrt((-rho * (const.g**2) * dptdp) / pt)

    @climo.register_derivation(('curvature', 'dcurvature'))
    def lapse_rate_curvature(self, name):
        # Curvature or curvature derivative using different diff methods
        # z = self.z if 'z' in vars else self.z0
        t = self.t
        z = self.coords.z0
        order = 2 if name == 'curvature' else 3
        data = climo.deriv_uneven(z, t, dim='lev', order=order, keepedges=True)
        return data

    @climo.register_derivation(('cs', 'css'))
    def climate_sensitivity(self, name):
        # Actual atmospheric climate sensitivity
        lev = 'avg' if name == 'cs' else SURFACE
        data = self.get('t_anomaly', lev=lev, area='avg')
        data.attrs.clear()  # remove 'suffix' instruction
        return data

    @climo.register_derivation(('rs', 'rss'))
    def relaxation_climate_sensitivity(self, name):
        # Climate sensitivity "predicted" by the reference feedback
        lev = 'avg' if name == 'rs' else SURFACE
        forcing = self.get('forcing_2', lev='int', area='avg')
        forcing = forcing.climo.to_units('W m^-2')
        ndamp = C_COLUMN * self.get('ndamp_1', lev=lev, area='avg')
        ndamp = ndamp.climo.to_units('W m^-2 K^-1')
        return (forcing / ndamp).climo.to_units('K')

    # ### Energy and momentum terms
    @climo.register_derivation('egf')
    def eddy_geopotential_flux(self):
        # Eddy geopotential flux
        warnings.warn('No egf data found. Assuming zero everywhere.')
        return 0 * ureg('m^2 s^-1') * self.ehf.climo.dequantify()

    @climo.register_derivation('evhf')
    def eddy_vertical_heat_flux(self):
        # Heat flux == omega * theta == (omega * T) * (p0 / p) ^ kappa
        # Since C(EPE, EKE) == Rd * (omega' * T') / p the vertical eddy heat flux
        # is stored *indirectly* in Lorenz term. See dry core for equation and
        return -self.pres * self.cpeke / (const.Rd * self.exner)

    @climo.register_derivation('epvf')
    def eddy_potential_vorticity_flux(self):
        # Eddy PV flux, masking out areas above 1.5PVU (stratosphere).
        pv = self.pv
        data = self.epvf.copy(deep=True)
        data[np.abs(pv) > 1.5 * ureg.PVU] = np.nan
        return data

    @climo.register_derivation('emf_cos')
    def eddy_angular_momentum_flux(self):
        # Eddy momentum flux scaled with cosine.
        # NOTE: Momentum fluxes are weighted by cos^2 so they reflect angular
        # momentum. Remember M = (omega * a * cos(phi) + u) * a * cos(phi)
        # and since Omega is constant an extra cosine scaling is sufficient.
        return self.cos * self.emf

    @climo.register_derivation('edsef')
    def eddy_static_energy_flux(self):
        # Eddy dry static energy flux
        data = self.ehf * const.cp + self.egf * const.g
        return data.climo.to_units('J kg^-1 m s^-1')

    @climo.register_derivation(re.compile(r'\Ac?(e|m)?(' + '|'.join(PARTS_FLUX) + r')\Z'))  # noqa: E501
    def eddy_mean_flux(self, name):
        # Eddy, mean, or total meridional flux, or convergence thereof
        # NOTE: Divergence
        flux = name[1:] if (convergence := name[0] == 'c') else name  # trim 'c'
        flux_name = flux if flux in PARTS_FLUX else flux[1:]  # trim '(e|m)'
        part1, part2 = PARTS_FLUX[flux_name]
        data = 0
        if flux in (flux_name, 'm' + flux_name):
            data += self[part1] * self[part2]
        if flux in (flux_name, 'e' + flux_name):
            data += self['e' + flux_name]
        if type(data) is int and data == 0:
            raise ValueError(f'Invalid flux {name!r}. This is impossible.')
        if convergence:
            if flux_name[0] == 'v':  # vertical flux
                data = -data.climo.derivative(lev=1)
            else:
                data = -data.climo.divergence(cos_power=2 if flux_name[0] == 'm' else 1)
        return data

    @climo.register_derivation(re.compile(r'\A(tdt|forcing|diabatic|rheating|cmdsef|cross)_approxdelta\Z'), assign_name=False)  # noqa: E501
    def barpanda_approx_storm_track_shift(self, name):
        # Barpanda et al. approx contribution of energy terms to storm track shift.
        # Return full latitude *predicted* by individual terms.
        # NOTE: Barpanda et al. write FMM as *divergence*. Switch the signs here.
        # f = -1 * self.get('cedsef_1', lev='int', standardize=True)  # div
        prefix, _ = name.split('_')
        dn = self.get(prefix + '_anomaly', lev='int')
        dn = dn.climo.to_units('W m^-2')
        f = self.get('edsef_1', lev='int')
        f = f.climo.divergence().climo.to_units('W m^-2')
        name = 'lat'
        dfdy = f.climo.derivative(y=1)  # divergence of divergence
        dndy = dn.climo.derivative(y=1)
        data = -1 * dn + f * dndy / dfdy + dn * dndy / dfdy
        data = data / (const.a * dfdy)
        data = data.climo.to_units('deg') + self.get('edsef_latitude_1')
        data.attrs['long_prefix'] = 'latitude shift due to'
        data.name = name.split('_')[0]
        return data

    @climo.register_derivation(re.compile(r'\A(tdt|forcing|diabatic|rheating|cmdsef)_exactdelta\Z'), assign_name=False)  # noqa: E501
    def barpanda_exact_storm_track_shift(self, name):
        # Barpanda et al. exact contribution of energy terms to storm track shift.
        # Return full latitude *predicted* by individual terms.
        # NOTE: Barpanda et al. write FMM as *divergence*. Switch the signs here.
        prefix, _ = name.split('_')
        n = self.get(prefix + '_anomaly', lev='int')
        n = n.climo.to_units('W m^-2')
        f = self.get('edsef_1', lev='int')
        f = f.climo.divergence().climo.to_units('W m^-2')
        name = 'lat'
        data = (f + n).climo.reduce(  # where - meridional gradient goes - to +
            lat='argzero', ntrack=1, which='posneg', lat_lim=(20, 70),
            dataset=self.data
        ).climo.to_units('deg')
        data.attrs['long_prefix'] = 'latitude shift due to'
        data.name = name.split('_')[0]
        return data

    @climo.register_derivation(re.compile(r'\A(ehf|edsef|epvf)_diffusive(_response)?\Z'), assign_name=False)  # noqa: E501
    def diffusive_energy_budget_response(self, name):
        # Estimate of anomaly using bulk diffusive assumption relative to reference.
        # NOTE: Unclear whether to use gradient at storm track position 2 or 1
        # for dfdy2. Play with both by commenting out the kw['lat'] line.
        # kw['lev_lim'] = (850, 1050) * ureg.hPa
        # kw = {'lev': 850 * ureg.hPa, 'lat': name + '_latitude_1'}
        name, _, *response = name.split('_')
        var = 't' if name == 'ehf' else name[1:-1]  # dse or pv
        kw1 = {'lev': 850 * ureg.hPa, 'lat': name + '_latitude_1'}
        # kw1 = {'lev': 'avg', 'lat': name + '_latitude_1', 'lev_lim': LEV_LIM}
        # kw1 = {'lev': 'avg', 'lat': name + '_latitude_1', 'lev_lim': (700, None)}
        kw2 = kw1.copy()
        kw2['lat'] = name + '_latitude_2'  # preserves original insertion order
        flux1 = self.get(name + '_strength_1')
        grad1 = self.get('d' + var + 'dy_1', **kw1)
        grad2 = self.get('d' + var + 'dy_2', **kw2)
        data = flux1 * grad2 / grad1
        data.climo.update_cell_attrs(flux1)
        if response:
            data = data - flux1  # difference relative to "control"
            data.attrs['long_suffix'] = 'response'
        data.name = name + '_diffusive'
        return data

    @climo.register_derivation(re.compile(r'\A(ehf|edsef|epvf)_diffusive_local(_response)?\Z'), assign_name=False)  # noqa: E501
    def diffusive_local_flux_response(self, name):
        # Estimate of anomaly using local diffusive assumption
        name, _, _, *response = name.split('_')
        grad = self['d' + ('t' if name == 'ehf' else name[1:-1]) + 'dy']
        flux = self[name].climo.sel_pair(1)
        data = flux * grad.climo.sel_pair(2) / grad.climo.sel_pair(1)
        data.data[np.abs(grad.climo.sel_pair(1).data) < 0.5e-3 * ureg('K km^-1')] = np.nan  # noqa: E501
        data.climo.update_cell_attrs(flux)
        if response:
            data = data - flux
            data.attrs['long_suffix'] = 'response'
        data.name = name + '_diffusive'
        return data

    @climo.register_derivation(re.compile(r'\Ai?tdt_(numerator|denominator)\Z'))
    def thermal_damping_numerator_denominator(self, name):
        # Contribution of teq and ntau to thermal damping
        # Numerator and denominator contribution
        name, part = name.split('_')
        num = -1 * self.get('deltat_' + ('2' if part == 'numerator' else '1'))
        denom = self.get('ndamp_' + ('1' if part == 'numerator' else '2'))
        data = num * denom
        if name[0] == 'i':
            data = -1 * data
        return data

    @climo.register_derivation(re.compile(r'\Ai?(e|m)?adiabatic\Z'))
    def adiabatic_heating(self, name):
        # Adiaatic heating == kappa * (omega * T) / p
        # Since C(EPE, EKE) == Rd * (omega' * T') / p the vertical eddy heating
        # is stored *indirectly* in Lorenz term. See dry core for equation and
        # hand written notes. Term appears when converting Dtheta/dp to DT/dp
        # in isentropic energy conservation equation.
        data = 0
        inverse = name[0] == 'i'
        if inverse:
            name = name[1:]
        if name in ('madiabatic', 'adiabatic'):
            data += const.kappa * self.t * self.omega / self.pres
        if name in ('eadiabatic', 'adiabatic'):
            data += -1 * self.cpeke / const.cp
        if inverse:
            data = -1 * data
        return data

    @climo.register_derivation(('eheating', 'mheating'))
    def eddy_mean_circulation_heating(self, name):
        # Total mean and eddy contributions to the heat budget
        # Since madiabatic is equivalent to cmgf, we only use cmhf here
        part = name[0]
        data = self[part + 'adiabatic'] + self['c' + part + 'hf']
        return data

    @climo.register_derivation(('diabatic', 'idiabatic'))
    def diabatic_heating(self, name):
        # Total diabatic heating i.e. damping plus forcing
        data = self.tdt + self.forcing
        if name == 'idiabatic':
            data = -data
        return data

    @climo.register_derivation(('forcing', 'iforcing', 'rforcing'))
    def external_forcing_heating(self, name):
        # Imposed forcing or "residual forcing" i.e. diabatic difference
        data = self.forcing
        if name == 'iforcing':
            data = -data
        elif name == 'rforcing':
            data += self.tdt
            data *= -1
        return data

    @climo.register_derivation(('rheating', 'irheating'))
    def residual_heating(self, name):
        # "Residual heating" leftover from diabatic heating and EHF. This is
        # "residual" in sense that expect most of EHF to be balanced by diabatic
        # cooling. Includes *numerical* residual described by 'aheating'
        # data = -(self['diabatic'] + self['cehf'])
        units = 'W kg^-1'
        data = -1 * sum(self[name].climo.to_units(units) for name in ('cedsef', 'diabatic'))  # noqa: E501
        if name == 'irheating':
            data = -1 * data
        return data

    @climo.register_derivation(('aheating', 'iaheating'))
    def all_heating(self, name):
        # *All* heating sources. This characterizes numerical residual term due to
        # numerical error/failure to close energy budget.
        # data = sum(self[name] for name in ('chf', 'adiabatic', 'diabatic'))
        units = 'W kg^-1'
        data = sum(self[name].climo.to_units(units) for name in ('cdsef', 'diabatic'))
        if name == 'iaheating':
            data = -1 * data
        return data

    @climo.register_derivation('dheating')
    def geopotential_adiabatic_heating_difference(self):
        # Difference between geopotential convergence and mean adiabatic
        # Should be roughly identical, this gives residual
        units = 'W kg^-1'
        cgf = self.cgf.climo.to_units(units)
        adiabatic = self.madiabatic.climo.to_units(units)
        return cgf - adiabatic

    @climo.register_derivation(re.compile(r'\A(t|pv|dse)_diffusivity\Z'))
    def local_diffusivity(self, name):
        # Eddy diffusivity, in units m^2/s, masking out regions with tiny gradients
        name, _ = name.split('_')
        grad = self['d' + name + 'dy']
        flux = self['e' + ('h' if name == 't' else name) + 'f']
        data = flux / grad
        mins = {
            't': ureg.Quantity(0.5e-3, 'K km^-1'),
            'pv': ureg.Quantity(0.5e-3, 'PVU km^-1'),
            'dse': ureg.Quantity(0.5, 'J kg^-1 km^-1'),
        }
        data.data[np.abs(grad.data) < mins[name]] = np.nan
        return data

    @climo.register_derivation('diss')
    def dissipative_heating(self):
        # Heat loss due to wind dissipation. Approximately derive from KE using:
        # -[u(t + 0.5*dt) * (-u(t) / tauf) + v(t + 0.5*dt) * (-v(t) / tauf)]
        # =~ (u(t)^2 + v(t)^2) / tauf = 2 * ke / tauf. Derive from KE term.
        return 2 * self.ke * self.rdamp

    @climo.register_derivation(tuple(desc.name for desc in vreg.lorenz))
    def lorenz_energy_budget(self, name):
        # Lorenz terms or sum of like terms
        # NOTE: Pint context permits transforming J / kg to J / m^2 Pa^2
        key = name
        if name in ('ckmpm', 'ckepe'):
            key = 'c' + name[3:] + name[1:3]
        if key in PARTS_LORENZ:
            data = sum(self[part] for part in PARTS_LORENZ[key])
        else:
            data = self[key]
        if name in ('ckmpm', 'ckepe'):
            data = -data  # fix sign
        if data.data._units.get('meter', None) == -2:
            data.climo.add_cell_methods(lev='int')
        return data

    @climo.register_derivation(('g/gpe', 'cpe/gpe'))
    def lorenz_energy_budget_ratio(self, name):
        # Lorenz ratios used as *timescales*
        name1, name2 = name.split('/')
        data = (
            self.get(name1, area='avg', lev='int')
            / self.get(name2, area='avg', lev='int')
        )
        return data

    @climo.register_derivation('a')
    def zonal_acceleration(self):
        # Eddy forcing of the zonal mean flow (udt should include cosine)
        return self.epd + self.torqueresid + self.udt + self.cmmf

    @climo.register_derivation('ucos')
    def cosine_weighted_zonal_wind(self):
        # Zonal wind scaled with cosine
        return self.u * self.cos

    @climo.register_derivation('uweight')
    def damping_weighted_zonal_wind(self):
        # Zonal wind weighted by damping timescale
        return self.cos * self.u.climo.average('lev', weight=self.rdamp)

    @climo.register_derivation('cemf_position', assign_name=False)
    def accurate_jet_position(self):
        # Highly accurate jet position, avoiding double finite differentiation
        cos = self.cos
        emf = self.emf.climo.truncate(lat_lim=(20, 70)).climo.integral('lev')
        cemf = emf.climo.convergence().climo.absmax('lat').climo.to_units('Pa')
        data = (cos ** 3 * emf).climo.derivative(y=2) / cos ** 3  # noqa: E501
        data = data.climo.argloc('lat', value=0, which='negpos', track=False)
        data = data.squeeze('track', drop=True)
        data = data.drop_vars(data.attrs['parent_name'])
        data.climo.update_cell_methods({'lat': 'argmax'})
        data.attrs['parent_name'] = 'cemf'
        data.coords['cemf'] = cemf.drop_vars('lat')
        return data

    @climo.register_derivation('L')
    def angular_momentum(self):
        # Angular momentum (u + Omega * a * cos phi) * a * cos phi
        return (self.u + const.Omega * const.a * self.cos) * const.a * self.cos

    @climo.register_derivation(('udt', 'iudt'))
    def mechanical_damping(self, name):
        # Drag weighted by cosine
        # data = self.udt * self.cos
        data = self.udt
        # data = self.udt
        if name[0] == 'i':
            data = -data
        return data

    @climo.register_derivation(('torque', 'torquee', 'torqueresid'))
    def zonal_coriolis_torque(self, name):
        # Zonal torque due to mean meridional wind
        return self.cor * self['v' + name[6:]]

    @climo.register_derivation('uthermal')
    def thermal_wind(self):
        # Zonal wind inferred from thermal wind. Integrate from surface
        return self.uthermal_integrand.climo.cumintegral('lev', reverse=True)

    @climo.register_derivation('uthermal_integrand')
    def thermal_wind_integrand(self):
        # Thermal wind integrand
        return (const.Rd / self.cor) * self.dtdy / self.pres

    @climo.register_derivation('shear')
    def vertical_zonal_wind_shear(self):
        # Vertical shear in m / s / km
        u = self.u
        rho = self.rho  # need rho to convert d/dp to d/dz
        return -const.g * rho * u.climo.derivative(lev=1)

    @climo.register_derivation('ehf_corr')
    def eddy_heat_flux_correlation_coefficient(self):
        # Correlation coefficient for heat flux
        return self.ehf / (self.vstd * self.tstd)

    @climo.register_derivation('emf_corr')
    def eddy_angular_momentum_flux_correlation_coefficient(self):
        # Correlation coefficient for momentum flux
        return self.ehf / (self.vstd * self.ustd)

    @climo.register_derivation(('ustd', 'vstd', 'tstd', 'zstd'))
    def zonal_standard_deviation(self, name):
        # Standard deviation
        data = np.sqrt(self[name[0] + 'var'])
        if attr := data.attrs.get('long_name'):
            data.attrs['long_name'] = attr.replace('variance', 'standard deviation')
        return data

    # ### Mass terms
    @climo.register_derivation('ve')
    def eddy_tem_meridional_wind(self):
        # Eddy-induced TEM meridional wind (see O'Gorman dynamics notes)
        # Use global mean potential temperature to scale
        # WARNING: This will rarely vertically integrate to zero because streamfunction
        # is heavily concentrated near surface (see O'Gorman Figure 5.1). Might also
        # get weird results when global mean inversions are present (as with fast
        # damping timescales).
        ehf = self.ehf
        dptdp = self.pt.climo.average('area').climo.derivative(lev=1)
        return (ehf / dptdp).climo.derivative(lev=1)

    @climo.register_derivation('omegae')
    def eddy_tem_vertical_wind(self):
        # Eddy-induced vertical wind (see O'Gorman dynamics notes)
        # Use global mean potential temperature to scale
        # Recall that dz / dt = dp / dt * 1 / (dp / dz) = dp / dt * (-1 / (rho * g))
        # Means that w = d(ehf / (dtheta / dz)) / dphi so that
        # omega = -(rho * g) * d(ehf / ((dtheta / dp) * (-rho * g))) / dphi
        # i.e. rhos cancel. The -ve should be there, original textbook has it
        ehf = self.ehf
        dptdp = self.pt.climo.average('area').climo.derivative(lev=1)
        return -1 * (ehf / dptdp).climo.divergence()
        # return -rho * (ehf / (rho * dptdp)).climo.divergence()

    @climo.register_derivation('vresid')
    def residual_tem_meridional_wind(self):
        # Residual (diabatic) meridional wind
        return self.v - self.ve

    @climo.register_derivation('omegaresid')
    def residual_tem_vertical_wind(self):
        # Residual (diabatic) vertical wind
        return self.omega - self.omegae

    @climo.register_derivation(('mpsi', 'mpsie', 'mpsiresid'))
    def meridional_mass_streamfunction(self, name):
        # Meridional mass streamfunction
        # Hartmann formula: Psi_M = (2 * pi * a * cos(phi)) / g * int_0^p [v] dp
        # SI units are kg/s, but want to convert to Tg/s using (1e3/1e12)
        v = 'v' if name == 'mpsi' else 've' if name == 'mpsie' else 'vresid'
        data = 2 * np.pi * const.a * self.cos * self[v]
        data = data.climo.cumintegral('lev')  # strat from TOA
        return -1 * data  # does not follow right-hand rule without this :/

    # ### Eliassen Palm
    @climo.register_derivation('qgpv')
    def ertel_isentropic_potential_vorticity(self):
        # TODO: Integrate with metpy
        # TODO: Add this
        raise NotImplementedError('Cannot get potential vorticity.')

    @climo.register_derivation('qgpv')
    def quasi_geostrophic_potential_vorticity(self):
        # TODO: Integrate with metpy
        # TODO: Add this
        raise NotImplementedError('Cannot get potential vorticity.')

    @climo.register_derivation('epy')
    def meridional_eliassan_palm_flux(self):
        # The full EP flux vector without QG approximation (see O'Gorman notes)
        return -1 * self.emf + self.u.climo.derivative(lev=1) * self.ehf / self.dptdp

    @climo.register_derivation('epz')
    def vertical_eliassan_palm_flux(self):
        # The full EP flux vector without QG approximation (see O'Gorman notes)
        # NOTE: This excludes the u*omega* component for now
        return (self.cor - self.u.climo.divergence()) * self.ehf / self.dptdp

    @climo.register_derivation('qgepy')
    def quasi_geostrophic_meridional_eliassan_palm_flux(self):
        # Meridional component of EP flux. Weighted by cosine for displaying
        # on plot with latitude axis unweighted by sine (Andrews et al.)
        return -1 * self.emf * self.cos / (const.a * np.pi)  # scaling for display

    @climo.register_derivation('qgepz')
    def quasi_geostrophic_vertical_eliassan_palm_flux(self):
        # Vertical component of EP flux. NOTE: Why cosine weighting here too?
        # Pressure is descending *up* in plots so multiply by -1.
        f = self.cor
        if self.cf.vertical_type == 'temperature':
            data = self.eqf + self.stress
            scale = 100 * ureg.K  # scaling for display?
        else:
            data = f * self.ehf / self.dptdp
            scale = 1e5 * ureg.Pa  # scaling for display?
        return -1 * data * self.cos / scale

    @climo.register_derivation(('epc', 'epd'))
    def eliassen_palm_flux_convergence_divergence(self, name):
        # Divergence and convergence of EP flux. See Andrews et al.
        # NOTE: When computing contribution of eddy fluxes to zonal wind budget,
        # end up with 2 cosines in both numerator and denominator (becuase we divide
        # by the cosine associated with angular momentum L on the LHS of equation).
        # Here, angular momentum (not zonal wind) is the quantity involved with wave
        # fluxes, so end up with 2 cosines in numerator and 1 in denominator.
        # NOTE: Units of py are m^2 / s^2, dervative has units m / s^2.
        # Units of pz are K m s^-1 * s^-1 * (Pa / K) = Pa m / s^2, vertical
        # derviative makes this m / s^2.
        data = self.epy.climo.divergence() + self.epz.climo.derivative(lev=1)
        if name == 'epc':
            data *= -1
        return data

    # ### Length and time scales
    @climo.register_derivation(('egrowth', 'etimescale'))
    def eady_growth_rate_timescale(self, name):
        # Eady growth rate or timescale
        b = self.b  # instability
        f = self.cor
        dudz = -1 * self.shear
        data = 0.3098 * dudz * f / b
        if name == 'etimescale':
            data = 1 / data
        return data

    @climo.register_derivation(('ks', 'ls'))
    def stationary_wavenumber_lengthscale(self, name):
        # Stationary wavenumber ks \equiv sqrt(beta/ubar)
        beta = self.beta
        u = self.u  # optionally use the thermal wind here
        if name == 'ks':  # want zonal wavenumber
            # Get cycles per zonal band.
            circum = 2 * np.pi * const.a * self.cos
            data = (circum / (2 * np.pi)) * np.sqrt(beta / u)
            data = data.climo.to_units('')
        else:
            # Get meters per cycle
            data = 2 * np.pi / np.sqrt(beta / u)
            data = data.climo.to_units('m')
        return data

    @climo.register_derivation(('ld', 'constld', 'massld'))
    def rossby_deformation_radius(self, name):  # noqa: U100
        # Rhossby deformation radius
        # TODO: Get ave scale height *up to tropopause*
        # b = self.get('b', lev='avg', lev_min=250 * ureg.hPa)
        f = self.cor
        b = self.get('b')
        ztrop = self.ztrop
        return b * ztrop / f  # radius deformation

    @climo.register_derivation('lr')
    def rhines_beta_scale(self):
        # Rhines beta-scale
        # vstd = self.get('vstd', lev='avg', lev_min=250 * ureg.hPa)
        beta = -1 * self.beta
        vstd = self.get('vstd')
        return np.sqrt(2) * vstd / np.sqrt(beta)

    @climo.register_derivation('ldisp')
    def barry_eddy_displacement_scale(self):
        # Barry et al. (2005) displacement length scale
        # tstd = self.get('tstd', lev='avg', lev_min=250 * ureg.hPa)
        # dtdy = self.get('dtdy', lev='avg', lev_min=250 * ureg.hPa)
        tstd = self.get('tstd')
        dtdy = self.get('dtdy')
        return tstd / np.abs(dtdy)

    # ### Tropopause metrics
    @climo.register_derivation('pvtrop')
    def potential_vorticity_tropopause(self):
        # 2 PVU tropopause
        if self.cf.vertical_type == 'temperature':
            raise NotImplementedError('Cannot get 2PVU tropopause.')
        else:
            raise NotImplementedError('Cannot get 2PVU tropopause.')

    @climo.register_derivation('mtrop')
    def mass_streamfunction_tropopause(self):
        # The overturning-circulation-determined tropopause height.
        mpsi = self.get('mpsiresid', area='avg', lat_lim=(10, 40))
        thresh = mpsi.min(dim='lev') * 0.1
        data = (mpsi - thresh).climo.reduce(
            lev='argzero', lev_max=500 * ureg.hPa, dataset=self.data,
        )
        return data.max(dim='line')

    @climo.register_derivation(('trop', 'ztrop'))
    def lapse_rate_tropopause(self, name):
        # Lapse rate tropopause: 2 K / km height. Look for points that go from less
        # negative to more negative lapse rates with increasing pressure.
        if self.cf.vertical_type == 'temperature':
            raise NotImplementedError('Cannot get tropopause for isentropic data.')
        trop = -2 * ureg.K / ureg.km
        dtdz = self.dtdz
        data = (dtdz - trop).climo.reduce(
            lev='argzero', lev_min=80 * ureg.hPa, lev_max=350 * ureg.hPa,
            dim_track='lat', which='posneg', dataset=self.data,
        )
        if 'track' in data.dims:
            data = data.min(dim='track')  # minimum of obtained values
        if name == 'ztrop':
            data = const.H * np.log(const.p0 / data)  # log-pressure coords
            data = data.climo.to_units('km')
        return data

    @climo.register_derivation(('ctrop', 'zctrop'))
    def thermal_curvature_tropopause(self, name):
        # Curvature tropopause: Where temperature is stabilizing fastest, i.e. rate
        # of rate of decrease is fastest, i.e. curvature has minimum.
        dcurv = self.dcurvature
        data = dcurv.climo.reduce(
            lev='argzero', lev_max=350 * ureg.hPa, dataset=self.data,
            dim_track='lat', ntrack=1, which='negpos',
            seed=150, sep=50,  # TODO: support units here
        )
        if 'track' in data.dims:
            data = data.min(dim='track')
        if name == 'zctrop':
            data = const.H * np.log(const.p0 / data)  # log-pressure coords
            data = data.climo.to_units('km')
        return data

    # ### Bulk criticality metrics
    @climo.register_derivation('slope_bulk')
    def bulk_isentropic_slope(self):
        # Bulk slope (should have units meters per meter)
        return self.dptdy_bulk / self.dptdz_bulk

    @climo.register_derivation('slope_diff')
    def diff_isentropic_slope(self):
        # The criticality parameter.
        pt = self.get('pt', lev='avg', lat=LAT_LIM, lev_lim=LEV_LIM)
        dptdy = pt.diff('lat').squeeze(drop=True)
        pt = self.get('pt', area='avg', lev=LEV_LIM, lat_lim=LAT_LIM)
        dptdz = pt.diff('lev').squeeze(drop=True)
        return dptdy / dptdz

    @climo.register_derivation('shear_bulk')
    def bulk_vertical_zonal_wind_shear(self):
        # Upper-lower difference
        data = (
            self.get('u', lev=250 * ureg.hPa)
            - self.get('u', lev=1000 * ureg.hPa)
        )
        return data

    @climo.register_derivation('ratio')
    def equilibrium_temperature_ratio(self):
        # Ratio between equilibrium difference and actual difference that was observed
        params = [
            -self.get(s, lev='avg', lat_lim=LAT_LIM, lev_lim=LEV_LIM)
            .diff('lat').squeeze(drop=True) for s in ('pt', 'pteq')
        ]
        return params[0] / params[1]

    @climo.register_derivation(('dtdy_bulk', 'dptdy_bulk', 'ddsedy_bulk'))
    def bulk_meridional_gradient(self, name):
        # Mean temperature gradient across box (calculated as difference)
        # NOTE: Used to just subtract gradients, but that failes to account
        # for cosine latitude weights! Invalid method!
        name = re.sub(r'\Ad(.*)dy_bulk\Z', r'\1', name)
        vertical = self.cf.vertical_type
        if vertical == 'temperature' and name == 'dse':
            raise NotImplementedError('Cannot get DSE gradient on isentropic levels.')
        elif vertical == 'temperature':
            # Use gradient of sea-level potential temp (close enough to temp)
            idx = np.round(0.5 * self.x.size).astype(int)
            dt = -1 * self.slth.isel(x=idx).climo.derivative(y=1)  # 50th percentile
            dt = dt.climo.average('area', lat_lim=LAT_LIM)
        else:
            # Option A: Vertical average of meridional derivative
            dt = -1 * self.get(name).climo.derivative(y=1)
            dt = dt.climo.average('volume', lat_lim=LAT_LIM, lev_lim=LEV_LIM)
            # Option B: Meridional derivative of vertical average
            # t = self.get(name, lev='avg', lev_lim=LEV_LIM)
            # dt = -1 * t.climo.derivative(y=1)
            # dt = dt.climo.average('area', lat_lim=LAT_LIM)
            # Option C: Non-area-weighted average
            # t = self.get(name, lat=LAT_LIM, lev='avg', lev_lim=LEV_LIM)
            # dt = -1 * t.diff('lat').squeeze(drop=True)
            # dt /= (0.5 * np.pi * const.a * np.diff(LAT_LIM)).to('km').squeeze()
        return dt

    @climo.register_derivation(('dtdz_bulk', 'dptdz_bulk'))
    def bulk_vertical_gradient(self, name):
        # The difference between mean surface temp and tropopause temp.
        # NOTE: Critical to interpolate to levels first
        name = re.sub(r'\Ad(.*)dz_bulk\Z', r'\1', name)
        # Option A: Horizontal average of vertical gradient
        dt = self.get(name).climo.derivative(z0=1)
        dt = dt.climo.average('volume', lat_lim=LAT_LIM, lev_lim=LEV_LIM)
        # Option B: Vertical gradient of horizontal average
        # t = self.get(name, area='avg', lat_lim=LAT_LIM)
        # dt = t.climo.derivative(z0=1)
        # dt = dt.climo.average('lev', lev_lim=LEV_LIM)
        # Option C: Height-weighted average
        # z = self.coords.z0
        # t = self.get(name, lev=LEV_LIM, area='avg', lat_lim=LAT_LIM)
        # # t = self.get(name)
        # # t = t.drop_vars('cell_height').climo.add_cell_measures()
        # # t = t.climo.reduce(lev=LEV_LIM, area='avg', lat_lim=LAT_LIM)
        # dt = t.diff(dim='lev').squeeze(drop=True)
        # dt /= z.climo.interp(lev=LEV_LIM).diff('lev').squeeze(drop=True)
        return dt

    # ### Bulk diffusivity metrics
    @climo.register_derivation(('ehf_bulk', 'edsef_bulk'))
    def bulk_eddy_flux(self, name):
        # Preset average for meridional eddy flux
        name, _ = name.split('_')
        return self.get(name, lev=850 * ureg.hPa, area='avg', lat_lim=LAT_LIM)
        # return self.get(name, lev='avg', lev_lim=LEV_LIM, area='avg', lat_lim=LAT_LIM)

    @climo.register_derivation(re.compile(r'\Z(t|dse)_diffusivity_bulk'))
    def bulk_eddy_diffusivity(self, name):
        # Diffusivity using bulk flux and gradient metrics
        name, _, _ = name.split('_')
        flux = self.ehf_bulk if name == 't' else self.edsef_bulk
        dfdy = self.get('d' + name + 'dy_bulk')
        return flux / dfdy

    @climo.register_derivation('behf')
    def barry_eddy_heat_flux(self):
        # Barry et al. (2005) heat flux param. TODO: Double check
        e = 0.75  # utilization coefficient
        q = self.get('tdt', lev='avg', lev_min=500 * ureg.hPa)
        k = self.get('corr', lev='avg', lev_min=500 * ureg.hPa)
        t0 = self.get('t', area='avg', lev='avg')
        dtdy = self.dtdy_bulk
        beta = self.beta
        return k * dtdy ** 1.6 * (e * const.a * q / t0) ** 0.6 * (2 / beta) ** 0.8

    @climo.register_derivation('bvstd')
    def barry_meridional_wind_zonal_standard_deviation(self):
        # Barry et al. (2005) RMS meridional wind param. TODO: Double check
        e = 0.75  # utilization coefficient
        q = self.get('tdt', lev='avg', lev_min=500 * ureg.hPa)
        t0 = self.get('t', area='avg', lev='avg')
        dtdy = self.get('dtdy', lev='avg', lev_min=500 * ureg.hPa)
        beta = self.beta
        return (e * const.a * dtdy * q / t0) ** 0.4 * (2 / beta) ** 0.2
