import os

import numpy as np
from numpy import pi, array, sort, multiply, sum
import pprint

from isca import SocratesCodeBase, DiagTable, Experiment, Namelist, GFDL_BASE
from isca.util import exp_progress

#  ----------- Global constants -----------
# AU definition [m]
AU = 149597870700.
# Universal gravitational constant [m^3.kg^-1.s^-2]
G_universal = 6.67430e-11
# Universal gas constant [J.mol^-1.K^-1]
R = 8.31446261815324 
# Number of seconds in a day
day_s = 24.0*3600.0

# ===== SUN/EARTH PARAMETERS =====
M_Sun        = 1988400e24 # Solar mass [kg]
L_Sun        = 3.828e26   # Solar luminosity [W]
radius_Earth = 6.371e6    # Earth radius [m]
mass_Earth   = 5.9722e24  # Earth mass [kg]
grav_Earth   = 9.807      # Earth surface gravity [m.s^-2]

def angular_velocity(orbital_period):
	return 2.0*pi/(orbital_period*day_s)

def weighted_mean(mix_ratios,y):
	return sum(multiply(array(mix_ratios),array(y))) 

def surface_gravity(mass,radius):
	g = G_universal*array(mass)*mass_Earth/(array(radius)*radius_Earth)**2
	if g.ndim > 0:
		g = sort(g)
	return g

# Sources: TRAPPIST-1: doi:10.3847/psj/abd022 ; Proxima-b: doi:10.3847/2041-8205/831/2/l16 (Figure 2) ; Teegarden: doi:10.1051/0004-6361/201935460 (Figure 12). 
planets = { 'Earth': {'Stellar Luminosity': L_Sun, 'Radius': radius_Earth, 'Gravity': grav_Earth, 'Semi-major axis': AU, 'Angular velocity': angular_velocity(1.0)},
            'TRAPPIST-1b': {'Stellar Luminosity': 0.000553*L_Sun, 'Radius': 1.116*radius_Earth, 'Gravity': 1.102*grav_Earth, 'Semi-major axis': 0.01154*AU, 'Angular velocity': angular_velocity(1.510826)},
			'TRAPPIST-1d': {'Stellar Luminosity': 0.000553*L_Sun, 'Radius': 0.788*radius_Earth, 'Gravity': 0.624*grav_Earth, 'Semi-major axis': 0.02227*AU, 'Angular velocity': angular_velocity(4.049219)},
			'TRAPPIST-1e': {'Stellar Luminosity': 0.000553*L_Sun, 'Radius': 0.920*radius_Earth, 'Gravity': 0.817*grav_Earth, 'Semi-major axis': 0.02925*AU, 'Angular velocity': angular_velocity(6.101013)},
			'Proxima-b':   {'Stellar Luminosity': 0.001567*L_Sun, 'Radius': array([0.94,1.4])*radius_Earth, 'Gravity': surface_gravity(1.07,[0.94,1.4]), 'Semi-major axis': 0.04856*AU, 'Angular velocity': angular_velocity(11.1868)},
			'Teegarden-b':   {'Stellar Luminosity': 0.00073*L_Sun, 'Radius': 1.02*radius_Earth, 'Gravity': surface_gravity(1.16,1.02), 'Semi-major axis': 0.0259*AU, 'Angular velocity': angular_velocity(4.90634)},
			'Teegarden-c':   {'Stellar Luminosity': 0.00073*L_Sun, 'Radius': 1.04*radius_Earth, 'Gravity': surface_gravity(1.05,1.04), 'Semi-major axis': 0.0455*AU, 'Angular velocity': angular_velocity(11.416)}}

atmospheres = { 'Earth': {'H2O': 1e-3, 'CO2': 400e-6, 'O3': 0.07e-6, 'N2O': 0.31e-6, 'CO': 0.001e-6, 'CH4': 1e-6, 'O2': 0.20947, 'NO': 0.0, 'SO2': 0.0, 'NO2': 0.02e-6, 'NH3': 1.0e-7, 'HNO3': 0.0, 'N2': 0.78084, 'H2': 0.03e-6, 'He': 5.24e-6, 'OCS': 0.0},
                'Pure Steam': {'H2O': 1.0, 'CO2': 0.0, 'O3': 0.0, 'N2O': 0.0, 'CO': 0.0, 'CH4': 0.0, 'O2': 0.0, 'NO': 0.0, 'SO2': 0.0, 'NO2': 0.0, 'NH3': 0.0, 'HNO3': 0.0, 'N2': 0.0, 'H2': 0.0, 'He': 0.0, 'OCS': 0.0},
				'Archean Earth': {'H2O': 1e-3, 'CO2': 20000e-6, 'O3': 0.0, 'N2O': 0.0, 'CO': 10e-6, 'CH4': 30e-6, 'O2': 0.20947e-6, 'NO': 0.0, 'SO2': 0.0, 'NO2': 0.0, 'NH3': 0.0, 'HNO3': 0.0, 'N2': 0.97991, 'H2': 50e-6, 'He': 0.0, 'OCS': 0.0},
				'Eocene Earth': {'H2O': 1e-3, 'CO2': 1500e-6, 'O3': 0.07e-6, 'N2O': 0.31e-6, 'CO': 0.10e-6, 'CH4': 5e-6, 'O2': 0.25, 'NO': 0.0, 'SO2': 0.0, 'NO2': 0.02e-6, 'NH3': 1.0e-7, 'HNO3': 0.0, 'N2': 0.78849, 'H2': 0.03e-6, 'He': 5.24e-6, 'OCS': 0.0}}

molecules = {'Molecular Weight': {'H2O': 0.018, 'CO2': 0.044, 'O3': 0.048, 'N2O': 0.044, 'CO': 0.028, 'CH4': 0.016, 'O2': 0.032, 'NO': 0.030, 'SO2': 0.064, 'NO2': 0.046, 'NH3': 0.017, 'HNO3': 0.063, 'N2': 0.028, 'H2': 0.002, 'He': 0.004, 'OCS': 0.060},
			 'Isobaric Heat Capacity': {'H2O': 1864.0, 'CO2': 849.0, 'O3': 819.375, 'N2O': 877.364, 'CO': 1040.0, 'CH4': 2232.0, 'O2': 918.0, 'NO': 995.0, 'SO2': 624.0625, 'NO2': 805.0, 'NH3': 2175.0, 'HNO3': 849.365, 'N2': 1040.0, 'H2': 14310.0, 'He': 5197.5, 'OCS': 41.592}}

molecular_weights_values = list(molecules['Molecular Weight'].values())
molecular_weights = molecules['Molecular Weight']

heat_capacities   = list(molecules['Isobaric Heat Capacity'].values())

def calculate_molar_mass(atmosphere, molecular_weights):
    # Calculate the molar mass of the atmosphere in kg/mol
    molar_mass_air = sum((fraction * molecular_weights[gas]) for gas, fraction in atmosphere.items())
    return molar_mass_air

def convert_ppmv_to_ppmm(atmosphere, molecular_weights, molar_mass_air):
    # Convert ppmv to ppmm using the molar mass of each gas and the molar mass of air
    atmosphere_ppmm = {}
    for gas, ppmv in atmosphere.items():
        molar_mass_gas = molecular_weights[gas]
        ppmm = (ppmv * molar_mass_gas) / molar_mass_air
        atmosphere_ppmm[gas] = ppmm
    return atmosphere_ppmm

# Calculate the ppmm for each atmosphere using your molecular weights
atmospheres_ppmm = {}
for name, atmosphere in atmospheres.items():
    molar_mass_air = calculate_molar_mass(atmosphere, molecular_weights)
    atmospheres_ppmm[name] = convert_ppmv_to_ppmm(atmosphere, molecular_weights, molar_mass_air)

# Print the result
pprint.pprint(atmospheres_ppmm)

# ===== SELECT PLANET AND ATMOSPHERE =====
planet     = 'Teegarden-b'
atmosphere = 'Earth'

# ===== BULK PLANET PARAMETERS =====
# Stellar mass [kg]
M_Star = 0.0970*M_Sun
# Stellar luminosity [W]
L_Star = planets[planet]['Stellar Luminosity']    
# Radius [m]
radius = planets[planet]['Radius']     
# Gravity [m.s^-2]
grav = planets[planet]['Gravity']
# Semi-major axis [m]
a_major = planets[planet]['Semi-major axis']
# Angular velocity [rad.s^-1]
omega = planets[planet]['Angular velocity']
# Orbital period [s]
orbital_period = 4.90634 * 86400.

# ===== ATMOSPHERE PARAMETERS =====
# Mixing ratios
atmospheric_composition = list(atmospheres[atmosphere].values())
# Mean molecular weight of air [g.mol^-1]
wtmair = weighted_mean(atmospheric_composition,molecular_weights_values)*1000.0 
# Gas constant [J.kg^-1.K^-1]
rdgas = R/weighted_mean(atmospheric_composition,molecular_weights_values) 
# Specific isobaric heat capacity of air at 300 K [J.kg^-1.K^-1]
cp_air = weighted_mean(atmospheric_composition,heat_capacities)       
# Lapse rate    
kappa = rdgas/cp_air
# Solar constant [W.m-2]
solar_constant = 1564.707 # (1.0 - 0.0) * L_Star / (4. * np.pi * a_b**2) = (1.0 - 0.0) * 0.00073*3.828e26 / (4. * np.pi * (0.0252*AU)**2)
# Surface pressure [Pa]
p_surf = 0.23e5


NCORES = 16
base_dir = os.path.dirname(os.path.realpath(__file__))
# a CodeBase can be a directory on the computer,
# useful for iterative development
cb = SocratesCodeBase.from_directory(GFDL_BASE)

# or it can point to a specific git repo and commit id.
# This method should ensure future, independent, reproducibility of results.
# cb = DryCodeBase.from_repo(repo='https://github.com/isca/isca', commit='isca1.1')

# compilation depends on computer specific settings.  The $GFDL_ENV
# environment variable is used to determine which `$GFDL_BASE/src/extra/env` file
# is used to load the correct compilers.  The env file is always loaded from
# $GFDL_BASE and not the checked out git repo.

# create an Experiment object to handle the configuration of model parameters
# and output diagnostics

exp = Experiment('planetb_ArcheanEarth_rot0_days', codebase=cb)
exp.clear_rundir()

inputfiles = [os.path.join(GFDL_BASE,'input/rrtm_input_files/ozone_1990_notime_zero.nc')]

#Tell model how to write diagnostics
diag = DiagTable()
diag.add_file('atmos_daily', 1, 'days', time_units='days')

#Write out diagnostics need for vertical interpolation post-processing
diag.add_field('dynamics', 'ps', time_avg=True)
diag.add_field('dynamics', 'bk')
diag.add_field('dynamics', 'pk')
diag.add_field('dynamics', 'zsurf')

#Tell model which diagnostics to write
diag.add_field('atmosphere', 'precipitation', time_avg=True)
diag.add_field('atmosphere', 'rh', time_avg=True)
diag.add_field('mixed_layer', 't_surf', time_avg=True)
diag.add_field('mixed_layer', 'flux_t', time_avg=True) #SH
diag.add_field('mixed_layer', 'flux_lhe', time_avg=True) #LH
diag.add_field('dynamics', 'sphum', time_avg=True)
diag.add_field('dynamics', 'ucomp', time_avg=True)
diag.add_field('dynamics', 'vcomp', time_avg=True)
diag.add_field('dynamics', 'omega', time_avg=True)
diag.add_field('dynamics', 'temp', time_avg=True)

#temperature tendency - units are K/s
diag.add_field('socrates', 'soc_tdt_lw', time_avg=True) # net flux lw 3d (up - down)
diag.add_field('socrates', 'soc_tdt_sw', time_avg=True)
diag.add_field('socrates', 'soc_tdt_rad', time_avg=True) #sum of the sw and lw heating rates

#net (up) and down surface fluxes
diag.add_field('socrates', 'soc_surf_flux_lw', time_avg=True)
diag.add_field('socrates', 'soc_surf_flux_sw', time_avg=True)
diag.add_field('socrates', 'soc_surf_flux_lw_down', time_avg=True)
diag.add_field('socrates', 'soc_surf_flux_sw_down', time_avg=True)
#net (up) TOA and downward fluxes
diag.add_field('socrates', 'soc_olr', time_avg=True)
diag.add_field('socrates', 'soc_toa_sw', time_avg=True) 
diag.add_field('socrates', 'soc_toa_sw_down', time_avg=True)
diag.add_field('socrates', 'soc_flux_lw_up', time_avg=True)
diag.add_field('socrates', 'soc_flux_lw_down', time_avg=True)
diag.add_field('socrates', 'soc_flux_sw_up', time_avg=True)
diag.add_field('socrates', 'soc_flux_sw_down', time_avg=True)
diag.add_field('socrates', 'soc_flux_direct', time_avg=True)

#diag.add_field('socrates', 'soc_cff', time_avg=True)

diag.add_field('socrates', 'soc_spectral_olr', time_avg=True)

#clear sky fluxes
diag.add_field('socrates', 'soc_surf_flux_lw_clr', time_avg=True)
diag.add_field('socrates', 'soc_surf_flux_sw_clr', time_avg=True)
diag.add_field('socrates', 'soc_surf_flux_lw_down_clr', time_avg=True)
diag.add_field('socrates', 'soc_surf_flux_sw_down_clr', time_avg=True)
diag.add_field('socrates', 'soc_olr_clr', time_avg=True)
diag.add_field('socrates', 'soc_toa_sw_clr', time_avg=True) 
diag.add_field('socrates', 'soc_toa_sw_down_clr', time_avg=True) 
diag.add_field('socrates', 'soc_flux_lw_up_clr', time_avg=True)
diag.add_field('socrates', 'soc_flux_lw_down_clr', time_avg=True)
diag.add_field('socrates', 'soc_flux_sw_up_clr', time_avg=True)
diag.add_field('socrates', 'soc_flux_sw_down_clr', time_avg=True)
diag.add_field('socrates', 'soc_flux_direct_clr', time_avg=True)

diag.add_field('cloud_simple', 'cf', time_avg=True)
diag.add_field('cloud_simple', 'reff_rad', time_avg=True)
diag.add_field('cloud_simple', 'frac_liq', time_avg=True)
diag.add_field('cloud_simple', 'qcl_rad', time_avg=True)
#diag.add_field('cloud_simple', 'simple_rhcrit', time_avg=True)
diag.add_field('cloud_simple', 'rh_min', time_avg=True)
diag.add_field('cloud_simple', 'rh_in_cf', time_avg=True)
diag.add_field('mixed_layer', 'albedo')


exp.diag_table = diag
exp.inputfiles = inputfiles

#Define values for the 'core' namelist
exp.namelist = namelist = Namelist({
    'main_nml':{
     'days'   : 1,
     'hours'  : 0,
     'minutes': 0,
     'seconds': 0,
     'dt_atmos':120,#450, #600, 
     'current_date' : [1,1,1,0,0,0],
     'calendar' : 'no_calendar' #'thirty_day'
    },
    'socrates_rad_nml': {
        'stellar_constant':solar_constant, 
        'lw_spectral_filename':"/proj/bolinc/users/x_ryabo/socrates_edited_for_isca/spectral_files_for_GCMs/miniSuran_lw.sf",
        'sw_spectral_filename':"/proj/bolinc/users/x_ryabo/socrates_edited_for_isca/spectral_files_for_GCMs/miniSuran_sw.sf",
        'do_read_ozone': True,
        'ozone_file_name':'ozone_1990_notime_zero',
        'ozone_field_name':'ozone_1990',
        'co2_ppmv': 20000.0,
        'dt_rad':3600,
        'store_intermediate_rad':True,
        'chunk_size': 16,
        'use_pressure_interp_for_half_levels':False,
        'tidally_locked':True,
        'solday':90,
        'co_mix_ratio': 9.881e-6, # Well mixed gas concentrations (kg / kg)
        'n2o_mix_ratio': 0.0,
        'ch4_mix_ratio': 13.698e-6,
        'o2_mix_ratio': 0.2314e-6,
        'so2_mix_ratio': 0.0,
        'h2_mix_ratio': 3.529e-6,
        'n2_mix_ratio': 0.96828
    }, 
    'idealized_moist_phys_nml': {
        'do_damping': True,
        'turb':True,
        'mixed_layer_bc':True,
        'do_virtual' :False,
        'do_simple': True,
        'roughness_mom':3.21e-05,
        'roughness_heat':3.21e-05,
        'roughness_moist':3.21e-05,            
        'two_stream_gray': False,     #Use the grey radiation scheme
        'do_socrates_radiation': True,
        'convection_scheme': 'SIMPLE_BETTS_MILLER', #Use simple Betts miller convection            
        'do_cloud_simple': True # this is where the clouds scheme is turned on
    },

    'cloud_spookie_nml': { #use all existing defaults as in code
        'spookie_protocol':2
    },

    'vert_turb_driver_nml': {
        'do_mellor_yamada': False,     # default: True
        'do_diffusivity': True,        # default: False
        'do_simple': True,             # default: False
        'constant_gust': 0.0,          # default: 1.0
        'use_tau': False 
    },

    'diffusivity_nml': {
        'do_entrain':False,
        'do_simple': True,
    },

    'surface_flux_nml': {
        'use_virtual_temp': False,
        'do_simple': True,
        'old_dtaudv': True    
    },

    'atmosphere_nml': {
        'idealized_moist_model': True
    },

    #Use a large mixed-layer depth, and the Albedo of the CTRL case in Jucker & Gerber, 2017
    'mixed_layer_nml': {
        'tconst' : 310., #285. The Archean should've had at most 313.15K/40C, though of course it will change with tidal locking.
        'prescribe_initial_dist':True,
        'evaporation':True,  
        'depth': 2.5,                          #Depth of mixed layer used
        'albedo_value': 0.3,                  #Albedo value used      
    },

    'qe_moist_convection_nml': {
        'rhbm':0.7,
        'Tmin':160.,
        'Tmax':400.#350.  
    },
    
    'lscale_cond_nml': {
        'do_simple':True,
        'do_evap':True
    },
    
    'sat_vapor_pres_nml': {
        'show_bad_value_count_by_slice':True,
        'show_all_bad_values':True, # Show all bad temperatures causing saturation vapor pressure table overflows
        'do_simple':True,
        'construct_table_wrt_liq_and_ice':False,
        'do_not_calculate': False, # Don't compute esat for non-Earth simulations
        'tcmin': -160, 
        'tcmax': 100, 
        'tcmin_simple': -273, #-173, 
        'tcmax_simple': 350 # in Celsius, so 623.15K
       },
    
    'damping_driver_nml': {
        'do_rayleigh': True,
        'trayfric': -0.5,              # neg. value: time in *days*
        'sponge_pbottom':  650.0*(p_surf/1e5), #150., #Setting the lower pressure boundary for the model sponge layer in Pa.
        'do_conserve_energy': True,       
    },

    # FMS Framework configuration
    'diag_manager_nml': {
        'mix_snapshot_average_fields': False  # time avg fields are labelled with time in middle of window
    },

    'fms_nml': {
        'domains_stack_size': 600000                        # default: 0
    },

    'fms_io_nml': {
        'threading_write': 'single',                         # default: multi
        'fileset_write': 'single',                           # default: multi
    },

    'spectral_dynamics_nml': {
        'damping_order': 4,             
        'water_correction_limit': 200.e2,
        'reference_sea_level_press':p_surf,
        'num_levels':40,      #How many model pressure levels to use
        'valid_range_t':[1.,800.], #[100.,800.], #[20.,2000.]
        'initial_sphum':[2.e-6],
        'vert_coord_option':'uneven_sigma',
        'surf_res':0.075,#0.2, #Parameter that sets the vertical distribution of sigma levels
        'scale_heights' : 7.6, #11.0,
        'exponent':4.0, #7.0,
        'robert_coeff':0.03
    },

    'astronomy_nml': {
    'ecc':0.0,
    'obliq' : 0.0,
    'period':orbital_period # specified length of year (orbital period) in seconds
    },

    'constants_nml': {
#Set target constants
        'radius':  radius,
        'grav':  grav,
        'omega':  omega,
        'orbital_period':  orbital_period,
        'PSTD':  p_surf*10.0, # mean sea level pressure [dynes/cm^2 = 0.1 Pa]
        'PSTD_MKS':  p_surf, # mean sea level pressure [Newtons/m^2 = Pa]
        'rdgas':  rdgas,
		'cp_air':  cp_air,
		'kappa': kappa,
		'solar_const': solar_constant,
		'orbit_radius': a_major/AU
    }
})

#Lets do a run!
if __name__=="__main__":

        cb.compile(debug=False)
        #Set up the experiment object, with the first argument being the experiment name.
        #This will be the name of the folder that the data will appear in.

        overwrite=False
        
        restart_files = '/proj/bolinc/users/x_ryabo/Isca_outputs/planetb_ArcheanEarth_rot0_days/run0546/res0546/*'
        exp.run(546, use_restart=restart_files, num_cores=NCORES, overwrite_data=overwrite)

        for i in range(547,720): # 60 years + 1 year: to 720, then to 732
            exp.run(i, num_cores=NCORES, overwrite_data=overwrite)
