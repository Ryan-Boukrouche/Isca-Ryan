import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import seaborn as sns # installing seaborn removes netcdf4 and libnetcdf ??
import lifesim
from lifesim.util.importer import SpectrumImporter
from spectres import spectres
import pandas as pd
import matplotlib.patches as mpatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
#import seaborn as sns
import cartopy.crs as ccrs
from netCDF4 import Dataset

save_figs = True

""" 
This script reads spectra from selected simulations and plots them with arbitrary error bars and significance graphs.
The spectral flux must be in [ph m-2 s-1 micron-1] as a function of [micron].
-------------------------------------------------------------------------------
                             Define constants 
-------------------------------------------------------------------------------
"""

# Global constants
h  = 6.62607015e-34 # Planck's constant [J.s]
c  = 2.99792458e8   # Speed of light [m/s] 
kb = 1.380649e-23   # Boltzmann constant [J/K]
c1 = 2*h*c**2       # First  radiation constant
c2 = h*c/kb         # Second radiation constant
light_year = 94607304725808000. # m
AU         = 149597870700.      # m
parsec     = 648000*AU / np.pi  # m
radius_Earth = 6.371e6          # m

# Target star parameters
teegarden_distance    = 3.832 # Distance to the target system in pc
teegarden_distance_m  = 3.832 * parsec # m
teegarden_temperature = 3034  # Star effective temperature in K
teegarden_radius      = 0.120 # Star effective radius in Solar radii

# Target planet parameters
teegarden_b_angular_separation = 0.0252*AU / teegarden_distance_m  # degrees
teegarden_b_angular_separation_asec = teegarden_b_angular_separation * 180.0 * 60 * 60 / np.pi # arcsec
radius_teegarden_b             = 1.02*radius_Earth
teegarden_c_angular_separation = 0.0443*AU / teegarden_distance_m  # degrees
teegarden_c_angular_separation_asec = teegarden_c_angular_separation * 180.0 * 60 * 60 / np.pi # arcsec
radius_teegarden_c             = 1.04*radius_Earth

""" 
-------------------------------------------------------------------------------
                    Set the LIFEsim simulation variables 
-------------------------------------------------------------------------------
"""

scenario = 'baseline' # Options: 'basline', 'pessimistic', 'optimistic'
spectral_resolution = 50  # Resolving power
minimum_wavelength = 4 # Minimum wavelength of the spectrometer in micron
maximum_wavelength = 18.5 # Maximum wavelength of the spectrometer in micron
integration_time = 1*24*60*60  # Overall integration time in s

planet_radius = radius_teegarden_b  # Planet effective radius in Earth radii
planet_angular_separation = teegarden_b_angular_separation_asec  # Planet angular separation from host star in arcsec
star_distance = teegarden_distance  # Distance to the target system in pc
star_temperature = teegarden_temperature  # Star effective temperature in K
star_radius = teegarden_radius  # Star effective radius in Solar radii
exozodi_level = 3  # Exozodi level in zodis

""" 
-------------------------------------------------------------------------------
                                 Functions 
-------------------------------------------------------------------------------
"""

def labels(CB, quantity):
    if quantity == 'Tfull':
        CB.set_label(r'Temperature, $T$ [K]', size=size)
    if quantity == 'precipitation':
        CB.set_label(r'Precipitation [mm day$^{-1}$]',size=size)    
    if quantity == 'relative_humidity':
        CB.set_label(r'Relative humidity, ${R_\mathrm{h}}$ [%]',size=size)   
    if quantity == 'specific_humidity':
        CB.set_label(r'Specific humidity, q [%]', size=size)
    if quantity == 'zonal_wind':
        CB.set_label(r'Zonal velocity, $u$ [m s$^{-1}$]',size=size)
    if quantity == 'meridional_wind':
        CB.set_label(r'Meridional velocity, $v$ [m s$^{-1}$]',size=size)
    if quantity == 'vertical_wind':
        CB.set_label(r'Vertical velocity, ${\omega}$ [m s$^{-1}$]',size=size)
    if quantity == 'soc_tdt_lw':
        CB.set_label(r'Temperature tendency due to LW radiation, ${\frac{\mathrm{dT}_{\mathrm{rad}}}{\mathrm{dt}}} \; [\mathrm{K} \; \mathrm{day}^{-1}]$', size=size)
    if quantity == 'soc_tdt_sw':
        CB.set_label(r'Temperature tendency due to SW radiation, ${\frac{\mathrm{dT}_{\mathrm{rad}}}{\mathrm{dt}}} \; [\mathrm{K} \; \mathrm{day}^{-1}]$', size=size)
    if quantity == 'soc_tdt_rad':
        CB.set_label(r'Temperature tendency due to radiation, ${\frac{\mathrm{dT}_{\mathrm{rad}}}{\mathrm{dt}}} \; [\mathrm{K} \; \mathrm{day}^{-1}]$', size=size)
    if quantity == 'soc_flux_lw_up':
        CB.set_label(r'Diffuse upward flux, ${F_{up}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_flux_lw_down':
        CB.set_label(r'Diffuse downward flux, ${F_{down}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_flux_sw_up':
        CB.set_label(r'Stellar upward flux, ${F_{up}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_flux_sw_down':
        CB.set_label(r'Stellar downward flux, ${F_{down}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_flux_direct':
        CB.set_label(r'Direct beam, ${F_{inc}}$ [W m$^{-2}$]', size=size)
    if quantity == 'flux_t':
        CB.set_label(r'Surface sensible heat flux, ${F_{t}}$ [W m$^{-2}$]', size=size)
    if quantity == 'flux_lhe':
        CB.set_label(r'Surface latent heat flux, ${F_{lhe}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_surf_flux_lw':
        CB.set_label(r'Socrates Net LW surface flux (up), ${F_{s,up}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_surf_flux_sw':
        CB.set_label(r'Socrates Net SW surface flux (down), ${F_{s,down}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_surf_flux_lw_down':
        CB.set_label(r'Socrates LW surface flux down, ${F_{s,down}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_surf_flux_sw_down':
        CB.set_label(r'Socrates SW surface flux down, ${F_{s,down}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_toa_sw':
        CB.set_label(r'Socrates Net TOA SW flux (down), ${F_{down}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_toa_sw_down':
        CB.set_label(r'Socrates TOA SW flux down, ${F_{down}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_olr':
        CB.set_label(r'Socrates OLR, ${F_{up}}$ [W m$^{-2}$]', size=size)
    if quantity == 'cloud_fraction':
        CB.set_label(r'Cloud fraction [%]', size=size)
    if quantity == 'droplet_radius':
        CB.set_label(r'Effective cloud particle radius [$\mu$m]', size=size)
    if quantity == 'frac_liq':
        CB.set_label(r'Liquid cloud fraction', size=size)
    if quantity == 'qcl_rad':
        CB.set_label(r'Specific humidity of cloud liquid [%]', size=size) 
    if quantity == 'rh_in_cloud':
        CB.set_label(r'Relative humidity in cloud [%]', size=size)
    if quantity == 'cff':
        CB.set_label(r'Flux contribution, ${\mathcal{CF}_\mathrm{F}}$ [W m$^{-2}$]', size=size)
    if quantity == 'p_surf':
        CB.set_label(r'Mean sea level pressure [Pa]', size=size)
    if quantity == 'T_surf':
        CB.set_label(r'Surface temperature [K]', size=size)
    if quantity == 'ozone':
        CB.set_label(r'Ozone mass mixing ratio [ppmm]', size=size)
        

def icm2imeter(x):
    # Convert inverse cm [cm-1] to inverse meter [m-1].
    return x * 1e2

def imeter2icm(x):
    # Convert inverse meter [m-1] to inverse cm [cm-1] 
    return x * 1e-2

def sigma2micron(x):
    return (x * 1e2)**(-1) * 1e6

def sigma2meter(x):
    return (x * 1e2)**(-1) 

def Wm2m_to_Wm2um(Wm2m):
    # Convert a radiant flux from W/m2/m to W/m2/μm
    return Wm2m * 1e-6 # Rationale: If we have one W/m2 across one meter, we must have a million times less in only one micron!

def Wm2um_to_Wm2m(Wm2um):
    # Convert a radiant flux from W/m2/micron to W/m2/m
    return Wm2um * 1e6

def Wm2cm_to_Wm2m(Wm2cm, wavenumber):
    # Convert a radiant flux from W/m2/cm-1 to W/m2/m. wavenumber is in cm-1.
    return 1e2 * wavenumber**2 * Wm2cm
    # F_sigma*d_sigma = F_lambda*dlambda --> F_lambda = F_sigma * (d_sigma/dlambda) 
    # = F_sigma * d(1/lambda)/dlambda = F_sigma * 1/lambda**2. 
    # F_lambda [W/m2/m] = F_sigma [W/m2/m-1] * 1/lambda[m]**2 = 1e-2 F_sigma [W/m2/cm-1] * sigma[m-1]**2
    # sigma[m-1]**2 = (sigma[cm-1]*1e2)**2 = sigma[cm-1]**2 * 1e4
    # Finally: F_lambda [W/m2/m] = 1e-2 F_sigma [W/m2/cm-1] * sigma[cm-1]**2 * 1e4 = 1e2 * sigma[cm-1]**2 * F_sigma [W/m2/cm-1] 

def Wm2cm_to_Wm2um(Wm2cm, wavenumber):
    # Convert a radiant flux from W/m2/cm-1 to W/m2/μm. wavenumber is in cm-1.
    return 1e-4 * wavenumber**2 * Wm2cm
    # F_sigma*d_sigma = F_lambda*dlambda --> F_lambda = F_sigma * (d_sigma/dlambda) 
    # = F_sigma * d(1/lambda)/dlambda = F_sigma * 1/lambda**2. 
    # F_lambda [W/m2/m] = F_sigma [W/m2/m-1] * 1/lambda[m]**2 = 1e-2 F_sigma [W/m2/cm-1] * sigma[m-1]**2
    # sigma[m-1]**2 = (sigma[cm-1]*1e2)**2 = sigma[cm-1]**2 * 1e4 and F_lambda [W/m2/m] = 1e6 * F_lambda [W/m2/um]
    # Finally: F_lambda [W/m2/um] = 1e-6 * 1e-2 F_sigma [W/m2/cm-1] * sigma[cm-1]**2 * 1e4 = 1e-4 * sigma[cm-1]**2 * F_sigma [W/m2/cm-1] 

def Wm2um_to_Wm2cm(Wm2um, wavelength):
    # Convert a radiant flux from W/m2/μm to W/m2/cm-1. wavelength is in micron.
    return 1e-4 * wavelength**2 * Wm2um 
    # F_sigma*d_sigma = F_lambda*dlambda --> F_sigma = F_lambda * (dlambda/d_sigma) 
    # = F_lambda * d(1/sigma)/dsigma = F_lambda * 1/sigma**2. 
    # F_sigma [W/m2/m-1] = F_lambda [W/m2/m] * 1/sigma[m-1]**2 = 1e6 F_lambda [W/m2/um] * lambda[m]**2
    # lambda[m]**2 = (lambda[um]*1e-6)**2 = lambda[um]**2 * 1e-12 and F_sigma [W/m2/m-1] = F_sigma [W/m2/cm-1] * 1e-2
    # F_sigma [W/m2/cm-1] = 1e6 F_lambda [W/m2/um] * lambda[um]**2 * 1e-12 * 1e2
    # Finally: F_sigma [W/m2/cm-1] = 1e-4 * F_lambda [W/m2/um] * lambda[um]**2

def Wm2m_to_Wm2cm(Wm2m, wavelength):
    # Convert a radiant flux from W/m2/m to W/m2/cm-1. wavelength is in meter.
    return 1e-2 * wavelength**2 * Wm2m

def radiant_flux_to_photon_flux(radiant_flux, wavelength):
    # Convert a radiant flux in W/m²/m to a photon flux in photon/m²/s/m. To convert to photon/m²/s/um just multiply by 1e-6
    E_photon = h * c / wavelength # Photon energy [J/photon], wavelength in m.
    photon_flux = radiant_flux / E_photon
    return photon_flux

def photon_flux_to_radiant_flux(photon_flux, wavelength):
    # Convert a photon flux in photon/m²/s/m to a radiant flux in W/m²/m    
    E_photon = h * c / wavelength # Photon energy [J/photon], wavelength in m.
    radiant_flux = photon_flux * E_photon
    return radiant_flux

def inverse_square_law(planet_flux, planet_radius, distance):
    # Compute the received flux from an emitted flux    
    flux_at_earth = planet_flux * (planet_radius/distance)**2 
    return flux_at_earth

def read_band_edges(sfpath:str) -> list:
    """Read the band edges from a spectral file, from spectral tools written by Harrison Nicholls

    Parameters
    ----------
    sfpath : str
        Path to file

    Returns
    -------
    list
        Band edges, ascending in units of [metres]
    """

    with open(sfpath,'r') as hdl:
        lines = hdl.readlines()
    nband = int(  str(lines[2]).split("=")[1].strip()  )
    band_edgesm = []
    block_idx:int = -999999999
    for l in lines:
        # We want block 1 data
        if "Band        Lower limit         Upper limit" in l:
            block_idx = 0 
        if (block_idx > 0) and ("*END" in l):
            break 
        
        # Read bands
        if (block_idx > 0):
            s = [float(ss.strip()) for ss in l.split()[1:]]
            # First edge
            if block_idx == 1:
                band_edgesm.append(s[0])  # Note that these are in units of [metres]
            # Upper edges
            band_edgesm.append(s[1])  # [metres]

        # Block index 
        block_idx += 1

    if len(band_edgesm)-1 != nband:
        raise Exception("Band edges could not be read from spectral file")
    
    return band_edgesm

def cell_area_calculate(lons, lats, lonb, latb, radius):
    """compute cell areas in metres**2. Taken from src/extra/python/scripts/cell_area.py."""
    nlon=lons.shape[0]
    nlat=lats.shape[0]

    area_array = np.zeros((nlat,nlon))
    area_array_2 = np.zeros((nlat,nlon))
    xsize_array = np.zeros((nlat,nlon))
    ysize_array = np.zeros((nlat,nlon))

    for i in np.arange(len(lons)):
        for j in np.arange(len(lats)):
            xsize_array[j,i] = radius*np.absolute(np.radians(lonb[i+1]-lonb[i])*np.cos(np.radians(lats[j])))
            ysize_array[j,i] = radius*np.absolute(np.radians(latb[j+1]-latb[j]))
            area_array[j,i] = xsize_array[j,i]*ysize_array[j,i]
            area_array_2[j,i] = (radius**2.)*np.absolute(np.radians(lonb[i+1]-lonb[i]))*np.absolute(np.sin(np.radians(latb[j+1]))-np.sin(np.radians(latb[j])))

    return area_array_2,xsize_array,ysize_array

def cell_area_all(radius=6376.0e3):
    filename   = 'atmos_monthly.nc'
    """read in grid from approriate file, and return 2D array of grid cell areas in metres**2. Taken from src/extra/python/scripts/cell_area.py."""
    resolution_file = Dataset(dirs['output']+filename, 'r', format='NETCDF3_CLASSIC')

    lons = resolution_file.variables['lon'][:]
    lats = resolution_file.variables['lat'][:]

    lonb = resolution_file.variables['lonb'][:]
    latb = resolution_file.variables['latb'][:]

    area_array,xsize_array,ysize_array = cell_area_calculate(lons, lats, lonb, latb, radius)


    return area_array,xsize_array,ysize_array

def cell_area(radius=6376.0e3):
    """wrapper for cell_area_all, such that cell_area only returns area array, and not xsize_array and y_size_array too. Taken from src/extra/python/scripts/cell_area.py."""
    area_array,xsize_array,ysize_array = cell_area_all(radius=radius)
    return area_array

""" 
-------------------------------------------------------------------------------
                                  Paths
------------------------------------------------------------------------------- 
"""

simulation   = 'planetb_EoceneEarth_rot0/ISR_1300' #'planetb_presentdayEarth_rot0/ISR_1361' #'Earth'
run          = 'run0226'
netcdf_file  = 'atmos_monthly.nc'
sf_path      = '/proj/bolinc/users/x_ryabo/socrates_edited_for_isca/spectral_files_for_GCMs/'
sf_name      = 'Suran_lw.sf'

isca_plots = '/proj/bolinc/users/x_ryabo/Isca-Ryan_plots'

dirs = {
    "isca_outputs": os.getenv('GFDL_DATA')+"/",
    "simulation": os.getenv('GFDL_DATA')+"/"+simulation+"/",
    "output": os.getenv('GFDL_DATA')+"/"+simulation+"/"+run+"/",
    "plot_output": isca_plots+"/"+simulation+"/"+run+"/LIFEsim/"
    }

if not os.path.exists(dirs["plot_output"]):
    os.makedirs(dirs["plot_output"])
    os.makedirs(dirs["plot_output"]+'data/')

""" 
-------------------------------------------------------------------------------
                  Spectral range and reference Planck curves
------------------------------------------------------------------------------- 
"""

n_band_edges_m = np.array(read_band_edges(os.path.join(sf_path, sf_name))) # [m]
band_widths_m  = np.diff(n_band_edges_m)                                     # [m]

n_band_edges_um = n_band_edges_m*1e6        # [microns]
band_widths_um  = np.diff(n_band_edges_um) # [microns]

n_band_edges_cm = np.flip(sigma2meter(n_band_edges_m)) # [cm-1]
band_widths_cm  = np.diff(n_band_edges_cm)           # [cm-1]

band_centres_m = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_bins_lw'].values # Socrates LW & SW spectral bin centers [m]
band_centres_um = band_centres_m * 1e6 # [microns]
band_centres_cm = np.flip(sigma2meter(band_centres_m))

nbands 	       = np.size(band_centres_m)-1

LIFE_range_indices = np.where(np.logical_and(band_centres_um >= 4, band_centres_um <= 18))
LIFE_band_centres_um = band_centres_um[LIFE_range_indices]

def surf_Planck_cm(ts,albedo_s):
    # Array, Planck function as a function of wavenumbers, W/m2/cm-1
    B   = np.zeros(len(band_centres_cm))
    for i in range(len(band_centres_cm)):
        sigma      = band_centres_cm[i]
        B[i]    = (c1*(icm2imeter(sigma))**3 / (np.exp(c2*icm2imeter(sigma)/ts)-1)) # converting nu and band_widths from cm-1 to m-1
    B   = (1.-albedo_s) * np.pi * B * icm2imeter(band_widths_cm) # units W/m * m-1 = W/m2
    return B

def surf_Planck_um(ts,albedo_s):
    # Array, Planck function as a function of wavelengths, W/m2/μm
    B   = np.zeros(len(band_centres_m))
    for i in range(len(band_centres_m)):
        lam     = band_centres_m[i]
        B[i]    = ( (c1/lam**5) / (np.exp(c2/(lam*ts))-1))
    B   = (1.-albedo_s) * np.pi * B * band_widths_m * 1e-6 # units W/m2/m * m = W/m2 ; convert final value from W/m2/m to W/m2/μm
    return B

def surf_Planck_nu(ts,albedo_s): # Reference for W/m2/cm-1
    B   = np.zeros(len(band_centres_cm))
    c1  = 1.191042e-5
    c2  = 1.4387752
    for i in range(len(band_centres_cm)):
        nu      = band_centres_cm[i]
        B[i]    = (c1*nu**3 / (np.exp(c2*nu/ts)-1))
    B   = (1.-albedo_s) * np.pi * B * band_widths_cm/1000.0
    return B

""" 
-------------------------------------------------------------------------------
                  Lat-lon grid, cell areas, face plots
------------------------------------------------------------------------------- 
"""

# Lon/Lat locations 
Substellar_latitude = 32 # lat[32,1] = 1.395 degrees
Substellar_longitude = 0 
# Terminator points: [0,any] (south pole) or [-1,any] (north pole) or [any,32] or [any,96]
# [32,32] is 90 degrees east of the substellar point on the Equator.
# [32,96] is 90 degrees west of the substellar point on the Equator (270 degrees east).
Terminator_morning_latitude = 32 
Terminator_morning_longitude = 96
Terminator_evening_latitude = 32 
Terminator_evening_longitude = 32
Antistellar_latitude = 32 # lat[32,:] = 1.395 degrees
Antistellar_longitude = 64 # lon[:,64] = 180 degrees  
North_pole_latitude = 63
North_pole_longitude = 0
South_pole_latitude = 0
South_pole_longitude = 0

lons = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lon'].values # Longitudes [degree]
lats = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lat'].values # Latitudes  [degree]
lon, lat = np.meshgrid(lons, lats)

lons_edges = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lonb'].values # Longitude edges [degree]
lats_edges = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['latb'].values # Latitude edges  [degree]
lonb, latb = np.meshgrid(lons_edges, lats_edges)

"""
# Cell area [m2]
lats_r  = np.radians(lats) 
lons_edges_r = np.radians(lonb) 
lats_edges_r = np.radians(latb) 

dlat = np.diff(lats_edges_r, axis=0)  # Differences along the latitude axis
dlon = np.diff(lons_edges_r, axis=1)  # Differences along the longitude axis
cos_lat = np.cos(lats_r).reshape(-1, 1)

area = planet_radius**2* ( cos_lat * dlat[:,:-1] * dlon[:-1,:] )
"""
# Computing the surface area of each cell of the Gaussian grid
area=np.array(cell_area(radius=planet_radius))

area_planet = np.sum(area) # close to the analytical 4.0*np.pi*(1.02*6.371e6)**2
"""
# Another way
latr = np.deg2rad(lats)
dlatr = np.diff(np.deg2rad(lats_edges))
weights = np.cos(latr) * 2. * np.sin(dlatr/2.)

sw = (ds.soc_toa_sw * weights[None,:,None]).mean(('lat','lon')) / np.mean(weights)
"""
# print(-lw.values, sw.values, (sw-lw).values)

def define_face(lat_center, lon_center, lat_range, lon_range, lons_wrap=False, all_latitudes=False, all_longitudes=False):
    if all_latitudes:
        lat_min = 0
        lat_max = 63
    else:
        lat_min = max(0, lat_center - lat_range)
        lat_max = min(63, lat_center + lat_range)
        
    if all_longitudes:
        lon_indices = np.arange(0, 128)
    else:
        if lons_wrap:
            lon_indices = np.concatenate([
                np.arange(lon_center, lon_center + lon_range + 1),
                np.arange(128 - lon_range + lon_center, 128)
            ])
        else:
            lon_min = max(0, lon_center - lon_range)
            lon_max = min(127, lon_center + lon_range)
            lon_indices = np.arange(lon_min, lon_max + 1)
            
    return lat_min, lat_max, lon_indices

lat_range = 16
lat_range_pole = 31 # 31 makes the north pole face start just above the equator.
lon_range = 32

faces = {
    'Substellar_point': define_face(Substellar_latitude, Substellar_longitude, lat_range, lon_range, lons_wrap=True, all_latitudes=True),
    'Antistellar_point': define_face(Antistellar_latitude, Antistellar_longitude, lat_range, lon_range, all_latitudes=True),
    'Morning_terminator': define_face(Terminator_morning_latitude, Terminator_morning_longitude, lat_range, lon_range, all_latitudes=True),
    'Evening_terminator': define_face(Terminator_evening_latitude, Terminator_evening_longitude, lat_range, lon_range, all_latitudes=True),
    'North_pole': define_face(North_pole_latitude, North_pole_longitude, lat_range_pole, lon_range, all_longitudes=True),
    'South_pole': define_face(South_pole_latitude, South_pole_longitude, lat_range_pole, lon_range, all_longitudes=True)
}

projections = {
    'Substellar_point': ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0),
    'Antistellar_point': ccrs.Orthographic(central_longitude=180.0, central_latitude=0.0),
    'Morning_terminator': ccrs.Orthographic(central_longitude=270.0, central_latitude=0.0),
    'Evening_terminator': ccrs.Orthographic(central_longitude=90.0, central_latitude=0.0),
    'North_pole': ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0),
    'South_pole': ccrs.Orthographic(central_longitude=180.0, central_latitude=-90.0)
}

# Print face boundaries
for face, bounds in faces.items():
    lat_min, lat_max, lon_indices = bounds
    print(f"{face} Face Boundaries (lat_min, lat_max, lon_indices): {lat_min, lat_max, lon_indices}")

# Define lat-lon slices for each face
lat_slices = {}
lon_slices = {}

for face, bounds in faces.items():
    lat_min, lat_max, lon_indices = bounds
    lat_slices[face] = slice(lat_min, lat_max + 1)
    lon_slices[face] = lon_indices

cmap = plt.cm.RdYlBu_r
resolution = 200
size = 10

pfull = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['pfull'].values # Approx full (midpoint)  pressure levels [mbar]
lev = 6 
print(pfull)
Plev = pfull[lev] # 10 mbar

Tfull = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['temp'].values[0,lev,:,:]    # Temperature [K]
cloud_fraction = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['cf'].values[0,lev,:,:] # Cloud fraction for the simple cloud scheme [0-1]
cloud_fraction = cloud_fraction*100. # converted to percent
soc_toa_sw = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_toa_sw'].values[0,:,:] # Socrates Net TOA SW flux (down) [W/m2]
soc_olr    = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_olr'].values[0,:,:]    # Socrates OLR [W/m2]

# Ozone file has its own lat-lon grid and pfull array. ozone has same number of latitudes but only 2 longitudes.
pfull_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['pfull'].values # Approx full (midpoint)  pressure levels [mbar]
lev_o3 = 27 
print(pfull_o3)
Plev_o3 = pfull_o3[lev_o3] # 11 mbar

lons_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['lon'].values # Longitudes [degree]
lats_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['lat'].values # Latitudes  [degree]
lon_o3, lat_o3 = np.meshgrid(lons_o3, lats_o3)

lons_edges_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['lonb'].values # Longitude edges [degree]
lats_edges_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['latb'].values # Latitude edges  [degree]
lonb_o3, latb_o3 = np.meshgrid(lons_edges_o3, lats_edges_o3)
ozone_1990 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['ozone_1990'].values[-1,lev_o3,:,:]    # Ozone mass mixing ratio [kg/kg] from Fortuin & Langematz 1995, 10.1117/12.198578
ozone_1990 = ozone_1990*1e6 # Converting from kg/kg to ppmm
ozone_1990 = np.repeat(ozone_1990, 128, axis=1) # Copy data across grid longitudes for plotting

# Maps are plotted at level pfull[6] = 10 mbar except for the OLR and ISR.
Global_quantities = {
    "Tfull": Tfull,
    "cloud_fraction": cloud_fraction,
    "soc_toa_sw": soc_toa_sw,
    "soc_olr": soc_olr,
    "ozone": ozone_1990
}

for quantity in Global_quantities:
        
    fig, axs = plt.subplots(3, 2, figsize=(8, 10), dpi=150, subplot_kw={'projection': None})

    # Flatten the axs array for easy iteration
    axs = axs.flatten()
    for i, face in enumerate(faces.keys()):
        lat_min, lat_max, lon_indices = faces[face]
        chosen_variable = Global_quantities[quantity][lat_slices[face], lon_slices[face]]

        ax = axs[i]
        ax = plt.subplot(3, 2, i + 1, projection=projections[face])
        gl = ax.gridlines(draw_labels=True)
        
        lon_plot = lon[lat_slices[face], lon_slices[face]]
        lat_plot = lat[lat_slices[face], lon_slices[face]]

        CS = ax.contourf(lon_plot, lat_plot, chosen_variable, resolution, extend='both', 
                        transform=ccrs.PlateCarree(), cmap=cmap, vmin=np.min(chosen_variable), vmax=np.max(chosen_variable))

        for coll in CS.collections:
            coll.set_edgecolor("face")

        CB = plt.colorbar(CS, ax=ax, orientation='horizontal', format="%0.0f")
        #mean_loc = (np.mean(chosen_variable) - CB.vmin) / (CB.vmax - CB.vmin)
        #CB.ax.plot([mean_loc] * 2, [1, 0], 'k')
        CB.ax.plot([np.mean(chosen_variable)]*2,[1, 0], 'k')

        labels(CB, quantity) 

        #CB.set_label(r'Stellar downward flux, ${F_{down}}$ [W m$^{-2}$]', size=size)
        CB.ax.tick_params(labelsize=7)
        ax.set_title(f'{face.replace("_", " ")} - Average = ' + '{:0.2e}'.format(np.mean(chosen_variable)), fontsize=size)
        ax.set_global()  # make the map global rather than have it zoom in to the extents of any plotted data

    plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
    if save_figs:
        plt.savefig(dirs["plot_output"] + 'Faces_'+quantity+'.pdf', bbox_inches='tight')
        plt.savefig(dirs["plot_output"] + 'Faces_'+quantity+'.png', bbox_inches='tight')
    plt.show()

""" 
-------------------------------------------------------------------------------
                Plotting reference Planck curves cm - um - ph
------------------------------------------------------------------------------- 
"""

T_surf_2D  = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['t_surf'].values[0,:,:] # Surface temperature [K]
T_surf     = np.mean(T_surf_2D[:,:]) # Average, for the Planck functions
albedo_isca = 0.3

# Checked against https://www.spectralcalc.com/blackbody_calculator/blackbody.php which yields max(B) = 22.42 W/m2/μm. So pi*B = 70.43 W/m2/μm.
B = 2 * h * c**2 / (band_centres_m)**5 / (np.exp(h * c / band_centres_m / kb / T_surf) - 1) # Reference for W/m2/μm. 
fig = plt.figure()
ax = fig.add_subplot(111)
y1 = np.pi*B*1e-6
y2 = surf_Planck_um(T_surf,0.0)/band_widths_m
ax.plot(band_centres_um, y1, 'r--', lw=2)
ax.plot(band_centres_um, y2, 'g', lw=1)
# 1 micron: band_centres_um[90] ; 5 microns: band_centres_um[205] ; 10 microns: band_centres_um[254] ; 15 microns: band_centres_um[283] ; 20 microns: band_centres_um[304]
ax.plot(1.0, np.pi*2.34246e-10, marker='x', markersize=8, color='red')
ax.plot(5.0, np.pi*10.9636, marker='x', markersize=8, color='red')
ax.plot(10.0, np.pi*20.5461, marker='x', markersize=8, color='red')
ax.plot(15.0, np.pi*11.0843, marker='x', markersize=8, color='red')
ax.plot(20.5, np.pi*5.21624, marker='x', markersize=8, color='red')
ax.set_xlabel(r'$\lambda\;[\mathrm{\mu m}]$')
ax.set_ylabel(r'$B(\lambda)\;[\mathrm{W\,m^{−2}\,\mu m^{−1}}]$')
ax.set_xlim(left=1, right=25)
ax.set_ylim(bottom=min(y1[58:302]), top=max(y1[58:302]))
ax.minorticks_on()
ax.grid(which='both')
#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'planck_microns.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'planck_microns.png',bbox_inches='tight')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(band_centres_um, np.pi*B, 'k--', lw=2)
#ax.plot(band_centres_um, surf_Planck_um(5778.0,0.0)/band_widths_m, 'b', lw=1)
y1 = surf_Planck_nu(T_surf,0.0)/band_widths_cm
y2 = surf_Planck_cm(T_surf,0.0)/band_widths_cm
ax.plot(band_centres_cm, y1, 'r--', lw=2)
ax.plot(band_centres_cm, y2, 'g', lw=1)
# 1 micron: band_centres_cm[311] = 10069 cm-1 ; 5 microns: band_centres_cm[196] = 2013 cm-1 ; 10 microns: band_centres_cm[146] = 999.86 cm-1 ; 15 microns: band_centres_cm[117] = 666.25 cm-1 ; 20 microns: band_centres_cm[96] = 496.56 cm-1
ax.plot(10069, np.pi*1.80492e-14, marker='x', markersize=8, color='red')
ax.plot(2013, np.pi*0.0265038, marker='x', markersize=8, color='red')
ax.plot(999.86, np.pi*0.205494, marker='x', markersize=8, color='red')
ax.set_xlabel(r'$\sigma\;[\mathrm{cm}^{−1}]$')
ax.set_ylabel(r'$B(\sigma)\;[\mathrm{W\,m^{−2}\,cm}]$')
ax.set_xlim(left=sigma2micron(1), right=sigma2micron(25))
ax.set_ylim(bottom=min(y1[15:260]), top=max(y1[15:260]))
ax.minorticks_on()
ax.grid(which='both')
#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'planck_cm.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'planck_cm.png',bbox_inches='tight')
plt.close()

""" 
-------------------------------------------------------------------------------
           Converting W/m2/cm-1 to W/m2/μm and ph m-2 s-1 micron-1
------------------------------------------------------------------------------- 
"""

cmap = plt.get_cmap('coolwarm_r')
Sfl_U_LW    = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_spectral_olr'].values[0,:,:,:] # W/m2/m

# Retrieving the 1D OLR value at the substellar point: 486.33487 W/m2 for Ts = 364.80417 K
print("Substellar soc_olr[32,0] = ", soc_olr[32,0])
print("np.sum(Sfl_U_LW[:,32,0]) = ", np.sum(Sfl_U_LW[:,32,0])) 
# Summing Sfl_U_LW [W/m2/m] at the substellar point yields 486.33487 W/m2

Sfl_U_LW_um = Wm2m_to_Wm2um(Sfl_U_LW) # W/m2/um. Widths are multiplied by 1e6, so the flux is divided by 1e6 to preserve F_lambda*dlambda
print("np.sum(Sfl_U_LW_um[:,32,0])*1e6 = ", np.sum(Sfl_U_LW_um[:,32,0])*1e6) #1e6 factor: same band widths except 1e-6*original width. 

m, n, p = Sfl_U_LW_um.shape
Sfl_U_LW_cm = np.empty((m, n, p))
for i in range(n):
    for j in range(p):
        #Sfl_U_LW_cm[:, i, j] = Wm2um_to_Wm2cm(Sfl_U_LW_um[:, i, j] / band_widths_um, band_centres_um)
        Sfl_U_LW_cm[:, i, j] = Wm2um_to_Wm2cm(Sfl_U_LW_um[:, i, j], band_centres_um) # Sfl_U_LW_um is in W/m2/um, so Sfl_U_LW_um / band_widths_um is in W/m2/um2. So we shouldn't divide by band_widths_um here.
Sfl_U_LW_cm = np.flipud(Sfl_U_LW_cm)
print("np.sum(Sfl_U_LW_cm[:,32,0]) = ", np.sum(Sfl_U_LW_cm[:,32,0]))  # This yields 2.978255384513222e-05 W/m2
print("np.sum(Wm2cm_to_Wm2um(Sfl_U_LW_cm[:,32,0], band_centres_cm))*1e6 = ", np.sum(Wm2cm_to_Wm2um(Sfl_U_LW_cm[:,32,0], band_centres_cm))*1e6) # Yields 486.3348687527299
print("np.sum(Wm2cm_to_Wm2m(Sfl_U_LW_cm[:,32,0], band_centres_cm)) = ", np.sum(Wm2cm_to_Wm2m(Sfl_U_LW_cm[:,32,0], band_centres_cm))) # Yields 486.3348687527299

spectral_photon_flux_m = np.empty((m, n, p))
spectral_photon_flux_um = np.empty((m, n, p))
for i in range(n):
    for j in range(p):
        spectral_photon_flux_m[:, i, j] = radiant_flux_to_photon_flux(Sfl_U_LW[:, i, j], band_centres_m) 
        spectral_photon_flux_um[:, i, j] = Wm2m_to_Wm2um(spectral_photon_flux_m[:, i, j])
        
print("np.sum(spectral_photon_flux_um[:,32,0]) = ", np.sum(spectral_photon_flux_um[:,32,0])) # Yields 3.994504829467879e+16

""" 
-------------------------------------------------------------------------------
           Plotting the 3D spectral OLR at the substellar point only
------------------------------------------------------------------------------- 
"""

# =================================================
#            Plot radiant flux in W/m2/cm-1
# =================================================

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

ax.set_xlim(left=1.0, right=10000.0) # Limit at 10000 cm-1
ax.set_ylim(bottom=1e-16, top=4e3)

alpha_domains    = 0.03
color_micrometer = 'purple'
color_IR         = 'red'
color_visible    = 'green'
color_uv         = 'blue'
ax.axvspan(1.,10.,alpha=alpha_domains, color=color_micrometer)     # Micrometer
ax.axvspan(10.,12500.,alpha=alpha_domains, color=color_IR)         # IR

xticks_array = np.arange(2000, 12000, 2000.0)
xticks_array = np.insert(xticks_array, 0, 1)
plt.xticks(xticks_array)

ticks = np.arange(1,10000,2000)[1:]-1
ticks = np.insert(ticks, 0, 1, axis=0)
ticks = np.append(ticks, 10000)

second_xaxis_mic = sigma2micron(ticks)
second_xaxis_mic = np.around(second_xaxis_mic,2)
second_xaxis_cm  = ticks

ax2=ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(second_xaxis_cm)
ax2.set_xticklabels(second_xaxis_mic)
ax2.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=14)
    
linewidth = 1.5 #+ i

ax.semilogy(band_centres_cm, Sfl_U_LW_cm[:,32,0]/band_widths_cm, lw=linewidth)
ax.semilogy(band_centres_cm, surf_Planck_cm(T_surf_2D[32,0],albedo_isca)/band_widths_cm, color='grey', lw=1.0, ls="--")

plot_mission = True
alpha_missions = 0.05
alpha_instruments = 0.01
ymin, ymax = ax.get_ylim()
height = ymax - ymin
# Calculate the y-coordinate for the center at 20% of the y-axis length in log scale
y_center = np.exp(0.0 * (np.log(ymax) - np.log(ymin)) + np.log(ymin))

LIFE_mission = {'LIFE' : mpatch.Rectangle((540.54,ymin), 1959.46, height, color='#079e65', alpha=alpha_missions)}

if plot_mission:
    for r in LIFE_mission:
        ax.add_artist(LIFE_mission[r])
        rx, ry = LIFE_mission[r].get_xy()
        cx = rx + LIFE_mission[r].get_width()/2.0
        cy = np.exp(np.log(ry) + 2.0) #ry + LIFE_mission[r].get_height()/2.0

        ax.annotate(r, (cx, cy), color='navy', fontweight=1000, 
                    fontsize=12, ha='center', va='center', fontfamily='serif')
        
ax.set_xlabel('Wavenumber ' + r'$\nu$ [cm$^{-1}$]', family='serif', fontsize=14)
#ax.set_ylabel('Spectral flux density ' + r'[W m$^{-2}$ cm]', family='serif', fontsize=14)
ax.set_ylabel('Spectral irradiance ' + r'[W m$^{-2}$ cm]', family='serif', fontsize=14)

ax.legend()
#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'spectrum_cm_SSP.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'spectrum_cm_SSP.png',bbox_inches='tight')
plt.close()


# =================================================
#            Plot radiant flux in W/m2/μm
# =================================================

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.set_xlim(left=band_centres_um[89], right=band_centres_um[319]) # Limits 1-25 microns
ax.set_ylim(bottom=1e-11, top=3e2) # Limits 1-25 microns

#ax.set_ylim(bottom=1e-16, top=1e1)
alpha_domains    = 0.03
color_micrometer = 'purple'
color_IR         = 'red'
color_visible    = 'green'
color_uv         = 'blue'
ax.axvspan(sigma2micron(10.),sigma2micron(1.),alpha=alpha_domains, color=color_micrometer)     # Micrometer
ax.axvspan(sigma2micron(12500.),sigma2micron(10.),alpha=alpha_domains, color=color_IR)         # IR

xticks_array = np.arange(band_centres_um[89], band_centres_um[319], 2.0)
xticks_array = np.append(xticks_array, 25)
plt.xticks(xticks_array)

ticks = np.arange(1,27,2)

second_xaxis_cm = sigma2micron(ticks)
second_xaxis_cm = np.around(second_xaxis_cm,2)
second_xaxis_mic = ticks

ax2=ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(second_xaxis_mic)
ax2.set_xticklabels(second_xaxis_cm)
ax2.set_xlabel('Wavenumber ' + r'$\nu$ [cm$^{-1}$]', family='serif', fontsize=14)
    
linewidth = 1.5 #+ i
ax.semilogy(band_centres_um, (Sfl_U_LW_um[:,32,0]/band_widths_um)*1e6, lw=linewidth)
ax.semilogy(band_centres_um, (surf_Planck_um(T_surf_2D[32,0],albedo_isca)/band_widths_m), color='grey', lw=1.0, ls="--")

plot_mission = True
alpha_missions = 0.05
alpha_instruments = 0.01
ymin, ymax = ax.get_ylim()
height = ymax - ymin
# Calculate the y-coordinate for the center at 20% of the y-axis length in log scale
y_center = np.exp(0.2 * (np.log(ymax) - np.log(ymin)) + np.log(ymin))

# Limits taken from 4 to 18.5 microns
LIFE_mission = {'LIFE' : mpatch.Rectangle((band_centres_um[188],ymin), band_centres_um[297]-band_centres_um[188], height, color='#079e65', alpha=alpha_missions)}

if plot_mission:
    for r in LIFE_mission:
        ax.add_artist(LIFE_mission[r])
        rx, ry = LIFE_mission[r].get_xy()
        cx = rx + LIFE_mission[r].get_width()/2.0
        cy = np.exp(np.log(ry) + 2.0) #ry + LIFE_mission[r].get_height()/2.0

        ax.annotate(r, (cx, cy), color='navy', fontweight=1000, 
                    fontsize=12, ha='center', va='center', fontfamily='serif')
        

ax.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=14)
#ax.set_ylabel('Spectral flux density ' + r'[W m$^{-2}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)
ax.set_ylabel('Spectral irradiance ' + r'[W m$^{-2}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)

ax.legend()
#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'spectrum_microns_SSP.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'spectrum_microns_SSP.png',bbox_inches='tight')
plt.close()

# =================================================
#     Plot photon flux in ph m-2 s-1 micron-1
# =================================================

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.set_xlim(left=band_centres_um[89], right=band_centres_um[319]) # Limits 1-25 microns
ax.set_ylim(bottom=1e8, top=1e22) # Limits 1-25 microns

#ax.set_ylim(bottom=1e-16, top=1e1)
alpha_domains    = 0.03
color_micrometer = 'purple'
color_IR         = 'red'
color_visible    = 'green'
color_uv         = 'blue'
ax.axvspan(sigma2micron(10.),sigma2micron(1.),alpha=alpha_domains, color=color_micrometer)     # Micrometer
ax.axvspan(sigma2micron(12500.),sigma2micron(10.),alpha=alpha_domains, color=color_IR)         # IR

xticks_array = np.arange(band_centres_um[89], band_centres_um[319], 2.0)
xticks_array = np.append(xticks_array, 25)
plt.xticks(xticks_array)

ticks = np.arange(1,27,2)

second_xaxis_cm = sigma2micron(ticks)
second_xaxis_cm = np.around(second_xaxis_cm,2)
second_xaxis_mic = ticks

ax2=ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(second_xaxis_mic)
ax2.set_xticklabels(second_xaxis_cm)
ax2.set_xlabel('Wavenumber ' + r'$\nu$ [cm$^{-1}$]', family='serif', fontsize=14)
    
linewidth = 1.5 #+ i
ax.semilogy(band_centres_um, (spectral_photon_flux_um[:,32,0]/band_widths_um)*1e6, lw=linewidth)
ax.semilogy(band_centres_um, radiant_flux_to_photon_flux(surf_Planck_um(T_surf_2D[32,0],albedo_isca)/band_widths_m, band_centres_m), color='grey', lw=1.0, ls="--")

plot_mission = True
alpha_missions = 0.05
alpha_instruments = 0.01
ymin, ymax = ax.get_ylim()
height = ymax - ymin
# Calculate the y-coordinate for the center at 20% of the y-axis length in log scale
y_center = np.exp(0.2 * (np.log(ymax) - np.log(ymin)) + np.log(ymin))

# Limits taken from 4 to 18.5 microns
LIFE_mission = {'LIFE' : mpatch.Rectangle((band_centres_um[188],ymin), band_centres_um[297]-band_centres_um[188], height, color='#079e65', alpha=alpha_missions)}

if plot_mission:
    for r in LIFE_mission:
        ax.add_artist(LIFE_mission[r])
        rx, ry = LIFE_mission[r].get_xy()
        cx = rx + LIFE_mission[r].get_width()/2.0
        cy = np.exp(np.log(ry) + 2.0) #ry + LIFE_mission[r].get_height()/2.0

        ax.annotate(r, (cx, cy), color='navy', fontweight=1000, 
                    fontsize=12, ha='center', va='center', fontfamily='serif')
        
ax.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=14)
#ax.set_ylabel('Spectral flux density ' + r'[ph m$^{-2}$ s$^{-1}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)
ax.set_ylabel('Spectral irradiance ' + r'[ph m$^{-2}$ s$^{-1}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)

#sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
ax.legend()
#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'spectrum_photons_SSP.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'spectrum_photons_SSP.png',bbox_inches='tight')
plt.close()

""" 
-------------------------------------------------------------------------------
        Computing weighted sums of fluxes and planck curves for plotting
------------------------------------------------------------------------------- 
"""

# Slice the arrays according to the latitude and longitude ranges of each face. 
T_surf_2D_faces = {}
Sfl_U_LW_cm_faces = {}
Sfl_U_LW_um_faces = {}
spectral_photon_flux_um_faces = {}
Sfl_U_LW_cm_faces_sum = {}
Sfl_U_LW_um_faces_sum = {}
spectral_photon_flux_um_faces_sum = {}
num_cells_faces = {}
area_faces = {}
area_faces_array = {}

Sfl_U_LW_cm_faces_sum_weighted = {}
Sfl_U_LW_um_faces_sum_weighted = {}
spectral_photon_flux_um_faces_sum_weighted = {}

planck_cm_plot = {} 
planck_um_plot = {}
planck_ph_plot = {}

# For debugging
numerator_cm = {}
numerator_um = {}
numerator_ph = {}

for face in faces.keys():
    T_surf_2D_faces[face] = T_surf_2D[lat_slices[face], lon_slices[face]]
    Sfl_U_LW_cm_faces[face] = Sfl_U_LW_cm[:, lat_slices[face], lon_slices[face]]
    Sfl_U_LW_um_faces[face] = Sfl_U_LW_um[:, lat_slices[face], lon_slices[face]]
    spectral_photon_flux_um_faces[face] = spectral_photon_flux_um[:, lat_slices[face], lon_slices[face]]

    Sfl_U_LW_cm_faces_sum[face] = np.sum(Sfl_U_LW_cm[:, lat_slices[face], lon_slices[face]], axis=(1, 2))
    Sfl_U_LW_um_faces_sum[face] = np.sum(Sfl_U_LW_um[:, lat_slices[face], lon_slices[face]], axis=(1, 2))
    spectral_photon_flux_um_faces_sum[face] = np.sum(spectral_photon_flux_um[:, lat_slices[face], lon_slices[face]], axis=(1, 2))

    lat_min, lat_max, lon_indices = faces[face]
    num_lat = lat_max - lat_min + 1
    num_lon = len(lon_indices)
    num_cells_faces[face] = num_lat * num_lon # Number of elements in the face

    area_faces_array[face] = area[lat_slices[face], lon_slices[face]]
    #area_faces[face] = np.sum(area[lat_slices[face], lon_slices[face]], axis=(0, 1)) # This yields slightly different values for each face
    area_faces[face] = (4.0*np.pi*radius_teegarden_b**2)/2 # half the area of the sphere?

    # Weighting the fluxes and Planck curves over each face
    m, n, p = spectral_photon_flux_um_faces[face].shape
    Sfl_U_LW_cm_faces_cells = np.empty((m, n, p))
    Sfl_U_LW_um_faces_cells = np.empty((m, n, p))
    spectral_photon_flux_um_faces_cells = np.empty((m, n, p))

    planck_cm_plot_cells = np.empty((m, n, p))
    planck_um_plot_cells = np.empty((m, n, p))
    planck_ph_plot_cells = np.empty((m, n, p))

    for i in range(num_lat):
        for j in range(num_lon):
            Sfl_U_LW_cm_faces_cells[:,i,j] = Sfl_U_LW_cm_faces[face][:,i,j]*area_faces_array[face][i,j]
            Sfl_U_LW_um_faces_cells[:,i,j] = Sfl_U_LW_um_faces[face][:,i,j]*area_faces_array[face][i,j]
            spectral_photon_flux_um_faces_cells[:,i,j] = spectral_photon_flux_um_faces[face][:,i,j]*area_faces_array[face][i,j]
            #print("np.shape(spectral_photon_flux_um_faces[face]) = ", np.shape(spectral_photon_flux_um_faces[face]))

            planck_cm_plot_cells[:,i,j] = ( surf_Planck_cm(T_surf_2D_faces[face][i,j],albedo_isca)/band_widths_cm ) * area_faces_array[face][i,j]
            planck_um_plot_cells[:,i,j] = ( surf_Planck_um(T_surf_2D_faces[face][i,j],albedo_isca)/band_widths_m  ) #* area_faces_array[face][i,j]
            planck_ph_plot_cells[:,i,j] = (radiant_flux_to_photon_flux(surf_Planck_um(T_surf_2D_faces[face][i,j],albedo_isca)/band_widths_m, band_centres_m)) * area_faces_array[face][i,j]

    Sfl_U_LW_cm_faces_sum_weighted[face] = np.sum(Sfl_U_LW_cm_faces_cells, axis=(1, 2))
    numerator_cm[face] = np.sum(Sfl_U_LW_cm_faces_cells, axis=(1, 2))
    Sfl_U_LW_cm_faces_sum_weighted[face] = Sfl_U_LW_cm_faces_sum_weighted[face] / area_faces[face]

    Sfl_U_LW_um_faces_sum_weighted[face] = np.sum(Sfl_U_LW_um_faces_cells, axis=(1, 2))
    numerator_um[face] = np.sum(Sfl_U_LW_um_faces_cells, axis=(1, 2))
    Sfl_U_LW_um_faces_sum_weighted[face] = Sfl_U_LW_um_faces_sum_weighted[face] / area_faces[face]

    spectral_photon_flux_um_faces_sum_weighted[face] = np.sum(spectral_photon_flux_um_faces_cells, axis=(1, 2))
    numerator_ph[face] = np.sum(spectral_photon_flux_um_faces_cells, axis=(1, 2))
    spectral_photon_flux_um_faces_sum_weighted[face] = spectral_photon_flux_um_faces_sum_weighted[face] / area_faces[face]

    planck_cm_plot[face] = np.sum(planck_cm_plot_cells, axis=(1, 2))
    planck_cm_plot[face] = planck_cm_plot[face] / area_faces[face]

    planck_um_plot[face] = np.sum(planck_um_plot_cells, axis=(1, 2))
    planck_um_plot[face] = planck_um_plot[face] #/ area_faces[face]

    planck_ph_plot[face] = np.sum(planck_ph_plot_cells, axis=(1, 2))
    planck_ph_plot[face] = planck_ph_plot[face] / area_faces[face]

# =================================================
#            Plot radiant flux in W/m2/cm-1
# =================================================

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

ax.set_xlim(left=1.0, right=10000.0) # Limit at 10000 cm-1
ax.set_ylim(bottom=1e-16, top=4e3)

alpha_domains    = 0.03
color_micrometer = 'purple'
color_IR         = 'red'
color_visible    = 'green'
color_uv         = 'blue'
ax.axvspan(1.,10.,alpha=alpha_domains, color=color_micrometer)     # Micrometer
ax.axvspan(10.,12500.,alpha=alpha_domains, color=color_IR)         # IR

xticks_array = np.arange(2000, 12000, 2000.0)
xticks_array = np.insert(xticks_array, 0, 1)
plt.xticks(xticks_array)

ticks = np.arange(1,10000,2000)[1:]-1
ticks = np.insert(ticks, 0, 1, axis=0)
ticks = np.append(ticks, 10000)

second_xaxis_mic = sigma2micron(ticks)
second_xaxis_mic = np.around(second_xaxis_mic,2)
second_xaxis_cm  = ticks

ax2=ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(second_xaxis_cm)
ax2.set_xticklabels(second_xaxis_mic)
ax2.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=14)
    
for i, face in enumerate(faces):
    print(faces.keys())
    color = cmap(i / len(faces))  # Normalize the index to [0, 1] to get a color from the colormap
    linewidth = 1.5 #+ i
    #ax.semilogy(band_centres_cm, Sfl_U_LW_cm_faces_sum[face] / band_widths_cm, label=face.replace('_', ' '), color=color, lw=linewidth)
    #ax.semilogy(band_centres_cm,num_cells_faces[face]*(surf_Planck_cm(np.min(T_surf_2D_faces[face]),0.0)/band_widths_cm), color=color, lw=2.0, ls="--")
    #ax.semilogy(band_centres_cm, Sfl_U_LW_cm_faces_sum[face], label=face.replace('_', ' '), color=color, lw=linewidth)
    #ax.semilogy(band_centres_cm,num_cells_faces[face]*(surf_Planck_cm(np.mean(T_surf_2D_faces[face]),albedo_isca)/band_widths_cm), color=color, lw=2.0, ls="--")

    ax.semilogy(band_centres_cm, Sfl_U_LW_cm_faces_sum_weighted[face]/band_widths_cm, label=face.replace('_', ' '), color=color, lw=linewidth)
    ax.semilogy(band_centres_cm, planck_cm_plot[face], color=color, lw=2.0, ls="--")

# Actually Tsurf should vary across a given face, so this Planck curve is exaggerated.
plot_mission = True
alpha_missions = 0.05
alpha_instruments = 0.01
ymin, ymax = ax.get_ylim()
height = ymax - ymin
# Calculate the y-coordinate for the center at 20% of the y-axis length in log scale
y_center = np.exp(0.0 * (np.log(ymax) - np.log(ymin)) + np.log(ymin))

LIFE_mission = {'LIFE' : mpatch.Rectangle((540.54,ymin), 1959.46, height, color='#079e65', alpha=alpha_missions)}

if plot_mission:
    for r in LIFE_mission:
        ax.add_artist(LIFE_mission[r])
        rx, ry = LIFE_mission[r].get_xy()
        cx = rx + LIFE_mission[r].get_width()/2.0
        cy = np.exp(np.log(ry) + 2.0) #ry + LIFE_mission[r].get_height()/2.0

        ax.annotate(r, (cx, cy), color='navy', fontweight=1000, 
                    fontsize=12, ha='center', va='center', fontfamily='serif')
        
ax.set_xlabel('Wavenumber ' + r'$\nu$ [cm$^{-1}$]', family='serif', fontsize=14)
#ax.set_ylabel('Spectral flux density ' + r'[W m$^{-2}$ cm]', family='serif', fontsize=14)
ax.set_ylabel('Spectral irradiance ' + r'[W m$^{-2}$ cm]', family='serif', fontsize=14)

ax.legend()
#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'spectrum_cm.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'spectrum_cm.png',bbox_inches='tight')
plt.close()


# =================================================
#            Plot radiant flux in W/m2/μm
# =================================================

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.set_xlim(left=band_centres_um[89], right=band_centres_um[319]) # Limits 1-25 microns
ax.set_ylim(bottom=1e-11, top=3e2) # Limits 1-25 microns

#ax.set_ylim(bottom=1e-16, top=1e1)
alpha_domains    = 0.03
color_micrometer = 'purple'
color_IR         = 'red'
color_visible    = 'green'
color_uv         = 'blue'
ax.axvspan(sigma2micron(10.),sigma2micron(1.),alpha=alpha_domains, color=color_micrometer)     # Micrometer
ax.axvspan(sigma2micron(12500.),sigma2micron(10.),alpha=alpha_domains, color=color_IR)         # IR

xticks_array = np.arange(band_centres_um[89], band_centres_um[319], 2.0)
xticks_array = np.append(xticks_array, 25)
plt.xticks(xticks_array)

ticks = np.arange(1,27,2)

second_xaxis_cm = sigma2micron(ticks)
second_xaxis_cm = np.around(second_xaxis_cm,2)
second_xaxis_mic = ticks

ax2=ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(second_xaxis_mic)
ax2.set_xticklabels(second_xaxis_cm)
ax2.set_xlabel('Wavenumber ' + r'$\nu$ [cm$^{-1}$]', family='serif', fontsize=14)
    
for i, face in enumerate(faces):
    color = cmap(i / len(faces))  # Normalize the index to [0, 1] to get a color from the colormap
    linewidth = 1.5 #+ i
    #ax.semilogy(band_centres_um, Sfl_U_LW_um_faces_sum[face], label=face.replace('_', ' '), color=color, lw=linewidth)
    #ax.semilogy(band_centres_um,num_cells_faces[face]*(surf_Planck_um(np.mean(T_surf_2D_faces[face]),albedo_isca)/band_widths_m), color=color, lw=2.0, ls="--")

    ax.semilogy(band_centres_um, (Sfl_U_LW_um_faces_sum_weighted[face]/band_widths_um)*1e6, label=face.replace('_', ' '), color=color, lw=linewidth)
    ax.semilogy(band_centres_um, planck_um_plot[face], color=color, lw=2.0, ls="--")

plot_mission = True
alpha_missions = 0.05
alpha_instruments = 0.01
ymin, ymax = ax.get_ylim()
height = ymax - ymin
# Calculate the y-coordinate for the center at 20% of the y-axis length in log scale
y_center = np.exp(0.2 * (np.log(ymax) - np.log(ymin)) + np.log(ymin))

# Limits taken from 4 to 18.5 microns
LIFE_mission = {'LIFE' : mpatch.Rectangle((band_centres_um[188],ymin), band_centres_um[297]-band_centres_um[188], height, color='#079e65', alpha=alpha_missions)}

if plot_mission:
    for r in LIFE_mission:
        ax.add_artist(LIFE_mission[r])
        rx, ry = LIFE_mission[r].get_xy()
        cx = rx + LIFE_mission[r].get_width()/2.0
        cy = np.exp(np.log(ry) + 2.0) #ry + LIFE_mission[r].get_height()/2.0

        ax.annotate(r, (cx, cy), color='navy', fontweight=1000, 
                    fontsize=12, ha='center', va='center', fontfamily='serif')
        

ax.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=14)
#ax.set_ylabel('Spectral flux density ' + r'[W m$^{-2}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)
ax.set_ylabel('Spectral irradiance ' + r'[W m$^{-2}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)

ax.legend()
#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'spectrum_microns.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'spectrum_microns.png',bbox_inches='tight')
plt.close()

# =================================================
#     Plot photon flux in ph m-2 s-1 micron-1
# =================================================

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.set_xlim(left=band_centres_um[89], right=band_centres_um[319]) # Limits 1-25 microns
ax.set_ylim(bottom=1e8, top=1e22) # Limits 1-25 microns

#ax.set_ylim(bottom=1e-16, top=1e1)
alpha_domains    = 0.03
color_micrometer = 'purple'
color_IR         = 'red'
color_visible    = 'green'
color_uv         = 'blue'
ax.axvspan(sigma2micron(10.),sigma2micron(1.),alpha=alpha_domains, color=color_micrometer)     # Micrometer
ax.axvspan(sigma2micron(12500.),sigma2micron(10.),alpha=alpha_domains, color=color_IR)         # IR

xticks_array = np.arange(band_centres_um[89], band_centres_um[319], 2.0)
xticks_array = np.append(xticks_array, 25)
plt.xticks(xticks_array)

ticks = np.arange(1,27,2)

second_xaxis_cm = sigma2micron(ticks)
second_xaxis_cm = np.around(second_xaxis_cm,2)
second_xaxis_mic = ticks

ax2=ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(second_xaxis_mic)
ax2.set_xticklabels(second_xaxis_cm)
ax2.set_xlabel('Wavenumber ' + r'$\nu$ [cm$^{-1}$]', family='serif', fontsize=14)
    
for i, face in enumerate(faces):
    color = cmap(i / len(faces))  # Normalize the index to [0, 1] to get a color from the colormap
    linewidth = 1.5 #+ i
    #ax.semilogy(band_centres_um, spectral_photon_flux_um_faces_sum[face], label=face.replace('_', ' '), color=color, lw=linewidth)
    #ax.semilogy(band_centres_um,num_cells_faces[face]*radiant_flux_to_photon_flux(surf_Planck_um(np.mean(T_surf_2D_faces[face]),albedo_isca)/band_widths_m, band_centres_m), color=color, lw=2.0, ls="--")

    ax.semilogy(band_centres_um, (spectral_photon_flux_um_faces_sum_weighted[face]/band_widths_um)*1e6, label=face.replace('_', ' '), color=color, lw=linewidth)
    ax.semilogy(band_centres_um, planck_ph_plot[face], color=color, lw=2.0, ls="--")

# temperatures = [200.0, 250.0, 300.0, 350.0, 400.0]
# for i, T in enumerate(temperatures):
#     ax.semilogy(band_centres_um, radiant_flux_to_photon_flux(surf_Planck_um(T,albedo_isca)/band_widths_m, band_centres_m), color='gray', lw=2.0, ls="--")
#     x_pos = 5.0
#     y_pos = spectral_photon_flux_um_faces_sum[face][np.argmin(np.abs(band_centres_um - x_pos))] * 1.1  # Slightly above the curve
#     ax.text(x_pos, y_pos, f'Blackbody {int(T)} K',
#                 fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

plot_mission = True
alpha_missions = 0.05
alpha_instruments = 0.01
ymin, ymax = ax.get_ylim()
height = ymax - ymin
# Calculate the y-coordinate for the center at 20% of the y-axis length in log scale
y_center = np.exp(0.2 * (np.log(ymax) - np.log(ymin)) + np.log(ymin))

# Limits taken from 4 to 18.5 microns
LIFE_mission = {'LIFE' : mpatch.Rectangle((band_centres_um[188],ymin), band_centres_um[297]-band_centres_um[188], height, color='#079e65', alpha=alpha_missions)}

if plot_mission:
    for r in LIFE_mission:
        ax.add_artist(LIFE_mission[r])
        rx, ry = LIFE_mission[r].get_xy()
        cx = rx + LIFE_mission[r].get_width()/2.0
        cy = np.exp(np.log(ry) + 2.0) #ry + LIFE_mission[r].get_height()/2.0

        ax.annotate(r, (cx, cy), color='navy', fontweight=1000, 
                    fontsize=12, ha='center', va='center', fontfamily='serif')
        
ax.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=14)
#ax.set_ylabel('Spectral flux density ' + r'[ph m$^{-2}$ s$^{-1}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)
ax.set_ylabel('Spectral irradiance ' + r'[ph m$^{-2}$ s$^{-1}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)

#sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
ax.legend()
#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'spectrum_photons.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'spectrum_photons.png',bbox_inches='tight')
plt.close()

# ==================================================================================================
#        Write photon flux for each cloud cover case scaled by the distance to the target
# ==================================================================================================

# Write the case with 10% cloud cover
for face in faces.keys():
    # Is it this?
    #flux_to_dilute = spectral_photon_flux_um_faces_sum_weighted[face][LIFE_range_indices] # around 1e14 ph m-2 s-1 micron-1
    # Or this?
    flux_to_dilute = (spectral_photon_flux_um_faces_sum_weighted[face][LIFE_range_indices]/band_widths_um[LIFE_range_indices])*1e6 # around 1e20 ph m-2 s-1 micron-1
    # Need to check.
    LIFE_spectral_photon_flux_um = inverse_square_law(flux_to_dilute, radius_teegarden_b, teegarden_distance_m) #spectral_photon_flux_um[LIFE_range_indices] 
    
    columns_to_write = np.column_stack((LIFE_band_centres_um, LIFE_spectral_photon_flux_um))
    np.savetxt(dirs["plot_output"]+'data/spectrum_'+face+'.txt', columns_to_write, delimiter=' ', header='', comments='')

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.semilogy(LIFE_band_centres_um, LIFE_spectral_photon_flux_um, lw=1.3)
    ax.semilogy(band_centres_um,inverse_square_law(num_cells_faces[face]*radiant_flux_to_photon_flux(surf_Planck_um(np.mean(T_surf_2D_faces[face]),albedo_isca)/band_widths_m, band_centres_m), radius_teegarden_b, teegarden_distance_m), color=color, lw=2.0, ls="--")
    ax.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=14)
    ax.set_ylabel('Spectral flux density received on Earth' + r'[Photons m$^{-2}$ s$^{-1}$ $\mathrm{\mu m}^{-1}$]', family='serif', fontsize=14)
    ax.set_xlim(4.0, 18.5)
    #ax.set_ylim(1e-5, 1e-1)
    if save_figs:
        fig.savefig(dirs["plot_output"]+'spectrum_'+face+'.pdf',bbox_inches='tight')
        fig.savefig(dirs["plot_output"]+'spectrum_'+face+'.png',bbox_inches='tight')
    #plt.show()
    plt.close()

""" 
-------------------------------------------------------------------------------
                            Set up LIFEsim Pipeline
-------------------------------------------------------------------------------
"""

# Create bus
bus = lifesim.Bus()

# Set the baseline scenario
bus.data.options.set_scenario(scenario)

# Set some options manually
bus.data.options.set_manual(spec_res=spectral_resolution)
bus.data.options.set_manual(wl_min=minimum_wavelength)
bus.data.options.set_manual(wl_max=maximum_wavelength)

# Create the instrument and add it to the bus
instrument = lifesim.Instrument(name='inst')
bus.add_module(instrument)

# Create the transmission map and add it to the bus
transm = lifesim.TransmissionMap(name='transm')
bus.add_module(transm)

# Create the noise sources and add them to the bus
exo = lifesim.PhotonNoiseExozodi(name='exo')
bus.add_module(exo)

local = lifesim.PhotonNoiseLocalzodi(name='local')
bus.add_module(local)

star = lifesim.PhotonNoiseStar(name='star')
bus.add_module(star)

# Connect all modules
bus.connect(('inst', 'transm'))
bus.connect(('inst', 'exo'))
bus.connect(('inst', 'local'))
bus.connect(('inst', 'star'))
bus.connect(('star', 'transm'))

""" 
-------------------------------------------------------------------------------
                        Import the Spectrum Into LIFEsim
-------------------------------------------------------------------------------
"""

# Import the spectrum for a single case, setting the planet properties
importer = SpectrumImporter()
importer.do_import(pathtotext=dirs["plot_output"]+r'data/spectrum_Substellar_point.txt',
                x_string='micron',
                y_string='ph m-2 s-1 micron-1',
                radius_p_spectrum=None,
                radius_p_target=planet_radius,
                distance_s_spectrum=10.,
                distance_s_target=star_distance,
                integration_time=0)

flux_planet_spectrum = [importer.x_data, importer.y_data]

# Plot input spectrum
fig = plt.figure(figsize=(10,8))
plt.plot(importer.x_data, importer.y_data, color='black')
plt.title('Input Spectrum')
plt.xlabel(f'Wavelength ({str(importer.x_data.unit)})')
plt.ylabel(f'Planet Flux ({str(importer.y_data.unit)})')
if save_figs:
    plt.savefig(dirs["plot_output"]+'input_spectrum_Substellar_point.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'input_spectrum_Substellar_point.png',bbox_inches='tight')
#plt.show()
plt.close()

""" 
-------------------------------------------------------------------------------
                                Run the Simulation
-------------------------------------------------------------------------------
"""

snr, flux, noise = instrument.get_spectrum(temp_s=star_temperature,
                                        radius_s=star_radius,
                                        distance_s=star_distance,
                                        lat_s=0.78, # In radians
                                        z=exozodi_level,
                                        angsep=planet_angular_separation,
                                        flux_planet_spectrum=flux_planet_spectrum, # In ph m-3 s-1 over m
                                        integration_time=integration_time,
                                        safe_mode=False)

""" 
-------------------------------------------------------------------------------
                    Plot the Simulated Observation Results
-------------------------------------------------------------------------------
"""

# Rescale original spectrum to match amount of bins for plotting
flux_planet_spectrum_rescaled = spectres(new_wavs=instrument.data.inst['wl_bin_edges'],
                                spec_wavs=flux_planet_spectrum[0].value,
                                spec_fluxes=flux_planet_spectrum[1].value,
                                edge_mode=True)

# Compute one draw from noise distribution
random_noise = np.random.normal(0,flux_planet_spectrum_rescaled/snr[1],
                                size = flux_planet_spectrum_rescaled.shape)

# Plot the input spectrum and measured spectrum
plt.fill_between(np.arange(len(flux)),
                flux_planet_spectrum_rescaled-flux_planet_spectrum_rescaled/snr[1],
                flux_planet_spectrum_rescaled+flux_planet_spectrum_rescaled/snr[1],
                color='black', alpha=0.1, label='1-$\sigma$')

plt.plot(flux_planet_spectrum_rescaled, color='black', label='Input Spectrum')

plt.scatter(np.arange(len(flux)), flux_planet_spectrum_rescaled+random_noise,
            color='black', marker='.', label='Sim. Observation')

plt.errorbar(np.arange(len(flux)), flux_planet_spectrum_rescaled+random_noise,
            yerr=flux_planet_spectrum_rescaled/snr[1], color='black', capsize=2, ls='none')

plt.title(f'Input Spectrum and Simulated Observation ({scenario})')
plt.xlabel('Wavelength ($\mu$m)')
plt.xticks(ticks=np.arange(len(flux))[::6], labels=np.round(snr[0][::6]*1e6, 1))
plt.ylabel('Flux (ph s$^{-1}$ m$^{-3}$)')
plt.legend()
if save_figs:
    plt.savefig(dirs["plot_output"]+'input_spectrum_sim_obs_Substellar_point.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'input_spectrum_sim_obs_Substellar_point.png',bbox_inches='tight')
#plt.show()
plt.close()

""" 
-------------------------------------------------------------------------------
                        Plot the SNR per Wavelength Bin
-------------------------------------------------------------------------------
"""
plt.step(snr[0]*1e6, snr[1], c='black')
plt.title('SNR per Wavelength Bin')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('SNR per Wavelength Bin')
if save_figs:
    plt.savefig(dirs["plot_output"]+'snr_Substellar_point.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'snr_Substellar_point.png',bbox_inches='tight')
#plt.show()
plt.close()

""" 
-------------------------------------------------------------------------------
    Extract "Wavelength" and "Spectral Irradiance" columns from txt-File
-------------------------------------------------------------------------------
"""

# Read in the file as a pandas dataframe
#dataframe = pd.read_csv(dirs["plot_output"]+'data/spectrum_Substellar_point.txt', sep='  ', header=None,
#                names=['wavelength', 'spectral_irradiance', '_', '__'], comment='#')

# Save the relevant columns as a new txt-file
#dataframe.loc[:, ['wavelength', 'spectral_irradiance']].to_csv(dirs["plot_output"]+'data/spectrum_Substellar_point.txt', header=None, sep='\t', index=False)

for face in faces:
    file_path = dirs["plot_output"] + 'data/spectrum_' + face + '.txt'
    
    # Read in the file as a pandas dataframe
    dataframe = pd.read_csv(file_path, sep=' ', header=None,
                            names=['wavelength', 'spectral_irradiance', '_', '__'], comment='#')
    
    # Save the relevant columns as a new txt-file
    output_path = dirs["plot_output"] + 'data/spectrum_' + face + '_extracted.txt'
    dataframe.loc[:, ['wavelength', 'spectral_irradiance']].to_csv(output_path, header=None, sep='\t', index=False)

""" 
-------------------------------------------------------------------------------
                            Set up LIFEsim Pipeline
-------------------------------------------------------------------------------
"""

# Create bus
bus = lifesim.Bus()

# Set the baseline scenario
bus.data.options.set_scenario(scenario)

# Set some options manually
bus.data.options.set_manual(spec_res=spectral_resolution)
bus.data.options.set_manual(wl_min=minimum_wavelength)
bus.data.options.set_manual(wl_max=maximum_wavelength)

# Create the instrument and add it to the bus
instrument = lifesim.Instrument(name='inst')
bus.add_module(instrument)

# Create the transmission map and add it to the bus
transm = lifesim.TransmissionMap(name='transm')
bus.add_module(transm)

# Create the noise sources and add them to the bus
exo = lifesim.PhotonNoiseExozodi(name='exo')
bus.add_module(exo)

local = lifesim.PhotonNoiseLocalzodi(name='local')
bus.add_module(local)

star = lifesim.PhotonNoiseStar(name='star')
bus.add_module(star)

# Connect all modules
bus.connect(('inst', 'transm'))
bus.connect(('inst', 'exo'))
bus.connect(('inst', 'local'))
bus.connect(('inst', 'star'))
bus.connect(('star', 'transm'))

""" 
-------------------------------------------------------------------------------
                        Import the Spectra Into LIFEsim
-------------------------------------------------------------------------------
"""

#colors = cm.PuBu(np.linspace(0.2, 1, len(faces))) # scale of blues - good for parameter sweeps

# Plot input spectrum
fig = plt.figure(figsize=(10,8))

for i, face in enumerate(faces.keys()): 
    color = cmap(i / len(faces))
    # Import the spectrum, setting the planet properties
    importer = SpectrumImporter()

    importer.do_import(pathtotext=dirs["plot_output"]+'data/spectrum_'+face+'_extracted.txt',
                    x_string='micron',
                    y_string='ph m-2 s-1 micron-1',
                    radius_p_spectrum=None,
                    radius_p_target=planet_radius,
                    distance_s_spectrum=10.,
                    distance_s_target=star_distance,
                    integration_time=0)

    flux_planet_spectrum = [importer.x_data, importer.y_data]

    plt.plot(importer.x_data, importer.y_data, label=face.replace('_', ' '), color=color) # colors[i] for PuBu
    
plt.title('Input Spectrum')
plt.xlabel(f'Wavelength ({str(importer.x_data.unit)})')
plt.ylabel(f'Planet Flux ({str(importer.y_data.unit)})')
plt.legend(title='Observed face')
if save_figs:
    plt.savefig(dirs["plot_output"]+'spectrum_all_cases.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'spectrum_all_cases.png',bbox_inches='tight')
#plt.show()
plt.close()

""" 
-------------------------------------------------------------------------------
        Run the Simulations and plot the Simulated Observation Results
-------------------------------------------------------------------------------
"""

fig = plt.figure(figsize=(10,8))

legend_labels = []
legend_handles = []

for i, face in enumerate(faces.keys()): 
    color = cmap(i / len(faces))

    importer = SpectrumImporter()

    importer.do_import(pathtotext=dirs["plot_output"]+r'data/spectrum_'+face+'_extracted.txt',
                    x_string='micron',
                    y_string='ph m-2 s-1 micron-1',
                    radius_p_spectrum=None,
                    radius_p_target=planet_radius,
                    distance_s_spectrum=10.,
                    distance_s_target=star_distance,
                    integration_time=0)

    flux_planet_spectrum = [importer.x_data, importer.y_data]

    snr, flux, noise = instrument.get_spectrum(temp_s=star_temperature,
                                            radius_s=star_radius,
                                            distance_s=star_distance,
                                            lat_s=0.78, # In radians
                                            z=exozodi_level,
                                            angsep=planet_angular_separation,
                                            flux_planet_spectrum=flux_planet_spectrum, # In ph m-3 s-1 over m
                                            integration_time=integration_time,
                                            safe_mode=False)

    # Rescale original spectrum to match amount of bins for plotting
    flux_planet_spectrum_rescaled = spectres(new_wavs=instrument.data.inst['wl_bin_edges'],
                                    spec_wavs=flux_planet_spectrum[0].value,
                                    spec_fluxes=flux_planet_spectrum[1].value,
                                    edge_mode=True)

    # Compute one draw from noise distribution
    random_noise = np.random.normal(0, flux_planet_spectrum_rescaled/snr[1],
                                    size = flux_planet_spectrum_rescaled.shape)

    # Plot the input spectra and the simulated observation results
    plt.fill_between(np.arange(len(flux)),
                    flux_planet_spectrum_rescaled-flux_planet_spectrum_rescaled/snr[1],
                    flux_planet_spectrum_rescaled+flux_planet_spectrum_rescaled/snr[1],
                    color=color, alpha=0.2)

    plt.plot(flux_planet_spectrum_rescaled, color=color, label=face.replace('_', ' '))

    plt.scatter(np.arange(len(flux)), flux_planet_spectrum_rescaled+random_noise,
                color=color, marker='.')
    plt.errorbar(np.arange(len(flux)), flux_planet_spectrum_rescaled+random_noise, yerr=flux_planet_spectrum_rescaled/snr[1],
                color=color, capsize=2, ls='none')

plt.title(f'Input Spectrum and Simulated Observation ({scenario})')
plt.xlabel('Wavelength ($\mu$m)')
plt.xticks(ticks=np.arange(len(flux))[::6], labels=np.round(snr[0][::6]*1e6, 1))
plt.ylabel('Flux (ph s$^{-1}$ m$^{-3}$)')
legend_elements = [Line2D([0], [0], color='k', marker='.', linestyle='', label='Simulated LIFE observation'), 
                   Patch(facecolor='black', edgecolor='none', alpha=0.3, label=r'1$\sigma$ confidence area')]
subtitle = plt.Line2D([], [], linestyle='none', label='Observed face')

# Add legends manually
plt.legend(handles=legend_elements + [subtitle] + plt.gca().get_legend_handles_labels()[0])
legend = plt.gca().get_legend()
for text_instance in legend.texts:
    if text_instance.get_text() == 'Observed face':
        text_instance.set_horizontalalignment('center')
if save_figs:
    plt.savefig(dirs["plot_output"]+'spectrum_obs_all_cases.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'spectrum_obs_all_cases.png',bbox_inches='tight')
#plt.show()
plt.close()

""" 
-------------------------------------------------------------------------------
                Statistical Significance of Detected Difference
-------------------------------------------------------------------------------
"""
# Which 2 cases to compare ?
key1 = 'Substellar_point'

fig = plt.figure(figsize=(10,8))

start_index = 0
for i, key2 in enumerate(faces.keys()): 
    color = cmap(i / len(faces))
    if i == start_index:
        continue

    # Create bus
    bus = lifesim.Bus()

    # Set the baseline scenario
    bus.data.options.set_scenario(scenario)

    # Set some options manually
    bus.data.options.set_manual(spec_res=spectral_resolution)
    bus.data.options.set_manual(wl_min=minimum_wavelength)
    bus.data.options.set_manual(wl_max=maximum_wavelength)

    # Create the instrument and add it to the bus
    instrument = lifesim.Instrument(name='inst')
    bus.add_module(instrument)

    # Create the transmission map and add it to the bus
    transm = lifesim.TransmissionMap(name='transm')
    bus.add_module(transm)

    # Create the noise sources and add them to the bus
    exo = lifesim.PhotonNoiseExozodi(name='exo')
    bus.add_module(exo)

    local = lifesim.PhotonNoiseLocalzodi(name='local')
    bus.add_module(local)

    star = lifesim.PhotonNoiseStar(name='star')
    bus.add_module(star)

    # Connect all modules
    bus.connect(('inst', 'transm'))
    bus.connect(('inst', 'exo'))
    bus.connect(('inst', 'local'))
    bus.connect(('inst', 'star'))
    bus.connect(('star', 'transm'))

    # Import the spectrum, setting the planet properties
    importer_1 = SpectrumImporter()
    importer_2 = SpectrumImporter()

    importer_1.do_import(pathtotext=dirs["plot_output"]+r'data/spectrum_'+key1+'_extracted.txt',
                    x_string='micron',
                    y_string='ph m-2 s-1 micron-1',
                    radius_p_spectrum=None,
                    radius_p_target=planet_radius,
                    distance_s_spectrum=10.,
                    distance_s_target=star_distance,
                    integration_time=0)

    importer_2.do_import(pathtotext=dirs["plot_output"]+r'data/spectrum_'+key2+'_extracted.txt',
                        x_string='micron',
                        y_string='ph m-2 s-1 micron-1',
                        radius_p_spectrum=None,
                        radius_p_target=planet_radius,
                        distance_s_spectrum=10.,
                        distance_s_target=star_distance,
                        integration_time=0)

    flux_planet_spectrum_1 = [importer_1.x_data, importer_1.y_data]
    flux_planet_spectrum_2 = [importer_2.x_data, importer_2.y_data]

    snr_1, flux_1, noise_1 = instrument.get_spectrum(temp_s=star_temperature,
                                            radius_s=star_radius,
                                            distance_s=star_distance,
                                            lat_s=0.78, # In radians
                                            z=exozodi_level,
                                            angsep=planet_angular_separation,
                                            flux_planet_spectrum=flux_planet_spectrum_1, # In ph m-3 s-1 over m
                                            integration_time=integration_time,
                                            safe_mode=False)

    snr_2, flux_2, noise_2 = instrument.get_spectrum(temp_s=star_temperature,
                                                        radius_s=star_radius,
                                                        distance_s=star_distance,
                                                        lat_s=0.78, # In radians
                                                        z=exozodi_level,
                                                        angsep=planet_angular_separation,
                                                        flux_planet_spectrum=flux_planet_spectrum_2, # In ph m-3 s-1 over m
                                                        integration_time=integration_time,
                                                        safe_mode=False)

    # Rescale original spectrum to match amount of bins for plotting
    flux_planet_spectrum_rescaled_1 = spectres(new_wavs=instrument.data.inst['wl_bin_edges'],
                                    spec_wavs=flux_planet_spectrum_1[0].value,
                                    spec_fluxes=flux_planet_spectrum_1[1].value,
                                    edge_mode=True)

    flux_planet_spectrum_rescaled_2 = spectres(new_wavs=instrument.data.inst['wl_bin_edges'],
                                                spec_wavs=flux_planet_spectrum_2[0].value,
                                                spec_fluxes=flux_planet_spectrum_2[1].value,
                                                edge_mode=True)

    # Plot the statistical significance
    flux_difference_in_sigma = np.divide(abs(flux_planet_spectrum_rescaled_1 - flux_planet_spectrum_rescaled_2),
    (flux_planet_spectrum_rescaled_1/snr_1[1]))

    plt.step(np.arange(len(flux_difference_in_sigma)), flux_difference_in_sigma, color=color, label=key1.replace('_', ' ')+'-'+key2.replace('_', ' '))

plt.title('Statistical Significance of Detected Difference')
plt.ylabel(r'$(F_{ref} - F)/\sigma_{ref}$')
plt.xlabel('Wavelength ($\mu$m)')
plt.xticks(ticks=np.arange(len(flux))[::6], labels=np.round(snr_1[0][::6]*1e6, 1))

plt.legend(title='Difference')
if save_figs:
    plt.savefig(dirs["plot_output"]+'statistical_difference.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'statistical_difference.png',bbox_inches='tight')
#plt.show()
plt.close()

""" 
-------------------------------------------------------------------------------
                                  Side by Side
-------------------------------------------------------------------------------
"""

#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,14), sharex=True)
#plt.subplots_adjust(hspace=0.)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

legend_labels = []
legend_handles = []

for i, face in enumerate(faces.keys()): 
    color = cmap(i / len(faces))

    importer = SpectrumImporter()

    importer.do_import(pathtotext=dirs["plot_output"]+r'data/spectrum_'+face+'_extracted.txt',
                    x_string='micron',
                    y_string='ph m-2 s-1 micron-1',
                    radius_p_spectrum=None,
                    radius_p_target=planet_radius,
                    distance_s_spectrum=10.,
                    distance_s_target=star_distance,
                    integration_time=0)

    flux_planet_spectrum = [importer.x_data, importer.y_data]

    snr, flux, noise = instrument.get_spectrum(temp_s=star_temperature,
                                            radius_s=star_radius,
                                            distance_s=star_distance,
                                            lat_s=0.78, # In radians
                                            z=exozodi_level,
                                            angsep=planet_angular_separation,
                                            flux_planet_spectrum=flux_planet_spectrum, # In ph m-3 s-1 over m
                                            integration_time=integration_time,
                                            safe_mode=False)

    # Rescale original spectrum to match amount of bins for plotting
    flux_planet_spectrum_rescaled = spectres(new_wavs=instrument.data.inst['wl_bin_edges'],
                                    spec_wavs=flux_planet_spectrum[0].value,
                                    spec_fluxes=flux_planet_spectrum[1].value,
                                    edge_mode=True)

    # Compute one draw from noise distribution
    random_noise = np.random.normal(0, flux_planet_spectrum_rescaled/snr[1],
                                    size = flux_planet_spectrum_rescaled.shape)

    # Plot the input spectra and the simulated observation results
    ax1.fill_between(np.arange(len(flux)),
                    flux_planet_spectrum_rescaled-flux_planet_spectrum_rescaled/snr[1],
                    flux_planet_spectrum_rescaled+flux_planet_spectrum_rescaled/snr[1],
                    color=color, alpha=0.2)

    ax1.plot(flux_planet_spectrum_rescaled, color=color, label=face.replace('_', ' '))

    ax1.scatter(np.arange(len(flux)), flux_planet_spectrum_rescaled+random_noise,
                color=color, marker='.')
    ax1.errorbar(np.arange(len(flux)), flux_planet_spectrum_rescaled+random_noise, yerr=flux_planet_spectrum_rescaled/snr[1],
                color=color, capsize=2, ls='none')

#sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
#ax1.set_title(f'Input Spectrum and Simulated Observation ({scenario})')
ax1.set_xlabel('Wavelength ($\mu$m)')
ax1.set_xticks(ticks=np.arange(len(flux))[::6], labels=np.round(snr[0][::6]*1e6, 1))
ax1.set_ylabel('Flux (ph s$^{-1}$ m$^{-3}$)')
legend_elements = [Line2D([0], [0], color='k', marker='.', linestyle='', label='Simulated LIFE observation'), 
                   Patch(facecolor='black', edgecolor='none', alpha=0.3, label=r'1$\sigma$ confidence area')]
subtitle = Line2D([], [], linestyle='none', label='Observed face')

# Add legends manually
# ax1.legend(handles=legend_elements + [subtitle] + ax1.get_legend_handles_labels()[0])
# legend = ax1.legend()
# for text_instance in legend.texts:
#     if text_instance.get_text() == 'Cloud cover fractions':
#         text_instance.set_horizontalalignment('center')

legend = ax1.legend(handles=legend_elements + [subtitle] + ax1.get_legend_handles_labels()[0])
for text_instance in legend.texts:
    if text_instance.get_text() == 'Observed face':
        text_instance.set_horizontalalignment('center')

#ax1.tick_params(axis='both', which='major', direction='inout', length=8, labelsize=14)
#ax1.tick_params(axis='both', which='minor', direction='inout', length=8, labelsize=14)

# Which 2 cases to compare ?
key1 = 'Substellar_point'

start_index = 0
for i, key2 in enumerate(faces.keys()): 
    color = cmap(i / len(faces))

    if i == start_index:
        continue

    # Create bus
    bus = lifesim.Bus()

    # Set the baseline scenario
    bus.data.options.set_scenario(scenario)

    # Set some options manually
    bus.data.options.set_manual(spec_res=spectral_resolution)
    bus.data.options.set_manual(wl_min=minimum_wavelength)
    bus.data.options.set_manual(wl_max=maximum_wavelength)

    # Create the instrument and add it to the bus
    instrument = lifesim.Instrument(name='inst')
    bus.add_module(instrument)

    # Create the transmission map and add it to the bus
    transm = lifesim.TransmissionMap(name='transm')
    bus.add_module(transm)

    # Create the noise sources and add them to the bus
    exo = lifesim.PhotonNoiseExozodi(name='exo')
    bus.add_module(exo)

    local = lifesim.PhotonNoiseLocalzodi(name='local')
    bus.add_module(local)

    star = lifesim.PhotonNoiseStar(name='star')
    bus.add_module(star)

    # Connect all modules
    bus.connect(('inst', 'transm'))
    bus.connect(('inst', 'exo'))
    bus.connect(('inst', 'local'))
    bus.connect(('inst', 'star'))
    bus.connect(('star', 'transm'))

    # Import the spectrum, setting the planet properties
    importer_1 = SpectrumImporter()
    importer_2 = SpectrumImporter()

    importer_1.do_import(pathtotext=dirs["plot_output"]+r'data/spectrum_'+key1+'_extracted.txt',
                    x_string='micron',
                    y_string='ph m-2 s-1 micron-1',
                    radius_p_spectrum=None,
                    radius_p_target=planet_radius,
                    distance_s_spectrum=10.,
                    distance_s_target=star_distance,
                    integration_time=0)

    importer_2.do_import(pathtotext=dirs["plot_output"]+r'data/spectrum_'+key2+'_extracted.txt',
                        x_string='micron',
                        y_string='ph m-2 s-1 micron-1',
                        radius_p_spectrum=None,
                        radius_p_target=planet_radius,
                        distance_s_spectrum=10.,
                        distance_s_target=star_distance,
                        integration_time=0)

    flux_planet_spectrum_1 = [importer_1.x_data, importer_1.y_data]
    flux_planet_spectrum_2 = [importer_2.x_data, importer_2.y_data]

    snr_1, flux_1, noise_1 = instrument.get_spectrum(temp_s=star_temperature,
                                            radius_s=star_radius,
                                            distance_s=star_distance,
                                            lat_s=0.78, # In radians
                                            z=exozodi_level,
                                            angsep=planet_angular_separation,
                                            flux_planet_spectrum=flux_planet_spectrum_1, # In ph m-3 s-1 over m
                                            integration_time=integration_time,
                                            safe_mode=False)

    snr_2, flux_2, noise_2 = instrument.get_spectrum(temp_s=star_temperature,
                                                        radius_s=star_radius,
                                                        distance_s=star_distance,
                                                        lat_s=0.78, # In radians
                                                        z=exozodi_level,
                                                        angsep=planet_angular_separation,
                                                        flux_planet_spectrum=flux_planet_spectrum_2, # In ph m-3 s-1 over m
                                                        integration_time=integration_time,
                                                        safe_mode=False)

    # Rescale original spectrum to match amount of bins for plotting
    flux_planet_spectrum_rescaled_1 = spectres(new_wavs=instrument.data.inst['wl_bin_edges'],
                                    spec_wavs=flux_planet_spectrum_1[0].value,
                                    spec_fluxes=flux_planet_spectrum_1[1].value,
                                    edge_mode=True)

    flux_planet_spectrum_rescaled_2 = spectres(new_wavs=instrument.data.inst['wl_bin_edges'],
                                                spec_wavs=flux_planet_spectrum_2[0].value,
                                                spec_fluxes=flux_planet_spectrum_2[1].value,
                                                edge_mode=True)

    # Plot the statistical significance
    flux_difference_in_sigma = np.divide(abs(flux_planet_spectrum_rescaled_1 - flux_planet_spectrum_rescaled_2),
    (flux_planet_spectrum_rescaled_1/snr_1[1]))

    ax2.step(np.arange(len(flux_difference_in_sigma)), flux_difference_in_sigma, color=color, label=key1.replace('_', ' ')+'-'+key2.replace('_', ' '))
    print("i, key2, flux_difference_in_sigma = ", i, key2, np.array2string(flux_difference_in_sigma, separator=',', suppress_small=True))

#ax2.set_title('Statistical Significance of Detected Difference')
#sns.despine(fig=None, ax=None, top=False, right=True, left=False, bottom=False, offset=None, trim=False)
ax2.set_yscale('log')
ax2.set_ylabel(r'$(F_{ref} - F)/\sigma_{ref}$')
ax2.set_xlabel(r'Wavelength [$\mu$m]')
ax2.set_xticks(ticks=np.arange(len(flux))[::6], labels=np.round(snr_1[0][::6]*1e6, 1))

ax2.legend(title='Difference')
if save_figs:
    plt.savefig(dirs["plot_output"]+'obs_stat_sidebyside_vert.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'obs_stat_sidebyside_vert.png',bbox_inches='tight')
#plt.show()
plt.close()
