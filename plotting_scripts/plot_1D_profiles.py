import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatch
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')

# Global constants
h  = 6.62607015e-34 # Planck's constant [J.s]
c  = 2.99792458e8   # Speed of light [m/s] 
kb = 1.380649e-23   # Boltzmann constant [J/K]
c1 = 2*h*c**2       # First  radiation constant
c2 = h*c/kb         # Second radiation constant
radius_Earth = 6.371e6   # Earth radius [m]
radius_Teegardenb = 1.02*6.371e6   # Teegarden's Star b radius [m]

# Figure settings
size = 10       # Default size of labels
linewidth = 1.5 

# Save figures
save_figs = True

def icm2imeter(x):
    # Convert inverse cm [cm-1] to inverse meter [m-1].
    return x * 1e2

def sigma2micron(x):
    return (x * 1e2)**(-1) * 1e6

def sigma2meter(x):
    return (x * 1e2)**(-1) 

def Wm2cm_to_Wm2um(Wm2cm, wavenumber):
    # Convert a radiant flux from W/m2/cm-1 to W/m2/μm. wavenumber is in cm-1.
    return 1e-4 * wavenumber**2 * Wm2cm

def Wm2um_to_Wm2cm(Wm2um, wavelength):
    # Convert a radiant flux from W/m2/um to W/m2/cm-1. wavelength is in microns.
    return 1e4 * wavelength**2 * Wm2um

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

def labels(ax, quantity):
    if quantity == 'Tfull':
        ax.set_xlabel(r'Temperature, $T$ [K]', size=size)   
    if quantity == 'precipitation':
        ax.set_xlabel(r'Precipitation [mm day$^{-1}$]',size=size)      
    if quantity == 'relative_humidity':
        ax.set_xlabel(r'Relative humidity, ${R_\mathrm{h}}$ [%]',size=size)   
    if quantity == 'specific_humidity':
        ax.set_xlabel(r'Specific humidity, $q$ [%]',size=size)  
    if quantity == 'zonal_wind':
        ax.set_xlabel(r'Zonal velocity, $u$ [$m \; s^{-1}$]',size=size)
    if quantity == 'meridional_wind':
        ax.set_xlabel(r'Meridional velocity, $v$ [$m \; s^{-1}$]',size=size)
    if quantity == 'vertical_wind':
        ax.set_xlabel(r'Vertical velocity, ${\omega}$ [$m \; s^{-1}$]',size=size)
    if quantity == 'soc_tdt_lw':
        ax.set_xlabel(r'Temperature tendency due to LW radiation, ${\frac{\mathrm{dT}_{\mathrm{rad}}}{\mathrm{dt}}} \; [\mathrm{K} \; \mathrm{day}^{-1}]$', size=size)
    if quantity == 'soc_tdt_sw':
        ax.set_xlabel(r'Temperature tendency due to SW radiation, ${\frac{\mathrm{dT}_{\mathrm{rad}}}{\mathrm{dt}}} \; [\mathrm{K} \; \mathrm{day}^{-1}]$', size=size)
    if quantity == 'soc_tdt_rad':
        ax.set_xlabel(r'Temperature tendency due to radiation, ${\frac{\mathrm{dT}_{\mathrm{rad}}}{\mathrm{dt}}} \; [\mathrm{K} \; \mathrm{day}^{-1}]$', size=size)
    if quantity == 'soc_flux_lw_up':
        ax.set_xlabel(r'Diffuse upward flux, ${F_{up}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_flux_lw_down':
        ax.set_xlabel(r'Diffuse downward flux, ${F_{down}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_flux_sw_up':
        ax.set_xlabel(r'Stellar scattered upward flux, ${F_{up,scattered}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_flux_sw_down':
        ax.set_xlabel(r'Stellar scattered downward flux, ${F_{down,scattered}}$ [W m$^{-2}$]', size=size)
    if quantity == 'soc_flux_direct':
        ax.set_xlabel(r'Direct beam, ${F_{inc}}$ [W m$^{-2}$]', size=size)    
    if quantity == 'net_flux':
        ax.set_xlabel(r'Net flux (down - up), ${F_{net}}$ [W m$^{-2}$]', size=size)
    if quantity == 'cloud_fraction':
        ax.set_xlabel(r'Cloud fraction', size=size)
    if quantity == 'droplet_radius':
        ax.set_xlabel(r'Effective cloud particle radius [$\mu$m]', size=size)
    if quantity == 'frac_liq':
        ax.set_xlabel(r'Liquid cloud fraction', size=size)
    if quantity == 'qcl_rad':
        ax.set_xlabel(r'Specific humidity of cloud liquid [%]', size=size) 
    if quantity == 'rh_in_cloud':
        ax.set_xlabel(r'Relative humidity in cloud [%]', size=size)
    if quantity == 'cff':
        ax.set_xlabel(r'Flux contribution, ${\mathcal{CF}_\mathrm{F}}$ [W m$^{-2}$]', size=size)
    if quantity == 'net_flux_sum':
        ax.set_xlabel(r'Global normalized net flux (down - up), ${F_{\mathrm{net}}}$ [W m$^{-2}$]', size=size)

simulation   = 'planetb_presentdayEarth_rot0/ISR_1400' #'planetb_EoceneEarth_rot0' #'planetb_ArcheanEarth_rot0' #'Earth'
run          = 'run0210'
netcdf_file  = 'atmos_monthly.nc'
sf_path      = '/proj/bolinc/users/x_ryabo/socrates_edited_for_isca/spectral_files_for_GCMs/'
sf_name      = 'miniSuran_lw.sf'
n_band_edges = np.array(read_band_edges(os.path.join(sf_path, sf_name))) # [m]
band_widths  = np.diff(n_band_edges)                                     # [m]

n_band_edges_um = n_band_edges*1e6        # [microns]
band_widths_um  = np.diff(n_band_edges_um) # [microns]

n_band_edges_cm = np.flip(sigma2meter(n_band_edges)) # [cm-1]
band_widths_cm  = np.diff(n_band_edges_cm)           # [cm-1]

isca_plots = '/proj/bolinc/users/x_ryabo/Isca-Ryan_plots'

dirs = {
    "isca_outputs": os.getenv('GFDL_DATA')+"/",
    "simulation": os.getenv('GFDL_DATA')+"/"+simulation+"/",
    "output": os.getenv('GFDL_DATA')+"/"+simulation+"/"+run+"/",
    "plot_output": isca_plots+"/"+simulation+"/"+run+"/columns/"
    }

if not os.path.exists(dirs["plot_output"]):
    os.makedirs(dirs["plot_output"])

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

# Time-step to probe
time=-1

# Clouds active ?
cloud = True


lons = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lon'].values # Longitudes [degree]
lats = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lat'].values # Latitudes  [degree]
lon, lat = np.meshgrid(lons, lats)

lons_edges = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lonb'].values # Longitude edges [degree]
lats_edges = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['latb'].values # Latitude edges  [degree]
lonb, latb = np.meshgrid(lons_edges, lats_edges)

# Cell area [m2]
rad_lat  = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lat'].values.astype(np.float64)  # latitudes  
rad_lon_b = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lonb'].values.astype(np.float64) # longitude edges    
rad_lat_b = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['latb'].values.astype(np.float64) # latitude edges 
rad_lonb, rad_latb = np.meshgrid(rad_lon_b, rad_lat_b)

# Convert to radians
rad_lat  = np.radians(rad_lat) 
rad_lonb = np.radians(rad_lonb) 
rad_latb = np.radians(rad_latb) 

dlat = np.diff(rad_latb, axis=0)  # Differences along the latitude axis
dlon = np.diff(rad_lonb, axis=1)  # Differences along the longitude axis
cos_lat = np.cos(rad_lat).reshape(-1, 1)

area = radius_Teegardenb**2* ( cos_lat * dlat[:,:-1] * dlon[:-1,:] )
area_planet = np.sum(area)


pfull = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['pfull'].values # Approx full (midpoint)  pressure levels [Pa]
phalf = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['phalf'].values # Approx half (interface) pressure levels [Pa]

if sf_name == 'Suran_lw.sf':
    soc_bins_lw = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_bins_lw'].values # Socrates LW & SW spectral bin centers [m]
    soc_bins_lw_um = soc_bins_lw * 1e6 # [microns]
    soc_bins_lw_cm = np.flip(sigma2meter(soc_bins_lw))

p_surf = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['ps'].values # Surface pressure [Pa]

precipitation = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['precipitation'].values # Precipitation from resolved, parameterised and snow [Kg/m2/s]
precipitation = precipitation*86400. # converted from kg/m2/s to mm/day. kg/m2/s = mm/s.

relative_humidity = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['rh'].values # Relative humidity [percent]

T_surf = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['t_surf'].values # Surface temperature [K]

flux_t = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['flux_t'].values # Sensible heat flux up at surface [W/m2]

flux_lhe = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['flux_lhe'].values # Latent heat flux up at surface [W/m2]

specific_humidity = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['sphum'].values # Specific humidity [kg/kg]
specific_humidity = specific_humidity*100. # converted to percent

zonal_wind      = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['ucomp'].values # Zonal wind component      [m/s]
meridional_wind = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['vcomp'].values # Meridional wind component [m/s]
vertical_wind   = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['omega'].values # Vertical wind component   [Pa/s]

Tfull = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['temp'].values # Temperature at full (midpoint) pressure levels [K]

soc_tdt_lw  = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_tdt_lw'].values  # Socrates temperature tendency due to LW radiation [K/s]
soc_tdt_sw  = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_tdt_sw'].values  # Socrates temperature tendency due to SW radiation [K/s]
soc_tdt_rad = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_tdt_rad'].values # Socrates temperature tendency due to radiation    [K/s]

soc_surf_flux_lw      = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_surf_flux_lw'].values      # Socrates Net LW surface flux (up)   [W/m2]
soc_surf_flux_sw      = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_surf_flux_sw'].values      # Socrates Net SW surface flux (down) [W/m2]
soc_surf_flux_lw_down = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_surf_flux_lw_down'].values # Socrates LW surface flux down       [W/m2]
soc_surf_flux_sw_down = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_surf_flux_sw_down'].values # Socrates SW surface flux down       [W/m2]

soc_olr = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_olr'].values # Socrates TOA LW flux (up) [W/m2]

soc_toa_sw      = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_toa_sw'].values # Socrates Net TOA SW flux (down) [W/m2]
soc_toa_sw_down = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_toa_sw'].values # Socrates TOA SW flux down [W/m2]

soc_flux_lw_up   = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_lw_up'].values   # Socrates LW flux up   [W/m2]
soc_flux_lw_down = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_lw_down'].values # Socrates LW flux down [W/m2]
soc_flux_sw_up   = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_sw_up'].values   # Socrates SW flux up   [W/m2]
soc_flux_sw_down = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_sw_down'].values # Socrates SW flux down [W/m2]
soc_flux_direct  = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_direct'].values  # Socrates Direct flux [W/m2]
net_flux         = soc_flux_lw_down + soc_flux_sw_down - soc_flux_lw_up - soc_flux_sw_up # positive downward

sum_NP = np.zeros(len(net_flux[0,:,0,0])) 
net_power = np.zeros(np.shape(net_flux[0,:,:,:]))
for j in range(len(net_flux[0,0,:,0])):
    for k in range(len(net_flux[0,0,0,:])):
        net_power[:,j,k] = net_flux[time,:,j,k]*area[j,k]
for i in range(len(net_flux[0,:,0,0])):
    sum_NP[i] = np.sum(net_power[i,:,:])
net_flux_sum = sum_NP/area_planet

if sf_name == 'Suran_lw.sf':
    soc_spectral_olr    = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_spectral_olr'].values # Socrates substellar LW OLR spectrum [W/m2/band]
    soc_spectral_olr_um = soc_spectral_olr*1e-6 # W/m2/m --> W/m2/um
    soc_spectral_olr_cm = np.array([ np.array([ np.array([ np.flipud(Wm2um_to_Wm2cm(soc_spectral_olr_um[i,:,j,k]/band_widths_um, soc_bins_lw_um)) for k in range(soc_spectral_olr_um.shape[3]) ]) for j in range(soc_spectral_olr_um.shape[2]) ]) for i in range(soc_spectral_olr_um.shape[0]) ])
    soc_spectral_olr_cm = np.transpose(soc_spectral_olr_cm, (0, 3, 1, 2))

if cloud:
    cloud_fraction = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['cf'].values # Cloud fraction for the simple cloud scheme [0-1]
    cloud_fraction = cloud_fraction*100. # converted to percent

    droplet_radius = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['reff_rad'].values # Effective cloud particle radius [microns]

    frac_liq = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['frac_liq'].values # Liquid cloud fraction (liquid, mixed-ice phase, ice) [0-1]
    frac_liq = frac_liq *100. # converted to percent

    qcl_rad = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['qcl_rad'].values # Specific humidity of cloud liquid [kg/kg]
    qcl_rad = qcl_rad*100. # converted to percent
    
    rh_in_cloud = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['rh_in_cf'].values # Relative humidity in cloud [percent]

surface_albedo = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['albedo'].values # Surface albedo

def surf_Planck_cm(ts,albedo_s):
    # Array, Planck function as a function of wavenumbers, W/m2/cm-1
    B   = np.zeros(len(soc_bins_lw_cm))
    for i in range(len(soc_bins_lw_cm)):
        sigma      = soc_bins_lw_cm[i]
        B[i]    = (c1*(icm2imeter(sigma))**3 / (np.exp(c2*icm2imeter(sigma)/ts)-1)) # converting nu and band_widths from cm-1 to m-1
    B   = (1.-albedo_s) * np.pi * B * icm2imeter(band_widths_cm) 
    return B

def surf_Planck_nu(ts,albedo_s):
    h   = 6.63e-34
    c   = 3.0e8
    kb  = 1.38e-23
    B   = np.zeros(len(soc_bins_lw_cm))
    c1  = 1.191042e-5
    c2  = 1.4387752
    for i in range(len(soc_bins_lw_cm)):
        nu      = soc_bins_lw_cm[i]
        B[i]    = (c1*nu**3 / (np.exp(c2*nu/ts)-1))
    B   = (1.-albedo_s) * np.pi * B * band_widths/1000.0
    return B

def surf_Planck_um(ts,albedo_s):
    # Array, Planck function as a function of wavelengths, W/m2/μm
    B   = np.zeros(len(soc_bins_lw))
    for i in range(len(soc_bins_lw)):
        lam     = soc_bins_lw[i]
        B[i]    = ( (c1/lam**5) / (np.exp(c2/(lam*ts))-1))
    B   = (1.-albedo_s) * np.pi * B * soc_bins_lw * 1e-6 # convert final value from W/m2/m to W/m2/μm
    return B

excluded_levels = 0
for i in range(len(Tfull[0,:,0,0])):
    if ((any(any(row>1e4) for row in Tfull[time,i,:,:])) or (any(any(row<=0.0) for row in Tfull[time,i,:,:]))):
        excluded_levels+=1

vertical_levels = slice(0,len(pfull)-excluded_levels) # Vertically regridded quantities may have no data near the surface

if cloud:
    column_quantities =     {
            "Tfull": Tfull[time,vertical_levels,:,:],
            "relative_humidity": relative_humidity[time,vertical_levels,:,:],
            "specific_humidity": specific_humidity[time,vertical_levels,:,:],
            "zonal_wind": zonal_wind[time,vertical_levels,:,:],
            "meridional_wind": meridional_wind[time,vertical_levels,:,:],
            "vertical_wind": vertical_wind[time,vertical_levels,:,:],
            "soc_tdt_lw": soc_tdt_lw[time,vertical_levels,:,:],
            "soc_tdt_sw": soc_tdt_sw[time,vertical_levels,:,:],
            "soc_tdt_rad": soc_tdt_rad[time,vertical_levels,:,:],
            "soc_flux_lw_up": soc_flux_lw_up[time,vertical_levels,:,:],
            "soc_flux_lw_down": soc_flux_lw_down[time,vertical_levels,:,:],
            "soc_flux_sw_up": soc_flux_sw_up[time,vertical_levels,:,:],
            "soc_flux_sw_down": soc_flux_sw_down[time,vertical_levels,:,:],
            "soc_flux_direct": soc_flux_direct[time,vertical_levels,:,:],
            "net_flux": net_flux[time,vertical_levels,:,:],
            "cloud_fraction": cloud_fraction[time,vertical_levels,:,:],
            "droplet_radius": droplet_radius[time,vertical_levels,:,:],
            "frac_liq": frac_liq[time,vertical_levels,:,:],
            "qcl_rad": qcl_rad[time,vertical_levels,:,:],
            "rh_in_cloud": rh_in_cloud[time,vertical_levels,:,:]
            }
else:
    column_quantities =     {
            "Tfull": Tfull[time,vertical_levels,:,:],
            "relative_humidity": relative_humidity[time,vertical_levels,:,:],
            "specific_humidity": specific_humidity[time,vertical_levels,:,:],
            "zonal_wind": zonal_wind[time,vertical_levels,:,:],
            "meridional_wind": meridional_wind[time,vertical_levels,:,:],
            "vertical_wind": vertical_wind[time,vertical_levels,:,:],
            "soc_tdt_lw": soc_tdt_lw[time,vertical_levels,:,:],
            "soc_tdt_sw": soc_tdt_sw[time,vertical_levels,:,:],
            "soc_tdt_rad": soc_tdt_rad[time,vertical_levels,:,:],
            "soc_flux_lw_up": soc_flux_lw_up[time,vertical_levels,:,:],
            "soc_flux_lw_down": soc_flux_lw_down[time,vertical_levels,:,:],
            "soc_flux_sw_up": soc_flux_sw_up[time,vertical_levels,:,:],
            "soc_flux_sw_down": soc_flux_sw_down[time,vertical_levels,:,:],
            "net_flux": net_flux[time,vertical_levels,:,:]
            }

format_quantities = {"Tfull": 'plain', "precipitation": 'plain', "relative_humidity": 'plain', "specific_humidity": 'plain', "zonal_wind": 'plain', 
                     "meridional_wind": 'plain', "vertical_wind": 'plain', "soc_tdt_lw": 'sci', "soc_tdt_sw": 'sci', "soc_tdt_rad": 'sci', 
                     "soc_flux_lw_up": 'plain', "soc_flux_lw_down": 'plain', "soc_flux_sw_up": 'plain', "soc_flux_sw_down": 'plain', "soc_flux_direct": 'plain', 
                     "net_flux": 'plain', "cloud_fraction": 'plain', "droplet_radius": 'plain', "frac_liq": 'plain', "qcl_rad": 'sci', "rh_in_cloud": 'plain'}

latitude_to_plot = 32 # lats[32] = 1.39 degrees, lats[43] = 29 degrees, lats[63] = 88 degrees

for quantity in column_quantities:
    chosen_variable=column_quantities[quantity][:,:,:]

    fig, ax = plt.subplots(dpi=150)
            
    x_substellar          = chosen_variable[:,latitude_to_plot,Substellar_longitude] 
    x_morning_terminator  = chosen_variable[:,latitude_to_plot,Terminator_morning_longitude]
    x_evening_terminator  = chosen_variable[:,latitude_to_plot,Terminator_evening_longitude] 
    x_antistellar         = chosen_variable[:,latitude_to_plot,Antistellar_longitude]

    y = pfull

    ax.semilogy(x_substellar,y,'r', lw=1, label = 'Substellar point',zorder=3)
    ax.semilogy(x_morning_terminator,y,'orange', lw=1 , label = 'Morning terminator',zorder=2)
    ax.semilogy(x_evening_terminator,y,'navy', lw=1, label = 'Evening terminator',zorder=1)
    ax.semilogy(x_antistellar,y,'k', lw=1, label = 'Antistellar point',zorder=0)
       
    ax.ticklabel_format(style=format_quantities[quantity], axis='x', scilimits=(0,0))
    ax.legend(frameon=True,loc='best',fontsize=8)
    ax.invert_yaxis()
    #sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    ax.set_ylabel(r"Pressure, $P \: [\mathrm{mbar}]$")

    plt.gca().set_yscale('log')
    plt.gca().invert_yaxis()
    plt.ylim([pfull[-1],pfull[0]])

    labels(ax, quantity)   

    if save_figs:
        plt.savefig(dirs["plot_output"]+'column_'+quantity+'.pdf',bbox_inches='tight')
        plt.savefig(dirs["plot_output"]+'column_'+quantity+'.png',bbox_inches='tight')

    # Show figure
    plt.show()
    #plt.close()












quantity = 'net_flux_sum'

fig, ax = plt.subplots(dpi=150)
        
x = net_flux_sum[vertical_levels]

y = pfull

ax.semilogy(x, y, lw=1)
    
ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
ax.invert_yaxis()
#sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

ax.set_ylabel(r"Pressure, $P \: [\mathrm{mbar}]$")

plt.gca().set_yscale('log')
plt.gca().invert_yaxis()
plt.ylim([pfull[-1],pfull[0]])

labels(ax, quantity)   

if save_figs:
    plt.savefig(dirs["plot_output"]+'column_'+quantity+'.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'column_'+quantity+'.png',bbox_inches='tight')

# Show figure
plt.show()
#plt.close()

if sf_name == 'Suran_lw.sf':
    # ======================== ENTIRE SPECTRAL RANGE ========================
    # fig = plt.figure(figsize=(20,10))
    # ax = fig.add_subplot(111)

    # ax.set_xlim(left=n_band_edges_cm[0], right=n_band_edges_cm[-1])
    # ax.set_ylim(bottom=1e-12, top=8e2)

    # alpha_domains    = 0.03
    # color_micrometer = 'purple'
    # color_IR         = 'red'
    # color_visible    = 'green'
    # color_uv         = 'blue'
    # ax.axvspan(1.,10.,alpha=alpha_domains, color=color_micrometer)     # Micrometer
    # ax.axvspan(10.,12500.,alpha=alpha_domains, color=color_IR)         # IR
    # ax.axvspan(12500.,25000.,alpha=alpha_domains, color=color_visible) # Visible
    # ax.axvspan(25000.,41300.,alpha=alpha_domains, color=color_uv)      # UV

    # ax.text(5200., 1e-11, 'Infrared\n(IR)', fontsize=18, family='serif', color='grey', alpha=0.8, multialignment='center')
    # ax.text(18104., 1e-11, 'Visible\n(VIS)', fontsize=18, family='serif', color='grey', alpha=0.8,multialignment='center')
    # ax.text(25685., 1e-11, 'Ultraviolet\n(UV)', fontsize=18, family='serif', color='grey', alpha=0.8, multialignment='center')

    # plt.xticks(np.arange(n_band_edges_cm[0], n_band_edges_cm[-1]+2000, 2000.0))

    # ticks = np.arange(1,30000,2000)[1:]-1
    # ticks = np.insert(ticks, 0, 1, axis=0)
    # ticks = np.append(ticks, 30000)

    # second_xaxis_mic = sigma2micron(ticks)
    # second_xaxis_mic = np.around(second_xaxis_mic,2)
    # second_xaxis_cm  = ticks

    # ax2=ax.twiny()
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_xticks(second_xaxis_cm)
    # ax2.set_xticklabels(second_xaxis_mic)
    # ax2.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=14)
        
    # ax.semilogy(soc_bins_lw_cm, soc_spectral_olr_cm[time,:,Substellar_latitude,Substellar_longitude] / band_widths_cm, lw=linewidth)

    # ax.semilogy(soc_bins_lw_cm,surf_Planck_nu(T_surf[time,Substellar_latitude,Substellar_longitude],surface_albedo[Substellar_latitude,Substellar_longitude])/band_widths_cm, color="#d5d5d5", lw=2.0, ls="--", label=r'Blackbody')

    # plot_mission = True
    # alpha_missions = 0.05
    # alpha_instruments = 0.01
    # ymin, ymax = ax.get_ylim()
    # height = ymax - ymin
    # # Calculate the y-coordinate for the center at 20% of the y-axis length in log scale
    # y_center = np.exp(0.2 * (np.log(ymax) - np.log(ymin)) + np.log(ymin))

    # LIFE_mission = {'LIFE' : mpatch.Rectangle((540.54,ymin), 1959.46, height, color='#079e65', alpha=alpha_missions)}

    # if plot_mission:
    #     for r in LIFE_mission:
    #         ax.add_artist(LIFE_mission[r])
    #         rx, ry = LIFE_mission[r].get_xy()
    #         cx = rx + LIFE_mission[r].get_width()/2.0
    #         cy = ry + LIFE_mission[r].get_height()/2.0

    #         ax.annotate(r, (cx, cy), color='navy', fontweight=1000, 
    #                     fontsize=12, ha='center', va='center', fontfamily='serif')
            
    # ax.set_xlabel('Wavenumber ' + r'$\nu$ [cm$^{-1}$]', family='serif', fontsize=14)
    # ax.set_ylabel('Spectral flux density ' + r'[W m$^{-2}$ cm]', family='serif', fontsize=14)

    # labels = [str(1)] + [str(int(tick)) for tick in ax.get_xticks()[1:]]
    # ax.set_xticklabels(labels)

    # #sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    # ax.legend()
    # #plt.show()
    # fig.savefig(dirs["plot_output"]+'spectrum.pdf',bbox_inches='tight')
    # fig.savefig(dirs["plot_output"]+'spectrum.png',bbox_inches='tight')
    # plt.close()


    # ======================== IR SPECTRAL RANGE ========================
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    ax.set_xlim(left=n_band_edges_cm[0], right=n_band_edges_cm[285]) # Limit at 7000 cm-1
    ax.set_ylim(bottom=1e-16, top=1e1)

    alpha_domains    = 0.03
    color_micrometer = 'purple'
    color_IR         = 'red'
    color_visible    = 'green'
    color_uv         = 'blue'
    ax.axvspan(1.,10.,alpha=alpha_domains, color=color_micrometer)     # Micrometer
    ax.axvspan(10.,7000.,alpha=alpha_domains, color=color_IR)         # IR (original limit = 12500.)
    #ax.axvspan(12500.,25000.,alpha=alpha_domains, color=color_visible) # Visible
    #ax.axvspan(25000.,41300.,alpha=alpha_domains, color=color_uv)      # UV

    ax.text(5200., 1e-2, 'Infrared\n(IR)', fontsize=18, family='serif', color='grey', alpha=0.8, multialignment='center')
    #ax.text(18104., 1e-11, 'Visible\n(VIS)', fontsize=18, family='serif', color='grey', alpha=0.8,multialignment='center')
    #ax.text(25685., 1e-11, 'Ultraviolet\n(UV)', fontsize=18, family='serif', color='grey', alpha=0.8, multialignment='center')

    plt.xticks(np.arange(n_band_edges_cm[0], n_band_edges_cm[285]+1000, 1000.0))

    ticks = np.arange(1,7000,1000)[1:]-1
    ticks = np.insert(ticks, 0, 1, axis=0)
    ticks = np.append(ticks, 7000)

    second_xaxis_mic = sigma2micron(ticks)
    second_xaxis_mic = np.around(second_xaxis_mic,2)
    second_xaxis_cm  = ticks

    ax2=ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(second_xaxis_cm)
    ax2.set_xticklabels(second_xaxis_mic)
    ax2.set_xlabel('Wavelength ' + r'$\lambda \; [\mathrm{\mu m}]$', family='serif', fontsize=16)
        
    ax2.tick_params(axis='both', which='major', direction='inout', length=8, labelsize=14)
    ax2.tick_params(axis='both', which='minor', direction='inout', length=8, labelsize=14)

    ax.semilogy(soc_bins_lw_cm, soc_spectral_olr_cm[time,:,Substellar_latitude,Substellar_longitude] / band_widths_cm, lw=linewidth)

    ax.semilogy(soc_bins_lw_cm,surf_Planck_cm(T_surf[time,Substellar_latitude,Substellar_longitude],surface_albedo[Substellar_latitude,Substellar_longitude])/band_widths_cm, color="#d5d5d5", lw=2.0, ls="--", label=r'Blackbody')

    plot_mission = True
    alpha_missions = 0.05
    alpha_instruments = 0.01
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin
    # Calculate the y-coordinate for the center at 20% of the y-axis length in log scale
    y_center = np.exp(0.2 * (np.log(ymax) - np.log(ymin)) + np.log(ymin))

    LIFE_mission = {'LIFE' : mpatch.Rectangle((540.54,ymin), 1959.46, height, color='#079e65', alpha=alpha_missions)}

    if plot_mission:
        for r in LIFE_mission:
            ax.add_artist(LIFE_mission[r])
            rx, ry = LIFE_mission[r].get_xy()
            cx = rx + LIFE_mission[r].get_width()/2.0
            cy = ry + LIFE_mission[r].get_height()/2.0

            ax.annotate(r, (cx, cy), color='navy', fontweight=1000, 
                        fontsize=12, ha='center', va='center', fontfamily='serif')
            
    ax.set_xlabel('Wavenumber ' + r'$\nu$ [cm$^{-1}$]', family='serif', fontsize=16)
    ax.set_ylabel('Spectral flux density ' + r'[W m$^{-2}$ cm]', family='serif', fontsize=16)

    labels = [str(1)] + [str(int(tick)) for tick in ax.get_xticks()[1:]]
    ax.set_xticklabels(labels)

    ax.tick_params(axis='both', which='major', direction='inout', length=8, labelsize=14)
    ax.tick_params(axis='both', which='minor', direction='inout', length=8, labelsize=14)
    #sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    ax.legend(fontsize=14)
    #plt.show()
    fig.savefig(dirs["plot_output"]+'spectrum.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'spectrum.png',bbox_inches='tight')
    plt.close()