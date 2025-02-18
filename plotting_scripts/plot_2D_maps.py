import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatch
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')

# Default size of labels
size = 10
# Save figures
save_figs = True
# Resolution of the 2D maps
resolution = 200
cmap = plt.cm.RdYlBu_r

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

simulation  = 'planetb_presentdayEarth_rot0/ISR_1400' #'planetb_EoceneEarth_rot0' #'planetb_ArcheanEarth_rot0' #'Earth'
run         = 'run0210'
netcdf_file = 'atmos_monthly.nc' #'atmos_monthly.nc' #'atmos_monthly.nc'
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
    "plot_output": isca_plots+"/"+simulation+"/"+run+"/maps/"
    }

if not os.path.exists(dirs["plot_output"]):
    os.makedirs(dirs["plot_output"]) 

lons = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lon'].values # Longitudes [degree]
lats = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lat'].values # Latitudes  [degree]
lon, lat = np.meshgrid(lons, lats)

lons_edges = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['lonb'].values # Longitude edges [degree]
lats_edges = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['latb'].values # Latitude edges  [degree]
lonb, latb = np.meshgrid(lons_edges, lats_edges)

pfull = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['pfull'].values # Approx full (midpoint)  pressure levels [Pa]
phalf = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['phalf'].values # Approx half (interface) pressure levels [Pa]

if sf_name == 'Suran_lw.sf':
    soc_bins_lw = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_bins_lw'].values # Socrates LW & SW spectral bin centers [m]
    soc_bins_lw_um = soc_bins_lw * 1e6 # [microns]
    soc_bins_lw_cm = np.flip(sigma2meter(soc_bins_lw))

p_surf = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['ps'].values # Surface pressure [Pa]

precipitation = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['precipitation'].values # Precipitation from resolved, parameterised and snow [kg/m2/s]
precipitation = precipitation*86400. # converted from kg/m2/s to mm/day. kg/m2/s = mm/s.

relative_humidity = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['rh'].values # Relative humidity [percent]

T_surf = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['t_surf'].values # Surface temperature [K]

flux_t = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['flux_t'].values # Sensible heat flux up at surface [W/m2]

flux_lhe = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['flux_lhe'].values # Latent heat flux up at surface [W/m2]

specific_humidity = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['sphum'].values # Specific humidity [kg/kg]
specific_humidity = specific_humidity*100. # converted to percent

zonal_wind      = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['ucomp'].values # Zonal wind component      [m/s]
meridional_wind = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['vcomp'].values # Meridional wind component [m/s]
vertical_wind   = -xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['omega'].values # Vertical wind component   [Pa/s], increases downward

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
soc_toa_sw_down = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_toa_sw_down'].values # Socrates TOA SW flux down [W/m2]

soc_flux_lw_up   = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_lw_up'].values   # Socrates LW flux up   [W/m2]
soc_flux_lw_down = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_lw_down'].values # Socrates LW flux down [W/m2]
soc_flux_sw_up   = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_sw_up'].values   # Socrates SW flux up   [W/m2]
soc_flux_sw_down = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_flux_sw_down'].values # Socrates SW flux down [W/m2]

if sf_name == 'Suran_lw.sf':
    soc_spectral_olr = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['soc_spectral_olr'].values # Socrates substellar LW OLR spectrum [W/m2/band]
    soc_spectral_olr_um = soc_spectral_olr*1e-6 # W/m2/m --> W/m2/um
    soc_spectral_olr_cm = np.array([ np.array([ np.array([ np.flipud(Wm2um_to_Wm2cm(soc_spectral_olr_um[i,:,j,k]/band_widths_um, soc_bins_lw_um)) for k in range(soc_spectral_olr_um.shape[3]) ]) for j in range(soc_spectral_olr_um.shape[2]) ]) for i in range(soc_spectral_olr_um.shape[0]) ])
    soc_spectral_olr_cm = np.transpose(soc_spectral_olr_cm, (0, 3, 1, 2))

cloud_fraction = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['cf'].values # Cloud fraction for the simple cloud scheme [0-1]
cloud_fraction = cloud_fraction*100. # converted to percent

droplet_radius = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['reff_rad'].values # Effective cloud particle radius [microns]

frac_liq = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['frac_liq'].values # Liquid cloud fraction (liquid, mixed-ice phase, ice) [0-1]
frac_liq = frac_liq *100. # converted to percent

qcl_rad = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['qcl_rad'].values # Specific humidity of cloud liquid [kg/kg]
qcl_rad = qcl_rad*100. # converted to percent
rh_in_cloud = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['rh_in_cf'].values # Relative humidity in cloud [percent]

surface_albedo = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['albedo'].values # Surface albedo

# Ozone file has its own lat-lon grid and pfull array. ozone has same number of latitudes but only 2 longitudes.
pfull_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['pfull'].values # Approx full (midpoint)  pressure levels [mbar]
lev_o3 = 27 
#print(pfull_o3)
Plev_o3 = pfull_o3[lev_o3] # 11 mbar

lons_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['lon'].values # Longitudes [degree]
lats_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['lat'].values # Latitudes  [degree]
lon_o3, lat_o3 = np.meshgrid(lons_o3, lats_o3)

lons_edges_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['lonb'].values # Longitude edges [degree]
lats_edges_o3 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['latb'].values # Latitude edges  [degree]
lonb_o3, latb_o3 = np.meshgrid(lons_edges_o3, lats_edges_o3)
ozone_1990 = xr.open_dataset(os.getenv('GFDL_BASE')+'/input/rrtm_input_files/ozone_1990.nc', decode_times=False)['ozone_1990'].values[-1,:,:,:]    # Ozone mass mixing ratio [kg/kg] from Fortuin & Langematz 1995, 10.1117/12.198578
ozone_1990 = ozone_1990*1e6 # Converting from kg/kg to ppmm
ozone_1990 = np.repeat(ozone_1990, 128, axis=2) # Copy data across grid longitudes for plotting

#========================================================
# Time-step to probe
time=-1
# Pressure level to probe, when applicable (along pfull)
lev = 6 
print(pfull)
Plev = pfull[lev] # 10 mbar

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

#========================================================

excluded_levels = 0
for i in range(len(Tfull[0,:,0,0])):
    if ((any(any(row>1e4) for row in Tfull[time,i,:,:])) or (any(any(row<=0.0) for row in Tfull[time,i,:,:]))):
        excluded_levels+=1

vertical_levels = slice(0,len(pfull)-excluded_levels) # Vertically regridded quantities may have no data near the surface

Global_quantities = {
    "Tfull": Tfull[time,vertical_levels,:,:],
    "precipitation": precipitation[time,:,:],
    "relative_humidity": relative_humidity[time,vertical_levels,:,:],
    "specific_humidity": specific_humidity[time,vertical_levels,:,:],
    "zonal_wind": zonal_wind[time,vertical_levels,:,:],
    "meridional_wind": meridional_wind[time,vertical_levels,:,:],
    "soc_tdt_lw": soc_tdt_lw[time,vertical_levels,:,:],
    "soc_tdt_sw": soc_tdt_sw[time,vertical_levels,:,:],
    "soc_tdt_rad": soc_tdt_rad[time,vertical_levels,:,:],
    "soc_flux_lw_up": soc_flux_lw_up[time,vertical_levels,:,:],
    "soc_flux_lw_down": soc_flux_lw_down[time,vertical_levels,:,:],
    "soc_flux_sw_up": soc_flux_sw_up[time,vertical_levels,:,:],
    "soc_flux_sw_down": soc_flux_sw_down[time,vertical_levels,:,:],
    "cloud_fraction": cloud_fraction[time,vertical_levels,:,:],
    "droplet_radius": droplet_radius[time,vertical_levels,:,:],
    "frac_liq": frac_liq[time,vertical_levels,:,:],
    "qcl_rad": qcl_rad[time,vertical_levels,:,:],
    "rh_in_cloud": rh_in_cloud[time,vertical_levels,:,:],
    "p_surf": p_surf[time,:,:],
    "T_surf": T_surf[time,:,:],
    "flux_t": flux_t[time,:,:],
    "flux_lhe": flux_lhe[time,:,:],
    "soc_surf_flux_lw": soc_surf_flux_lw[time,:,:],
    "soc_surf_flux_sw": soc_surf_flux_sw[time,:,:],
    "soc_surf_flux_lw_down": soc_surf_flux_lw_down[time,:,:],
    "soc_surf_flux_sw_down": soc_surf_flux_sw_down[time,:,:],
    "soc_olr": soc_olr[time,:,:],
    "soc_toa_sw": soc_toa_sw[time,:,:],
    "soc_toa_sw_down": soc_toa_sw_down[time,:,:],
    "ozone": ozone_1990[:,:,:]
}

format_quantities = {"Tfull": "%0.1f", "precipitation": "%0.1f", "relative_humidity": "%0.1f", "specific_humidity": "%0.3f", "zonal_wind": "%0.1f", "meridional_wind": "%0.1f",
                     "soc_tdt_lw": "%0.2e", "soc_tdt_sw": "%0.2e", "soc_tdt_rad": "%0.2e", "soc_flux_lw_up": "%0.1f", "soc_flux_lw_down": "%0.1f",
                     "soc_flux_sw_up": "%0.1f", "soc_flux_sw_down": "%0.1f", "cloud_fraction": "%0.1f", "droplet_radius": "%0.1f", "frac_liq": "%0.1f",
                     "qcl_rad": "%0.2e", "rh_in_cloud": "%0.1f", "p_surf": "%0.2e", "T_surf": "%0.1f", "flux_t": "%0.1f", "flux_lhe": "%0.1f",
                     "soc_surf_flux_lw": "%0.1f", "soc_surf_flux_sw": "%0.1f", "soc_surf_flux_lw_down": "%0.1f", "soc_surf_flux_sw_down": "%0.1f",
                     "soc_olr": "%0.1f", "soc_toa_sw": "%0.1f", "soc_toa_sw_down": "%0.1f", "ozone": "%0.1f"}

omega = vertical_wind[time,vertical_levels,:,:]*1e3 # Artificial scaling factor to emphasize the ascent and descent on plots where the y axis is logarithmic pressure.
omega_lin = vertical_wind[time,vertical_levels,:,:]*1e2

# First plot the Orthographic plots for a more natural look

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

# for quantity in Global_quantities:#['Tfull']:
        
#     fig, axs = plt.subplots(3, 2, figsize=(8, 10), dpi=150, subplot_kw={'projection': None})

#     # Flatten the axs array for easy iteration
#     axs = axs.flatten()
#     for i, face in enumerate(faces.keys()):
#         lat_min, lat_max, lon_indices = faces[face]
#         chosen_variable = Global_quantities[quantity]
#         if chosen_variable.ndim == 3:
#             chosen_variable = Global_quantities[quantity][lev, :, :]
#         elif (quantity == 'precipitation' or quantity == 'p_surf' or quantity == 'T_surf' or quantity == 'flux_t' or quantity == 'flux_lhe' 
#                 or quantity == 'soc_surf_flux_lw' or quantity == 'soc_surf_flux_sw' or quantity == 'soc_surf_flux_lw_down' or quantity == 'soc_surf_flux_sw_down'):
#             chosen_variable = Global_quantities[quantity]
#         elif (quantity == 'soc_olr' or quantity == 'soc_toa_sw' or quantity == 'soc_toa_sw_down'):
#             chosen_variable = Global_quantities[quantity]

#         if quantity == 'ozone':
#             chosen_variable = Global_quantities[quantity][lev_o3, :, :]

#         # Apparently if I have [lev, slice(lat), array(lon)] numpy creates a new dimension which yields shape (65,64) instead of (64,65)...
#         chosen_variable = chosen_variable[lat_slices[face], lon_slices[face]] 

#         ax = axs[i]
#         ax = plt.subplot(3, 2, i + 1, projection=projections[face])
#         gl = ax.gridlines(draw_labels=True)
        
#         lon_plot = lon[lat_slices[face], lon_slices[face]]
#         lat_plot = lat[lat_slices[face], lon_slices[face]]

#         CS = ax.contourf(lon_plot, lat_plot, chosen_variable, resolution, extend='both', 
#                         transform=ccrs.PlateCarree(), cmap=cmap, vmin=np.min(chosen_variable), vmax=np.max(chosen_variable))

#         for coll in CS.collections:
#             coll.set_edgecolor("face")

#         CB = plt.colorbar(CS, ax=ax, orientation='horizontal', format="%0.0f")
#         CB.ax.plot([np.mean(chosen_variable)]*2,[1, 0], 'k')
#         labels(CB, quantity) 
#         CB.ax.tick_params(labelsize=7)
#         ax.set_title(f'{face.replace("_", " ")} - Average = ' + '{:0.2e}'.format(np.mean(chosen_variable)), fontsize=size)
#         ax.set_global()  # make the map global rather than have it zoom in to the extents of any plotted data

#     plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
#     plt.savefig(dirs["plot_output"] + 'Ortho_'+quantity+'.pdf', bbox_inches='tight')
#     plt.savefig(dirs["plot_output"] + 'Ortho_'+quantity+'.png', bbox_inches='tight')
#     #plt.show()
#     plt.close()

# Now the Mollweide maps
projection = ccrs.Mollweide(central_longitude=0)

loop_over_pfull = True
if loop_over_pfull:
    pressures = enumerate(pfull[vertical_levels])
else:
    pressures = enumerate([pfull[lev]]) # Wrap in a list to make it iterable

for quantity in Global_quantities:
    if Global_quantities[quantity].ndim == 2:
        if (quantity == 'precipitation' or quantity == 'p_surf' or quantity == 'T_surf' or quantity == 'flux_t' or quantity == 'flux_lhe' 
            or quantity == 'soc_surf_flux_lw' or quantity == 'soc_surf_flux_sw' or quantity == 'soc_surf_flux_lw_down' or quantity == 'soc_surf_flux_sw_down'):
            pressures = enumerate([pfull[-1]])
        elif (quantity == 'soc_olr' or quantity == 'soc_toa_sw' or quantity == 'soc_toa_sw_down'):
            pressures = enumerate([pfull[0]])

    if quantity == 'ozone':
        if loop_over_pfull:
            pressures = enumerate(pfull_o3)
        else:
            pressures = enumerate([pfull_o3[lev_o3]])

    for levi, Plevi in pressures:
        if np.any(Global_quantities[quantity]):
            print("Processing quantity "+quantity)    
            chosen_variable=Global_quantities[quantity]
            if chosen_variable.ndim == 3:
                chosen_variable = chosen_variable[levi, :, :]

            fig = plt.figure(dpi=150)
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            gl = ax.gridlines(draw_labels=True)
                
            CS = ax.contourf(lon,lat,chosen_variable,resolution,extend='both',transform=ccrs.PlateCarree(),transform_first=False,cmap=cmap,lw=0,ls=None,latlon=True,vmin=np.min(chosen_variable),vmax=np.max(chosen_variable), zorder=1)
            skip = (slice(None, None, 3), slice(None, None, 3))
            if quantity != 'ozone':
                ax.quiver(lon[skip],lat[skip],Global_quantities['zonal_wind'][levi,:,:][skip],Global_quantities['meridional_wind'][levi,:,:][skip],transform=ccrs.PlateCarree(),cmap=cmap, zorder=2)
                    
            # Colour bar and formatting
            for c in CS.collections:
                c.set_edgecolor("face")
            
            CB = plt.colorbar(CS, orientation='horizontal', format=format_quantities[quantity])
            #mean_loc = (np.mean(chosen_variable) - CB.vmin) / (CB.vmax - CB.vmin)
            #CB.ax.plot([mean_loc]*2,[1, 0], 'k')
            CB.ax.plot([np.mean(chosen_variable)]*2,[1, 0], 'k')

            labels(CB, quantity) 
            
            CB.ax.tick_params(labelsize=size)

            if format_quantities[quantity] == "%0.2e": # rotate if using scientific notation
                CB.ax.tick_params(rotation=45)
            
            if Global_quantities[quantity].ndim == 2:
                plt.title('Average = ' + '{:0.2e}'.format(np.mean(chosen_variable)),fontsize=size)
            elif Global_quantities[quantity].ndim == 3:
                #plt.title('{:0.1f}'.format(Plevi) + ' mbar. Average = ' + '{:0.2e}'.format(np.mean(chosen_variable)),fontsize=size)
                plt.title(f'{Plevi:0.1f} mbar. Average = {np.mean(chosen_variable):0.2e}', fontsize=size)

            # Increase size of labels
            plt.tick_params(axis='both', which='both', labelsize=size)
            
            # Save figure
            plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
            if save_figs:
                file_base = dirs["plot_output"] + f'Global_{quantity}_{levi:03}_P{Plevi:0.1f}'
                plt.savefig(f'{file_base}.pdf', bbox_inches='tight')
                plt.savefig(f'{file_base}.png', bbox_inches='tight')

            ax.set_global() # make the map global rather than have it zoom in to the extents of any plotted data
            plt.show()
            #plt.close()


# Plotting the meridional maps, that is the crest of latitudes around the planet, [0,-90]U[-90,0]U[0,90]U[90,0]
lats_full = np.concatenate((lats[:32][::-1], lats[:32], lats[32:], lats[32:][::-1])) # initial values
lats_full_diff = np.diff(lats_full)
lats_full = np.zeros(128)
lats_full[0] = lats[0]*2
for i in range(1,128):
    lats_full[i] = lats_full[i-1]+abs(lats_full_diff[i-1])
# Remove the zero, add the final latitude
lats_full = np.delete(lats_full, 64)
lats_full = np.append(lats_full, lats_full[-1]+abs(lats_full_diff[-1]))

#meridional_wind_full = np.concatenate((Global_quantities['meridional_wind'][vertical_levels, :, Substellar_longitude], Global_quantities['meridional_wind'][vertical_levels, ::-1, Antistellar_longitude]), axis=1)
a = np.flip(Global_quantities['meridional_wind'][vertical_levels, :32, Antistellar_longitude], axis=1) # Nightside, from the antistellar point to the south pole
b = Global_quantities['meridional_wind'][vertical_levels, :32, Substellar_longitude] # Dayside, from the south pole to the substellar point 
c = Global_quantities['meridional_wind'][vertical_levels, 32:, Substellar_longitude] # Dayside, from the substellar point to the north pole 
d = np.flip(Global_quantities['meridional_wind'][vertical_levels, 32:, Antistellar_longitude], axis=1) # Nightside, from the north pole to the antistellar point
meridional_wind_full = np.concatenate((a,b,c,d),axis=1)

a = np.flip(omega[:, :32, Antistellar_longitude], axis=1) 
b = omega[:, :32, Substellar_longitude] 
c = omega[:, 32:, Substellar_longitude] 
d = np.flip(omega[:, 32:, Antistellar_longitude], axis=1)
omega_full = np.concatenate((a,b,c,d),axis=1)

a = np.flip(omega_lin[:, :32, Antistellar_longitude], axis=1) 
b = omega_lin[:, :32, Substellar_longitude] 
c = omega_lin[:, 32:, Substellar_longitude] 
d = np.flip(omega_lin[:, 32:, Antistellar_longitude], axis=1)
omega_lin_full = np.concatenate((a,b,c,d),axis=1)

for quantity, array in Global_quantities.items():
    if array.ndim == 3: # Meridional means only for quantities with a vertical dimension
        if np.any(Global_quantities[quantity]):
            print("Processing quantity "+quantity)    
            a = np.flip(Global_quantities[quantity][:, :32, Antistellar_longitude], axis=1) 
            b = Global_quantities[quantity][:, :32, Substellar_longitude] 
            c = Global_quantities[quantity][:, 32:, Substellar_longitude] 
            d = np.flip(Global_quantities[quantity][:, 32:, Antistellar_longitude], axis=1)
            chosen_variable = np.concatenate((a,b,c,d),axis=1)
            
            fig, ax = plt.subplots(dpi=150)
                
            if quantity == 'ozone':
                y = pfull_o3
                y_bottom = pfull_o3[-1]
                y_top = pfull_o3[0]
            else:
                y = pfull[vertical_levels]
                y_bottom = pfull[vertical_levels.stop-1]
                y_top = pfull[0]

            plot_filled = ax.contourf(lats_full,y,chosen_variable,resolution,extend='both',cmap=cmap, zorder=1)
            skip = (slice(None, None, 3), slice(None, None, 3))
            ax.quiver(lats_full[skip[0]],pfull[vertical_levels][skip[0]],meridional_wind_full[skip],omega_full[skip],color='black', zorder=2)
                
            ax.set_xlabel(r'Latitude [$\degree$]', size=size)
            ax.set_ylabel(r'Pressure [mbar]', size=size)
            
            plt.gca().set_yscale('log')
            plt.gca().invert_yaxis()
            plt.ylim([y_bottom,y_top])

            for c in plot_filled.collections:
                c.set_edgecolor("face")
            CB = plt.colorbar(plot_filled, orientation='horizontal', format=format_quantities[quantity])

            labels(CB, quantity) 
            
            CB.ax.tick_params(labelsize=size)

            if format_quantities[quantity] == "%0.2e": # rotate if using scientific notation
                CB.ax.tick_params(rotation=45)

            # Increase size of labels
            plt.tick_params(axis='both', which='both', labelsize=size)
            
            # Save figure
            plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
            if save_figs:
                plt.savefig(dirs["plot_output"]+'Meridional_'+quantity+'.pdf',bbox_inches='tight')
                plt.savefig(dirs["plot_output"]+'Meridional_'+quantity+'.png',bbox_inches='tight')

            # Show figure
            plt.show()
            #plt.close()

# And finally, the zonal maps, that is the crest of longitudes around the planet.
for quantity, array in Global_quantities.items():
    if array.ndim == 3: # Zonal means only for quantities with a vertical dimension
        if np.any(Global_quantities[quantity]):
            print("Processing quantity "+quantity)    
            chosen_variable=np.mean(Global_quantities[quantity],axis=1)

            fig, ax = plt.subplots(dpi=150)
                
            if quantity == 'ozone':
                y = pfull_o3
                y_bottom = pfull_o3[-1]
                y_top = pfull_o3[0]
            else:
                y = pfull[vertical_levels]
                y_bottom = pfull[vertical_levels.stop-1]
                y_top = pfull[0]
            
            plot_filled = ax.contourf(lons,y,chosen_variable,resolution,extend='both',cmap=cmap, zorder=1)
            skip = (slice(None, None, 3), slice(None, None, 3))
            ax.quiver(lons[skip[0]],pfull[vertical_levels][skip[0]],np.mean(Global_quantities['zonal_wind'],axis=1)[skip],np.mean(omega[:,:,:],axis=1)[skip],color='black', zorder=2)
                
            ax.set_xlabel(r'Longitude [$\degree$]', size=size)
            ax.set_ylabel(r'Pressure [mbar]', size=size)
            
            plt.gca().set_yscale('log')
            plt.gca().invert_yaxis()
            plt.ylim([y_bottom,y_top])

            for c in plot_filled.collections:
                c.set_edgecolor("face")
            CB = plt.colorbar(plot_filled, orientation='horizontal', format=format_quantities[quantity])

            labels(CB, quantity) 
            
            CB.ax.tick_params(labelsize=size)

            if format_quantities[quantity] == "%0.2e": # rotate if using scientific notation
                CB.ax.tick_params(rotation=45)

            # Increase size of labels
            plt.tick_params(axis='both', which='both', labelsize=size)
            
            # Save figure
            plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
            if save_figs:
                plt.savefig(dirs["plot_output"]+'Zonal_'+quantity+'.pdf',bbox_inches='tight')
                plt.savefig(dirs["plot_output"]+'Zonal_'+quantity+'.png',bbox_inches='tight')

            # Show figure
            plt.show()
            #plt.close()

#---------------------------------------------------------------------------------------------------------
#         Plotting the meridional and zonal maps with linear pressure scales for interpretation
#---------------------------------------------------------------------------------------------------------

for quantity, array in Global_quantities.items():
    if array.ndim == 3: # Meridional means only for quantities with a vertical dimension
        if np.any(Global_quantities[quantity]):
            print("Processing quantity "+quantity)    
            a = np.flip(Global_quantities[quantity][:, :32, Antistellar_longitude], axis=1) 
            b = Global_quantities[quantity][:, :32, Substellar_longitude] 
            c = Global_quantities[quantity][:, 32:, Substellar_longitude] 
            d = np.flip(Global_quantities[quantity][:, 32:, Antistellar_longitude], axis=1)
            chosen_variable = np.concatenate((a,b,c,d),axis=1)
            
            fig, ax = plt.subplots(dpi=150)
                
            if quantity == 'ozone':
                y = pfull_o3
                y_bottom = pfull_o3[-1]
                y_top = pfull_o3[0]
            else:
                y = pfull[vertical_levels]
                y_bottom = pfull[vertical_levels.stop-1]
                y_top = pfull[0]

            plot_filled = ax.contourf(lats_full,y,chosen_variable,resolution,extend='both',cmap=cmap, zorder=1)
            skip = (slice(None, None, 3), slice(None, None, 3))
            ax.quiver(lats_full[skip[0]],pfull[vertical_levels][skip[0]],meridional_wind_full[skip],omega_lin_full[skip],color='black', zorder=2)
                
            ax.set_xlabel(r'Latitude [$\degree$]', size=size)
            ax.set_ylabel(r'Pressure [mbar]', size=size)
            
            plt.gca().invert_yaxis()
            plt.ylim([y_bottom,y_top])

            for c in plot_filled.collections:
                c.set_edgecolor("face")
            CB = plt.colorbar(plot_filled, orientation='horizontal', format=format_quantities[quantity])

            labels(CB, quantity) 
            
            CB.ax.tick_params(labelsize=size)

            if format_quantities[quantity] == "%0.2e": # rotate if using scientific notation
                CB.ax.tick_params(rotation=45)

            # Increase size of labels
            plt.tick_params(axis='both', which='both', labelsize=size)
            
            # Save figure
            plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
            if save_figs:
                plt.savefig(dirs["plot_output"]+'Meridional_'+quantity+'_lin.pdf',bbox_inches='tight')
                plt.savefig(dirs["plot_output"]+'Meridional_'+quantity+'_lin.png',bbox_inches='tight')

            # Show figure
            plt.show()
            #plt.close()

# And finally, the zonal maps, that is the crest of longitudes around the planet.
for quantity, array in Global_quantities.items():
    if array.ndim == 3: # Zonal means only for quantities with a vertical dimension
        if np.any(Global_quantities[quantity]):
            print("Processing quantity "+quantity)    
            chosen_variable=np.mean(Global_quantities[quantity],axis=1)

            fig, ax = plt.subplots(dpi=150)
                
            if quantity == 'ozone':
                y = pfull_o3
                y_bottom = pfull_o3[-1]
                y_top = pfull_o3[0]
            else:
                y = pfull[vertical_levels]
                y_bottom = pfull[vertical_levels.stop-1]
                y_top = pfull[0]
            
            plot_filled = ax.contourf(lons,y,chosen_variable,resolution,extend='both',cmap=cmap, zorder=1)
            skip = (slice(None, None, 3), slice(None, None, 3))
            ax.quiver(lons[skip[0]],pfull[vertical_levels][skip[0]],np.mean(Global_quantities['zonal_wind'],axis=1)[skip],np.mean(omega_lin[:,:,:],axis=1)[skip],color='black', zorder=2)
                
            ax.set_xlabel(r'Longitude [$\degree$]', size=size)
            ax.set_ylabel(r'Pressure [mbar]', size=size)
            
            plt.gca().invert_yaxis()
            plt.ylim([y_bottom,y_top])

            for c in plot_filled.collections:
                c.set_edgecolor("face")
            CB = plt.colorbar(plot_filled, orientation='horizontal', format=format_quantities[quantity])

            labels(CB, quantity) 
            
            CB.ax.tick_params(labelsize=size)

            if format_quantities[quantity] == "%0.2e": # rotate if using scientific notation
                CB.ax.tick_params(rotation=45)

            # Increase size of labels
            plt.tick_params(axis='both', which='both', labelsize=size)
            
            # Save figure
            plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
            if save_figs:
                plt.savefig(dirs["plot_output"]+'Zonal_'+quantity+'_lin.pdf',bbox_inches='tight')
                plt.savefig(dirs["plot_output"]+'Zonal_'+quantity+'_lin.png',bbox_inches='tight')

            # Show figure
            plt.show()
            #plt.close()
#---------------------------------------------------------------------------------------------------------

# If the spectral resolution is high enough, plot the meridional variation of the OLR across the spectrum.
if sf_name == 'Suran_lw.sf':
    spectral_olr = soc_spectral_olr_um[:,:,:]

    fig, ax = plt.subplots(dpi=150)
            
    plot_filled = plt.contourf(lats,soc_bins_lw_um,spectral_olr[:,:,Substellar_longitude],resolution,extend='both',cmap=cmap)

    ax.set_xlabel(r'Latitude [$\degree$]', size=size)
    ax.set_ylabel(r'Wavelength [$\mu$m]', size=size)

    ax.set_yscale('log')
    plt.ylim([soc_bins_lw_um[0],soc_bins_lw_um[-1]])
        
    for c in plot_filled.collections:
        c.set_edgecolor("face")
    CB = plt.colorbar(plot_filled, orientation='horizontal', format="%0.1e")

    CB.set_label(r'Spectral OLR [$W \; \mathrm{m}^{-2} \; \mu \mathrm{m}$]', size=size)
    CB.ax.tick_params(rotation=45)

    ax.axhline(y=4.0, color='k', linestyle='-')
    ax.axhline(y=18.5, color='k', linestyle='-')

    if save_figs:
        plt.savefig(dirs["plot_output"]+'Zonal_mean_spectral_olr.pdf',bbox_inches='tight')
        plt.savefig(dirs["plot_output"]+'Zonal_mean_spectral_olr.png',bbox_inches='tight')

    plt.show()
    #plt.close()