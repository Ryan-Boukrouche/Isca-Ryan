import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatch
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

# Function to load a specific variable (e.g., temp, precipitation) for a specific run
def load_variable(run, dirs, variable_name, simulation):
    file = f"run{str(run).zfill(4)}/"+netcdf_file  # Ensure run number is zero-padded
    dataset = xr.open_dataset(os.path.join(dirs[simulation], file), decode_times=False)
    variable_data = dataset[variable_name].values  # Load the variable by its name
    return variable_data

# Generalized function to compute the average of any quantity over a range of runs
def average_quantity(start_run, end_run, dirs, variable_name, simulation):
    all_runs = []
    
    for run in range(start_run, end_run + 1):  # Loop over the defined run range
        variable_run = load_variable(run, dirs, variable_name, simulation)  # Load variable data
        all_runs.append(variable_run)
    
    # Stack all the runs along the time axis and compute the average over the time axis (axis=0)
    stacked_runs = np.concatenate(all_runs, axis=0)  # Combine time steps from all runs
    average_variable = np.mean(stacked_runs, axis=0)  # Average over the time dimension
    
    return average_variable

# Global constants
h  = 6.62607015e-34 # Planck's constant [J.s]
c  = 2.99792458e8   # Speed of light [m/s] 
kb = 1.380649e-23   # Boltzmann constant [J/K]
c1 = 2*h*c**2       # First  radiation constant
c2 = h*c/kb         # Second radiation constant
radius_Earth      = 6.371e6   # Earth radius [m]
radius_Teegardenb = 1.02*6.371e6   # Teegarden's Star b radius [m]

# Figure settings
size = 10       # Default size of labels
linewidth = 1.5 

# Save figures
save_figs = True

simulation = 'Earth'
netcdf_file  = 'atmos_monthly.nc'
isca_plots = '/proj/bolinc/users/x_ryabo/Isca-Ryan_plots'

# Specify the range of the climatology
start_run1 = 100 # Start where the climatology starts
end_run1   = 719 # Exclude the last year if it's using Suran instead of miniSuran

forcing_index   = 720 # Forcing before readjustment
newT_surf_index = 1000 # Readjusted surface temperature

dirs = {
    "output": os.getenv('GFDL_DATA')+"/"+simulation+"/",
    "plot_output": isca_plots+"/"+simulation+"/"
    }

if not os.path.exists(dirs["plot_output"]):
    os.makedirs(dirs["plot_output"])


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
    """read in grid from approriate file, and return 2D array of grid cell areas in metres**2. Taken from src/extra/python/scripts/cell_area.py."""
    resolution_file = Dataset(dirs['output']+'run'+f'{int(start_run1):04}'+'/'+netcdf_file, 'r', format='NETCDF3_CLASSIC')

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

area        = np.array(cell_area(radius=radius_Teegardenb))
area_planet = 4.0*np.pi*(radius_Teegardenb)**2

variables_to_average = ['t_surf', 'temp', 'soc_flux_lw_up', 'soc_flux_lw_down', 'soc_flux_sw_up', 'soc_flux_sw_down', 'soc_surf_flux_lw', 'soc_surf_flux_sw', 'soc_surf_flux_lw_down',
                        'soc_surf_flux_sw_down']  

# simulation1
#---------------------------------------------------------------------------------------------------------------------
averaged_quantities = {}
for variable_name in tqdm(variables_to_average, desc="Processing variables", unit="var"):
    averaged_quantities[variable_name] = average_quantity(start_run1, end_run1, dirs, variable_name, "output")

lons = xr.open_dataset(dirs["output"]+'run0001/'+netcdf_file, decode_times=False)['lon'].values # Longitudes [degree]
lats = xr.open_dataset(dirs["output"]+'run0001/'+netcdf_file, decode_times=False)['lat'].values # Latitudes  [degree]
lon, lat = np.meshgrid(lons, lats)

lons_edges = xr.open_dataset(dirs["output"]+'run0001/'+netcdf_file, decode_times=False)['lonb'].values # Longitude edges [degree]
lats_edges = xr.open_dataset(dirs["output"]+'run0001/'+netcdf_file, decode_times=False)['latb'].values # Latitude edges  [degree]
lonb, latb = np.meshgrid(lons_edges, lats_edges)

pfull = xr.open_dataset(dirs["output"]+'run0001/'+netcdf_file, decode_times=False)['pfull'].values # Approx full (midpoint)  pressure levels [Pa]
phalf = xr.open_dataset(dirs["output"]+'run0001/'+netcdf_file, decode_times=False)['phalf'].values # Approx half (interface) pressure levels [Pa]

# Climatologies
T_surf1 = averaged_quantities['t_surf'] # Surface temperature [K]
Tfull1  = averaged_quantities['temp'] # Temperature at full (midpoint) pressure levels [K]

soc_surf_flux_lw1      = averaged_quantities['soc_surf_flux_lw']      # Socrates Net LW surface flux (up)   [W/m2]
soc_surf_flux_sw1      = averaged_quantities['soc_surf_flux_sw']      # Socrates Net SW surface flux (down) [W/m2]
soc_surf_flux_lw_down1 = averaged_quantities['soc_surf_flux_lw_down'] # Socrates LW surface flux down       [W/m2]
soc_surf_flux_sw_down1 = averaged_quantities['soc_surf_flux_sw_down'] # Socrates SW surface flux down       [W/m2]

soc_flux_lw_up1   = averaged_quantities['soc_flux_lw_up']   # Socrates LW flux up   [W/m2]
soc_flux_lw_down1 = averaged_quantities['soc_flux_lw_down'] # Socrates LW flux down [W/m2]
soc_flux_sw_up1   = averaged_quantities['soc_flux_sw_up']   # Socrates SW flux up   [W/m2]
soc_flux_sw_down1 = averaged_quantities['soc_flux_sw_down'] # Socrates SW flux down [W/m2]

ASR1   = soc_flux_sw_down1[0,:,:] - soc_flux_sw_up1[0,:,:]
OTR1   = soc_flux_lw_up1[0,:,:] - soc_flux_lw_down1[0,:,:] 
ASP1 = np.zeros_like(ASR1)
sum_ASP1 = np.zeros(ASR1.shape[0])
OTP1 = np.zeros_like(OTR1)
sum_OTP1 = np.zeros(OTR1.shape[0])

for j in range(len(lats_edges)-1):
    for k in range(len(lons_edges)-1):
        ASP1[j,k] = ASR1[j,k]*area[j,k]
        OTP1[j,k] = OTR1[j,k]*area[j,k]
sum_ASP1 = np.sum(ASP1)
sum_OTP1 = np.sum(OTP1)

ASR1 = sum_ASP1/area_planet
OTR1 = sum_OTP1/area_planet
Imbalance1 = ASR1 - OTR1 # -0.4169752693478017

# simulation2
#---------------------------------------------------------------------------------------------------------------------

# Climatologies
T_surf2 = xr.open_dataset(dirs["output"]+'run'+f'{int(newT_surf_index):04}'+'/'+netcdf_file, decode_times=False)['t_surf'].values # Surface temperature [K]
Tfull2  = xr.open_dataset(dirs["output"]+'run'+f'{int(newT_surf_index):04}'+'/'+netcdf_file, decode_times=False)['temp'].values # Temperature at full (midpoint) pressure levels [K]

soc_surf_flux_lw2      = xr.open_dataset(dirs["output"]+'run'+f'{int(forcing_index):04}'+'/'+netcdf_file, decode_times=False)['soc_surf_flux_lw'].values      # Socrates Net LW surface flux (up)   [W/m2]
soc_surf_flux_sw2      = xr.open_dataset(dirs["output"]+'run'+f'{int(forcing_index):04}'+'/'+netcdf_file, decode_times=False)['soc_surf_flux_sw'].values      # Socrates Net SW surface flux (down) [W/m2]
soc_surf_flux_lw_down2 = xr.open_dataset(dirs["output"]+'run'+f'{int(forcing_index):04}'+'/'+netcdf_file, decode_times=False)['soc_surf_flux_lw_down'].values # Socrates LW surface flux down       [W/m2]
soc_surf_flux_sw_down2 = xr.open_dataset(dirs["output"]+'run'+f'{int(forcing_index):04}'+'/'+netcdf_file, decode_times=False)['soc_surf_flux_sw_down'].values # Socrates SW surface flux down       [W/m2]

soc_flux_lw_up2      = xr.open_dataset(dirs["output"]+'run'+f'{int(forcing_index):04}'+'/'+netcdf_file, decode_times=False)['soc_flux_lw_up'].values     # Socrates LW flux up   [W/m2]
soc_flux_lw_down2      = xr.open_dataset(dirs["output"]+'run'+f'{int(forcing_index):04}'+'/'+netcdf_file, decode_times=False)['soc_flux_lw_down'].values # Socrates LW flux down [W/m2]
soc_flux_sw_up2      = xr.open_dataset(dirs["output"]+'run'+f'{int(forcing_index):04}'+'/'+netcdf_file, decode_times=False)['soc_flux_sw_up'].values     # Socrates SW flux up   [W/m2]
soc_flux_sw_down2      = xr.open_dataset(dirs["output"]+'run'+f'{int(forcing_index):04}'+'/'+netcdf_file, decode_times=False)['soc_flux_sw_down'].values # Socrates SW flux down [W/m2]

ASR2   = soc_flux_sw_down2[0,0,:,:] - soc_flux_sw_up2[0,0,:,:]
OTR2   = soc_flux_lw_up2[0,0,:,:] - soc_flux_lw_down2[0,0,:,:] 
ASP2 = np.zeros_like(ASR2)
sum_ASP2 = np.zeros(ASR2.shape[0])
OTP2 = np.zeros_like(OTR2)
sum_OTP2 = np.zeros(OTR2.shape[0])

for j in range(len(lats_edges)-1):
    for k in range(len(lons_edges)-1):
        ASP2[j,k] = ASR2[j,k]*area[j,k]
        OTP2[j,k] = OTR2[j,k]*area[j,k]
sum_ASP2 = np.sum(ASP2)
sum_OTP2 = np.sum(OTP2)

ASR2 = sum_ASP2/area_planet
OTR2 = sum_OTP2/area_planet
Imbalance2 = ASR2 - OTR2 # 6.07087503176831

Radiative_forcing = Imbalance2-Imbalance1 # 6.4878503011161115 W/m2

surface_dT = np.mean(T_surf2) - np.mean(T_surf1) # 5.9724426 K 

Climate_Sensitivity = 3.7*surface_dT/Radiative_forcing # 3.406064673829637 K : the temperature change for a (canonical) 3.7 W/m² forcing