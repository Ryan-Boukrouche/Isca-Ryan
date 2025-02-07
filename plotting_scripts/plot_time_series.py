import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatch
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns

save_figs = True

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
    resolution_file = Dataset(dirs['output']+'run0001/'+filenames[0], 'r', format='NETCDF3_CLASSIC')

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

def find_last_non_nan_index(arr):
    """ Find the last non-NaN index along the second axis for each [i, j, k] combination """
    valid_mask = ~np.isnan(arr)
    indices = np.where(valid_mask.any(axis=1), valid_mask.shape[1] - np.argmax(valid_mask[:, ::-1, :, :], axis=1) - 1, -1)
    return indices

simulation   = 'planetb_presentdayEarth_rot0' #'planetb_ArcheanEarth_rot0' #'planetb_EoceneEarth_rot0' #'Earth'

isca_plots = '/proj/bolinc/users/x_ryabo/Isca-Ryan_plots'

dirs = {
    "output": os.getenv('GFDL_DATA')+"/"+simulation+"/",
    "plot_output": isca_plots+"/"+simulation+"/"
    }

if not os.path.exists(dirs["plot_output"]):
    os.makedirs(dirs["plot_output"])

step_sizes = ['monthly', 'daily', 'hourly', 'minutely', 'secondly'] 
step_size_unit = ['month', 'day', 'hour', 'minute', 'second'] 
transitions = []        
step_sizes_detected = []

start_run = 0#301
end_run   = 386
filenames = ['atmos_monthly.nc', 'atmos_daily.nc', 'atmos_hourly.nc', 'atmos_minute.nc', 'atmos_seconds.nc']
soc_flux_lw_up     = []
soc_flux_lw_down   = []
soc_flux_sw_up     = []
soc_flux_sw_down   = []
soc_flux_direct    = []
temp               = []
tsurf              = []
soc_tdt_rad        = []
dt_tg_convection   = []
dt_tg_condensation = []

# Cell area [m2]
radius = 1.02*6376.0e3 # default Earth
area = np.array(cell_area(radius=radius)) # computing the surface area of each cell of the Gaussian grid
area_planet = 4.0*np.pi*(radius)**2
pressures = xr.open_dataset(dirs['output']+'run'+f'{int(start_run):04}'+'/'+filenames[0], decode_times=False)['pfull'].values.astype(np.float64)*1e2  # midpoint pressures  

lat  = xr.open_dataset(dirs['output']+'run'+f'{int(start_run):04}'+'/'+filenames[0], decode_times=False)['lat'].values.astype(np.float64)  # latitudes  
lon_b = xr.open_dataset(dirs['output']+'run'+f'{int(start_run):04}'+'/'+filenames[0], decode_times=False)['lonb'].values.astype(np.float64) # longitude edges    
lat_b = xr.open_dataset(dirs['output']+'run'+f'{int(start_run):04}'+'/'+filenames[0], decode_times=False)['latb'].values.astype(np.float64) # latitude edges 
lonb, latb = np.meshgrid(lon_b, lat_b)

nb_steps_array = np.arange(start_run,end_run+1,1).astype(int)

# Save stepsize transitions 
current_step_size = 'month'  
for i in nb_steps_array:
    run_folder = f'run{i:04d}'

    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            if step_size != current_step_size:
                transitions.append(i)  # Record the run index where the step size change occurs
                step_sizes_detected.append(step_size)
                current_step_size = step_size
            break  

# Socrates Upward LW flux
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                soc_flux_lw_up_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_lw_up'].values 
            except:
                soc_flux_lw_up_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {soc_flux_lw_up_i.shape} and step size {step_size}")
            soc_flux_lw_up.append(soc_flux_lw_up_i)
    counter += 1
    
for idx, arr in enumerate(soc_flux_lw_up):
    print(f"Array {idx} shape: {arr.shape}")

soc_flux_lw_up = np.concatenate(soc_flux_lw_up,axis=0)
print(f"Final concatenated shape: {soc_flux_lw_up.shape}")

#last_non_nan_indices_lwup = find_last_non_nan_index(soc_flux_lw_up)
#print(np.shape(last_non_nan_indices))

# Socrates Downward LW flux
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                soc_flux_lw_down_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_lw_down'].values 
            except:
                soc_flux_lw_down_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {soc_flux_lw_down_i.shape} and step size {step_size}")
            soc_flux_lw_down.append(soc_flux_lw_down_i)
    counter += 1
    
for idx, arr in enumerate(soc_flux_lw_down):
    print(f"Array {idx} shape: {arr.shape}")

soc_flux_lw_down = np.concatenate(soc_flux_lw_down,axis=0)
print(f"Final concatenated shape: {soc_flux_lw_down.shape}")

# Socrates Upward SW flux
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                soc_flux_sw_up_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_sw_up'].values 
            except:
                soc_flux_sw_up_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {soc_flux_sw_up_i.shape} and step size {step_size}")
            soc_flux_sw_up.append(soc_flux_sw_up_i)
    counter += 1
    
for idx, arr in enumerate(soc_flux_sw_up):
    print(f"Array {idx} shape: {arr.shape}")

soc_flux_sw_up = np.concatenate(soc_flux_sw_up,axis=0)
print(f"Final concatenated shape: {soc_flux_sw_up.shape}")

# Socrates Downward SW flux
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                soc_flux_sw_down_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_sw_down'].values 
            except:
                soc_flux_sw_down_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {soc_flux_sw_down_i.shape} and step size {step_size}")
            soc_flux_sw_down.append(soc_flux_sw_down_i)
    counter += 1
    
for idx, arr in enumerate(soc_flux_sw_down):
    print(f"Array {idx} shape: {arr.shape}")

soc_flux_sw_down = np.concatenate(soc_flux_sw_down,axis=0)
print(f"Final concatenated shape: {soc_flux_sw_down.shape}")

# Socrates Direct flux
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                soc_flux_direct_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_direct'].values 
            except:
                soc_flux_direct_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {soc_flux_direct_i.shape} and step size {step_size}")
            soc_flux_direct.append(soc_flux_direct_i)
    counter += 1
    
for idx, arr in enumerate(soc_flux_direct):
    print(f"Array {idx} shape: {arr.shape}")

soc_flux_direct = np.concatenate(soc_flux_direct,axis=0)
print(f"Final concatenated shape: {soc_flux_direct.shape}")

# Temperature
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                temp_i = xr.open_dataset(filepath, decode_times=False)['temp'].values 
            except:
                temp_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {temp_i.shape} and step size {step_size}")
            temp.append(temp_i)
    counter += 1
    
for idx, arr in enumerate(temp):
    print(f"Array {idx} shape: {arr.shape}")

temp = np.concatenate(temp,axis=0)
print(f"Final concatenated shape: {temp.shape}")

# Surface Temperature
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                tsurf_i = xr.open_dataset(filepath, decode_times=False)['t_surf'].values 
            except:
                tsurf_i = np.full((1, len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {tsurf_i.shape} and step size {step_size}")
            tsurf.append(tsurf_i)
    counter += 1
    
for idx, arr in enumerate(tsurf):
    print(f"Array {idx} shape: {arr.shape}")

tsurf = np.concatenate(tsurf,axis=0)
print(f"Final concatenated shape: {tsurf.shape}")

# Temperature tendency due to radiation
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                soc_tdt_rad_i = xr.open_dataset(filepath, decode_times=False)['soc_tdt_rad'].values 
            except:
                soc_tdt_rad_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {soc_tdt_rad_i.shape} and step size {step_size}")
            soc_tdt_rad.append(soc_tdt_rad_i)
    counter += 1
    
for idx, arr in enumerate(soc_tdt_rad):
    print(f"Array {idx} shape: {arr.shape}")

soc_tdt_rad = np.concatenate(soc_tdt_rad,axis=0)
print(f"Final concatenated shape: {soc_tdt_rad.shape}")

# Temperature tendency due to convection
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                dt_tg_convection_i = xr.open_dataset(filepath, decode_times=False)['dt_tg_convection'].values 
            except:
                dt_tg_convection_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {dt_tg_convection_i.shape} and step size {step_size}")
            dt_tg_convection.append(dt_tg_convection_i)
    counter += 1
    
for idx, arr in enumerate(dt_tg_convection):
    print(f"Array {idx} shape: {arr.shape}")

dt_tg_convection = np.concatenate(dt_tg_convection,axis=0)
print(f"Final concatenated shape: {dt_tg_convection.shape}")

# Temperature tendency due to condensation
counter = 0
for i in nb_steps_array:
    print(counter) 
    run_folder = f'run{i:04d}'
    for filename, step_size in zip(filenames, step_sizes):
        filepath = dirs['output'] + run_folder + '/' + filename
        if os.path.exists(filepath):
            try:
                dt_tg_condensation_i = xr.open_dataset(filepath, decode_times=False)['dt_tg_condensation'].values 
            except:
                dt_tg_condensation_i = np.full((1, len(pressures), len(lat_b)-1, len(lon_b)-1), np.nan)
            print(f"Appending dataset for run {i} with shape {dt_tg_condensation_i.shape} and step size {step_size}")
            dt_tg_condensation.append(dt_tg_condensation_i)
    counter += 1
    
for idx, arr in enumerate(dt_tg_condensation):
    print(f"Array {idx} shape: {arr.shape}")

dt_tg_condensation = np.concatenate(dt_tg_condensation,axis=0)
print(f"Final concatenated shape: {dt_tg_condensation.shape}")

# ----------------------------------------------------------------------

# Time series

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

num_latitudes  = len(latb[:,0])
num_longitudes = len(latb[0,:])
num_pressures  = len(temp[0,:,0,0])
colors_lat = cm.Blues(np.linspace(0.2, 1, num_latitudes))
colors_lon = cm.Blues(np.linspace(0.2, 1, num_longitudes))

# Plot of the time series of the global UP and DOWN fluxes to probe for runaway
fig, ax = plt.subplots()

ASR   = soc_flux_direct[:,0,:,:] + soc_flux_sw_down[:,0,:,:]   + soc_flux_sw_up[:,0,:,:]
OTR   = soc_flux_direct[:,0,:,:] + soc_flux_sw_down[:,0,:,:]   + soc_flux_sw_up[:,0,:,:]
ASP = np.zeros_like(ASR)
sum_ASP = np.zeros_like(ASR.shape[0])
OTP = np.zeros_like(OTR)
sum_OTP = np.zeros_like(OTR.shape[0])

for i in nb_steps_array:
    for j in range(len(lat_b)-1):
        for k in range(len(lon_b)-1):
            ASP[i,j,k] = ASR[i,j,k]*area[j,k]
            OTP[i,j,k] = OTR[i,j,k]*area[j,k]
    sum_ASP[i] = np.sum(ASP[i,:,:])
    sum_OTP[i] = np.sum(OTP[i,:,:])

ASR = sum_ASP/area_planet
OTR = sum_OTP/area_planet

ax.plot(ASR, label='ASR')
ax.plot(OTR, label='OTR')

for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
for idx, step_size in enumerate(step_sizes_detected):
    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.33, step_size, fontsize=8, family='serif', rotation=90)

ax.legend(frameon=True,loc='best')

ax.set_xlabel("Time")
ax.set_ylabel(r'Global flux [W m$^{-2}$]')
ax.set_xlim(-10, None)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'OTR_ASR_timeseries.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'OTR_ASR_timeseries.png',bbox_inches='tight')
#plt.close()

# Plot of the time series of the TOA fluxes, up and down, on a sweep of latitudes
fig, ax = plt.subplots()

toa_up_total   = soc_flux_lw_up[:,0,:,:]   + soc_flux_sw_up[:,0,:,:]

for i in range(num_latitudes-1):
    ax.plot(toa_up_total[:,i,Substellar_longitude], color=colors_lat[i], label='Longitude 0' if i == 0 else None)

for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
for idx, step_size in enumerate(step_sizes_detected):
    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.33, step_size, fontsize=8, family='serif', rotation=90)

ax.legend(frameon=True,loc='best')

ax.set_xlabel("Time")
ax.set_ylabel(r'OLR [W m$^{-2}$]')
ax.set_xlim(-10, None)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'flux_toa_lon0_timeseries.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'flux_toa_lon0_timeseries.png',bbox_inches='tight')
#plt.close()

# Plot of the time series of the global minimum and maximum temperature
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(np.min(temp, axis=(1, 2, 3)), label='Global min temperature')
ax.plot(np.max(temp, axis=(1, 2, 3)), label='Global max temperature')
for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
for idx, step_size in enumerate(step_sizes_detected):
    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.33, step_size, fontsize=8, family='serif', rotation=90)

ax.axhline(y=623.15, color='k', linestyle='-', linewidth=0.5)
ax.legend(frameon=True,loc='best')

ax.set_xlabel("Time")
ax.set_ylabel('Temperature [K]')
ax.set_xlim(-10, None)
#ax.set_xlim(250, None)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'global_minmax_temp_timeseries.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'global_minmax_temp_timeseries.png',bbox_inches='tight')
#plt.close()

# Plot of the time series of the global minimum, mean, and maximum surface temperature
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(np.min(tsurf, axis=(1, 2)), label='Global min surface temperature')
ax.plot(np.mean(tsurf, axis=(1, 2)), label='Global mean surface temperature')
ax.plot(np.max(tsurf, axis=(1, 2)), label='Global max surface temperature')
for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
#for idx, step_size in enumerate(step_sizes_detected):
#    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.9, step_size, fontsize=8, family='serif', rotation=90)

#ax.axhline(y=623.15, color='k', linestyle='-', linewidth=0.5)
ax.legend(frameon=True,loc='best')

ax.set_xlabel("Time")
ax.set_ylabel('Temperature [K]')
ax.set_xlim(-10, None)
#ax.set_xlim(250, None)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'global_minmax_ts_timeseries.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'global_minmax_ts_timeseries.png',bbox_inches='tight')
#plt.close()
"""
# Plot of the time series of the global minimum and maximum temperature tendency
fig, ax = plt.subplots(figsize=(10, 5))

y_rad  = np.concatenate((np.full((start_run-1, soc_tdt_rad.shape[1], soc_tdt_rad.shape[2], soc_tdt_rad.shape[3]), np.nan), soc_tdt_rad), axis=0)
y_conv = np.concatenate((np.full((start_run-1, dt_tg_convection.shape[1], dt_tg_convection.shape[2], dt_tg_convection.shape[3]), np.nan), dt_tg_convection), axis=0)
y_cond = np.concatenate((np.full((start_run-1, dt_tg_condensation.shape[1], dt_tg_condensation.shape[2], dt_tg_condensation.shape[3]), np.nan), dt_tg_condensation), axis=0)

ax.plot(np.min(y_rad, axis=(1, 2, 3)), linestyle='--', color='r', linewidth=0.8, zorder=0, label='Global min dt_rad')
ax.plot(np.max(y_rad, axis=(1, 2, 3)), linestyle='-.', color='r', linewidth=0.8, zorder=0,label='Global max dt_rad')

ax.plot(np.min(y_conv, axis=(1, 2, 3)), linestyle='--', color='b', linewidth=0.6, zorder=1, label='Global min dt_conv')
ax.plot(np.max(y_conv, axis=(1, 2, 3)), linestyle='-.', color='b', linewidth=0.6, zorder=1, label='Global max dt_conv')

ax.plot(np.min(y_cond, axis=(1, 2, 3)), linestyle='--', color='g', linewidth=0.4, zorder=2, label='Global min dt_cond')
ax.plot(np.max(y_cond, axis=(1, 2, 3)), linestyle='-.', color='g', linewidth=0.4, zorder=2, label='Global max dt_cond')

for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
for idx, step_size in enumerate(step_sizes_detected):
    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.33, step_size, fontsize=8, family='serif', rotation=90)

ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.legend(frameon=True,loc='best')

ax.set_xlabel("Time")
ax.set_ylabel('Temperature tendency [K/s]')
#ax.set_xlim(-10, None)
ax.set_xlim(250, None)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'global_minmax_soc_tdt_timeseries.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'global_minmax_soc_tdt_timeseries.png',bbox_inches='tight')
#plt.close()

# Where is the maximum located?
max_idx = np.unravel_index(np.argmax(temp[-1,:,:,:]), temp[-1,:,:,:].shape)  
p_idx, lat_idx, lon_idx = max_idx
print(f"Maximum value is at indices: Pressure index = {p_idx}, Latitude index = {lat_idx}, Longitude index = {lon_idx}")
print(f"Maximum value is at indices: Pressure = {pressures[p_idx]}, Latitude = {lat_b[lat_idx]}, Longitude = {lon_b[lon_idx]}")
print(f"Maximum value: {temp[-1, p_idx, lat_idx, lon_idx]}")

# Plot of the time series of the temperature on a sweep of latitudes
fig, ax = plt.subplots()

for i in range(num_latitudes-1):
    ax.plot(temp[:,0,i,Substellar_longitude], color=colors_lat[i], label='TOA temperature, longitude 0' if i == 0 else None)

for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
for idx, step_size in enumerate(step_sizes_detected):
    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.77, step_size, fontsize=8, family='serif', rotation=90)

ax.legend(frameon=True,loc='best')

ax.set_xlabel("Time")
ax.set_ylabel('Temperature [K]')
#ax.set_xlim(75, None)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'temp_toa_lon0_timeseries.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'temp_toa_lon0_timeseries.png',bbox_inches='tight')
#plt.close()

# Plot of the time series of the temperature on a sweep of longitudes
fig, ax = plt.subplots()

for i in range(num_longitudes-1):
    ax.plot(temp[:,0,Substellar_latitude,i], color=colors_lon[i], label='TOA temperature, latitude 0' if i == 0 else None)

for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
for idx, step_size in enumerate(step_sizes_detected):
    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.77, step_size, fontsize=8, family='serif', rotation=90)

ax.legend(frameon=True,loc='best')

ax.set_xlabel("Time")
ax.set_ylabel('Temperature [K]')
#ax.set_xlim(75, None)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'temp_toa_lat0_timeseries.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'temp_toa_lat0_timeseries.png',bbox_inches='tight')
#plt.close()

# 3D time series - p,lat,lon
time = np.arange(temp.shape[0])
plt.ion()

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# for i in range(num_latitudes-1):
#     for p in range(num_pressures):
#         ax.plot(time, temp[:, p, i, Substellar_longitude], pressures[p], color=colors_lat[i])

for i in range(num_latitudes-1):
    for j in range(num_longitudes-1):
        #ax.plot(time, temp[:, -1, i, j], lat[i], color=colors_lat[i])
        ax.plot(time, temp[:, 6, i, j], lat[i], color=colors_lat[i])

# Set axis labels
ax.set_xlabel(f"Time")
ax.set_ylabel('Temperature [K]')
#ax.set_zlabel('Pressure [Pa]')
ax.set_zlabel(r'Latitude [${\degree}$]')
ax.invert_zaxis()

#elev: Sets the elevation angle in the z-plane (default is 30 degrees).
#azim: Sets the azimuthal angle in the x-y plane (default is -60 degrees)
ax.view_init(elev=30, azim=-20)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"] + 'temp_ps_timeseries_3D.pdf', bbox_inches='tight')
    fig.savefig(dirs["plot_output"] + 'temp_ps_timeseries_3D.png', bbox_inches='tight')

max_value = np.max(temp)  # Get the maximum value
max_location = np.unravel_index(np.argmax(temp), temp.shape)  # Get the indices of the maximum value

print(f"Maximum value: {max_value}")
print(f"Location of the maximum value: {max_location}")
"""