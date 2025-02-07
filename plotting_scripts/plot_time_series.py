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

def find_last_non_nan_index(arr):
    """ Find the last non-NaN index along the second axis for each [i, j, k] combination """
    valid_mask = ~np.isnan(arr)
    indices = np.where(valid_mask.any(axis=1), valid_mask.shape[1] - np.argmax(valid_mask[:, ::-1, :, :], axis=1) - 1, -1)
    return indices

simulation   = 'planetb_presentdayEarth_rot0' #'Earth' #'planetb_EoceneEarth_rot0' #'planetb_ArcheanEarth_rot0'  

isca_plots = '/proj/bolinc/users/x_ryabo/Isca-Ryan_plots'

dirs = {
    "output": os.getenv('GFDL_DATA')+"/"+simulation+"/",
    "plot_output": isca_plots+"/"+simulation+"/"
    }

if not os.path.exists(dirs["plot_output"]):
    os.makedirs(dirs["plot_output"])

step_sizes = ['monthly', 'daily', 'hourly', 'secondly'] 
step_size_unit = ['month', 'day', 'hour', 'second'] 
transitions = []        
step_sizes_detected = []

start_run = 0
end_run   = 1078
filenames = ['atmos_monthly.nc', 'atmos_daily.nc', 'atmos_hourly.nc', 'atmos_seconds.nc']
soc_flux_lw_up   = []
soc_flux_lw_down = []
soc_flux_sw_up   = []
soc_flux_sw_down = []
soc_flux_direct  = []

# Cell area [m2]
radius = 1.02*6376.0e3 # default Earth
lat  = xr.open_dataset(dirs['output']+'run'+f'{int(start_run):04}'+'/'+filenames[0], decode_times=False)['lat'].values.astype(np.float64)  # latitudes  
lon_b = xr.open_dataset(dirs['output']+'run'+f'{int(start_run):04}'+'/'+filenames[0], decode_times=False)['lonb'].values.astype(np.float64) # longitude edges    
lat_b = xr.open_dataset(dirs['output']+'run'+f'{int(start_run):04}'+'/'+filenames[0], decode_times=False)['latb'].values.astype(np.float64) # latitude edges 
lonb, latb = np.meshgrid(lon_b, lat_b)

# Convert to radians
lat  = np.radians(lat) 
lonb = np.radians(lonb) 
latb = np.radians(latb) 

dlat = np.diff(latb, axis=0)  # Differences along the latitude axis
dlon = np.diff(lonb, axis=1)  # Differences along the longitude axis
cos_lat = np.cos(lat).reshape(-1, 1)

area = radius**2* ( cos_lat * dlat[:,:-1] * dlon[:-1,:] )

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
            soc_flux_lw_up_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_lw_up'].values 
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
            soc_flux_lw_down_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_lw_down'].values 
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
            soc_flux_sw_up_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_sw_up'].values 
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
            soc_flux_sw_down_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_sw_down'].values 
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
            soc_flux_direct_i = xr.open_dataset(filepath, decode_times=False)['soc_flux_direct'].values 
            print(f"Appending dataset for run {i} with shape {soc_flux_direct_i.shape} and step size {step_size}")
            soc_flux_direct.append(soc_flux_direct_i)
    counter += 1
    
for idx, arr in enumerate(soc_flux_direct):
    print(f"Array {idx} shape: {arr.shape}")

soc_flux_direct = np.concatenate(soc_flux_direct,axis=0)
print(f"Final concatenated shape: {soc_flux_direct.shape}")

# ----------------------------------------------------------------------
net_F = soc_flux_sw_down + soc_flux_lw_down - soc_flux_lw_up - soc_flux_sw_up # Positive downward

plev = 0 # TOA
fig, ax = plt.subplots()

area_planet = np.sum(area) #4.0*np.pi*(1.116*6.371e6)**2
sum_NP = np.zeros(len(net_F[:,plev,0,0])) 
net_power = np.zeros(np.shape(net_F[:,plev,:,:]))

for i in range(0,end_run-start_run+1):
    for j in range(len(net_F[0,0,:,0])):
        for k in range(len(net_F[0,0,0,:])):
            net_power[i,j,k] = net_F[i,plev,j,k]*area[j,k]
    sum_NP[i] = np.sum(net_power[i,:,:])

ax.plot(sum_NP/area_planet)
for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
for idx, step_size in enumerate(step_sizes_detected):
    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.33, step_size, fontsize=8, family='serif', rotation=90)

ax.set_xlabel("Time ["+step_size+"s]")
ax.set_ylabel(r'TOA imbalance [W m$^{-2}$]')
ax.set_xlim(75, None)

#plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'toa_equilibrium_evolution.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'toa_equilibrium_evolution.png',bbox_inches='tight')
plt.close()


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

fig, ax = plt.subplots()

toa_up_total = soc_flux_lw_up[:,0,:,:] + soc_flux_sw_up[:,0,:,:]

sum = np.zeros(len(toa_up_total[:,0,0])) 
toa_up_weighted = np.zeros(np.shape(toa_up_total))

for i in range(0,end_run-start_run+1):
    for j in range(len(toa_up_total[0,:,0])):
        for k in range(len(toa_up_total[0,0,:])):
            toa_up_weighted[i,j,k] = toa_up_total[i,j,k]*area[j,k]
    sum[i] = np.sum(toa_up_weighted[i,:,:])

ax.plot(sum/area_planet, label='Global normalized')
ax.plot(toa_up_total[:,Substellar_latitude,Substellar_longitude], label='Substellar')
ax.plot(toa_up_total[:,Antistellar_latitude,Antistellar_longitude], label='Antistellar')

for transition in transitions:
    ax.axvline(x=transition, color='red', linestyle='--', linewidth=0.5)
for idx, step_size in enumerate(step_sizes_detected):
    ax.text(transitions[idx]+1, ax.get_ylim()[1] * 0.33, step_size, fontsize=8, family='serif', rotation=90)

ax.legend(frameon=True,loc='best')

ax.set_xlabel("Time ["+step_size+"s]")
ax.set_ylabel(r'OLR [W m$^{-2}$]')
ax.set_xlim(75, None)

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"]+'toa_olr_timeseries.pdf',bbox_inches='tight')
    fig.savefig(dirs["plot_output"]+'toa_olr_timeseries.png',bbox_inches='tight')
#plt.close()

# Exploration across lat and lon
num_latitudes  = len(latb[:,0])
num_longitudes = len(latb[0,:])
time = np.arange(soc_flux_lw_up.shape[0])
colors_lat = cm.Blues(np.linspace(0.2, 1, num_latitudes))
colors_lon = cm.Blues(np.linspace(0.2, 1, num_longitudes))

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

ax.plot(time, sum/area_planet, lat[0], color='k', label='Global normalized', zorder=10)
for i in range(num_latitudes-1):
    for j in range(num_longitudes-1):
        ax.plot(time, toa_up_total[:, i, j], lat[i], color=colors_lat[i])

# Set axis labels
ax.set_xlabel(f"Time [{step_size}s]")
ax.set_ylabel(r'OLR [W m$^{-2}$]')
#ax.set_zlabel('Pressure [Pa]')
ax.set_zlabel(r'Latitude [${\degree}$]')
ax.invert_zaxis()

plt.show()
if save_figs:
    fig.savefig(dirs["plot_output"] + 'toa_olr_3D_timeseries.pdf', bbox_inches='tight')
    fig.savefig(dirs["plot_output"] + 'toa_olr_3D_timeseries.png', bbox_inches='tight')

# Neil's script
# ds = xr.open_dataset(dirs['output']+'run0001/'+filename, decode_times=False)

# lat = ds.lat.values 
# latb = ds.latb.values 
# latr = np.deg2rad(lat)
# dlatr = np.diff(np.deg2rad(latb))
# weights = np.cos(latr) * 2. * np.sin(dlatr/2.)

# soc_toa_sw  = []
# counter = 0
# for i in nb_months_array:
#     print(counter) 
#     run_folder = f'run{i:04d}'
#     soc_toa_sw_i = xr.open_dataset(dirs['output']+run_folder+'/'+filename, decode_times=False)['soc_toa_sw'].values[0,:,:]
#     print(f"Appending dataset for day {i} with shape {soc_toa_sw_i.shape}")
#     soc_toa_sw.append(soc_toa_sw_i)
#     counter += 1

# soc_olr  = []
# counter = 0
# for i in nb_months_array:
#     print(counter) 
#     run_folder = f'run{i:04d}'
#     soc_olr_i = xr.open_dataset(dirs['output']+run_folder+'/'+filename, decode_times=False)['soc_olr'].values[0,:,:]
#     print(f"Appending dataset for day {i} with shape {soc_olr_i.shape}")
#     soc_olr.append(soc_olr_i)
#     counter += 1

# one_month = 719 #359
# one_year  = 708 #348
# ten_years = 600 #240
# fifteen_years = 540 #180
# soc_toa_sw_mean = np.mean(np.array(soc_toa_sw)[ten_years:,:,:],axis=0)
# soc_olr_mean    = np.mean(np.array(soc_olr)[ten_years:,:,:],axis=0)

# sw = np.mean(soc_toa_sw_mean * weights[:,None]) / np.mean(weights)
# lw = np.mean(soc_olr_mean    * weights[:,None]) / np.mean(weights)

# # sw - lw should be close to 0 if averaged over some time.
# print(-lw, sw, (sw-lw))
