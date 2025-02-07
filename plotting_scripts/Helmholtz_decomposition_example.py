import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from windspharm.xarray import VectorWind

simulation  = 'Earth' #'planetb_presentdayEarth_rot0'
run         = 'run0113'
netcdf_file = 'atmos_monthly.nc'

isca_plots = '/proj/bolinc/users/x_ryabo/Isca_plots'

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

zonal_wind      = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['ucomp'].values # Zonal wind component      [m/s]
meridional_wind = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['vcomp'].values # Meridional wind component [m/s]
vertical_wind   = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['omega'].values # Vertical wind component   [Pa/s]

Tfull = xr.open_dataset(dirs["output"]+netcdf_file, decode_times=False)['temp'].values # Temperature at full (midpoint) pressure levels [K]

# Time-step to probe
time=-1

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

excluded_levels = 0
for i in range(len(Tfull[0,:,0,0])):
    if ((any(any(row>1e4) for row in Tfull[time,i,:,:])) or (any(any(row<=0.0) for row in Tfull[time,i,:,:]))):
        excluded_levels+=1

vertical_levels = slice(0,len(pfull)-excluded_levels) # Vertically regridded quantities may have no data near the surface

omega = vertical_wind[time,vertical_levels,:,:]*1e3 # Artificial scaling factor to emphasize the ascent and descent on plots where the y axis is logarithmic pressure.


meridional_wind_slice = meridional_wind[time, :, :]  
vertical_wind_slice = vertical_wind[time, :, :]

# Create a VectorWind instance to handle the wind data
w = VectorWind(meridional_wind_slice, vertical_wind_slice)

# Perform Helmholtz decomposition with spherical harmonic truncation (n=21)
psi, chi = w.helmholtz(truncation=21)

# psi is the rotational (streamfunction)
# chi is the divergent (velocity potential)

# Plotting the streamfunction (rotational component)
plt.figure(figsize=(10, 6))
plt.contourf(lats, pfull, psi, cmap='viridis')
plt.colorbar(label='Streamfunction [psi]')
plt.title('Rotational Component (Streamfunction)')
plt.xlabel('Latitude')
plt.ylabel('Pressure')
plt.show()

# Plotting the velocity potential (divergent component)
plt.figure(figsize=(10, 6))
plt.contourf(lats, pfull, chi, cmap='coolwarm')
plt.colorbar(label='Velocity Potential [chi]')
plt.title('Divergent Component (Velocity Potential)')
plt.xlabel('Latitude')
plt.ylabel('Pressure')
plt.show()


save_figs = True
size = 10 # Default size of labels
resolution = 200
cmap = plt.cm.RdYlBu_r

fig, ax = plt.subplots(dpi=150)
    
plot_filled = ax.contourf(lats,pfull,psi,resolution,extend='both',cmap=cmap, zorder=1)
skip = (slice(None, None, 3), slice(None, None, 3))
ax.quiver(lats[skip[0]],pfull[vertical_levels][skip[0]],meridional_wind[skip],omega[skip],color='black', zorder=2)
    
ax.set_xlabel(r'Latitude [$\degree$]', size=size)
ax.set_ylabel(r'Pressure [mbar]', size=size)

plt.title('Rotational Component (Streamfunction)')

plt.gca().set_yscale('log')
plt.gca().invert_yaxis()
plt.ylim([pfull[vertical_levels.stop-1],pfull[0]])

for c in plot_filled.collections:
    c.set_edgecolor("face")
CB = plt.colorbar(plot_filled, orientation='horizontal', format="%0.1f")
CB.set_label(r'Streamfunction [kg s$^{-1}$]', size=size)
CB.ax.tick_params(labelsize=size)

plt.tick_params(axis='both', which='both', labelsize=size)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
if save_figs:
    plt.savefig(dirs["plot_output"]+'rotational.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'rotational.png',bbox_inches='tight')

plt.close()




fig, ax = plt.subplots(dpi=150)
    
plot_filled = ax.contourf(lats,pfull,psi,resolution,extend='both',cmap=cmap, zorder=1)
skip = (slice(None, None, 3), slice(None, None, 3))
ax.quiver(lats[skip[0]],pfull[vertical_levels][skip[0]],meridional_wind[skip],omega[skip],color='black', zorder=2)
    
ax.set_xlabel(r'Latitude [$\degree$]', size=size)
ax.set_ylabel(r'Pressure [mbar]', size=size)

plt.title('Divergent Component (Velocity Potential)')

plt.gca().set_yscale('log')
plt.gca().invert_yaxis()
plt.ylim([pfull[vertical_levels.stop-1],pfull[0]])

for c in plot_filled.collections:
    c.set_edgecolor("face")
CB = plt.colorbar(plot_filled, orientation='horizontal', format="%0.1f")
CB.set_label(r'Streamfunction [kg s$^{-1}$]', size=size)
CB.ax.tick_params(labelsize=size)

plt.tick_params(axis='both', which='both', labelsize=size)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
if save_figs:
    plt.savefig(dirs["plot_output"]+'divergent.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'divergent.png',bbox_inches='tight')

plt.close()