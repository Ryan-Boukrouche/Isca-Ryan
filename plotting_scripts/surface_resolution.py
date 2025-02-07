# p/p0 = exp[-scale_heights*(surf_res*x + (1-surf_res)*x^exponent)]
import os
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('Agg')

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
    resolution_file = Dataset(dirs['output']+'run0001/'+filename, 'r', format='NETCDF3_CLASSIC')

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

isca_plots = '/proj/bolinc/users/x_ryabo/Isca_plots'
simulation = 'planetb_presentdayEarth_rot0'
filename   = 'atmos_monthly.nc'

dirs = {
    "output": os.getenv('GFDL_DATA')+"/"+simulation+"/",
    "plot_output": isca_plots+"/"
    }

save_figs = True

radius_Earth   = 6.371e6    # Earth radius [m]
radius_planets = {"Earth": radius_Earth, "TRAPPIST-1b": 1.116*radius_Earth, "TRAPPIST-1c": 1.097*radius_Earth, "TRAPPIST-1d": 0.788*radius_Earth, "TRAPPIST-1e": 0.920*radius_Earth, "Proxima-b": np.array([0.94,1.4])*radius_Earth,
                     "Teegarden-b": 1.02*radius_Earth, "Teegarden-c": 1.04*radius_Earth}
radius = radius_planets["Teegarden-b"]

# Computing the surface area of each cell of the Gaussian grid
area_array=np.array(cell_area(radius=radius))

# Plotting the Gaussian grid
lat  = xr.open_dataset(dirs['output']+'run0001/'+filename, decode_times=False)['lat'].values.astype(np.float64)  # latitudes  
lon_b = xr.open_dataset(dirs['output']+'run0001/'+filename, decode_times=False)['lonb'].values.astype(np.float64) # longitude edges    
lat_b = xr.open_dataset(dirs['output']+'run0001/'+filename, decode_times=False)['latb'].values.astype(np.float64) # latitude edges 
lonb, latb = np.meshgrid(lon_b, lat_b)

# Convert to radians
lat_rad  = np.radians(lat) 
lonb_rad = np.radians(lonb) 
latb_rad = np.radians(latb) 

lon_max = 128
lat_max = 64

# Resolutions in degree for longitudes and latitudes
res_lon_degree = np.diff(lon_b)
res_lat_degree = np.diff(lat_b)
# Resolutions in radian for longitudes and latitudes
res_lon_rad = np.radians(res_lon_degree)
res_lat_rad = np.radians(res_lat_degree)
# Resolutions in km for longitudes and latitudes
res_lon_km = np.array([np.array([2*np.pi*radius*np.cos(lat_rad[i])*(res_lon_degree[j]/360.) for j in range(lon_max)]) for i in range(lat_max)])*1e-3
res_lat_km = 2*np.pi*radius*(res_lat_degree/360.)*1e-3

# Angular resolution along the zonal (longitudes) direction: res_lon_degree[0]
# Angular resolution along the meridional (latitudes) direction: np.min(res_lat_degree) to np.max(res_lat_degree) from equator to pole
# Resolution in km along the zonal (longitudes) direction: res_lon_km[32,0] to res_lon_km[0,0] from equator to pole
# Resolution in km along the meridional (latitudes) direction: np.min(res_lat_km) to np.max(res_lat_km) from equator to pole

# Convert spherical coordinates to Cartesian for plotting
x = radius * np.cos(latb_rad) * np.cos(lonb_rad)
y = radius * np.cos(latb_rad) * np.sin(lonb_rad)
z = radius * np.sin(latb_rad)

lonb_shifted = np.sort(((lonb + 180) % 360) - 180)
lonb_shifted_rad = np.radians(lonb_shifted) 

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="mollweide")

# Plot the grid lines for the Gaussian grid
for i in range(latb_rad.shape[0]):  # Latitude lines
    ax.plot(lonb_shifted_rad[i, :], latb_rad[i, :], color="blue", lw=1, alpha=0.7)

for j in range(lonb_rad.shape[1]):  # Longitude lines
    ax.plot(lonb_shifted_rad[:, j], latb_rad[:, j], color="red", lw=1, alpha=0.7)

# Set titles and labels
ax.set_title("Gaussian grid T42 (Mollweide Projection)", fontsize=14)
ax.set_xticks(np.radians([-180, -120, -60, 0, 60, 120, 180]))  # Correct tick locations
ax.set_xticklabels(["-180°", "-120°", "-60°", "0°", "60°", "120°", "180°"], fontsize=10)

# Add annotation
text = (
    f"Planet: Teegarden b\n"
    f"Radius: {radius / 1000:.2f} km\n"
    f"Angular Resolutions: {res_lon_degree[0]:.1f}° (lon.), {np.min(res_lat_degree):.1f}° - {np.max(res_lat_degree):.1f}° (lat.)\n"
    f"Distances: {res_lon_km[32,0]:.0f}-{res_lon_km[0,0]:.0f} km (lon.), {np.min(res_lat_km):.0f} - {np.max(res_lat_km):.0f} km (lat.)"
)
ax.annotate(
    text,
    xy=(0.5, -0.2),
    xycoords="axes fraction",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7),
)
plt.grid(True, linestyle="--", alpha=0.5)

# Save the figure if desired
if save_figs:
    fig.savefig(dirs["plot_output"] + "mollweide_gaussian_grid.pdf", bbox_inches="tight")
    fig.savefig(dirs["plot_output"] + "mollweide_gaussian_grid.png", bbox_inches="tight")

# ---------------------------------------------------------------------------------------
# Interactable version of the plot
fig_plotly = go.Figure()

# Add the Gaussian grid lines (latitude and longitude)
for i in range(latb_rad.shape[0]):  # Latitude lines
    fig_plotly.add_trace(go.Scatter3d(
        x=x[i, :], y=y[i, :], z=z[i, :], 
        mode='lines', line=dict(color='blue', width=2)
    ))

for j in range(lonb_rad.shape[1]):  # Longitude lines
    fig_plotly.add_trace(go.Scatter3d(
        x=x[:, j], y=y[:, j], z=z[:, j], 
        mode='lines', line=dict(color='red', width=2)
    ))

# Add annotation for Teegarden b and its radius
fig_plotly.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[radius * 1.2], 
    mode='text', text=[f"Planet \nradius: {radius/1000:.1f} km"], 
    textposition="top center", 
    showlegend=False, 
    marker=dict(size=0)
))

# Add annotations for the angular resolutions and distances
fig_plotly.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[radius * 1.1], 
    mode='text', text=[f"Longitude resolution: {res_lon_degree[0]:.1f}° ({res_lon_km[32,0]:.0f}-{res_lon_km[0,0]:.0f} km equator to pole)"], 
    textposition="top center", 
    showlegend=False, 
    marker=dict(size=0)
))

fig_plotly.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[radius], 
    mode='text', text=[f"Latitude resolution: {np.min(res_lat_degree):.1f}-{np.max(res_lat_degree):.1f}°\n ({np.min(res_lat_km):.0f}-{np.max(res_lat_km):.0f} km equator to pole)"], 
    textposition="top center", 
    showlegend=False, 
    marker=dict(size=0)
))

# Update the layout to make it more visually appealing
fig_plotly.update_layout(
    scene=dict(
        xaxis_title='X (km)', 
        yaxis_title='Y (km)', 
        zaxis_title='Z (km)', 
        aspectmode='data'
    ),
    title="Gaussian grid T42",
    title_x=0.5,  # Center the title
    title_y=0.95,
    title_font=dict(size=20),
    font=dict(size=12),
    margin=dict(l=0, r=0, b=0, t=60)  # Adjust margins for better layout
)

# Show or save the plot
if save_figs:
    fig_plotly.write_html(dirs["plot_output"]+"teegarden_b_gaussian_grid.html")
    print("Interactive plot saved as 'teegarden_b_gaussian_grid.html'. Open this file in a browser to view.")

# ---------------------------------------------------------------------------------------
# Adding the vertical grid cells to the 3D plot

# Parameters
surf_res = 0.06
scale_heights = 7.0
exponent = 3.0
xp = np.arange(0, 1.025, 0.025)
p0 = 1e5  # Reference pressure in Pa
height_scale = 1e6  # Scaling factor for plotting visible atmospheric layers

# Calculate the pressure array
sigma = np.exp(-scale_heights * (surf_res * xp + (1. - surf_res) * xp**exponent))
p = sigma * p0  # pressure array

fig_plotly = go.Figure()

# Add the Gaussian grid lines (latitude and longitude)
for i in range(latb_rad.shape[0]):  # Latitude lines
    fig_plotly.add_trace(go.Scatter3d(
        x=x[i, :], y=y[i, :], z=z[i, :], 
        mode='lines', line=dict(color='rgba(0, 204, 0, 1.0)', width=2)
    ))

for j in range(lonb_rad.shape[1]):  # Longitude lines
    fig_plotly.add_trace(go.Scatter3d(
        x=x[:, j], y=y[:, j], z=z[:, j], 
        mode='lines', line=dict(color='rgba(0, 204, 0, 1.0)', width=2)
    ))

# Add vertical and horizontal grid lines for atmospheric layers
vertical_x = []
vertical_y = []
vertical_z = []

horizontal_x = []
horizontal_y = []
horizontal_z = []

for k, sigma_value in enumerate(sigma):
    scaled_radius = radius + (height_scale * sigma_value)  # Ensure layers are outside the sphere
    layer_x = scaled_radius * np.cos(latb_rad) * np.cos(lonb_rad)
    layer_y = scaled_radius * np.cos(latb_rad) * np.sin(lonb_rad)
    layer_z = scaled_radius * np.sin(latb_rad)

    # Add horizontal grid lines for this layer
    for i in range(layer_x.shape[0]):
        horizontal_x.extend(layer_x[i, :].tolist() + [None])
        horizontal_y.extend(layer_y[i, :].tolist() + [None])
        horizontal_z.extend(layer_z[i, :].tolist() + [None])

    for j in range(layer_x.shape[1]):
        horizontal_x.extend(layer_x[:, j].tolist() + [None])
        horizontal_y.extend(layer_y[:, j].tolist() + [None])
        horizontal_z.extend(layer_z[:, j].tolist() + [None])

    # Add vertical lines connecting this layer to the previous one
    if k == 0:
        # First layer connects to the surface
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                vertical_x.extend([x[i, j], layer_x[i, j], None])
                vertical_y.extend([y[i, j], layer_y[i, j], None])
                vertical_z.extend([z[i, j], layer_z[i, j], None])


    else:
        # Subsequent layers connect to the previous layer
        prev_scaled_radius = radius + (height_scale * sigma[k-1])
        prev_layer_x = prev_scaled_radius * np.cos(latb_rad) * np.cos(lonb_rad)
        prev_layer_y = prev_scaled_radius * np.cos(latb_rad) * np.sin(lonb_rad)
        prev_layer_z = prev_scaled_radius * np.sin(latb_rad)
        for i in range(layer_x.shape[0]):
            for j in range(layer_x.shape[1]):
                vertical_x.extend([prev_layer_x[i, j], layer_x[i, j], None])
                vertical_y.extend([prev_layer_y[i, j], layer_y[i, j], None])
                vertical_z.extend([prev_layer_z[i, j], layer_z[i, j], None])

fig_plotly.add_trace(go.Scatter3d(
    x=vertical_x, y=vertical_y, z=vertical_z, 
    mode='lines', line=dict(color='rgba(14, 22, 50, 0.2)', width=0.5), name='Vertical Grid'
))

fig_plotly.add_trace(go.Scatter3d(
    x=horizontal_x, y=horizontal_y, z=horizontal_z, 
    mode='lines', line=dict(color='rgba(14, 22, 50, 0.2)', width=0.5), name='Horizontal Grid'
))

# Add annotation for Teegarden b and its radius
fig_plotly.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[radius * 1.3], 
    mode='text', text=[f"Planet \nradius: {radius/1000:.1f} km"], 
    textposition="top center", 
    showlegend=False, 
    marker=dict(size=0)
))

# Add annotations for the angular resolutions and distances
fig_plotly.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[radius * 1.2], 
    mode='text', text=[f"Longitude resolution: {res_lon_degree[0]:.1f}° ({res_lon_km[32,0]:.0f}-{res_lon_km[0,0]:.0f} km equator to pole)"], 
    textposition="top center", 
    showlegend=False, 
    marker=dict(size=0)
))

fig_plotly.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[radius * 1.11], 
    mode='text', text=[f"Latitude resolution: {np.min(res_lat_degree):.1f}-{np.max(res_lat_degree):.1f}°\n ({np.min(res_lat_km):.0f}-{np.max(res_lat_km):.0f} km equator to pole)"], 
    textposition="top center", 
    showlegend=False, 
    marker=dict(size=0)
))

# Update the layout to make it more visually appealing
fig_plotly.update_layout(
    scene=dict(
        xaxis_title='X (km)', 
        yaxis_title='Y (km)', 
        zaxis_title='Z (km)', 
        aspectmode='data'
    ),
    title="Gaussian grid T42",
    title_x=0.5,  # Center the title
    title_y=0.95,
    title_font=dict(size=20),
    font=dict(size=12),
    margin=dict(l=0, r=0, b=0, t=60)  # Adjust margins for better layout
)

# Show or save the plot
if save_figs:
    fig_plotly.write_html(dirs["plot_output"]+"teegarden_b_full_gaussian_grid.html")
    print("Interactive plot saved as 'teegarden_b_full_gaussian_grid.html'. Open this file in a browser to view.")