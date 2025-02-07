import numpy as np

# Define the dimensions of the Fortran array
nx, ny, nz = 128, 4, 40  # Replace with the actual dimensions of temp

# Load the binary data
temp = np.fromfile("/proj/bolinc/users/x_ryabo/Isca_outputs/planetb_presentdayEarth_rot0/run0341/temp_data.bin", dtype=np.float32)  # or np.float64 if needed
temp = temp.reshape((nx, ny, nz))  # Reshape to 3D

# Analyze or visualize the data
print("Loaded data shape:", temp.shape)
print("Max value in temp:", temp.max())
print("Number of unphysical values:", np.sum(temp > 1000))

# Example: Visualize a slice
import matplotlib.pyplot as plt
plt.imshow(temp[:, :, 0], cmap='hot')
plt.colorbar(label='Temperature')
plt.title('Slice of Temp (k=0)')
plt.show()