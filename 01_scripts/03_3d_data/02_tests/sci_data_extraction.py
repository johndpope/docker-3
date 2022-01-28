import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Define filename of .sci File that has been changed to a .txt file
filename = "E:\\ukr_data\\Einzelzaehne\\1917_1\\1917_1\\1917_1_1v.txt"

# Initialize empty list
coordinates_scan = []

# Open File
with open(filename, "r", encoding="UTF-16") as f:
    for line in f:
        # only read lines that start with "Vec3.."
        if line.find("<Vec3 x=") != -1:
            # Extract coordinates and save as float
            x = float(line.split()[1].replace('x="', "").replace('"', ""))
            y = float(line.split()[2].replace('y="', "").replace('"', ""))
            z = float(line.split()[3].replace('z="', "").replace('"', ""))
            # Append coordinates to list
            coordinates_scan.append([x, y, z])

# Convert list to array
coordinates_scan_arr = np.array(coordinates_scan)

# Initialize figure
fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection="3d")

# Plot the 3D-coordinates
ax.scatter(
    coordinates_scan_arr[:, 0], coordinates_scan_arr[:, 1], coordinates_scan_arr[:, 2]
)

# Set title and show
ax.set_title("3D Plot of .sci Scan-Data: 1917_1_1v")
plt.show()
