import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import time

# Initial time
start = time.time()

# File and pathnames
filename = "1917_2_2v.stl"
save_filename = filename.replace(".stl", "_bearbeitet.stl")
load_pathname = "E:\\ukr_data\\Einzelzaehne\\1917_2\\"
save_pathname = "E:\\ukr_data\\bearbeitet\\"

# Load the STL file
tooth_mesh = mesh.Mesh.from_file(load_pathname + filename)

# Threshold for cut in z direction
cut_threshold = -2

# All Meshdata:
# Contains: facet, vectors
meshdata = tooth_mesh.data

# Get all vectors
mesh_vectors = tooth_mesh.vectors

# Print Shape
print(f"Shape of vector-array: {np.shape(mesh_vectors)}")

# Get the smallest z-value
min_z = mesh_vectors[:, :, 2].min()
print(f"Smallest z-value: {min_z}")

# Get all points smaller than cut threshold
newvectors_bool = mesh_vectors[:, :, 2] < cut_threshold

# Init new array
combined_newvector_bool = np.array([], dtype=bool)
# If one of the three triangle vectors has a z-value smaller than threshold, then the whole triangle will be set to True
for line in newvectors_bool:
    combined_newvector_bool = np.append(combined_newvector_bool, line.any())

# Make new vector array
# only use triangles bigger than threshold
# combined_newvector_bool == True for all triangles with smaller than threshold values --> inversion of combined_newvector_bool
newvectors = mesh_vectors[~combined_newvector_bool]

# Print shape of new vector-array
print(f"Shape of new vector-array: {np.shape(newvectors)}")

# Create new mesh
new_mesh = mesh.Mesh(np.zeros(newvectors.shape[0], dtype=mesh.Mesh.dtype))

# Assign new vectors
new_mesh.vectors = newvectors

# Save new mesh
new_mesh.save(save_pathname + save_filename)

# end time
end = time.time()

# Print elapsed time
print(f"Elapsed time: {end-start}")

# # Plot the data
# # Create a new plot
# figure = pyplot.figure()
# axes = mplot3d.Axes3D(figure)

# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(newvectors))

# # Auto scale to the mesh size
# # scale = tooth_mesh.points.flatten()
# scale = newvectors.flatten()
# axes.auto_scale_xyz(scale, scale, scale)

# # Show the plot to the screen
# pyplot.show()
