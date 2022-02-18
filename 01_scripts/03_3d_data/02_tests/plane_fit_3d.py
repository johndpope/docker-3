import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import open3d as o3d
import copy
import time

def fit_plane_to_pcd(mat):
    """
    Returns xyz-pcd of optimal fit plane for mat
    A B C opt for AX + BY + CZ + 1 = 0
    Z = alpha*X + beta*Y + gamma
    Optimal fit for point cloud in mat
    """
    meanmat = mat.mean(axis=0)
    meanmat = np.array([1.8, 1.8, 15.5])
    xm = meanmat[0]
    ym = meanmat[1]
    zm = meanmat[2]

    mat_new = mat-meanmat
    E = np.matmul(mat_new.T, mat_new)
    print(f"{E = }")
    w, v = np.linalg.eig(E)
    print(f"{w = }")
    print(f"{v = }")

    A = np.empty(shape=(3))
    B = np.empty(shape=(3))
    C = np.empty(shape=(3))
    delta = np.empty(shape=(3))

    for ctr in range(3):
        Vi = v[:,ctr]
        # roh = - (xm*Vi[0] + ym*Vi[1] + zm*Vi[2])/(Vi[0]**2 + Vi[1]**2 + Vi[2]**2)
        # ui = -Vi[0]*roh
        # vi = -Vi[1]*roh
        # wi = -Vi[2]*roh
        print(f"Test: {Vi[0]**2 + Vi[1]**2 + Vi[2]**2} must be 1") 
        ai = Vi[0]/Vi[2]
        bi = Vi[1]/Vi[2]
        denom = (ai*xm+bi*ym+zm)
        Ai = -ai/denom
        Bi = -bi/denom
        Ci = -1/denom

        delta_sum_part = 0
        for x,y,z in mat:
            delta_sum_part += (Ai*x + Bi*y +Ci*z +1)**2

        delta[ctr] = 1/(Ai**2+Bi**2+Ci**2)* delta_sum_part
        A[ctr] = Ai
        B[ctr] = Bi
        C[ctr] = Ci

    min_indx = np.argmin(delta)
    Aopt = A[min_indx]
    Bopt = B[min_indx]
    Copt = C[min_indx]
    alpha = -Aopt/Copt
    beta = -Bopt/Copt
    gamma = -1/Copt

    plane_mat = np.tile(mat, (len(delta),1,1))

    for ctr in range(len(delta)):
        alpha = -A[ctr]/C[ctr]
        beta = -B[ctr]/C[ctr]
        gamma = -1/C[ctr]
        plane_mat[ctr,:,2] = mat[:,0]*alpha + mat[:,1]*beta + np.ones(shape=(mat.shape[0]))*gamma  

    # plane_mat = copy.deepcopy(mat)
    # plane_mat[:,2] = mat[:,0]*alpha + mat[:,1]*beta + np.ones(shape=(mat.shape[0]))*gamma

    return plane_mat
 
xk=np.array([1,1,1,2,2,2,3,3,3])
yk=[1,2,3,1,2,3,1,2,3]
zk=[9,14,20,11,17,23,15,20,26]
mat = np.array([xk, yk, zk]).T

t0 = time.time()
plane_mat = fit_plane_to_pcd(mat)
print(time.time()-t0)

print(plane_mat.shape)
print(plane_mat[0])
print(plane_mat[1])
print(plane_mat[2])



# pcd_old = o3d.geometry.PointCloud()
# pcd_old.points = o3d.utility.Vector3dVector(mat)
# pcd_old.paint_uniform_color([0.5, 0.5, 0])

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(plane_mat)
# pcd.paint_uniform_color([1, 0.706, 0])

# obb = pcd.get_oriented_bounding_box()
# obb_old = pcd_old.get_oriented_bounding_box()
# print(obb.R)
# print(obb_old.R)

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

# o3d.visualization.draw_geometries([pcd, pcd_old, obb, frame], width=720, height=720, window_name=f"Rotated Tooth", left=1000, top=300)

