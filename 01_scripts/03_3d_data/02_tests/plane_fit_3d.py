import numpy as np


def fit_plane(mat):
    """
    Return A B C opt for AX + BY + CZ + 1 = 0
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
        roh = - (xm*Vi[0] + ym*Vi[1] + zm*Vi[2])/(Vi[0]**2 + Vi[1]**2 + Vi[2]**2)
        ui = -Vi[0]*roh
        vi = -Vi[1]*roh
        wi = -Vi[2]*roh
        print(f"Test: {Vi[0]**2 + Vi[1]**2 + Vi[2]**2} must be 1") 
        ai = Vi[0]/Vi[2]
        bi = Vi[1]/Vi[2]
        denom = (ai*xm+bi*ym+zm)
        Ai = -ai/denom
        Bi = -bi/denom
        Ci = -1/denom

        num = 0
        for x,y,z in mat:
            num += (Ai*x + Bi*y +Ci*z +1)**2

        delta[ctr] = 1/(Ai**2+Bi**2+Ci**2)* num
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

    return Aopt, Bopt, Copt, alpha, beta, gamma
 
xk=np.array([1,1,1,2,2,2,3,3,3])
yk=[1,2,3,1,2,3,1,2,3]
zk=[9,14,20,11,17,23,15,20,26]
mat = np.array([xk, yk, zk]).T

print(fit_plane(mat))
