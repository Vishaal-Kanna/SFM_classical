import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as oR

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.shape[0] * 2
    n = n_cameras * 7 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.shape[0])
    for s in range(7):
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 7 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1

    return A

def normalise_quat(Q):
    # Q is of shape (num_of_cameras,4)
    Q = Q.astype(float)
    norm = np.linalg.norm(Q,axis=1).reshape(-1,1)
    Q = Q/norm 
    return Q

def project(points_3d, camera_extrinsics, K):
    Q = camera_extrinsics[:,0:4]
    C = camera_extrinsics[:,4:]
    r = oR.from_quat(Q)
    points_camera_3d = r.apply(points_3d) + C
    points_img_homo = points_camera_3d @ K.T
    points_proj = (points_img_homo/(points_img_homo[:,2].reshape(-1,1)))[:,0:2]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    camera_extrinsics = params[:n_cameras * 7].reshape((n_cameras, 7))
    # converting to unit quaternions
    camera_extrinsics[:,0:4] = normalise_quat(camera_extrinsics[:,0:4])
    points_3d = params[n_cameras * 7:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_extrinsics[camera_indices],K)
    return (points_proj - points_2d).ravel()

def bundle_adjustment(Cset,Rset,X,K,points_2d,V):
    print(Cset)
    C = np.array(Cset)
    r = oR.from_matrix(Rset)
    Q = r.as_quat()
    camera_extrinsics = np.hstack((Q,C))
    x0 = np.hstack((camera_extrinsics.ravel(), X.ravel()))

    n_cameras = len(Cset)
    n_points = X.shape[0]
    indices = np.argwhere(V)
    camera_indices = indices[:,1]
    point_indices = indices[:,0]
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))

    params = res.x
    camera_extrinsics = params[:n_cameras * 7].reshape((n_cameras, 7))
    # converting to unit quaternions
    C = camera_extrinsics[:,4:]
    Q = normalise_quat(camera_extrinsics[:,0:4])
    r = oR.from_quat(Q)
    R = r.as_matrix()
    Cset_BA = []
    Rset_BA = []
    for i in range(n_cameras):
        Cset_BA.append(C[i])
        Rset_BA.append(R[i])

    X_BA = params[n_cameras * 7:].reshape((n_points, 3))

    return Cset_BA, Rset_BA, X_BA