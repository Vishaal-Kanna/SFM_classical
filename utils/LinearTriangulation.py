import numpy as np
import cv2

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    sz = x1.shape[0]
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)
    P1 = K @ np.hstack((R1, C1))
    P2 = K @ np.hstack((R2, C2))

    X1 = np.hstack((x1, np.ones((sz, 1))))
    X2 = np.hstack((x2, np.ones((sz, 1))))

    X = np.zeros((sz, 3))

    for i in range(sz):
        A = np.zeros((4,4))
        A[0,:] = x1[i,0]*P1[2,:] - P1[0,:]
        A[1,:] = x1[i,1]*P1[2,:] - P1[1,:]
        A[2,:] = x2[i,0]*P2[2,:] - P2[0,:]
        A[3,:] = x2[i,1]*P2[2,:] - P2[1,:]
        _, _, Vt = np.linalg.svd(A)
        x = Vt[3,:]
        x = x.reshape(4,)
        x /= x[3]
        X[i, :] = x[0:3].reshape(1,3)

    return X
