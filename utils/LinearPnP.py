import numpy as np

def LinearPnP(X, x, K):

    N = X.shape[0]
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    x = np.hstack((x, np.ones((x.shape[0], 1))))

    x = np.transpose(np.dot(np.linalg.inv(K), x.T))
    A = []
    for i in range(N):
        xt = X[i, :].reshape((1, 4))
        z = np.zeros((1, 4))
        p = x[i, :]

        a1 = np.hstack((np.hstack((z, -xt)), p[1] * xt))
        a2 = np.hstack((np.hstack((xt, z)), -p[0] * xt))
        a3 = np.hstack((np.hstack((-p[1] * xt, p[0] * xt)), z))
        a = np.vstack((np.vstack((a1, a2)), a3))

        if (i == 0):
            A = a
        else:
            A = np.vstack((A, a))

    _, _, v = np.linalg.svd(A)
    P = v[-1].reshape((3, 4))
    R = P[:, 0:3]
    t = P[:, 3]
    U, s, Vt = np.linalg.svd(R)

    R = U @ Vt
    t = (P[:,3]/s[0]).reshape(3,1)
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    return R,t

def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))
    I = np.identity(3)
    P = K @ np.hstack((R, C))
    return P

def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))
