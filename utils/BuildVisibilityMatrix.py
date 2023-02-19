import numpy as np

def getV(X_Mask, Mask, img_id):
    idx_any = np.any(Mask[:,0:img_id+1],axis=1)
    idx_vis = np.argwhere(np.logical_and(X_Mask,idx_any)).reshape(-1)    
    V = Mask[idx_vis,0:img_id+1]

    return V, idx_vis