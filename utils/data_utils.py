import numpy as np

def get_features_from_matching_files(datapath, num_imgs):
    # fU[j,i] : x img coord of jth feature in ith image, NaN if not in ith image
    # fV[j,i] : y img coord of jth feature in ith image, NaN if not in ith image
    # Mask : if jth feature exists in ith image Mask[j,i] = 1 else 0
    fU = []
    fV = []
    Mask = []

    for i in range(1, num_imgs):
        # i represents ith image/matching file
        filepath = datapath + "matching" + str(i) + ".txt"
        lines = list(open(filepath,"r"))
        for line in lines[1:]:
            # j represents jth row in matching file
            U_j = np.empty((num_imgs))
            V_j = np.empty((num_imgs))
            Mask_j = np.zeros((num_imgs), dtype = int)

            data_j = np.array([float(x) for x in line.split()])
            # num. of matches for jth feature
            num_matches_j = int(data_j[0])-1

            U_j[i-1] = data_j[4]
            V_j[i-1] = data_j[5]
            Mask_j[i-1] = 1
            
            # matches in other images
            for k in range(num_matches_j):
                other_i = int(data_j[6+3*k])
                U_j[other_i-1] = data_j[7+3*k]
                V_j[other_i-1] = data_j[8+3*k]
                Mask_j[other_i-1] = 1

            fU.append(U_j)
            fV.append(V_j)
            Mask.append(Mask_j)
    
    fU = np.array(fU)
    fV = np.array(fV)
    Mask = np.array(Mask,dtype=int)

    return fU, fV, Mask


def get_correspondances(Mx,My,M,id1,id2):
    inlier_indices = np.logical_and(M[:, id1], M[:, id2])
    indices = np.where(inlier_indices == True)
    pts1 = np.hstack((Mx[indices, id1].reshape((-1, 1)), My[indices, id1].reshape((-1, 1))))
    pts2 = np.hstack((Mx[indices, id2].reshape((-1, 1)), My[indices, id2].reshape((-1, 1))))

    return pts1, pts2, indices, inlier_indices