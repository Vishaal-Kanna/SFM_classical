#!/usr/bin/env python

"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project3: SfM:

Author(s): 
Pavan Mantripragada (mppavan@umd.edu) 
Masters in Robotics,
University of Maryland, College Park

Vishaal Kanna Sivakumar (vishaal@umd.edu)
Masters in Robotics,
University of Maryland, College Park
"""

import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from utils.data_utils import *
from utils.GetInliersRANSAC import *
from utils.ExtractCameraPose import ExtractCameraPose
from utils.DrawCorrespondence import DrawCorrespondence
from utils.EssentialMatrixFromFundamentalMatrix import *
from utils.LinearTriangulation import *
from utils.DisambiguateCameraPose import *
from utils.NonLinearTriangulation import *
from utils.EstimateFundamentalMatrix import *
from utils.PnPRANSAC import *
from utils.NonLinearPnP import *
from utils.BuildVisibilityMatrix import *
from utils.BundleAdjustment import *
from utils.visualization_utils import *

def main():

	K = np.array([[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]])

	n_images = 6

	Mx, My, M = get_features_from_matching_files('Data/',n_images)

	w=1280
	h= 960
	
	# Using pre-computed inliers uncomment below two lines to re-compute
	M = np.load('./Data/M.npy')
	# M, outlier_indices = inlier_filter(Mx, My, M, n_images,1280, 960)
	# np.save('M',M)
	
	pts1,pts2,indices,X_Mask = get_correspondances(Mx,My,M,0,1)

	# Computing Fundamental Matrix
	best_F,_ = get_F(np.float32(pts1), np.float32(pts2), w,h)
	# Computing Essential Matrix
	E = EssentialMatrixFromFundamentalMatrix(best_F, K)
	# Recover Poses
	R_set, C_set = ExtractCameraPose(E)
	X_set = []
	for n in range(0, 4):
		X1 = LinearTriangulation(K, np.zeros((3, 1)), np.identity(3), C_set[n].T, R_set[n], np.float32(pts1), np.float32(pts2))
		X_set.append(X1)

	# Display all poses
	visualizer = CameraPoseVisualizer([-30, 30], [-30, 30], [-10, 50])
	colors = ['r','g','b','y']
	for i in range(len(C_set)):
		RC = np.eye(4)
		RC[0:3,0:3] = R_set[i].T
		RC[0:3,3] = -(R_set[i].T @ C_set[i].reshape(-1,1)).reshape(-1)
		visualizer.plot_points(X_set[i],colors[i])
		visualizer.extrinsic2pyramid(RC, colors[i], 5)
	RC = np.eye(4)
	RC[0:3, 3] = 0
	visualizer.extrinsic2pyramid(RC, 'k', 5)
	visualizer.show()

	# Disambiguate poses
	X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)

	# Non-linear triangulation
	X = NonLinearTriangulation(K,np.float32(pts1),np.float32(pts2),X,np.eye(3),np.zeros((3,1)),R,C)
	X_3D = np.zeros((M.shape[0], 3))
	X_3D[indices, :] = X
	
	# Start with first two images
	Cset = [np.zeros(3,),C.reshape(3,)]
	Rset = [np.identity(3),R]
	
	# Register rest of the images
	for i in range(2,n_images):
		# Find common 3D points between registered images and current image
		output = np.logical_and(X_Mask, M[:, i])
		indices = np.argwhere(output).reshape(-1)
		
		# If less than 6 points skip this image (don't register)
		if indices.shape[0]<6:
			continue

		x = np.transpose([Mx[indices, i], My[indices, i]])
		X = X_3D[indices, :]
		
		# Register the image using Linear PnP with RANSAC 
		C,R = PnPRANSAC(X, x, K)

		# _, rvec, C, _ = cv2.solvePnPRansac(X, x, K, None)
		# R = cv2.Rodrigues(rvec)[0]

		# Register the image using Non-linear PnP
		C, R = NonLinearPnP(X, x, K, C, R)
		Cset.append(C.reshape(3,))
		Rset.append(R)

		# Triangulate 
		for j in range(0,len(Cset)-1):
			pts1,pts2,pts_indices,inlier_indices = get_correspondances(Mx,My,M,j,i)
			if len(pts1)<=5:
				continue
			Xnew = LinearTriangulation(K, Cset[j].T, Rset[j], Cset[len(Cset)-1].T, Rset[len(Cset)-1], np.float32(pts1), np.float32(pts2))
			Xnew = NonLinearTriangulation(K,np.float32(pts1),np.float32(pts2),Xnew,Rset[j],Cset[j].reshape(3,1),Rset[len(Cset)-1],Cset[len(Cset)-1])
			X_3D[pts_indices,:] = Xnew
			X_Mask = np.logical_or(X_Mask,inlier_indices)

		V, idx_vis = getV(X_Mask, M, i)
		X = X_3D[idx_vis]
		mask_2d = np.logical_and(X_Mask.reshape(-1,1),M[:,0:i+1])
		points_2d = np.hstack((Mx[:,0:i+1][mask_2d == True].reshape(-1,1), My[:,0:i+1][mask_2d == True].reshape(-1,1)))
		Cset, Rset, X_BA = bundle_adjustment(Cset,Rset,X,K,points_2d,V)
		X_3D[idx_vis] = X_BA

	visualizer1 = CameraPoseVisualizer([-30, 30], [-30, 30], [-1, 50])
	visualizer1.plot_points(X_3D[X_Mask == 1], 'k')
	colors = ['k','r','g','b','y','c']
	for i in range(len(Cset)):
		RC = np.eye(4)
		RC[0:3,0:3] = Rset[i].T
		RC[0:3,3] = -(Rset[i].T @ Cset[i].reshape(-1,1)).reshape(-1,)
		visualizer1.extrinsic2pyramid(RC, colors[i], 1)
	visualizer.show()


if __name__ == '__main__':
	main()


