import numpy as np
import torch
import torch.utils.data as data
import os
import glob
from numpy import linalg as LA

def origin_mass_center2(pcd):
    expectation = np.mean(pcd, axis=0)
    centered_pcd = pcd - expectation
    return centered_pcd, expectation


def normalize2(points, unit_ball = False):
    normalized_points, center = origin_mass_center2(points)
    #normalized_points = points
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)

    if unit_ball:
        scale = max_distance
        normalized_points = normalized_points/(max_distance)
    else:
        scale = 2 * max_distance
        normalized_points = normalized_points/(2 * max_distance)

    return normalized_points, center, scale

def remove_duplicates_with_dict(arr):
    return list(dict.fromkeys(arr))

def reflect(Data,cent,sym):
    Data = Data.reshape(4096,3)
    cent = cent.reshape(3)
    sym = sym.reshape(-1, 3)
    reflect_points = np.zeros((4096,sym.shape[0],3))
    for j in range(sym.shape[0]):
        x1 = cent[0]
        y1 = cent[1]
        z1 = cent[2]
        a = sym[j,0]
        b = sym[j,1]
        c = sym[j,2]
        ref_point = np.zeros(Data.shape)
        for i in range(0, Data.shape[0]):
            d = a*x1+b*y1+c*z1
            t = (d-(a*Data[i][0]+b*Data[i][1]+c*Data[i][2]))/(a*a+b*b+c*c) #点到平面距离
            sym_x = 2 * a * t + Data[i][0]
            sym_y = 2 * b * t + Data[i][1]
            sym_z = 2 * c * t + Data[i][2]
            ref_point[i,:] = np.array([sym_x, sym_y, sym_z])
        reflect_points[:,j,:] = ref_point
    return reflect_points