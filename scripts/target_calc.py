import numpy as np
import cv2

def RMatrixCalc(roll, pitch, yaw):

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Rr = np.array([[0,1,0],[-1,0,0],[0,0,-1]])
    return Rr @ Rx @ Ry @ Rz 

D = [0,0,0,0,0]
MD = np.array(D, dtype=float)

def CoordinateTransfer(R, cx, cy, a, b, h, K):
    """
    计算世界坐标 [Xw, Yw, Zw]^T。

    参数:
        R: 3x3 旋转矩阵 (numpy.ndarray)
        T: 3x1 平移矩阵 (numpy.ndarray)
        uv: 目标像素坐标 [u, v] (list 或 numpy.ndarray)
        K: 3x3 相机内参矩阵 (numpy.ndarray)

    返回:
        numpy.ndarray: 世界坐标 [Xw, Yw, Zw]^T
    """
     
    P = np.array([[0,-1,0],[1,0,0],[0,0,-1]])
    R = np.array(R)
    T = np.array([[a], [b], [h]])
    K = np.array(K)
    w = np.array([0,0,0])
    R = P @ R
    T = R @ T
     
    assert R.shape == (3, 3), "R 必须是 3x3 矩阵"
    assert T.shape == (3, 1), "T 必须是 3x1 矩阵"
    assert K.shape == (3, 3), "K 必须是 3x3 矩阵"
    
     
    uv_h = np.array([cx, cy, 1.0]).reshape(3, 1)
    
     
    R_inv = np.linalg.inv(R)
    K_inv = np.linalg.inv(K)
    M = R_inv @ K_inv @ uv_h   
    N = R_inv @ T             
    
     
    M3 = M[2, 0]
    N3 = N[2, 0]
    
     
    factor = N3 / M3
    print(f'uv {uv_h}')
    print(f'factor {factor}')
    world_coordinates = (factor * M) - N   
    w = world_coordinates.flatten()
    w[2] = factor
    
    w = K_inv * factor @ uv_h
    
def CoordinateTransfer(R, cx, cy, a, b, h, K):
    """
    计算世界坐标 [Xw, Yw, Zw]^T。

    参数:
        R: 3x3 旋转矩阵 (numpy.ndarray)
        T: 3x1 平移矩阵 (numpy.ndarray)
        uv: 目标像素坐标 [u, v] (list 或 numpy.ndarray)
        K: 3x3 相机内参矩阵 (numpy.ndarray)

    返回:
        numpy.ndarray: 世界坐标 [Xw, Yw, Zw]^T
    """
     
    P = np.array([[0,-1,0],[1,0,0],[0,0,-1]])
    R = np.array(R)
    T = np.array([[a], [b], [h]])
    K = np.array(K)
    w = np.array([0,0,0])
    R = P @ R
    T = R @ T
     
    assert R.shape == (3, 3), "R 必须是 3x3 矩阵"
    assert T.shape == (3, 1), "T 必须是 3x1 矩阵"
    assert K.shape == (3, 3), "K 必须是 3x3 矩阵"
    
     
    uv_h = np.array([cx, cy, 1.0]).reshape(3, 1)
    
     
    R_inv = np.linalg.inv(R)
    K_inv = np.linalg.inv(K)
    M = R_inv @ K_inv @ uv_h   
    N = R_inv @ T             
    
     
    M3 = M[2, 0]
    N3 = N[2, 0]
    
     
    factor = N3 / M3
    print(f'uv {uv_h}')
    print(f'factor {factor}')
    world_coordinates = (factor * M) - N   
    w = world_coordinates.flatten()
    w[2] = factor
    
    w = K_inv * factor @ uv_h
    
    pts_uv = cv2.undistortPoints(np.array([cx,cy]), K, MD) * factor 
    
    tar = np.array([[pts_uv[0][0][1]],[pts_uv[0][0][0]],[h]])
     
    w = R_inv @ (T - tar)
    
    w_rounded = np.round(w.flatten(), 2)
    
    return w_rounded



def covert(point, z,K):    
    point = np.array(point, dtype=float)    
    pts_uv = cv2.undistortPoints(point, K, MD) * z   
    return pts_uv[0][0]
    


