# Virtual stereo camera calibration script (based on monocular with plane mirror reflection)
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 1. Chessboard parameters ===
chessboard_size = (9, 6)  # 9x6 corners
square_size = 14.2  # mm per square

# === 2. Prepare 3D points for the chessboard (0,0,0), (1,0,0), ... ===
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# === 3. Store 2D image points for left and right images ===
objpoints = []  # 3D points (world coordinates)
imgpoints_left = []  # Left image 2D points
imgpoints_right = []  # Right image 2D points

# === 4. Read left and right image pairs ===
images_left = sorted(glob.glob('./left/*.bmp'))
images_right = sorted(glob.glob('./right/*.bmp'))

assert len(images_left) == len(images_right), "The number of left and right images do not match!"

for img_left_path, img_right_path in zip(images_left, images_right):
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 5. Find chessboard corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, flags)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, flags)

    if ret_left and ret_right:
        objpoints.append(objp)

        # Subpixel optimization
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        os.makedirs('./corner', exist_ok=True)

        cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)

        base_left = os.path.basename(img_left_path)
        base_right = os.path.basename(img_right_path)

        save_left_path = os.path.join('./corner', f'corner_left_{base_left}')
        save_right_path = os.path.join('./corner', f'corner_right_{base_right}')

        cv2.imwrite(save_left_path, img_left)
        cv2.imwrite(save_right_path, img_right)

cv2.destroyAllWindows()

# === 6. Calibrate left camera ===
ret_left, K1, dist1, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_left.shape[::-1], None, None
)

# Compute reprojection error for left camera
total_error_left = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_left[i], tvecs_left[i], K1, dist1)
    error = cv2.norm(imgpoints_left[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error_left += error
mean_error_left = total_error_left / len(objpoints)
print(f"Left camera reprojection error: {mean_error_left:.4f} pixels")

# === 7. Calibrate right camera ===
ret_right, K2, dist2, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_right.shape[::-1], None, None
)

# Compute reprojection error for right camera
total_error_right = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_right[i], tvecs_right[i], K2, dist2)
    error = cv2.norm(imgpoints_right[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error_right += error
mean_error_right = total_error_right / len(objpoints)
print(f"Right camera reprojection error: {mean_error_right:.4f} pixels")

# === 8. Stereo calibration (DO NOT fix intrinsics) ===
flags = 0  # Do not fix intrinsic parameters
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 1e-5)
ret_stereo, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left, imgpoints_right,
    K1, dist1, K2, dist2,
    gray_left.shape[::-1],
    criteria=criteria,
    flags=flags
)

# === 9. Output calibration results ===
print("Left camera intrinsic parameters K1:\n", K1)
print("Left camera distortion dist1:\n", dist1.ravel())
print("Right camera intrinsic parameters K2:\n", K2)
print("Right camera distortion dist2:\n", dist2.ravel())
print("Stereo rotation R:\n", R)
print("Stereo translation T:\n", T.ravel())

np.savez('stereo_calib_params.npz', K1=K1, dist1=dist1, K2=K2, dist2=dist2, R=R, T=T, E=E, F=F)

print("Calibration complete, parameters saved to stereo_calib_params.npz")

# === 10. Triangulate and reconstruct 3D points for each stereo image pair ===
def fit_plane(points_3d):
    """ Fit a plane to a set of 3D points using least squares method. """
    A = np.c_[points_3d[:, 0], points_3d[:, 1], np.ones(points_3d.shape[0])]
    plane_params = np.linalg.lstsq(A, points_3d[:, 2], rcond=None)[0]
    return plane_params

def compute_plane_deviation(points_3d, plane_params):
    """ Compute the deviation of points from the fitted plane. """
    A, B, C = plane_params
    deviations = np.abs(A * points_3d[:, 0] + B * points_3d[:, 1] + C - points_3d[:, 2]) / np.sqrt(A**2 + B**2 + 1)
    return deviations

# Compute projection matrices using calibrated K1, K2
R1 = np.eye(3)
T1 = np.zeros((3,1))

P1 = K1 @ np.hstack((R1, T1))
P2 = K2 @ np.hstack((R, T))

all_3d_points = []
deviations_list = []

for i in range(len(objpoints)):
    pts1 = imgpoints_left[i].reshape(-1, 2).T
    pts2 = imgpoints_right[i].reshape(-1, 2).T

    pts4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T

    plane_params = fit_plane(pts3d)
    deviations = compute_plane_deviation(pts3d, plane_params)

    all_3d_points.append(pts3d)
    deviations_list.append(deviations)

# Visualize 3D reconstructed point cloud
all_3d_points = np.vstack(all_3d_points)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_3d_points[:,0], all_3d_points[:,1], all_3d_points[:,2], c='r', marker='o', s=10)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Virtual Stereo 3D Reconstruction Point Cloud')
plt.show()

# Visualize deviation histogram
deviations_all = np.hstack(deviations_list)
plt.figure(figsize=(8,6))
plt.hist(deviations_all, bins=50, color='b', alpha=0.7)
plt.title('Deviation of 3D points from fitted plane')
plt.xlabel('Deviation (mm)')
plt.ylabel('Frequency')
plt.show()

# Print average deviation per stereo pair
for i, deviations in enumerate(deviations_list):
    avg_dev = np.mean(deviations)
    print(f"Average deviation for stereo pair {i+1}: {avg_dev:.4f} mm")
