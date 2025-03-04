import cv2
import numpy as np
import os
import glob
from scipy.optimize import least_squares

def read_image_corners(folder_path):

    # checkboard_corners = [np.zeros(len(folder_path))]
    checkboard_corners = []

    # print(checkboard_corners)
    # print(folder_path)

    for i, img_path in enumerate(folder_path):

        img = cv2.imread(img_path)

        # print(img)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray_img, patternSize=(6, 9))

        if found:

            # checkboard_corners[i] = corners
            checkboard_corners.append(corners)
    
    return checkboard_corners

# def compute_H(obj_coords, img_coords):

#     num_points = obj_coords.shape[0]
#     A = []

#     for i in range(num_points):

#         X, Y = obj_coords[i, 0], obj_coords[i, 1]
#         u, v = img_coords[i, 0], img_coords[i, 1]

#         A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
#         A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
    
#     A = np.array(A)

#     # print(A)

#     _, _, Vt = np.linalg.svd(A)

#     H = Vt[-1].reshape(3, 3)

#     # print(H)

#     H = H/H[2,2]

#     return H

def compute_H(obj_coords, img_coords):
    # Step 1: Normalization
    # Only use x,y coordinates for mean and scale calculation
    obj_mean = np.mean(obj_coords[:, :2], axis=0)
    img_mean = np.mean(img_coords[:, :2], axis=0)
    
    # Calculate scaling factors using only x,y coordinates
    obj_scale = np.sqrt(2) / np.std(obj_coords[:, :2] - obj_mean)
    img_scale = np.sqrt(2) / np.std(img_coords[:, :2] - img_mean)
    
    # Create normalization matrices
    T1 = np.array([[obj_scale, 0, -obj_scale*obj_mean[0]],
                   [0, obj_scale, -obj_scale*obj_mean[1]],
                   [0, 0, 1]])
    
    T2 = np.array([[img_scale, 0, -img_scale*img_mean[0]],
                   [0, img_scale, -img_scale*img_mean[1]],
                   [0, 0, 1]])
    
    # Convert to homogeneous coordinates and normalize
    # Take only x,y coordinates and add homogeneous coordinate
    obj_homog = np.column_stack([obj_coords[:, :2], np.ones(len(obj_coords))])
    img_homog = np.column_stack([img_coords[:, :2], np.ones(len(img_coords))])
    
    # Apply normalization
    obj_normalized = (T1 @ obj_homog.T).T
    img_normalized = (T2 @ img_homog.T).T
    
    # Step 2: Compute H using normalized coordinates
    num_points = obj_normalized.shape[0]
    A = []
    
    for i in range(num_points):
        X, Y = obj_normalized[i, 0], obj_normalized[i, 1]
        u, v = img_normalized[i, 0], img_normalized[i, 1]
        
        A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
    
    A = np.array(A)
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    H_normalized = Vt[-1].reshape(3, 3)
    H_normalized = H_normalized / H_normalized[2, 2]
    
    # Step 3: Denormalize
    H = np.linalg.inv(T2) @ H_normalized @ T1
    
    return H

def compute_intrinsics(H):

    V = []

    for h in H:
        h1, h2, h3 = h[:, 0], h[:, 1], h[:, 2]

        V.append([
            h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], 
            h1[1] * h2[1], h1[2] * h2[0] + h1[0] * h2[2], 
            h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]
        ])

        V.append([
            h1[0]**2 - h2[0]**2, 2*(h1[0]*h1[1] - h2[0]*h2[1]),
            h1[1]**2 - h2[1]**2, 2*(h1[2]*h1[0] - h2[2]*h2[0]),
            2*(h1[2]*h1[1] - h2[2]*h2[1]), h1[2]**2 - h2[2]**2
        ])

    V = np.array(V)

    U, S, Vt = np.linalg.svd(V)

    # print(Vt)

    b = Vt[-1]

    # print(b)

    B11, B12, B22, B13, B23, B33 = b

    # print(B11)

    c_x = (B12*B23 - B13*B22) / (B11*B22 - B12**2)
    lambda_ = B33 - (B13**2 + c_x*(B12*B13 - B11*B23)) / B11

    # print(lambda_)

    f_x = np.sqrt(lambda_ / B11)
    f_y = np.sqrt(lambda_ * B11 / (B11*B22 - B12**2))
    gamma = 0
    c_y = gamma * c_x / f_y - B23 * f_x**2 / (lambda_ * f_y)

    A = np.array([[f_x, gamma, c_x], [0, f_y, c_y], [0, 0, 1]])

    return A

def compute_extrinsic(H, A):

    A_inv = np.linalg.inv(A)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    lambda_ = 1/np.linalg.norm(np.matmul(A_inv, h1))

    r1 = lambda_ * (np.matmul(A_inv, h1))
    r2 = lambda_ * (np.matmul(A_inv, h2))

    r3 = np.cross(r1, r2)

    t = lambda_ * (np.matmul(A_inv, h3))

    R = np.column_stack((r1, r2, r3))

    U, _ , Vt = np.linalg.svd(R)

    R_corrected = np.dot(U, Vt)

    if np.linalg.det(R_corrected) < 0:
        U[:, -1] *= -1  # Flip last column
        R_corrected = np.dot(U, Vt)

    return R_corrected, t

# def apply_radial_distortion(x, y, k1, k2, k3):
#     r2 = x**2 + y**2
#     radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
#     return x * radial, y * radial


def visualize_reprojection(img_path, original_corners, reprojected_points):
    """
    Visualize original corners and reprojected points on the image
    """
    import matplotlib.pyplot as plt
    
    # Read and convert image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15,10))
    plt.imshow(img_rgb)
    
    # Plot original corners (red)
    plt.scatter(original_corners[:,0], original_corners[:,1], 
               c='red', s=25, marker='o', label='Original corners')
    
    # Plot reprojected points (blue)
    plt.scatter(reprojected_points[:,0], reprojected_points[:,1], 
               c='blue', s=25, marker='x', label='Reprojected points')
    
    # Draw lines between corresponding points to show error
    for orig, proj in zip(original_corners, reprojected_points):
        plt.plot([orig[0], proj[0]], [orig[1], proj[1]], 'g-', alpha=0.5)
    
    plt.legend()
    plt.title('Corner Reprojection Visualization')
    plt.axis('off')
    plt.show()

def compute_reprojection_error(coords_3d, img_coords, A, R, t, k1, k2, k3):
    projected_points = []

    for P in coords_3d:
        P_cam = np.matmul(R, P[:3]) + t

        x = P_cam[0]/P_cam[2]
        y = P_cam[1]/P_cam[2]

        r2 = x**2 + y**2
        distortion = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2

        x_distorted = x*distortion
        y_distorted = y*distortion

        u_proj = A[0, 0] * x_distorted + A[0, 2]
        v_proj = A[1, 1] * y_distorted + A[1, 2]

        projected_points.append([u_proj, v_proj])

    projected_points = np.array(projected_points)
    error = np.linalg.norm((projected_points - img_coords), axis=1)
    
    return np.mean(error), projected_points

def reprojection_error(P, obj_points, img_points, img_indices, num_images):

    # Extract intrinsic parameters
    f_x, f_y, gamma, c_x, c_y, k1, k2 = P[:7]
    A = np.array([[f_x, gamma, c_x], [0, f_y, c_y], [0, 0, 1]])
    
    # Extract extrinsics (each image has 6 parameters: 3 for rotation, 3 for translation)
    extrinsics = P[7:].reshape(num_images, 6)
    
    errors = []
    
    for i in range(num_images):
        R_vec = extrinsics[i, :3]  # Rodrigues rotation vector
        t = extrinsics[i, 3:]

        # Convert Rodrigues vector back to rotation matrix
        R, _ = cv2.Rodrigues(R_vec)

        # Project points using updated camera parameters
        projected_points = []
        for P in obj_points:
            P_cam = np.matmul(R, P) + t  # Transform to camera coordinates
            x = P_cam[0] / P_cam[2]  # Normalize
            y = P_cam[1] / P_cam[2]
            
            r2 = x**2 + y**2
            distortion = 1 + k1*r2 + k2*r2**2  # Apply radial distortion
            
            x_distorted = x * distortion
            y_distorted = y * distortion

            # Convert back to pixel coordinates
            u_proj = f_x * x_distorted + gamma * y_distorted + c_x
            v_proj = f_y * y_distorted + c_y

            projected_points.append([u_proj, v_proj])

        projected_points = np.array(projected_points)

        # Compute difference between projected and actual image points
        error = projected_points - img_points[i]
        errors.append(error.flatten())

    return np.hstack(errors)  # Return a 1D array of all errors


def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    img_folder_path = os.path.join(current_dir, "Calibration_Imgs", "*.jpg")
    img_files = []
    img_files = glob.glob(img_folder_path)

    checkerboard_corners = np.array(read_image_corners(img_files))

    reshaped_checkerboard = checkerboard_corners.reshape(len(img_files), 54, 2)

    chessboard_size = (6, 9)

    square_size = 21.50

    coords_3d = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)

    coords_3d[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    coords_3d = coords_3d[::-1]

    for i in range(0, coords_3d.shape[0], chessboard_size[0]):
        coords_3d[i:i+chessboard_size[0]] = coords_3d[i:i+chessboard_size[0]][::-1]

    coords_3d += square_size
    coords_3d[:,2] = 0

    coords_3d = coords_3d[::-1]

    # visualize_coords_and_corners(coords_3d, img_files, checkerboard_corners, chessboard_size)

    print(coords_3d)

    H = []

    for i in range(len(img_files)):

        h_i = compute_H(coords_3d, reshaped_checkerboard[i])
        H.append(h_i)

    A = compute_intrinsics(H)

    print(A)

    extrinsics = []

    for h_i in H:

        R_i, t_i = compute_extrinsic(h_i, A)

        I_check = np.dot(R_i.T, R_i)

        det = np.linalg.det(R_i)

        error = np.linalg.norm(I_check - np.eye(3), ord='fro')

        if error < 0.001 and (abs(det - 1)) < 0.001:
            extrinsics.append([R_i, t_i])
            # print(f"Saved for extrinsic parameters of {h_i}")
        else:
            print(f"Not the correct rotation matrix for {h_i}")
    
    k1, k2, k3 = 0.0, 0.0, 0.0

    total_error = []

    for i, (R, t) in enumerate(extrinsics):
        error, projected_points = compute_reprojection_error(coords_3d, reshaped_checkerboard[i], A, R, t, k1, k2, k3)

        total_error.append(error)

        print(f"Reprojection error for image {i} before optimization: {error:.4f}")

        visualize_reprojection(
            img_files[i], 
            reshaped_checkerboard[i], 
            projected_points
        )
    
    unopt_error = np.mean(total_error)

    print(f"Total error before optimization: {unopt_error}")


    a = [A[0,0], A[1,1], A[0,1], A[0,2], A[1,2], k1, k2]

    extrinsics_opt = []

    for i, (R, t) in enumerate(extrinsics):
        R_vec, _ = cv2.Rodrigues(R)
        # extrinsics_opt.extend(R_vec.flatten())
        # extrinsics_opt.extend(t.flatten())
        extrinsics_opt.append(np.concatenate([R_vec.flatten(), t.flatten()]))
    
    # extrinsics_opt = np.hstack(extrinsics_opt)
    P_init = np.concatenate([a, np.concatenate(extrinsics_opt)])

    num_images = len(img_files)  # Total number of images
    num_corners = chessboard_size[0]*chessboard_size[1]  # Chessboard corners detected per image

    # Create img_indices: It repeats the image index for each corner detected
    img_indices = np.repeat(np.arange(num_images), num_corners)

    result = least_squares(reprojection_error, P_init, method='lm',
                           args=(coords_3d, reshaped_checkerboard, img_indices,len(img_files)))
    
    P_opt = result.x
    A_opt = np.array([[P_opt[0], P_opt[2], P_opt[3]], 
                      [0, P_opt[1], P_opt[4]], 
                      [0, 0, 1]])
    
    print(A_opt)

    dist_coeffs_opt = np.array([P_opt[5], P_opt[6], 0, 0, 0])

    extrinsics_opt = P_opt[7:].reshape(num_images, 6)

    P_opt_extrinsics = []
    for i in range(num_images):
        R_vec = extrinsics_opt[i, :3]
        t_opt = extrinsics_opt[i, 3:]

        # Convert back from Rodrigues vector to rotation matrix
        R_opt, _ = cv2.Rodrigues(R_vec)
        
        P_opt_extrinsics.append([R_opt, t_opt])


    k1, k2, k3 = dist_coeffs_opt[0], dist_coeffs_opt[1], dist_coeffs_opt[2]

    print(k1)
    print(k2)
    print(k3)

    opt_total_error = []

    for i, (R, t) in enumerate(P_opt_extrinsics):
        opt_error, opt_projected_points = compute_reprojection_error(coords_3d, reshaped_checkerboard[i], A_opt, R, t, k1, k2, k3)

        opt_total_error.append(opt_error)

        print(f"Reprojection error for image {i} after optimization: {opt_error:.4f}")

        visualize_reprojection(
            img_files[i], 
            reshaped_checkerboard[i], 
            opt_projected_points
        )
    
    opt_error = np.mean(opt_total_error)

    print(f"Total error after optimization: {opt_error}")


if __name__ == "__main__":
    main()
