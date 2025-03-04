# **Camera Calibration - README**

## **Overview**
This script estimates both **intrinsic** and **extrinsic** parameters of a camera using a **checkerboard pattern**. It follows the **Zhang’s Camera Calibration Method**, utilizing **homography estimation, intrinsic matrix computation, and reprojection error minimization** using **Levenberg-Marquardt optimization**.

## **Dependencies**
Ensure you have the following Python libraries installed:
```bash
pip install numpy opencv-python matplotlib scipy
```

## **Usage Instructions**
### **1. Prepare Calibration Images**
- Place checkerboard images in a directory named **`Calibration_Imgs`** inside the project folder.
- Images should be in **.jpg** format.
- Make sure the checkerboard pattern is **clearly visible** in each image.

### **2. Run the Calibration Script**
Execute the script using:
```bash
python Wrapper.py
```

### **3. Output**
- The script will compute and display:
  - **Intrinsic Camera Matrix (K)**
  - **Extrinsic Parameters (Rotation and Translation Matrices)**
  - **Before & After Optimization Reprojection Error**
  - **Visualization of Checkerboard Corner Detection**
- The optimized intrinsic parameters will be printed after **Levenberg-Marquardt optimization**.

### **4. Expected Results**
- The **before optimization error** will be high.
- The **after optimization error** should be significantly lower.
- The script will **display images** with detected corners before and after optimization.

## **Key Functions**
### **Corner Detection**
- `read_image_corners(folder_path)`: Detects checkerboard corners in images.
  
### **Homography Estimation**
- `compute_H(obj_coords, img_coords)`: Computes homography matrix between 3D world and 2D image points.

### **Intrinsic Matrix Computation**
- `compute_intrinsics(H)`: Estimates **intrinsic matrix (K)** using a closed-form solution.

### **Extrinsic Parameter Computation**
- `compute_extrinsic(H, A)`: Computes **rotation (R) and translation (t) matrices** for each image.

### **Reprojection Error Calculation**
- `compute_reprojection_error()`: Computes **error between detected and projected points**.

### **Optimization using Levenberg-Marquardt**
- `least_squares(reprojection_error, P_init, method='lm', args=...)`: Minimizes reprojection error.

## **Example Intrinsic Matrix Output**
After optimization, you should see an output like:
```plaintext
Optimized Intrinsic Matrix (K_opt):
[[2045.84   -1.641  759.42]
 [   0.    2037.55 1345.26]
 [   0.       0.       1.  ]]
```

## **Notes**
- The **checkerboard pattern should be well-lit** in images.
- If errors are high, try **capturing more images from different angles**.
- If corners are not detected properly, adjust **checkerboard grid size** in the script (`patternSize=(6, 9)`).

---
This README should help users understand the script and run it successfully. 🚀

