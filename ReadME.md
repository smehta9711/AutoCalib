# **Camera Calibration - README**

## **Overview**
This script estimates both **intrinsic** and **extrinsic** parameters of a camera using a **checkerboard pattern**. It follows the **Zhang’s Camera Calibration Method**, utilizing **homography estimation, intrinsic matrix computation, and reprojection error minimization** using **Levenberg-Marquardt optimization**.


### **1. Prepare Calibration Images**
- Place checkerboard images in a directory named **`Calibration_Imgs`** inside the project folder.
- Images should be in **.jpg** format.
- Make sure the checkerboard pattern is **clearly visible** in each image.

### **2. Run the Calibration Script**
Execute the script using:
```bash
python3 Wrapper.py
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


## **Example Intrinsic Matrix Output**
After optimization
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

