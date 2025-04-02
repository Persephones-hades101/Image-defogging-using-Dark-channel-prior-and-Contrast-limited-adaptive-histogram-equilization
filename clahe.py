import cv2
import csv
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to calculate PSNR
def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Function to calculate SSIM
def calculate_ssim(original, processed):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    return ssim(original_gray, processed_gray)

# CLAHE method for defogging
def defog_using_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img

# Store results for CLAHE
def store_clahe_results(image_paths, csv_filename='clahe_results.csv'):
    results = []
    for i, img_path in enumerate(image_paths, start=1):
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Image not found at the path: {img_path}")
            continue
        result_clahe = defog_using_clahe(original_img)

        # Save the defogged CLAHE image
        cv2.imwrite(f'clahe_defogged_image_{i}.jpg', result_clahe)

        psnr_clahe = calculate_psnr(original_img, result_clahe)
        ssim_clahe = calculate_ssim(original_img, result_clahe)
        results.append({"image": img_path, "psnr_clahe": psnr_clahe, "ssim_clahe": ssim_clahe})
        print(f"Image {i}: CLAHE - PSNR: {psnr_clahe}, SSIM: {ssim_clahe}")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["image", "psnr_clahe", "ssim_clahe"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"CLAHE results have been written to {csv_filename}")

# Example usage for CLAHE
image_paths = [f'image{i}.jpg' for i in range(1, 13)]
store_clahe_results(image_paths, 'clahe_defogging_results.csv')
