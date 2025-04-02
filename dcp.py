import cv2
import numpy as np
import csv
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

# Function to calculate dark channel
def dark_channel(img, size=15):
    min_img = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

# Estimate atmospheric light
def estimate_atmospheric_light(img, dark_channel):
    h, w = img.shape[:2]
    num_pixels = h * w
    top_pixels = int(max(np.floor(num_pixels * 0.001), 1))

    dark_flat = dark_channel.ravel()
    indices = np.argsort(dark_flat)[-top_pixels:]
    
    atmospheric_light = np.mean(img.reshape(num_pixels, 3)[indices], axis=0)
    return atmospheric_light

# Recover scene radiance for DCP
def recover_scene(img, dark_channel, atmospheric_light, omega=0.95, t_min=0.1, size=15):
    transmission = 1 - omega * dark_channel / atmospheric_light.max()
    transmission = np.clip(transmission, t_min, 1)
    transmission = cv2.dilate(transmission, np.ones((size, size), np.uint8))
    atmospheric_light_reshaped = atmospheric_light.reshape(1, 1, 3)
    recovered_img = (img - atmospheric_light_reshaped) / np.expand_dims(transmission, axis=2) + atmospheric_light_reshaped
    recovered_img = np.clip(recovered_img, 0, 255).astype(np.uint8)
    return recovered_img

# Dark Channel Prior (DCP) method
def defog_using_dcp(img):
    img_float = img.astype(np.float64)
    dark = dark_channel(img_float)
    atmospheric_light = estimate_atmospheric_light(img_float, dark)
    recovered_img = recover_scene(img_float, dark, atmospheric_light)
    return recovered_img

# Store results for DCP
def store_dcp_results(image_paths, csv_filename='dcp_results.csv'):
    results = []
    for i, img_path in enumerate(image_paths, start=1):
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Image not found at the path: {img_path}")
            continue
        result_dcp = defog_using_dcp(original_img)

        # Save the defogged DCP image
        cv2.imwrite(f'dcp_defogged_image_{i}.jpg', result_dcp)

        psnr_dcp = calculate_psnr(original_img, result_dcp)
        ssim_dcp = calculate_ssim(original_img, result_dcp)
        results.append({"image": img_path, "psnr_dcp": psnr_dcp, "ssim_dcp": ssim_dcp})
        print(f"Image {i}: DCP - PSNR: {psnr_dcp}, SSIM: {ssim_dcp}")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["image", "psnr_dcp", "ssim_dcp"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"DCP results have been written to {csv_filename}")

# Example usage for DCP
image_paths = [f'image{i}.jpg' for i in range(1, 13)]
store_dcp_results(image_paths, 'dcp_defogging_results.csv')
