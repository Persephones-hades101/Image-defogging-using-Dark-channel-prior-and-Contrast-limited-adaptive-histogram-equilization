import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import csv
import matplotlib.pyplot as plt

def plot_images(original_img, dcp_img, clahe_img, img_num):
    # Convert images from BGR to RGB format for proper display with matplotlib
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    dcp_rgb = cv2.cvtColor(dcp_img, cv2.COLOR_BGR2RGB)
    clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)
    
    # Create a figure to hold the subplots
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title(f'Original Image {img_num}')
    plt.axis('off')
    
    # Plot DCP processed image
    plt.subplot(1, 3, 2)
    plt.imshow(dcp_rgb)
    plt.title(f'DCP Image {img_num}')
    plt.axis('off')
    
    # Plot CLAHE processed image
    plt.subplot(1, 3, 3)
    plt.imshow(clahe_rgb)
    plt.title(f'CLAHE Image {img_num}')
    plt.axis('off')
    
    # Show the figure
    plt.show()

# Function to apply defogging methods and plot images
def process_and_plot_image(img_path, img_num):
    original_img = cv2.imread(img_path)
    
    # Check if image is loaded properly
    if original_img is None:
        raise FileNotFoundError(f"Image not found at the path: {img_path}")
    
    # Apply both defogging methods
    dcp_img = defog_using_dcp(original_img)
    clahe_img = defog_using_clahe(original_img)
    
    # Plot the original, DCP, and CLAHE images
    plot_images(original_img, dcp_img, clahe_img, img_num)


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

# Recover scene radiance
def recover_scene(img, dark_channel, atmospheric_light, omega=0.95, t_min=0.1, size=15):
    # Normalize dark channel
    transmission = 1 - omega * dark_channel / atmospheric_light.max()  # Use max atmospheric light value for scaling
    transmission = np.clip(transmission, t_min, 1)
    
    transmission = cv2.dilate(transmission, np.ones((size, size), np.uint8))
    
    # Reshape atmospheric light to match the image dimensions
    atmospheric_light_reshaped = atmospheric_light.reshape(1, 1, 3)
    
    # Recover the scene radiance
    recovered_img = (img - atmospheric_light_reshaped) / np.expand_dims(transmission, axis=2) + atmospheric_light_reshaped
    recovered_img = np.clip(recovered_img, 0, 255).astype(np.uint8)
    
    return recovered_img


# Main defogging function using Dark Channel Prior
def defog_using_dcp(img):
    img_float = img.astype(np.float64)

    dark = dark_channel(img_float)
    atmospheric_light = estimate_atmospheric_light(img_float, dark)
    recovered_img = recover_scene(img_float, dark, atmospheric_light)

    return recovered_img

# CLAHE method for defogging
def defog_using_clahe(img):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE on the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_img


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

def store_result_and_write_to_csv(image_paths, csv_filename='results.csv'):
    # List to store the results for writing to CSV
    results = []

    for i, img_path in enumerate(image_paths, start=1):
        original_img = cv2.imread(img_path)

        # Check if image is loaded properly
        if original_img is None:
            print(f"Image not found at the path: {img_path}")
            continue

        # Apply both defogging methods
        result_dcp = defog_using_dcp(original_img)
        result_clahe = defog_using_clahe(original_img)

        # Calculate PSNR for both methods
        psnr_dcp = calculate_psnr(original_img, result_dcp)
        psnr_clahe = calculate_psnr(original_img, result_clahe)

        # Calculate SSIM for both methods
        ssim_dcp = calculate_ssim(original_img, result_dcp)
        ssim_clahe = calculate_ssim(original_img, result_clahe)

        # Append the results to the list
        results.append({
            "image": img_path,
            "psnr_dcp": psnr_dcp,
            "ssim_dcp": ssim_dcp,
            "psnr_clahe": psnr_clahe,
            "ssim_clahe": ssim_clahe
        })

        # Print the results (optional)
        print(f"Image {i}:")
        print(f"PSNR for DCP: {psnr_dcp}")
        print(f"SSIM for DCP: {ssim_dcp}")
        print(f"PSNR for CLAHE: {psnr_clahe}")
        print(f"SSIM for CLAHE: {ssim_clahe}")

    # Write results to CSV file
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["image", "psnr_dcp", "ssim_dcp", "psnr_clahe", "ssim_clahe"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results have been written to {csv_filename}")


# Example usage
image_paths = [f'image{i}.jpg' for i in range(1, 13)]  # List of image paths
store_result_and_write_to_csv(image_paths, 'defogging_results.csv')