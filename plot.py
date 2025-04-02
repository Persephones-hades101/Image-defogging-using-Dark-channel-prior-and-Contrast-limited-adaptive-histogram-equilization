import pandas as pd
import matplotlib.pyplot as plt

# Function to plot PSNR and SSIM graphs from the CSV file
def plot_psnr_ssim(csv_filename):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_filename)

    # Extract image numbers
    image_numbers = range(1, len(data) + 1)

    # Plot PSNR
    plt.figure(figsize=(5, 5))
    plt.plot(image_numbers, data['psnr_dcp'], marker='o', label='DCP', color='blue')
    plt.plot(image_numbers, data['psnr_clahe'], marker='o', label='CLAHE', color='orange')
    plt.title('PSNR Comparison')
    plt.xlabel('Image Number')
    plt.ylabel('PSNR (dB)')
    plt.xticks(image_numbers)
    plt.legend()
    plt.grid()
    plt.show()

    # Plot SSIM
    plt.figure(figsize=(5, 5))
    plt.plot(image_numbers, data['ssim_dcp'], marker='o', label='DCP', color='blue')
    plt.plot(image_numbers, data['ssim_clahe'], marker='o', label='CLAHE', color='orange')
    plt.title('SSIM Comparison')
    plt.xlabel('Image Number')
    plt.ylabel('SSIM')
    plt.xticks(image_numbers)
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
plot_psnr_ssim('defogging_results.csv')
