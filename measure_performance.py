import pandas as pd

# Function to compute metrics from CSV and display results
def compute_metrics_and_display_table(csv_filename='defogging_results.csv'):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_filename)

    # Initialize a results dictionary to hold the metrics
    results = {
        "Metric": ["Average PSNR", "Max PSNR", "Min PSNR", "Median PSNR", "Std Dev PSNR",
                   "Average SSIM", "Max SSIM", "Min SSIM", "Median SSIM", "Std Dev SSIM"],
        "DCP": [
            data['psnr_dcp'].mean(),
            data['psnr_dcp'].max(),
            data['psnr_dcp'].min(),
            data['psnr_dcp'].median(),
            data['psnr_dcp'].std(),
            data['ssim_dcp'].mean(),
            data['ssim_dcp'].max(),
            data['ssim_dcp'].min(),
            data['ssim_dcp'].median(),
            data['ssim_dcp'].std()
        ],
        "CLAHE": [
            data['psnr_clahe'].mean(),
            data['psnr_clahe'].max(),
            data['psnr_clahe'].min(),
            data['psnr_clahe'].median(),
            data['psnr_clahe'].std(),
            data['ssim_clahe'].mean(),
            data['ssim_clahe'].max(),
            data['ssim_clahe'].min(),
            data['ssim_clahe'].median(),
            data['ssim_clahe'].std()
        ]
    }

    # Convert the results into a DataFrame for better display
    results_df = pd.DataFrame(results)

    # Display the results table
    print(results_df)

# Example usage
compute_metrics_and_display_table('defogging_results.csv')
