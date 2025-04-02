# Image Defogging using DCP and CLAHE

This repository implements image defogging techniques using **Dark Channel Prior
(DCP)** and **Contrast Limited Adaptive Histogram Equalization (CLAHE)**. The
goal of this project is to enhance the visibility of foggy or hazy images by
applying these methods, which improve image clarity and contrast.

## Methods Used

### 1. **Dark Channel Prior (DCP)**

DCP is a well-known method for single image defogging. It is based on the
observation that the minimum intensity value in a local patch of a foggy image
is non-zero in the absence of fog. The defogging process involves estimating the
atmospheric light and transmission map, which are then used to recover the scene
radiance.

### 2. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**

CLAHE is used for enhancing the contrast of an image. Unlike global histogram
equalization, CLAHE works by dividing the image into small tiles and applying
histogram equalization locally. It is particularly useful in improving the
contrast of the defogged image, especially in low-contrast regions.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
