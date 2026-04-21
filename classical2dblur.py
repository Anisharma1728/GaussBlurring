import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import convolve1d

image = np.zeros((100, 100))

image[0:50, 0:50] = 10.0


kernel_size = 50 # Must be smaller than or equal to image dimensions
sigma = 2.0

def blur_fft(img, k_size, sig):

    ax = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel_2d = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel_2d = kernel_2d / np.sum(kernel_2d) 
    
    padded_kernel = np.zeros_like(img)
    start_x = (img.shape[1] // 2) - (k_size // 2)
    start_y = (img.shape[0] // 2) - (k_size // 2)
    padded_kernel[start_y:start_y+k_size, start_x:start_x+k_size] = kernel_2d
    
    kernel_fft = fft2(ifftshift(padded_kernel))
    img_fft = fft2(img)
    blurred_fft = img_fft * kernel_fft
    
    blurred_img = np.real(ifft2(blurred_fft))
    
    return blurred_img

def blur_separable_fft(img, k_size, sig):
    rows, cols = img.shape
    
    # Create and pad the 1D kernel for rows
    ax_x = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
    kernel_1d_x = np.exp(-0.5 * np.square(ax_x) / np.square(sig))
    kernel_1d_x = kernel_1d_x / np.sum(kernel_1d_x)
    
    padded_kernel_x = np.zeros(cols)
    start_x = (cols // 2) - (k_size // 2)
    padded_kernel_x[start_x : start_x+k_size] = kernel_1d_x
    
    kernel_fft_x = fft(ifftshift(padded_kernel_x))
    
    img_fft_x = fft(img, axis=1)
    
    # Point-wise multiplication
    blurred_fft_x = img_fft_x * kernel_fft_x[np.newaxis, :]
    
    blurred_rows = np.real(ifft(blurred_fft_x, axis=1))

    # Create and pad the 1D kernel for columns
    ax_y = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
    kernel_1d_y = np.exp(-0.5 * np.square(ax_y) / np.square(sig))
    kernel_1d_y = kernel_1d_y / np.sum(kernel_1d_y)
    
    padded_kernel_y = np.zeros(rows)
    start_y = (rows // 2) - (k_size // 2)
    padded_kernel_y[start_y : start_y+k_size] = kernel_1d_y

    kernel_fft_y = fft(ifftshift(padded_kernel_y))
    
    img_fft_y = fft(blurred_rows, axis=0)
    
    # Point-wise multiplication
    blurred_fft_y = img_fft_y * kernel_fft_y[:, np.newaxis]

    final_blurred_img = np.real(ifft(blurred_fft_y, axis=0))
    
    return final_blurred_img

fft_result = blur_fft(image, kernel_size, sigma)
separable_fft_result = blur_separable_fft(image, kernel_size, sigma)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_kwargs = {'cmap': 'viridis', 'interpolation': 'nearest'}

# 1. Original Image
im0 = axes[0].imshow(image, **plot_kwargs, vmin=0, vmax=10)
axes[0].set_title('Original 10x10 Image')
axes[0].axis('off')
fig.colorbar(im0, ax=axes[0])

# 2. Flatten 1D FFT Result
im1 = axes[1].imshow(separable_fft_result, **plot_kwargs)
axes[1].set_title('Flatten 1D FFT Convolution')
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1])

# 3. 2D FFT Result
im2 = axes[2].imshow(fft_result, **plot_kwargs)
axes[2].set_title('Blurred (2D FFT Convolution)')
axes[2].axis('off')
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()