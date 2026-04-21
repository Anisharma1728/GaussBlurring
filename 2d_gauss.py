
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from qiskit import QuantumCircuit
from qiskit.circuit.library import *
from qiskit.quantum_info import Statevector

# For each standard deviation compare truncation
# For Gray scale image 
#One Single QFT
#CIRCULAR OR linear convo
# single 1d kernel to reduce no. of MCRy gates and no. of qubits
'''def truncate_kernel_fft(kernel_fft_2d, threshold=1e-3):
    """
    Point-wise compares each element of the FFT kernel to a threshold.
    Returns a new kernel where elements below the threshold are exactly 0.0.
    """
    # Find max to normalize the threshold comparison (so 1e-3 is universally scaled)
    max_val = np.max(np.abs(kernel_fft_2d))
    
    # Point-wise condition: True if (magnitude / max) > threshold
    condition = (np.abs(kernel_fft_2d) / max_val) > threshold
    
    # np.where applies the condition to every element: 
    # keep original value if True, otherwise set to 0.0
    truncated_kernel = np.where(condition, kernel_fft_2d, 0.0)
    
    return truncated_kernel'''
def run_spatial_truncation_sweep():
    
    qubits_per_dim = 4
    dim_size = 2**qubits_per_dim
    total_index_qubits = qubits_per_dim * 2
    N_total = 2**total_index_qubits
    
    image = np.zeros((dim_size, dim_size))
    image[4:12, 4:12] = 10.0  
    
    # Keeping sigma constant to see the effect of the spatial cutoff
    sigma = 1.5 
    
    x = np.arange(-dim_size//2, dim_size//2)
    y = np.arange(-dim_size//2, dim_size//2)
    x_grid, y_grid = np.meshgrid(x, y)
    radius_grid = np.sqrt(x_grid**2 + y_grid**2)
    raw_kernel = np.exp(-(radius_grid**2) / (2 * (sigma**2)))

    flat_image = image.flatten()
    norm_image = np.linalg.norm(flat_image)
    flat_image_normalized = flat_image / norm_image
    
    cutoffs = [1, 2, 3, 4, 5]
    results = []

    print(f"{'Cutoff':<8} | {'Gates Applied (No Threshold)':<30}")
    print("-" * 40)

    for k in cutoffs:
        # 1. Spatial Truncation
        cutoff_radius = k * sigma
        mask = radius_grid <= cutoff_radius
        truncated_kernel = raw_kernel * mask
        truncated_kernel = truncated_kernel / np.sum(truncated_kernel)
        kernel = np.fft.ifftshift(truncated_kernel)
        
        target_h_2d = np.real(ifft2(fft2(image) * fft2(kernel)))

        # 2. Quantum Encoding 
        kernel_fft_2d = fft2(kernel)
        flat_kernel_fft = kernel_fft_2d.flatten()
        max_k_hat = np.max(np.abs(flat_kernel_fft))
        k_hat_encoded = flat_kernel_fft / max_k_hat

        mags = np.clip(np.abs(k_hat_encoded), 0.0, 1.0) 
        thetas = np.arcsin(mags)
        
        ctrl_strs = [bin(i)[2:].zfill(total_index_qubits) for i in range(N_total)]

        # 3. Build Circuit
        qc = QuantumCircuit(total_index_qubits + 1, total_index_qubits + 1)
        row_qubits = list(range(qubits_per_dim))            
        col_qubits = list(range(qubits_per_dim, total_index_qubits)) 
        all_index_qubits = row_qubits + col_qubits           
        ancilla = total_index_qubits                        

        padded_image = np.zeros(2**(total_index_qubits + 1))
        padded_image[:N_total] = flat_image_normalized
        qc.initialize(padded_image, qc.qubits)

        qc.append(QFTGate(qubits_per_dim), row_qubits)
        qc.append(QFTGate(qubits_per_dim), col_qubits)

        gates_applied = 0
        
        # EXACT REQUEST: No 1e-3 frequency threshold. We apply the gate for all non-zero frequencies.
        # We use 1e-12 purely to avoid adding literal 0.0 rotation gates which do nothing mathematically.
        for i in range(N_total):
            if thetas[i] > 1e-12: 
                mcry = RYGate(2 * thetas[i]).control(total_index_qubits, ctrl_state=ctrl_strs[i])
                qc.append(mcry, all_index_qubits + [ancilla])
                gates_applied += 1

        qc.append(QFTGate(qubits_per_dim).inverse(), col_qubits)
        qc.append(QFTGate(qubits_per_dim).inverse(), row_qubits)

        # 4. Statevector Retrieval
        final_state = Statevector(qc)
        amplitudes = np.array(final_state)

        ancilla_1_amplitudes = np.real(amplitudes[N_total:])
        reconstructed_flat = ancilla_1_amplitudes * norm_image * max_k_hat
        reconstructed_2d = reconstructed_flat.reshape((dim_size, dim_size))
        
        print(f"{k} sigma  | {gates_applied} / {N_total}")
        
        results.append((np.fft.fftshift(kernel), target_h_2d, reconstructed_2d))

    # =========================================================
    # 5. Plotting the Comparison Grid
    # =========================================================
    fig, axs = plt.subplots(len(cutoffs), 3, figsize=(15, 20))
    fig.suptitle(f'Spatial Kernel Truncation Comparison (σ\sigma = {sigma})', fontsize=16)

    for i, k in enumerate(cutoffs):
        kernel_vis, target, recon = results[i]
        vmax_val = np.max(target)
        
        im0 = axs[i, 0].imshow(kernel_vis, cmap='magma')
        axs[i, 0].set_title(f'Truncated Kernel ({k}σ\sigma)')
        fig.colorbar(im0, ax=axs[i, 0], fraction=0.046, pad=0.04)

        im1 = axs[i, 1].imshow(target, cmap='viridis', vmin=0, vmax=vmax_val)
        axs[i, 1].set_title(f'Classical Blur ({k}σ\sigma)')
        fig.colorbar(im1, ax=axs[i, 1], fraction=0.046, pad=0.04)

        im2 = axs[i, 2].imshow(recon, cmap='viridis', vmin=0, vmax=vmax_val)
        axs[i, 2].set_title(f'Quantum Blur ({k}σ\sigma)')
        fig.colorbar(im2, ax=axs[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95) 
    plt.show()

if __name__ == "__main__":
    run_spatial_truncation_sweep()
