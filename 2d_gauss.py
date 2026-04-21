
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

def run_2d_quantum_gaussian_blur():
    
    qubits_per_dim = 4
    dim_size = 2**qubits_per_dim
    total_index_qubits = qubits_per_dim * 2
    N_total = 2**total_index_qubits
    
    # Create an 16x16 image: A solid square block
    image = np.zeros((dim_size, dim_size))
    image[4:12, 4:12] = 10.0  # 8x8 square of bright pixels
    
    # Create a 2D Gaussian Kernel
    sigma = 5.0
    x = np.arange(-dim_size//2, dim_size//2)
    y = np.arange(-dim_size//2, dim_size//2)
    x_grid, y_grid = np.meshgrid(x, y)
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * (sigma**2)))
    kernel = np.fft.ifftshift(kernel) 

    flat_image = image.flatten()
    norm_image = np.linalg.norm(flat_image)
    flat_image_normalized = flat_image / norm_image
    
    target_h_2d = np.real(ifft2(fft2(image) * fft2(kernel)))

    # ---------------------------------------------------------
    '''kernel_fft_2d_truncated = truncate_kernel_fft(kernel_fft_2d, threshold=1e-3)
    target_h_2d_truncated = np.real(ifft2(fft2(image) * kernel_fft_2d_truncated))

    # Find max of the truncated kernel for quantum amplitude encoding
    max_k_hat = np.max(np.abs(kernel_fft_2d_truncated))'''
    
    kernel_fft_2d = fft2(kernel)
    flat_kernel_fft = kernel_fft_2d.flatten()
    
    max_k_hat = np.max(np.abs(flat_kernel_fft))
    k_hat_encoded = flat_kernel_fft / max_k_hat

    mags = np.abs(k_hat_encoded)
    mags = np.clip(mags, 0.0, 1.0) 
    '''phases = np.angle(k_hat_encoded)'''
    
    thetas = np.arcsin(mags)
    
    ctrl_strs = [bin(k)[2:].zfill(total_index_qubits) for k in range(N_total)]

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
    threshold = 1e-3
    
    for k in range(N_total):
        if thetas[k] > threshold:  # TRUNCATION THRESHOLD
            mcry = RYGate(2 * thetas[k]).control(total_index_qubits, ctrl_state=ctrl_strs[k])
            qc.append(mcry, all_index_qubits + [ancilla])
            gates_applied += 1

    #Phase encoding (PhaseGate) completely removed here because 
    # Gaussian FFT elements are purely real and positive!
    '''qc.x(ancilla)
    for k in range(N_total):
    if np.abs(phases[k]) > 1e-8:
    mcphase = PhaseGate(phases[k]).control(total_index_qubits, ctrl_state=ctrl_strs[k])
    qc.append(mcphase, all_index_qubits + [ancilla])
    qc.x(ancilla)'''

    qc.append(QFTGate(qubits_per_dim).inverse(), col_qubits)
    qc.append(QFTGate(qubits_per_dim).inverse(), row_qubits)

    qc.draw("mpl", filename='circuit.png')
    print(f"Total possible Multi-Controlled RY gates: {N_total}")
    print(f"Gates actually applied: {gates_applied}")
    print(f"Gates skipped due to threshold: {N_total - gates_applied}")
    print(f"Final Circuit Depth: {qc.depth()}")

    print("\nCalculating exact statevector...")
    final_state = Statevector(qc)
    amplitudes = np.array(final_state)

    # Because we used arcsin, our data is now in the Ancilla = |1> state.
    # In Qiskit's little-endian ordering, Ancilla=|1> occupies the exact SECOND half of the array.
    ancilla_1_amplitudes = np.real(amplitudes[N_total:])
    
    reconstructed_flat = ancilla_1_amplitudes * norm_image * max_k_hat
    reconstructed_2d = reconstructed_flat.reshape((dim_size, dim_size))
    
    difference = np.abs(target_h_2d - reconstructed_2d)
    print(f"Maximum absolute error due to truncation: {np.max(difference):.6f}")

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axs[0].imshow(image, cmap='viridis')
    axs[0].set_title(f'Original {dim_size}x{dim_size} Image')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(target_h_2d, cmap='viridis')
    axs[1].set_title('Classical 2D')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    '''im1 = axs[1].imshow(target_h_2d_truncated, cmap='viridis')
    axs[1].set_title('Classical Target (Truncated)')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)'''

    im2 = axs[2].imshow(reconstructed_2d, cmap='viridis')
    axs[2].set_title('Quantum 2D')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # Difference plot will now show the actual truncation error!
    im3 = axs[3].imshow(difference, cmap='magma')
    axs[3].set_title(f'Absolute Difference\n(Max: {np.max(difference):.4f})')
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_2d_quantum_gaussian_blur()