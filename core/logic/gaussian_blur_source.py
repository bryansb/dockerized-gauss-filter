from pycuda.compiler import SourceModule

import numpy as np
import scipy.stats as st
import math

def get_gaussian_global_func(w, h, ksize=3):
    n = w * h
    name = 'apply_gaussian_blur'
    kernel_code_template = """
    #include <math.h>

    __device__ bool get_submatrix(int x, int y, int current, int k_size, 
        int *p_image, int p_subimage[%(SIZE)s][%(SIZE)s]) 
    {   
        int i_aux = 0;
        int j_aux = 0;
        int k_len = trunc(k_size/2.0);

        for (int i = -k_len; i <= k_len; i++) {
            int x_i = x + i;
            j_aux = 0;
            if (x_i < 0 || x_i > %(ROW)s) {
                return false;
            }
            for (int j = -k_len; j <= k_len; j++) {
                int y_i = y + j;
                int offset = x_i + y_i * current;
                if (y_i < 0 || y_i > %(COL)s) {
                    return false;
                }
                if (offset >= 0 && offset < %(N)s) {
                    p_subimage[j_aux][i_aux] = p_image[offset];
                } else {
                    return false;
                }
                j_aux += 1;
            }
            i_aux += 1;
        }
        
        return true;
    }

    __global__ void apply_gaussian_blur(int *p_image, double *p_kernel, int *output)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
	    int y = threadIdx.y + blockIdx.y * blockDim.y;
        
	    int offset = x + y * blockDim.x * gridDim.x;

        if ( ( offset < %(N)s) ) {
            int p_subimage [%(SIZE)s][%(SIZE)s];
            bool got_it = get_submatrix(x, y, (blockDim.x * gridDim.x), %(SIZE)s, p_image, p_subimage);

            int i_k = 0;
            float pixel = 0;
            if (got_it) {
                for (int i = 0; i < %(SIZE)s; i++) {
                    for (int j = 0; j < %(SIZE)s; j++) {
                        pixel += ((float)((float)(p_subimage[i][j]) * p_kernel[i_k]));
                        i_k += 1;
                    }
                }
                output[offset] = pixel > 255 ? 255 : pixel;
            } else {
                output[offset] = p_image[offset];
            }
        }
    }
    """
    kernel_code = kernel_code_template % {
        'SIZE': str(ksize),
        'N': str(n),
        'COL': w,
        'ROW': h
    }
    return (SourceModule(kernel_code), name)

def get_gaussian_kernel(kernlen=3, nsig=1):
    """
    Returns a 2D Gaussian kernel array.
    Source: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    kernel = kern2d/kern2d.sum()
    return kernel.astype(np.double)

def get_gaussian_kernel_x3():
    return np.array(
        [[1, 2, 1],
        [2, 4, 2], 
        [1, 2, 1]]
    )

def get_gaussian_kernel_x5():
    return np.array(
        [[1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4], 
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]]
    )