import pycuda.driver as cuda
# import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context

import numpy as np
import time
from PIL import Image

from . import gaussian_blur_source as gbs

import os
from django.conf import settings

MAX_BLOCK_SIZE = 16

OUTPUT_IMAGE_RESULT = 'core/static/img/results'
URL_IMAGE_RESULT = '/static/img/results'

gdim = None
bdim = None
kernel = None
sigma = None
pathName = None
image_size = None

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def imread(path):
    image = np.array(Image.open(path), dtype=np.uint8)
    return (image, image.shape)

def save_image(image, name='gauss_image.png'):
    image = Image.fromarray(image)
    image.save(name)

def split_channels(img):
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]
    return (r_channel.astype(np.int32), g_channel.astype(np.int32), b_channel.astype(np.int32))

def merge_channels(r_channel, g_channel, b_channel, img_size=(1920, 1080)):
    img = np.array( [ [ np.zeros(3, dtype=np.uint8) ] * img_size[1] ] * img_size[0] )
    img[:, :, 0] = r_channel[:]
    img[:, :, 1] = g_channel[:]
    img[:, :, 2] = b_channel[:]
    return img

def apply_gauss_cuda(channel, image_size, kernel):
    
    cuda.init()
    device = cuda.Device(0) 
    ctx = device.make_context() 
    print('[INFO] Device init')

    start_time = time.time()
    # Reserva y copia los datos en memoria
    channel_gpu = cuda.mem_alloc(channel.nbytes)
    cuda.memcpy_htod(channel_gpu, channel)

    # Reserva y copia el kernel gausiano
    kernel_gpu = cuda.mem_alloc(kernel.nbytes)
    cuda.memcpy_htod(kernel_gpu, kernel)

    # Reserva y copia la imagen filtrada
    channel_out = np.empty_like(channel)
    channel_out_gpu = cuda.mem_alloc(channel_out.nbytes)

    # Carga la función correspondiente
    mod, name = gbs.get_gaussian_global_func(image_size[0], image_size[1], kernel.shape[0])
    func = mod.get_function(name)

    # Ejecuta las funciones en Cuda
    bdim = (MAX_BLOCK_SIZE, MAX_BLOCK_SIZE, 1)
    gdim = (round(image_size[1]/MAX_BLOCK_SIZE), round(image_size[0]/MAX_BLOCK_SIZE))
    
    func(channel_gpu, kernel_gpu, channel_out_gpu, block=bdim, grid=gdim)

    # Copia desde el device la imagen filtrada
    cuda.memcpy_dtoh(channel_out, channel_out_gpu)

    end_time = time.time()
    final_time = end_time - start_time

    channel_gpu.free()
    kernel_gpu.free()
    channel_out_gpu.free()
    ctx.pop()

    return channel_out, final_time, bdim, gdim

def run_with_cuda(path, kernel_size_i, sigma_value, name='gauss_image_cuda.png'):
    global gdim, bdim, kernel, sigma, pathName, image_size
    kernel_size = kernel_size_i
    sigma = sigma_value
    
    # lee una imagen y las separa en canales
    path = filter_relative_path(path)
    image, image_size = imread( os.path.join(settings.BASE_DIR, "core", path) )
    # image = image.astype(np.int32)
    
    #print('IMG SHAPE:', image.shape)
    image_r, image_g, image_b = split_channels(image)
    
    
    kernel = gbs.get_gaussian_kernel(kernlen=kernel_size, nsig=sigma)
    #print(kernel)
    
    # image_out, final_time = apply_gauss_cuda(image, image_size, kernel)
    #print(image_out)
    image_r_out, time_r, bdim, gdim = apply_gauss_cuda(image_r, image_size, kernel)
    image_g_out, time_g, bdim, gdim = apply_gauss_cuda(image_g, image_size, kernel)
    image_b_out, time_b, bdim, gdim = apply_gauss_cuda(image_b, image_size, kernel)

    final_time = time_r + time_g + time_b

    print('[INFO] Process finished')
    
    image = merge_channels(image_r_out, image_g_out, image_b_out, image_size)
    file_name = get_image_name(path)
    print('[INFO] File Name: ', file_name)

    pathName = os.path.join(settings.BASE_DIR, OUTPUT_IMAGE_RESULT, file_name)
    
    save_image(image, name=pathName)
    print('[INFO]', f'{image_size[0]},{image_size[1]},{final_time}')

    pathURL= os.path.join(URL_IMAGE_RESULT, file_name)
    return  {
                "imgUrl": pathURL, 
                "final_time": final_time,
                "kernel": kernel_size, 
                "sigma": sigma_value, 
                "gdim": gdim, 
                "bdim": bdim
            }

def get_image_name(path):
    spt = path.split('/')
    return spt[len(spt) - 1]

def filter_relative_path(path):
    if path[0] == '/' or path[0] == '\\':
        return path[1:]
    else:
        return path
def get_gauss_value(channel, row, col, kernel):
    k_len = int(kernel.shape[0]/2)
    start_row = row - k_len
    end_row = row + k_len

    start_col = col - k_len
    end_col = col + k_len

    subimage = None
    if start_row >= 0 and end_row < channel.shape[0] \
        and start_col >= 0 and end_col < channel.shape[1]:
        subimage = channel[start_row:end_row + 1, start_col:end_col + 1]

        subimage = subimage * kernel
    return np.sum(subimage)

def apply_gauss_cpu(channel, image_size, kernel):
    k_size = kernel.shape[0]
    channel_out = np.empty_like(channel)

    start_time = time.time()
    for row in range(channel.shape[0]):
        for col in range(channel.shape[1]):
            value = get_gauss_value(channel, row, col, kernel)
            if value != None:
                channel_out[row, col] = value
            else:
                channel_out[row, col] = channel[row, col]
    end_time = time.time()

    final_time = end_time - start_time

    return channel_out, final_time

def run_with_cpu(path, kernel_size, sigma_value, name='gauss_image_cpu.png'):
    image, image_size = imread(path)
    image = image.astype(np.int32)
    #print('IMG SHAPE:', image.shape)
    #image_r, image_g, image_b = split_channels(image)

    kernel = gbs.get_gaussian_kernel(kernlen=kernel_size, nsig=sigma_value)

    image_out, final_time = apply_gauss_cpu(image, image_size, kernel)
    #image_r, time_r = apply_gauss_cpu(image_r, image_size, kernel)
    #image_g, time_g = apply_gauss_cpu(image_g, image_size, kernel)
    #image_b, time_b = apply_gauss_cpu(image_b, image_size, kernel)

    #final_time = time_r + time_g + time_b

    #image = merge_channels(image_r, image_g, image_b, image_size)
    #pathName = 'images_results/'+name
    pathName = 'images_results_gray/'+name
    save_image(image_out, name=pathName)
    print(f'{image_size[0]},{image_size[1]},{final_time}')
    return image

create_dir(os.path.join(settings.BASE_DIR, OUTPUT_IMAGE_RESULT))