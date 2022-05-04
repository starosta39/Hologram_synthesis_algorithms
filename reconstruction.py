import numpy as np 
import cv2
import matplotlib.pyplot as plt
def prepare_for_frenel ( input_matrix):
        return np.exp(1j * input_matrix * 2 * np.pi / 256)


def frenel_transform (pixel_holo_size_h, pixel_holo_size_w, distance, wavelenght, input_matrix):
    N_h, N_w = input_matrix.shape
    scale_h = distance * wavelenght / ( N_h * pixel_holo_size_h**2)
    scale_w = distance * wavelenght / ( N_w * pixel_holo_size_w**2)
    first_frenel_factor = np.exp(
                    np.array([
                        [
                            1j * np.pi * (scale_h * (i- int(N_h / 2))**2 / N_h  
                            + scale_w * (j - int(N_w / 2))**2 / N_w)
                            for j in range(N_w)
                        ]
                        for i in range(N_h)
                    ])
                ) 
    second_frenel_factor = np.exp(
                    np.array([
                        [
                            1j * np.pi * ((i- int(N_h / 2))**2 /( N_h * scale_h) 
                            + (j - int(N_w / 2))**2 / (N_w * scale_w))
                            for j in range(N_w)
                        ]
                        for i in range(N_h)
                    ])
                )     
    fourier = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(input_matrix * first_frenel_factor)))
    return fourier * second_frenel_factor

def inverse_frenel_transform (pixel_holo_size_h, pixel_holo_size_w, distance, wavelenght, input_matrix):
    N_h, N_w = input_matrix.shape
    scale_h = distance * wavelenght / ( N_h * pixel_holo_size_h**2)
    scale_w = distance * wavelenght / ( N_w * pixel_holo_size_w**2)
    inverse_first_frenel_factor = np.exp(
                    np.array([
                        [
                            -1j * np.pi * ((i- int(N_h / 2))**2 /( N_h * scale_h) 
                            + (j - int(N_w / 2))**2 / (N_w * scale_w))
                            for j in range(N_w)          
                        ]
                        for i in range(N_h)
                    ])
                ) 
    inverse_second_frenel_factor = np.exp(
                    np.array([
                        [
                            -1j * np.pi * (scale_h * (i- int(N_h / 2))**2 / N_h  
                            + scale_w * (j - int(N_w / 2))**2 / N_w)
                            for j in range(N_w)
                        ]
                        for i in range(N_h)
                    ])
                ) 

    inv_fourier = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(input_matrix * inverse_first_frenel_factor)))
    return inv_fourier * inverse_second_frenel_factor

def angle_spect_transform(pixel_holo_size_h, pixel_holo_size_w, distance, wavelength, input_matrix):
    N_h, N_w = input_matrix.shape
    scale_h = distance * wavelength / ( N_h * pixel_holo_size_h**2)
    scale_w = distance * wavelength / ( N_w * pixel_holo_size_w**2)
    chirp_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * (scale_h * (i- int(N_h / 2))**2 / N_h  
                            + scale_w * (j - int(N_w / 2))**2 / N_w)
                            for j in range(N_w)
                        ]
                        for i in range(N_h)
                    ])
                )
            )
    transform = np.fft.ifftshift(
            np.fft.fft2(
                np.fft.fftshift(
                    chirp_factor * np.fft.ifftshift(
                        np.fft.ifft2(
                            np.fft.ifftshift(
                                            input_matrix    
                            )
                        )
                    )
                )
            )
    )
        
    return  transform

def inverse_angle_spect_transform(pixel_holo_size_h, pixel_holo_size_w, distance, wavelength, input_matrix):
    N_h, N_w = input_matrix.shape
    scale_h = distance * wavelength / ( N_h * pixel_holo_size_h**2)
    scale_w = distance * wavelength / ( N_w * pixel_holo_size_w**2)
    inverse_chirp_factor = (
                np.exp(
                    np.array([
                        [
                            -1j * np.pi * (scale_h * (i- int(N_h / 2))**2 / N_h  
                            + scale_w * (j - int(N_w / 2))**2 / N_w)
                            for j in range(N_w)
                        ]
                        for i in range(N_h)
                    ])
                )
    )
    inverse_transform = np.fft.ifftshift(
            np.fft.fft2(
                np.fft.fftshift(
                    inverse_chirp_factor * np.fft.ifftshift(
                        np.fft.ifft2(
                            np.fft.ifftshift(
                                            input_matrix    
                            )
                        )
                    )
                )
            )
    )
        
    return  inverse_transform

holo = cv2.imread("C:\\Users\\minik\\Desktop\\lena_holo.bmp", cv2.IMREAD_GRAYSCALE)
pixel_holo_size_h = 8e-6
pixel_holo_size_w = 8e-6
dist = 0.5
wavelenght =532e-9

img = abs(angle_spect_transform(pixel_holo_size_h, pixel_holo_size_w, dist, wavelenght, prepare_for_frenel(holo)))**2
# img[512-300:512+300, 512-300:512+300] = 0
plt.plot(img)
plt.show()
plt.imshow(img * 255/img.max(), cmap="gray")
plt.show()