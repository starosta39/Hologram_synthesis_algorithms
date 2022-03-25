import numpy as np 
import cv2
import matplotlib.pyplot as plt
import cmath
import time 
class GS_synthesis(object):
    
    def __init__(self,  holo_pixel_size, distance, wavelength, input_matrix_size, error, iter_limit, img_size_for_syn = None):
        
        self.name = 'Gerchberg_Sexton'
        self._first_frenel_factor = None
        self._second_frenel_factor = None
        self._first_inv_frenel_factor = None
        self._second_inv_frenel_factor = None
        self.holo_pixel_size = holo_pixel_size
        self.distance = distance
        self.wavelength = wavelength
        self.input_matrix_size = input_matrix_size 
        self.error = error
        self.iter_limit = iter_limit 
        self.error_lists = []
        self.img_pixel_size = self.wavelength * self.distance / (self.input_matrix_size * self.holo_pixel_size)
        self.scale = self.img_pixel_size / self.holo_pixel_size
        
        if img_size_for_syn  is None:
            self.img_size_for_syn = self.input_matrix_size - int(self.input_matrix_size/2)
        elif self.input_matrix_size % self.reshape_img_size != 0:
            print("Некорректный ввод  нового размера изображения, исходный размер  должен делиться на него без остатка")
            exit()
        else: self.img_size_for_syn = img_size_for_syn
            
                
    def reshape_img_for_syn(self, input_matrix):
        print(self.img_size_for_syn)
        new_img = np.zeros((self.input_matrix_size, self.input_matrix_size))
        reshape_img = cv2.resize(input_matrix, (self.img_size_for_syn,self.img_size_for_syn))
        index = int((self.input_matrix_size - self.img_size_for_syn) / 2)
        new_img[index:index + self.img_size_for_syn, index:index + self.img_size_for_syn] = reshape_img
        self.reshape_img = reshape_img
        self.new_img = new_img
        return new_img

    def matrix_normalization(self, input_matrix):
        norm_matrix = np.zeros((self.input_matrix_size, self.input_matrix_size)) 
        for i in range (self.input_matrix_size):
            for j in range (self.input_matrix_size):
                element = cmath.phase(input_matrix[i,j])
                if element < 0:
                    element = 2 * np.pi + element
                element = element/(2 * np.pi) * 256
                norm_matrix[i,j] = element
        return norm_matrix.astype('uint8')

        
    

    @property
    def first_frenel_factor(self):
        if self._first_frenel_factor is None:
            self._first_frenel_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * self.scale * ((i- int(self.input_matrix_size / 2))**2 + (j - int(self.input_matrix_size / 2))**2) / self.input_matrix_size
                            for j in range(self.input_matrix_size)
                        ]
                        for i in range(self.input_matrix_size)
                    ])
                )
            ) 

        return self._first_frenel_factor

    @property
    def second_frenel_factor(self):
        if self._second_frenel_factor is None:
            self._second_frenel_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * ((i- int(self.input_matrix_size / 2))**2 + (j - int(self.input_matrix_size / 2))**2) / (self.input_matrix_size * self.scale)
                                   for j in range(self.input_matrix_size)
                        ] 
                        for i in range(self.input_matrix_size)
                    ])
                )
            )
        return self._second_frenel_factor
      
    @property
    def first_inv_frenel_factor(self):
        if self._first_inv_frenel_factor is None:
            self._first_inv_frenel_factor = np.exp(
                np.array([
                    [
                        -1j * np.pi * ((i- int(self.input_matrix_size / 2))**2 + (j - int(self.input_matrix_size / 2))**2)/(self.scale * self.input_matrix_size)
                        for j in range(self.input_matrix_size)
                    ]
                    for i in range(self.input_matrix_size)
                ])
            )
        return self._first_inv_frenel_factor

    @property
    def second_inv_frenel_factor(self):
        if self._second_inv_frenel_factor is None:
            self._second_inv_frenel_factor = np.exp(
                np.array([
                    [
                        -1j * np.pi * self.scale *  ((i - int(self.input_matrix_size / 2))**2 + (j - int(self.input_matrix_size / 2))**2) / (self.input_matrix_size)
                        for j in range(self.input_matrix_size)
                    ]
                    for i in range(self.input_matrix_size)
                ])
            )
            

        return self._second_inv_frenel_factor

    def frenel_transform(self,input_matrix):
        fourier = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(input_matrix * self.first_frenel_factor)))
        return  self.second_frenel_factor * fourier
    
    def inverse_frenel_transform(self,input_matrix):
        inv_fourier = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(input_matrix * self.first_inv_frenel_factor)))
        return self.second_inv_frenel_factor * inv_fourier
    
    
    def initial_approx(self):
        return np.exp(1j * np.random.rand(self.input_matrix_size, self.input_matrix_size))
    
    def prepare_for_frenel (self, input_matrix):
        return np.exp(1j * input_matrix * 2 * np.pi / 255)

    def calc_error(self, input_matrix, real_img):
        input_matrix = np.float64(abs(input_matrix))
        real_img = np.float64(abs(real_img))
        a = np.sum(input_matrix**2)
        b = np.sum(real_img**2)
        ab = np.sum(input_matrix*real_img)
        return np.sqrt(1-(ab*ab)/(a*b))

    def img_recovery(self, holo):
        rec_img = abs(self.frenel_transform(self.prepare_for_frenel(holo)))
        self.recovery_img  = rec_img   
        index = int((self.input_matrix_size - self.img_size_for_syn) / 2)
        informative_img_zone = rec_img[index:index + self.img_size_for_syn, index:index + self.img_size_for_syn]
        self.informative_img_zone = informative_img_zone
        
    def __call__(self, input_matrix):
        error_list = []
        start_time = time.time()
        holo = self.initial_approx()
        img = self.reshape_img_for_syn(input_matrix)
        error = float('inf')
        i = 0
        while error > self.error and i < self.iter_limit:
            ref_img = self.frenel_transform(holo)
            error = self.calc_error(ref_img, img)
            ref_img = np.sqrt(img) * np.exp(1j*np.angle(ref_img))
            holo = self.inverse_frenel_transform(ref_img)
            holo = np.exp(1j*np.angle(holo))
            i += 1 
            error_list.append(error)
            print("Iteration ", i, "error ", error)
            
        self.iteration = i
        self.error_lists.append(error_list)
        holo = self.matrix_normalization(holo)
        self.img_recovery(holo)
        self.holo = holo
        print("--- %s seconds ---" % (time.time() - start_time))
        return holo




# img_1 = cv2.imread("C:\\Users\\minik\\Desktop\\timur_4.jpg", cv2.IMREAD_GRAYSCALE)
# img_3 = cv2.imread("C:\\Users\\minik\\Desktop\\hello.jpg", cv2.IMREAD_GRAYSCALE)

# transform = GS_synthesis(
#     holo_pixel_size = 8e-6,
#     distance = 10,
#     wavelength = 532e-9,
#     input_matrix_size = 1024,
#     error = 1e-9,
#     iter_limit = 50
# )

# holo_1 = transform(img_1)
# holo_3 = transform(img_3)

