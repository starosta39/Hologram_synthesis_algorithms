import numpy as np 
import cv2
import matplotlib.pyplot as plt
import cmath
import time 
class GS_Frenel_synthesis(object):
    
    def __init__(self, 
                 holo_pixel_size, 
                 distance, wavelength, 
                 holo_size, error_dif, 
                 iter_limit, holo_type, 
                 dynamic_range,
                  restored_img_size = None
                  ):

        self.dynamic_range = dynamic_range
        self.holo_type = holo_type
        self.name = 'Gerchberg_Sexton'
        self._first_frenel_factor = None
        self._second_frenel_factor = None
        self._first_inv_frenel_factor = None
        self._second_inv_frenel_factor = None
        self.holo_pixel_size = holo_pixel_size
        self.distance = distance
        self.wavelength = wavelength
        self.holo_size = holo_size 
        self.error_dif= error_dif
        self.iter_limit = iter_limit 
        self.error_lists = []
        self.img_pixel_size = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size)
        self.scale = self.img_pixel_size / self.holo_pixel_size
        
        if restored_img_size  is None:
            self.restored_img_size = int(self.holo_size/2)
        elif self.holo_size % restored_img_size != 0 and self.holo_type == 'phase':
            print("Incorrect input of the new image size, the original size must be divided by it without remainder")
            exit()
        else: self.restored_img_size = restored_img_size
            
                
    def reshape_img_for_syn(self, input_matrix):
        
        new_img = np.zeros((self.holo_size, self.holo_size))
        reshape_img = cv2.resize(input_matrix, (self.restored_img_size,self.restored_img_size))
        if self.holo_type == "phase":
            index = int((self.holo_size - self.restored_img_size) / 2)
            new_img[index:index + self.restored_img_size, index:index + self.restored_img_size] = reshape_img
        elif self.holo_type == "amplitude":
            new_img[
                self.reshaped_img_coord_h:self.reshaped_img_coord_h + self.restored_img_size, 
                self.reshaped_img_coord_w:self.reshaped_img_coord_w + self.restored_img_size,
                ] = reshape_img
        self.reshape_img = reshape_img
        self.new_img = new_img
        return new_img

    def informative_zone (self, input_matrix):
        if self.holo_type == "phase":
            index = int((self.holo_size - self.restored_img_size) / 2)
            res  = input_matrix[index:index + self.restored_img_size, index:index + self.restored_img_size]
        elif self.holo_type == "amplitude":
            res = input_matrix[
                self.reshaped_img_coord_h:self.reshaped_img_coord_h + self.restored_img_size, 
                self.reshaped_img_coord_w:self.reshaped_img_coord_w + self.restored_img_size,
                ]
        return res
        
    # function for translating an argument from a range [0 2*pi) to  (-pi,pi]
    def zero_to_two_pi_range(self, phase): 
        return (phase + 2 * np.pi) * (phase < 0) + (phase) * (phase >= 0)
    
    def matrix_normalization(self, input_matrix):
        if self.holo_type == "phase":
            norm_matrix = np.angle(input_matrix)
            norm_matrix = self.zero_to_two_pi_range(norm_matrix)
            norm_matrix = np.uint8(norm_matrix * 255 / (2 * np.pi)) 
            if self.dynamic_range == "bin":
                norm_matrix = cv2.threshold(norm_matrix, 0, 127, cv2.THRESH_OTSU)
                norm_matrix = norm_matrix[1]
        elif self.holo_type == "amplitude":
            norm_matrix = np.uint8(input_matrix * 255 / input_matrix.max())
            if self.dynamic_range == "bin":
                norm_matrix = cv2.threshold(norm_matrix, 0, 127, cv2.THRESH_OTSU)
                norm_matrix = norm_matrix[1]
        return norm_matrix
 

    @property
    def first_frenel_factor(self):
        if self._first_frenel_factor is None:
            self._first_frenel_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * self.scale * ((i- int(self.holo_size / 2))**2 + (j - int(self.holo_size / 2))**2) / self.holo_size
                            for j in range(self.holo_size)
                        ]
                        for i in range(self.holo_size)
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
                            1j * np.pi * ((i- int(self.holo_size / 2))**2 + (j - int(self.holo_size / 2))**2) / (self.holo_size * self.scale)
                                   for j in range(self.holo_size)
                        ] 
                        for i in range(self.holo_size)
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
                        -1j * np.pi * ((i- int(self.holo_size / 2))**2 + (j - int(self.holo_size / 2))**2)/(self.scale * self.holo_size)
                        for j in range(self.holo_size)
                    ]
                    for i in range(self.holo_size)
                ])
            )
        return self._first_inv_frenel_factor

    @property
    def second_inv_frenel_factor(self):
        if self._second_inv_frenel_factor is None:
            self._second_inv_frenel_factor = np.exp(
                np.array([
                    [
                        -1j * np.pi * self.scale *  ((i - int(self.holo_size / 2))**2 + (j - int(self.holo_size / 2))**2) / (self.holo_size)
                        for j in range(self.holo_size)
                    ]
                    for i in range(self.holo_size)
                ])
            )
        return self._second_inv_frenel_factor

    def zero_to_two_pi_range(self, phase): 
        return (phase + 2 * np.pi) * (phase < 0) + (phase) * (phase >= 0)

    def frenel_transform(self,input_matrix):
        fourier = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(input_matrix * self.first_frenel_factor)))
        return  self.second_frenel_factor * fourier
    
    def inverse_frenel_transform(self,input_matrix):
        inv_fourier = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(input_matrix * self.first_inv_frenel_factor)))
        return self.second_inv_frenel_factor * inv_fourier
    
    
    def initial_approx(self):
        return np.exp(1j * np.random.rand(self.holo_size, self.holo_size))
    
    def prepare_for_frenel (self, input_matrix):
        return np.exp(1j * input_matrix * 2 * np.pi / 256)

    def calc_error(self, input_matrix, real_img):
        input_matrix = np.float64(abs(input_matrix))
        real_img = np.float64(abs(real_img))
        a = np.sum(input_matrix**2)
        b = np.sum(real_img**2)
        ab = np.sum(input_matrix*real_img)
        return np.sqrt(1-(ab*ab)/(a*b))

    def img_recovery(self, holo):
        if self.holo_type == "phase":
            rec_img = abs(self.frenel_transform(self.prepare_for_frenel(holo)))
        elif self.holo_type == "amplitude":
            rec_img = abs(self.frenel_transform(holo))
        rec_img  = rec_img**2   
        self.rec_img = rec_img
        plt.imshow(rec_img, cmap = "gray")
        plt.show()
        self.informative_img_zone = self.informative_zone(rec_img)
        plt.imshow(self.informative_img_zone, cmap = "gray")
        plt.show()

        
    def __call__(self, input_matrix, control, reshaped_img_coord_h_w = (None,None)):
        self.reshaped_img_coord_h, self.reshaped_img_coord_w = reshaped_img_coord_h_w
        error_list = []
        start_time = time.time()
        holo = self.initial_approx()
        img = self.reshape_img_for_syn(input_matrix)
        plt.imshow(img, cmap="gray")
        plt.show()
        error_dif = float('inf')
        last_error_dif = 0
        i = 0
        while error_dif > self.error_dif and i < self.iter_limit:
            ref_img = self.frenel_transform(holo)
            error = self.calc_error(
                                    abs(self.informative_zone(ref_img))**2,
                                    self.informative_zone(img)
                                    )                 
            ref_img = np.sqrt(img) * np.exp(1j*np.angle(ref_img))
            holo = self.inverse_frenel_transform(ref_img)
            if self.holo_type == "phase":
                if self.dynamic_range == "bin":
                    holo = self.zero_to_two_pi_range(np.angle(holo))
                    holo = cv2.threshold(np.uint8(holo * 255 / (2 * np.pi)), 0, 127, cv2.THRESH_OTSU)[1]
                    holo = np.exp(1j * holo * np.pi / 127 )
                else:
                    holo = np.exp(1j * np.angle(holo))
            elif self.holo_type == "amplitude": 
                
                holo = abs(holo)
            i += 1 
            error_list.append(error)
            print("Iteration ", i, "error(informative zone)", error) 
            
        self.iteration = i
        self.error_lists.append(error_list)
        holo = self.matrix_normalization(holo)
        if control:
            self.img_recovery(holo)
        self.holo = holo
        print("--- %s seconds ---" % (time.time() - start_time))
        return holo

transform = GS_Frenel_synthesis(
    holo_pixel_size = 8e-6,
    distance = 0.5,
    wavelength = 532e-9,
    holo_size = 1024,
    error_dif = 1e-4,
    holo_type = 'amplitude',
    dynamic_range = "bin",
    iter_limit = 20,
    restored_img_size=400
)

img = cv2.imread("C:\\Users\\minik\\Desktop\\lena.jpg", cv2.IMREAD_GRAYSCALE)
holo = transform(img, reshaped_img_coord_h_w = (100,100),  control=True)
# cv2.imwrite("C:\\Users\\minik\\Desktop\\lena_holo.bmp", holo)
#  # print(holo)
# # # plt.imshow(holo, cmap = 'gray')
# # plt.show()
# holo = cv2.imread("C:\\Users\\minik\\Desktop\\lena_holo.bmp", cv2.IMREAD_GRAYSCALE)
# plt.imshow((abs(transform.frenel_transform(holo)))**2, cmap = 'gray')
# plt.show()
# # plt.plot(abs(transform.frenel_transform(np.exp(1j*holo)))**2)
# # plt.show()
