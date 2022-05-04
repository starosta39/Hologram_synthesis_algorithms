import numpy as np 
import cv2
import matplotlib.pyplot as plt
import cmath
import time 

class AS_synthesis(object):
    
    def __init__(self, 
                holo_pixel_size,
                distance, 
                wavelength, 
                holo_size,
                error,
                iter_limit, 
                holo_type,
                dynamic_range,
                restored_img_size = None):
        self.dynamic_range = dynamic_range
        self.holo_type = holo_type
        self.name = 'Angular_spectrum'
        self._chirp_factor = None
        self._inv_chirp_factor = None
        self.holo_pixel_size = holo_pixel_size
        self.distance = distance
        self.wavelength = wavelength
        self.holo_size= holo_size
        self.error = error
        self.iter_limit = iter_limit 
        self.error_lists = []
        self.scale = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size**2)       
        
        

        if restored_img_size  is None:
            self.restored_img_size = int(holo_size / 2 )
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

    def del_central_zone(self,input_matrix):
        central_point = int(self.holo_size / 2)
        input_matrix[central_point - 1:central_point + 1, central_point - 1:central_point + 1] = 0
        return input_matrix       

    def matrix_normalization(self, input_matrix):
        if self.holo_type == "phase":
            norm_matrix = np.angle(input_matrix)
            norm_matrix = self.zero_to_two_pi_range(norm_matrix)
            norm_matrix = np.uint8(norm_matrix * 256 / (2 * np.pi)) 
            if self.dynamic_range == "bin":
                norm_matrix = cv2.threshold(norm_matrix, 0, 127, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                norm_matrix = norm_matrix[1]
        elif self.holo_type == "amplitude":
            norm_matrix = np.uint8(input_matrix * 256 / input_matrix.max())
            if self.dynamic_range == "bin":
                norm_matrix = cv2.threshold(norm_matrix, 0, 127, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                norm_matrix = norm_matrix[1]
        return norm_matrix

    @property
    def chirp_factor(self):
        if self._chirp_factor is None:
            self._chirp_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * self.scale * ((i- int(self.holo_size/ 2))**2 + (j - int(self.holo_size/ 2))**2) / self.holo_size
                            for j in range(self.holo_size)
                        ]
                        for i in range(self.holo_size)
                    ])
                )
            ) 
        return self._chirp_factor
     
    @property
    def inv_chirp_factor(self):
        if self._inv_chirp_factor is None:
            self._inv_chirp_factor = np.conj(self.chirp_factor)
        return self._inv_chirp_factor
    
    def angle_spect_transform(self,input_matrix):
        
        transform = np.fft.ifftshift(
            np.fft.fft2(
                np.fft.fftshift(
                    self.chirp_factor * np.fft.ifftshift(
                        np.fft.ifft2(
                            np.fft.ifftshift(
                                input_matrix)
                        )
                    )
                )
            )
        )
        
        return  transform
    
    def inverse_angle_spect_transform(self,input_matrix):
        inv_transform = np.fft.ifftshift(
            np.fft.ifft2(
                np.fft.ifftshift(
                    self.inv_chirp_factor * np.fft.ifftshift(
                        np.fft.fft2(
                            np.fft.fftshift(
                                input_matrix 
                            )
                        )
                    )
                )
            )
        )
        return inv_transform
      
    def initial_approx(self):
        if self.holo_type == "phase":
            return np.exp(1j * np.random.rand(self.holo_size, self.holo_size))
        elif self.holo_type == "amplitude":
            return  np.random.rand(self.holo_size, self.holo_size)  

    
    def prepare_for_angle_spec_transform (self, input_matrix):
        return np.exp(1j * input_matrix * 2 * np.pi / 256)
        

    def img_recovery(self, holo):
        if self.holo_type == "phase":
            rec_img = abs(self.angle_spect_transform(self.prepare_for_angle_spec_transform(holo)))
        elif self.holo_type == "amplitude":
            rec_img = abs(self.angle_spect_transform(holo))
            # rec_img = self.del_central_zone(rec_img)
        rec_img  = rec_img**2   
        plt.imshow(rec_img, cmap = "gray")
        plt.show()

        self.informative_img_zone = self.informative_zone(rec_img)
        plt.imshow(self.informative_img_zone, cmap = "gray")
        plt.show()

    # function for translating an argument from a range (-pi,pi] to  [0 2*pi)
    def zero_to_two_pi_range(self, phase): 
        return (phase + 2 * np.pi) * (phase < 0) + (phase) * (phase >= 0)
    
    # function for translating an argument from a range [0 2*pi) to  (-pi,pi]
    def pi_to_pi_range(self, phase):
        res = (( phase - 2*np.pi) * (phase > np.pi) 
                + (2 * np.pi + phase) * (phase <= -np.pi) 
                + phase * (phase <= np.pi)*(phase > -np.pi)
                )
        return res

    def calc_error(self, input_matrix, real_img):
        input_matrix = np.float64(input_matrix)
        real_img = np.float64(real_img)
        a = np.sum(input_matrix**2)
        b = np.sum(real_img**2)
        ab = np.sum(input_matrix*real_img)
        return np.sqrt(1-(ab*ab)/(a*b))


    def __call__(self, input_matrix,  control=False, reshaped_img_coord_h_w = (None, None)):
        self.reshaped_img_coord_h, self.reshaped_img_coord_w = reshaped_img_coord_h_w
        error_list = []
        start_time = time.time()
        holo = self.initial_approx()
        img = self.reshape_img_for_syn(input_matrix)
        plt.imshow(img, cmap="gray")
        plt.show()
        i = 0
        error = float("inf") 
        while error > self.error and i < self.iter_limit:
            ref_img = self.angle_spect_transform(holo)
            error = self.calc_error(
                                    abs(self.informative_zone(ref_img))**2,
                                    self.informative_zone(img)
                                    )

            # plt.imshow(abs(ref_img)**2) 
            # plt.show()                      
            ref_img = np.sqrt(img) * np.exp(1j*np.angle(ref_img))
            holo = self.inverse_angle_spect_transform(ref_img)
            if self.holo_type == "phase":
                holo = np.exp(1j * np.angle(holo))
            elif self.holo_type == "amplitude": 
                holo = abs(holo)
            i += 1 
            error_list.append(error)
            print("Iteration ", i, "error ", error) 

        self.iteration = i
        self.error_lists.append(error_list)
        holo = self.matrix_normalization(holo)
        if control:
            self.img_recovery(holo)

        self.holo = holo
        print("--- %s seconds ---" % (time.time() - start_time))
        return holo

transform = AS_synthesis(
    holo_pixel_size = 8e-6,
    distance = 0.1,
    wavelength = 532e-9,
    holo_size = 1024,
    error = 1e-9,
    iter_limit = 20,
    restored_img_size = 400,
    holo_type = "amplitude",
    dynamic_range = 'bin'
)



img = cv2.imread("C:\\Users\\minik\\Desktop\\lena.jpg", cv2.IMREAD_GRAYSCALE)
holo = transform(img, reshaped_img_coord_h_w = (100, 100), control=True )
cv2.imwrite("C:\\Users\\minik\\Desktop\\lena_holo.bmp", holo)



# plt.imshow(holo, cmap = "gray")
# plt.show() 