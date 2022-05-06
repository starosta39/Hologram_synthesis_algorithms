import numpy as np 
import cv2
import matplotlib.pyplot as plt
import time 

class BP_synthesis(object): 
    def __init__(self,
                 del_area,
                 wavelength, 
                 holo_size, 
                 holo_type,
                 rand_phase_mask,                 
                 dynamic_range,
                 near_zone = True,
                 distance = None,
                 holo_pixel_size = None
                ):
        self.rand_phase_mask = rand_phase_mask
        self.dynamic_range = dynamic_range
        self.del_area = del_area
        self.holo_type = holo_type
        self.name = 'Bipolar_intensity'
        self.near_zone = near_zone
        self.holo_size = holo_size
        self.distance = distance
        self.wavelength = wavelength
        self.error_list = []
        
        if near_zone:
            self._first_fresnel_factor = None
            self._second_fresnel_factor = None
            self._first_inv_fresnel_factor = None
            self._second_inv_fresnel_factor = None
            self.holo_pixel_size =  holo_pixel_size
            self.scale_h = self.wavelength * self.distance / ( self.holo_size[0] * self.holo_pixel_size[0]**2)
            self.scale_w = self.wavelength * self.distance / ( self.holo_size[1] * self.holo_pixel_size[1]**2)
        
            
            
    def reshape_img_for_syn(self, input_matrix, restored_img_size, reshaped_img_position_coord_h_w):

        if restored_img_size  is None:
            print("Incorre input, the argument 'restored_img_size' must't be None" ) 
            exit()
        else: self.restored_img_size = restored_img_size
        
        new_img = np.zeros(self.holo_size)
        reshape_img = cv2.resize(input_matrix, (self.restored_img_size[1],self.restored_img_size[0]))
        new_img[
                reshaped_img_position_coord_h_w[0]:reshaped_img_position_coord_h_w [0]+ self.restored_img_size[0], 
                reshaped_img_position_coord_h_w[1]:reshaped_img_position_coord_h_w[1] + self.restored_img_size[1],
                ] = reshape_img
        self.reshape_img = reshape_img
        self.new_img = new_img 
        return new_img  
        
    
    @property
    def first_fresnel_factor(self):
        if self._first_fresnel_factor is None:
            self._first_fresnel_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * (self.scale_h * (i- int(self.holo_size[0] / 2))**2 / self.holo_size[0] 
                            + self.scale_w * (j - int(self.holo_size[1] / 2))**2 / self.holo_size[1])
                            for j in range(self.holo_size[1])
                        ]
                        for i in range(self.holo_size[0])
                    ])
                ) 
            )
        return self._first_fresnel_factor

    @property
    def second_fresnel_factor(self):
        if self._second_fresnel_factor is None:
            self._second_fresnel_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * ((i- int(self.holo_size[0] / 2))**2 /( self.holo_size[0] * self.scale_h) 
                            + (j - int(self.holo_size[1] / 2))**2 / (self.holo_size[1] * self.scale_w))
                            for j in range(self.holo_size[1])          
                        ]
                        for i in range(self.holo_size[0])
                    ])
                ) 
            )
        return self._second_fresnel_factor
      
    @property
    def first_inv_fresnel_factor(self):
        if self._first_inv_fresnel_factor is None:
            self._first_inv_fresnel_factor = np.exp(
                 np.array([
                        [
                            -1j * np.pi * ((i- int(self.holo_size[0] / 2))**2 /( self.holo_size[0] * self.scale_h) 
                            + (j - int(self.holo_size[1] / 2))**2 / (self.holo_size[1] * self.scale_w))
                            for j in range(self.holo_size[1])          
                        ]
                        for i in range(self.holo_size[0])
                    ])
                )
        return self._first_inv_fresnel_factor

    @property
    def second_inv_fresnel_factor(self):
        if self._second_inv_fresnel_factor is None:
            self._second_inv_fresnel_factor = np.exp(
                np.array([
                        [
                            -1j * np.pi * (self.scale_h * (i- int(self.holo_size[0] / 2))**2 / self.holo_size[0] 
                            + self.scale_w * (j - int(self.holo_size[1] / 2))**2 / self.holo_size[1])
                            for j in range(self.holo_size[1])
                        ]
                        for i in range(self.holo_size[0])
                    ])
                ) 
        return self._second_inv_fresnel_factor
    
    def fresnel_transform(self,input_matrix):
        fourier = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(input_matrix * self.first_fresnel_factor)))
        return  self.second_fresnel_factor * fourier
    
    def inverse_fresnel_transform(self,input_matrix):
        inv_fourier = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(input_matrix * self.first_inv_fresnel_factor)))
        return self.second_inv_fresnel_factor * inv_fourier
    
    def fourier_transform(self,input_matrix):
        return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(input_matrix)))
    
    def inverse_fourier_transform(self,input_matrix):
        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(input_matrix)))
    
    def del_central_zone(self,input_matrix):
        central_point = (int(self.holo_size[0] / 2),int(self.holo_size[1] / 2))
        input_matrix[
            central_point[0] - self.del_area:central_point[0] + self.del_area, central_point[1] - self.del_area:central_point[1] + self.del_area
        ] = 0
        return input_matrix
    
    def informative_zone (self, input_matrix, restored_img_size, reshaped_img_position_coord_h_w):
        res = input_matrix[
                reshaped_img_position_coord_h_w[0]:reshaped_img_position_coord_h_w[0] + restored_img_size[0], 
                reshaped_img_position_coord_h_w[1]:reshaped_img_position_coord_h_w[1] + restored_img_size[1],
                ]
        return res

    def img_recovery(self, holo,restored_img_size, reshaped_img_position_coord_h_w ):
        if self.near_zone:
            if self.holo_type == "phase":
                rec_img = abs(self.inverse_fresnel_transform(self.prepare_for_transform(holo)))
            elif self.holo_type == "amplitude":
                rec_img = abs(self.inverse_fresnel_transform(holo))
            else: 
                print("Incorre input, the argument 'holo_type' must be equal phase/amplitude, but not ", self.holo_type)
                exit()
            rec_img = rec_img**2
            rec_img = self.del_central_zone(rec_img)

            plt.plot(rec_img)
            plt.show()
            plt.imshow(rec_img, cmap = 'gray')
            plt.show()
        else:
            if self.holo_type == "phase":
                rec_img = abs(self.fourier_transform(self.prepare_for_transform(holo)))
            elif self.holo_type == "amplitude":
                rec_img = abs(self.fourier_transform(holo))
            else: 
                print("Incorre input, the argument 'holo_type' must be equal phase/amplitude, but not ", self.holo_type)
            rec_img = self.del_central_zone(rec_img)
            plt.plot(rec_img)
            plt.show()
            plt.imshow(rec_img, cmap = 'gray')
            plt.show()
        self.recovery_img  = rec_img   
        self.informative_img_zone = self.informative_zone(rec_img ,restored_img_size, reshaped_img_position_coord_h_w)
    
    
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
    
    def prepare_for_transform (self, input_matrix):
        return np.exp(1j * input_matrix * 2 * np.pi / 255)
    
    def calc_error(self, input_matrix, real_img):
        input_matrix = np.float64(abs(input_matrix))
        real_img = np.float64(abs(real_img))
        a = np.sum(input_matrix**2)
        b = np.sum(real_img**2)
        ab = np.sum(input_matrix*real_img)
        return np.sqrt(1-(ab*ab)/(a*b))
       
    def phase_mask(self, input_matrix):
        mask = np.int8(np.random.rand(self.holo_size[0],self.holo_size[1]) * 2)       
        return input_matrix * np.exp(1j * np.pi * mask)    
    
    def __call__(
                    self, 
                    input_matrix,
                    control=False,
                    restored_img_size=None,
                    reshaped_img_position_coord_h_w = (None,None)
                ):
        start_time = time.time()
        error = None
        self.reshaped_img_coord_h,self.reshaped_img_coord_w = reshaped_img_position_coord_h_w
        img = self.reshape_img_for_syn(input_matrix, restored_img_size, reshaped_img_position_coord_h_w)

        if control:
           plt.imshow(img, cmap='gray') 
           plt.show()

        if self.rand_phase_mask:
            img  = self.phase_mask(img)
        if self.near_zone:
            holo = self.fresnel_transform(img)
        else :
            holo = self.fourier_transform(img)

        holo = 2 * (np.real(holo) - np.real(holo).min())
        holo = np.uint8(holo * 255 /holo.max())
        if self.dynamic_range == "bin":
            holo = cv2.threshold(holo, 0, 127,  cv2.THRESH_OTSU)[1]
        if control:
            self.img_recovery(holo, restored_img_size, reshaped_img_position_coord_h_w)
            error = self.calc_error(self.informative_img_zone, self.reshape_img)
        print("Error ", error)  
        self.error_list.append(error)
        print("Time --- %s seconds ---" % (time.time() - start_time))
        self.holo  = holo
        return holo
            

transform = BP_synthesis(
del_area = 80,
wavelength = 532e-9,
holo_size = (1024,1024),
near_zone = False,
holo_type = "amplitude",
dynamic_range = 'bin',
rand_phase_mask  = True,
distance = 1.5,
holo_pixel_size = (8e-6,8e-6)   
)
img = cv2.imread("C:\\Users\\minik\\Desktop\\lena.jpg", cv2.IMREAD_GRAYSCALE)
holo = transform(
                    img,
                    restored_img_size=(int(img.shape[0] / 3), int(img.shape[1] / 3)), 
                    reshaped_img_position_coord_h_w=(100,100), 
                    control=True
                )
plt.imshow(holo, cmap = "gray")
plt.show()