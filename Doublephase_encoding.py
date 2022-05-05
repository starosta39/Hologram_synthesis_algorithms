import numpy as np 
import cv2
import matplotlib.pyplot as plt
import time 
import cmath
import random


class DE_syntesis(object):

    def __init__(self,  
                 del_area,
                 wavelength, 
                 holo_size, 
                 mod,
                 near_zone = True,
                 distance = None,
                 holo_pixel_size = None
                ):
        self.del_area = del_area 
        self.mod = mod
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
            self.scale = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size**2)
            self.scale_h = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size**2)
            self.scale_w = self.wavelength *  self.distance / (self.holo_size  * 4 *   self.holo_pixel_size**2)
            self.scale_h_inverse = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size**2)
            self.scale_w_inverse = self.wavelength * self.distance / (self.holo_size * 2 * self.holo_pixel_size**2)
        
            
    def reshape_img_for_syn(self, input_matrix, position, restored_img_size, reshaped_img_position_coord_h_w):

        if restored_img_size  is None:
            self.restored_img_size = int(self.holo_size / 2 )
        else: self.restored_img_size = restored_img_size
        
        new_img = np.zeros((self.holo_size, self.holo_size)) 
        reshape_img = cv2.resize(input_matrix, (self.restored_img_size, self.restored_img_size))
        self.reshape_img = reshape_img

        if position == "centre":
            index = int((self.holo_size - self.restored_img_size) / 2)
            new_img[index:index + self.restored_img_size, index:index + self.restored_img_size] = reshape_img     
        elif position == "free":
            new_img[
                reshaped_img_position_coord_h_w[0]:reshaped_img_position_coord_h_w [0]+ self.restored_img_size, 
                reshaped_img_position_coord_h_w[1]:reshaped_img_position_coord_h_w[1] + self.restored_img_size,
                ] = reshape_img
        else:
            print("Incorre input, the argument 'position' must be equal center/free, but not ", position)           
        self.new_img = new_img 
        return new_img
        
    
    @property
    def first_fresnel_factor(self):
        if self._first_fresnel_factor is None and self.near_zone:
            self._first_fresnel_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * (self.scale_h * ((i- int(self.holo_size / 2))**2 + 
                            self.scale_w * (j - int(self.holo_size / 2))**2)) / self.holo_size
                            for j in range(self.holo_size)
                        ]
                        for i in range(self.holo_size)
                    ])
                )
            ) 

        return self._first_fresnel_factor

    @property
    def second_fresnel_factor(self):
        if self._second_fresnel_factor is None and self.near_zone:
            self._second_fresnel_factor = (
                np.exp(
                    np.array([
                        [
                            1j * np.pi * ((i- int(self.holo_size / 2))**2 / self.scale_h + 
                            (j - int(self.holo_size / 2))**2 / self.scale_w) / (self.holo_size)
                                   for j in range(self.holo_size)
                        ] 
                        for i in range(self.holo_size)
                    ])
                )
            )
        return self._second_fresnel_factor
    
    @property
    def first_inv_fresnel_factor(self):
        if self._first_inv_fresnel_factor is None and self.near_zone:
            self._first_inv_fresnel_factor = np.exp(
                np.array([
                    [
                        -1j * np.pi * ((i- int(self.holo_size / 2 ))**2 /(self.scale_h_inverse * self.holo_size)  
                        + (j - int(self.holo_size))**2 /(self.scale_w_inverse * self.holo_size * 2))
                        for j in range(self.holo_size * 2)
                    ]
                    for i in range(self.holo_size)
                ])
            )
            
        return self._first_inv_fresnel_factor

    @property
    def second_inv_fresnel_factor(self):
        if self._second_inv_fresnel_factor is None and self.near_zone:
            self._second_inv_fresnel_factor = np.exp(
                np.array([
                    [
                        -1j * np.pi * (self.scale_h_inverse *(i - int(self.holo_size / 2))**2 /(self.holo_size) 
                        + self.scale_w_inverse * (j - int(self.holo_size ))**2 / (self.holo_size * 2)) 
                        for j in range(self.holo_size * 2)
                    ]
                    for i in range(self.holo_size)
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
        if self.del_area > 0  or self.del_area != None:
            central_point_h = int(input_matrix.shape[0] / 2)
            central_point_w = int(input_matrix.shape[1] / 2)
            input_matrix[central_point_h - self.del_area: central_point_h + self.del_area, central_point_w - 1:central_point_w + 1 ] = 0
        return input_matrix

    def img_recovery(self, holo):
        if self.near_zone:
            rec_img = abs(self.inverse_fresnel_transform(self.prepare_for_transform(holo)))
            rec_img = self.del_central_zone(rec_img)
            
        else:
            rec_img = abs(self.inverse_fourier_transform(self.prepare_for_transform(holo)))
            rec_img = self.del_central_zone(rec_img)

        plt.plot(rec_img)
        plt.show()

        plt.imshow(rec_img, cmap = 'gray')
        plt.show()

         
    
    
    def matrix_normalization(self, input_matrix):
        norm_matrix = self.zero_to_two_pi_range(input_matrix)
        norm_matrix = np.uint8(input_matrix * 255 /(2 * np.pi))
        return norm_matrix
    

    # Загоняет интовскую матрицу в экспоненту и нормирует на 2 пи 
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
        mask = np.int8(np.random.rand(self.holo_size,self.holo_size) * 2)       
        return input_matrix * np.exp(1j * np.pi * mask)   

    def zero_to_two_pi_range(self, angles):
        return (2*np.pi + angles) * (angles < 0) + angles*(angles >= 0)

    
    def generate_doubph_enc_matrix(self, teta_1, teta_2, size, mod):        
        dbphase_holo = np.zeros((size[0], size[1]))
        min_size = min(size)
        last = True
        for i in range (min_size): 
            for j in range(min_size):
                if mod == 'def':
                    dbphase_holo[ i ,2 * j] = teta_1[i,j]
                    dbphase_holo[ i ,2 * j + 1 ] = teta_2[i,j]
                elif mod == 'rand':
                    a = [teta_1[i,j], teta_2[i,j]] 
                    random.shuffle(a)
                    dbphase_holo[ i ,2 * j] = a.pop()
                    dbphase_holo[ i ,2 * j + 1 ] = a.pop()
                elif mod == "chess":
                    if last:
                        dbphase_holo[ i ,2 * j] = teta_1[i,j]
                        dbphase_holo[ i ,2 * j + 1 ] = teta_2[i,j]
                        last = False
                    else:
                        dbphase_holo[ i ,2 * j] = teta_2[i,j]
                        dbphase_holo[ i ,2 * j + 1 ] = teta_1[i,j]
                        last = True
                else:
                    print("Unknown mod")
                    exit()
        return dbphase_holo



            
    

    def __call__(self, input_matrix, position = None,  restored_img_size=None, reshaped_img_position_coord_h_w=(None, None),control=False ):
        start_time = time.time()
        img = self.reshape_img_for_syn(input_matrix, position, restored_img_size, reshaped_img_position_coord_h_w )
        if control: 
            plt.imshow(img, cmap='gray')
            plt.show()
        img  = self.phase_mask(img)
        if self.near_zone:
            holo = self.fresnel_transform(img)
        else :
            holo = self.fourier_transform(img)

        
        holo = holo/abs(holo).max()

        del_phi = np.arccos(abs(holo) / 2)
        phase = np.angle(holo)
        teta_1 =  phase + del_phi
        teta_2 = phase - del_phi 
        dbphase_holo = self.generate_doubph_enc_matrix(teta_1, teta_2, (self.holo_size, self.holo_size * 2), self.mod )
        dbphase_holo = self.matrix_normalization(dbphase_holo)
        # до этого момента все отрабатывает нормально 
        if control:
            self.img_recovery(dbphase_holo)
        # часть, которая восстанавливает, чтобы проверить 
        # a = self.inverse_fresnel_transform(self.prepare_for_transform(dbphase_holo))
        # a = self.inverse_fresnel_transform(dbphase_holo)
        # a = self.del_central_zone(a)

        # plt.plot(abs(a))
        # plt.show()
        # a = abs(a) * 255/abs(a).max()

        # cv2.imwrite("C:\\Users\\minik\\Desktop\\def_2.bmp", a )
        # plt.imshow(a, cmap = 'gray')
        # plt.show()
        # plt.plot(abs(a))
        # plt.show()


 
        # self.img_recovery(dbphase_holo)
        # # error = self.calc_error( self.informative_img_zone, self.reshape_img)
        # # print("Error ", error)
        # print(self.recovery_img.shape)
        # plt.imshow(self.recovery_img, cmap = 'gray')
        # plt.show()
        # self.error_list.append(error)
        print("Time --- %s seconds ---" % (time.time() - start_time))
        self.holo  =  dbphase_holo
        return dbphase_holo

syntesis = DE_syntesis(
                        del_area = 1,
                        wavelength = 532e-9,
                        holo_size = 1024,
                        near_zone = True,
                        mod  = 'def',
                        holo_pixel_size = 8e-6,
                        distance= 1,

)
img_2 = cv2.imread("C:\\Users\\minik\\Desktop\\lena.jpg", cv2.IMREAD_GRAYSCALE)
holo = syntesis(img_2, position='centre', restored_img_size=200, reshaped_img_position_coord_h_w=(200, 200), control=True)