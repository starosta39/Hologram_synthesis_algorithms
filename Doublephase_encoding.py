import numpy as np 
import cv2
import matplotlib.pyplot as plt
import time 
import cmath
import copy
import random
from Bipolar_intensity import BP_synthesis


class DE_syntesis(BP_synthesis):

    def __init__(self,   wavelength, 
                 holo_size, 
                 mod,
                 reshaped_img_coord_x = None,
                 reshaped_img_coord_y = None,
                 near_zone = True,
                 distance = None,
                 img_size_for_syn = None,
                 holo_pixel_size = None
                ):
        self.mod = mod
        self.name = 'Bipolar_intensity'
        self.near_zone = near_zone
        self.holo_size = holo_size
        self.distance = distance
        self.wavelength = wavelength
        self.reshaped_img_coord_x = reshaped_img_coord_x
        self.reshaped_img_coord_y = reshaped_img_coord_y 
        self.error_list = []
        
        if near_zone:
            self._first_frenel_factor = None
            self._second_frenel_factor = None
            self._first_inv_frenel_factor = None
            self._second_inv_frenel_factor = None
            self.holo_pixel_size =  holo_pixel_size
            self.scale = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size**2)
            self.scale_h = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size**2)
            self.scale_w = self.wavelength *  self.distance / (self.holo_size  * 4 *   self.holo_pixel_size**2)
            self.scale_h_inverse = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size**2)
            self.scale_w_inverse = self.wavelength * self.distance / (self.holo_size * 2 * self.holo_pixel_size**2)
        
        if img_size_for_syn  is None:
            self.img_size_for_syn = self.holo_size - int(self.holo_size/2)
        else: self.img_size_for_syn = img_size_for_syn
            
            
    def reshape_img_for_syn(self, input_matrix):
        new_img = np.zeros((self.holo_size, self.holo_size)) 
        reshape_img = cv2.resize(input_matrix, (self.img_size_for_syn,self.img_size_for_syn))
        self.reshape_img = reshape_img
        if self.reshaped_img_coord_x is not None and self.reshaped_img_coord_y is not None:  
            new_img[
                    self.reshaped_img_coord_y:self.reshaped_img_coord_y + self.img_size_for_syn, 
                    self.reshaped_img_coord_x:self.reshaped_img_coord_x + self.img_size_for_syn
                    ] = reshape_img     
        else:
            index = int((self.holo_size - self.img_size_for_syn) / 2)
            new_img[
                    index:index + self.img_size_for_syn, index:index + self.img_size_for_syn
                    ] = reshape_img            
        self.new_img = new_img 
        return new_img
        
    
    @property
    def first_frenel_factor(self):
        if self._first_frenel_factor is None and self.near_zone:
            self._first_frenel_factor = (
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

        return self._first_frenel_factor

    @property
    def second_frenel_factor(self):
        if self._second_frenel_factor is None and self.near_zone:
            self._second_frenel_factor = (
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
        return self._second_frenel_factor
    
    @property
    def first_inv_frenel_factor(self):
        if self._first_inv_frenel_factor is None and self.near_zone:
            self._first_inv_frenel_factor = np.exp(
                np.array([
                    [
                        -1j * np.pi * ((i- int(self.holo_size / 2 ))**2 /(self.scale_h_inverse * self.holo_size)  
                        + (j - int(self.holo_size))**2 /(self.scale_w_inverse * self.holo_size * 2))
                        for j in range(self.holo_size * 2)
                    ]
                    for i in range(self.holo_size)
                ])
            )
            plt.imshow(np.angle(self._first_inv_frenel_factor))
            plt.show()
            
        return self._first_inv_frenel_factor

    @property
    def second_inv_frenel_factor(self):
        if self._second_inv_frenel_factor is None and self.near_zone:
            self._second_inv_frenel_factor = np.exp(
                np.array([
                    [
                        -1j * np.pi * (self.scale_h_inverse *(i - int(self.holo_size / 2))**2 /(self.holo_size) 
                        + self.scale_w_inverse * (j - int(self.holo_size ))**2 / (self.holo_size * 2)) 
                        for j in range(self.holo_size * 2)
                    ]
                    for i in range(self.holo_size)
                ])
            )
            

        return self._second_inv_frenel_factor
    
    def frenel_transform(self,input_matrix):
        fourier = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(input_matrix * self.first_frenel_factor)))
        return  self.second_frenel_factor * fourier
    
    def inverse_frenel_transform(self,input_matrix):
        inv_fourier = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(input_matrix * self.first_inv_frenel_factor)))
        return self.second_inv_frenel_factor * inv_fourier
    
    def fourier_transform(self,input_matrix):
        return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(input_matrix)))
    
    def inverse_fourier_transform(self,input_matrix):
        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(input_matrix)))
    
    def del_central_zone(self,input_matrix):
        central_point_x = int(input_matrix.shape[0] / 2)
        central_point_y = int(input_matrix.shape[1] / 2)
        input_matrix[central_point_x - 1:central_point_x + 1, central_point_y - 1:central_point_y + 1 ] = 0
        return input_matrix
    # central_point_y - 10
    # central_point_y + 10
    def img_recovery(self, holo):
        if self.near_zone:
            rec_img = abs(self.inverse_frenel_transform(self.prepare_for_transform(holo)))
            rec_img = self.del_central_zone(rec_img)
            
        else:
            rec_img = abs(self.inverse_fourier_transform(self.prepare_for_transform(holo)))
            rec_img = self.del_central_zone(rec_img)

        self.recovery_img  = rec_img 
        if self.reshaped_img_coord_x is not None and self.reshaped_img_coord_y is not None:
            informative_img_zone = rec_img [
                                        self.reshaped_img_coord_y:self.reshaped_img_coord_y + self.img_size_for_syn, 
                                        self.reshaped_img_coord_x:self.reshaped_img_coord_x + self.img_size_for_syn        
                                        ]
        else:
            index = int((self.holo_size - self.img_size_for_syn) / 2)
            informative_img_zone = rec_img [
                                            index:index + self.img_size_for_syn, index:index + self.img_size_for_syn
                                            ]
        self.informative_img_zone = informative_img_zone
        # plt.plot(self.recovery_img )
        # plt.show()
    
    
    def matrix_normalization(self, input_matrix):
        norm_matrix = np.zeros((self.holo_size, self.holo_size)) 
        for i in range (self.holo_size):
            for j in range (self.holo_size):
                element = cmath.phase(input_matrix[i,j])
                if element < 0:
                    element = 2 * np.pi + element
                element = element/(2 * np.pi) * 256
                norm_matrix[i,j] = element
        return norm_matrix.astype('uint8')
    

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

    def other_angle(self, angles):
        return (2*np.pi + angles) * (angles < 0) + angles*(angles > 0)

    
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



            
    

    def __call__(self, input_matrix):
        start_time = time.time()
        img = self.reshape_img_for_syn(input_matrix)
        img  = self.phase_mask(img)
        if self.near_zone:
            holo = self.frenel_transform(img)
        else :
            holo = self.fourier_transform(img)

        
        holo = holo/abs(holo).max()

        del_phi = np.arccos(abs(holo) / 2)
        phase = np.angle(holo)
        teta_1 =  phase + del_phi
        teta_2 = phase - del_phi 
        dbphase_holo = self.generate_doubph_enc_matrix(teta_1, teta_2, (self.holo_size, self.holo_size * 2), self.mod )
        dbphase_holo = self.other_angle(dbphase_holo)
        dbphase_holo = np.uint8(dbphase_holo * 255 /(2 * np.pi))
        # до этого момента все отрабатывает нормально 
       


        # часть, которая восстанавливает, чтобы проверить 
        a = self.inverse_frenel_transform(self.prepare_for_transform(dbphase_holo))
        # a = self.inverse_frenel_transform(dbphase_holo)
        # a = self.del_central_zone(a)

        plt.plot(abs(a))
        plt.show()
        a = abs(a) * 255/abs(a).max()

        cv2.imwrite("C:\\Users\\minik\\Desktop\\def_2.bmp", a )
        plt.imshow(a, cmap = 'gray')
        plt.show()
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
                        wavelength = 532e-9,
                        holo_size = 1024,
                        # reshaped_img_coord_x = 200,
                        # reshaped_img_coord_y = 200,
                        near_zone = True,
                        mod  = 'def',
                        holo_pixel_size = 8e-6,
                        distance= 1,
                        img_size_for_syn = 512,

)
img_2 = cv2.imread("C:\\Users\\minik\\Desktop\\timur1.jpg", cv2.IMREAD_GRAYSCALE)
holo = syntesis(img_2)