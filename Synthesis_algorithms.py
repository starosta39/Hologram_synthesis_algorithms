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

class AS_synthesis(object):
    
    def __init__(self,  holo_pixel_size, distance, wavelength, input_matrix_size, error, iter_limit, img_size_for_syn = None):
        
        self.name = 'Angular_spectrum'
        self._chirp_factor = None
        self._inv_chirp_factor = None
        self.holo_pixel_size = holo_pixel_size
        self.distance = distance
        self.wavelength = wavelength
        self.input_matrix_size = input_matrix_size 
        self.error = error
        self.iter_limit = iter_limit 
        self.error_lists = []
        self.scale = self.wavelength * self.distance / (self.input_matrix_size * self.holo_pixel_size**2)       
        
        if img_size_for_syn  is None:
            self.img_size_for_syn = self.input_matrix_size - int(self.input_matrix_size/2)
        elif self.input_matrix_size % self.reshape_img_size != 0:
            print("Некорректный ввод  нового размера изображения, исходный размер  должен делиться на него без остатка")
            exit()
        else: self.img_size_for_syn = img_size_for_syn
                            
    def reshape_img_for_syn(self, input_matrix):
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
    def chirp_factor(self):
        if self._chirp_factor is None:
            self._chirp_factor = (
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
        return np.exp(1j * np.random.rand(self.input_matrix_size, self.input_matrix_size))
    
    def prepare_for_angle_spec_transform (self, input_matrix):
        return np.exp(1j * input_matrix * 2 * np.pi / 255)

    def img_recovery(self, holo):
        rec_img = abs(self.angle_spect_transform(self.prepare_for_angle_spec_transform(holo)))
        self.recovery_img  = rec_img 
        index = int((self.input_matrix_size - self.img_size_for_syn) / 2)
        informative_img_zone = rec_img[index:index + self.img_size_for_syn, index:index + self.img_size_for_syn]
        self.informative_img_zone = informative_img_zone
    
    
    def calc_error(self, input_matrix, real_img):
        input_matrix = np.float64(abs(input_matrix))
        real_img = np.float64(abs(real_img))
        a = np.sum(input_matrix**2)
        b = np.sum(real_img**2)
        ab = np.sum(input_matrix*real_img)
        return np.sqrt(1-(ab*ab)/(a*b))

    def __call__(self, input_matrix):
        error_list = []
        start_time = time.time()
        holo = self.initial_approx()
        img = self.reshape_img_for_syn(input_matrix)
        error = float('inf')
        i = 0
        while error > self.error and i < self.iter_limit:
            ref_img = self.angle_spect_transform(holo)
            error = self.calc_error(ref_img, img)
            ref_img = np.sqrt(img) * np.exp(1j*np.angle(ref_img))
            holo = self.inverse_angle_spect_transform(ref_img)
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

class BP_synthesis(object):
    
    def __init__(self,   wavelength, 
                 holo_size, 
                 reshaped_img_coord_x,
                 reshaped_img_coord_y,
                 near_zone = True,
                 distance = None,
                 img_size_for_syn = None,
                 holo_pixel_size = None
                ):
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
            self.holo_pixel_size =  holo_pixel_size
            self.scale = self.wavelength * self.distance / (self.holo_size * self.holo_pixel_size**2)
        
        if img_size_for_syn  is None:
            self.img_size_for_syn = self.holo_size - int(self.holo_size/2)
        else: self.img_size_for_syn = img_size_for_syn
            
            
    def reshape_img_for_syn(self, input_matrix):
        new_img = np.zeros((self.holo_size, self.holo_size)) 
        reshape_img = cv2.resize(input_matrix, (self.img_size_for_syn,self.img_size_for_syn))
        self.reshape_img = reshape_img
        new_img[
                self.reshaped_img_coord_y:self.reshaped_img_coord_y + self.img_size_for_syn, 
                self.reshaped_img_coord_x:self.reshaped_img_coord_x + self.img_size_for_syn
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
        if self._second_frenel_factor is None and self.near_zone:
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
        if self._first_inv_frenel_factor is None and self.near_zone:
            self._first_inv_frenel_factor = np.exp(
                np.array([
                    [
                        -1j * np.pi * ((i- int(self.holo_size / 2))**2 + (j - int(self.holo_sizee / 2))**2)/(self.scale * self.holo_size)
                        for j in range(self.holo_size)
                    ]
                    for i in range(self.holo_size)
                ])
            )
        return self._first_inv_frenel_factor

    @property
    def second_inv_frenel_factor(self):
        if self._second_inv_frenel_factor is None and self.near_zone:
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
        central_point = int(self.holo_size / 2)
        input_matrix[central_point - 1:central_point + 1, central_point - 1:central_point + 1] = 0
        return input_matrix
    
    def img_recovery(self, holo):
        if self.near_zone:
            rec_img = abs(self.inverse_fourier_transform(self.prepare_for_transform(holo)))
            rec_img = self.del_central_zone(rec_img)
        else:
            rec_img = abs(self.fourier_transform(self.prepare_for_transform(holo)))
            rec_img = self.del_central_zone(rec_img)
        self.recovery_img  = rec_img   
        informative_img_zone = rec_img [
                                    self.reshaped_img_coord_y:self.reshaped_img_coord_y + self.img_size_for_syn, 
                                    self.reshaped_img_coord_x:self.reshaped_img_coord_x + self.img_size_for_syn
            
                                    ]
        
        self.informative_img_zone = informative_img_zone
    
    
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
    
    def __call__(self, input_matrix):
        start_time = time.time()
        img = self.reshape_img_for_syn(input_matrix)
        img  = self.phase_mask(img)
        if self.near_zone:
            holo = self.frenel_transform(img)
        else :
            holo = self.fourier_transform(img)
        holo = 2 * (np.real(holo) - np.real(holo).min())
        holo = np.uint8(holo * 255 /holo.max())
        self.img_recovery(holo)
        error = self.calc_error( self.informative_img_zone, self.reshape_img)
        print("Error ", error)
        
        self.error_list.append(error)
        print("Time --- %s seconds ---" % (time.time() - start_time))
        self.holo  = holo
        return holo


def show_error(class_obj, last_img = True):
    error_lists = class_obj.error_lists
    
    if last_img:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.array(error_lists[-1]), '-o')
        ax.grid()
        ax.set_xlabel('Iteration',{'fontname':'Times New Roman'},fontweight='light',color='k', fontsize=24)
        ax.set_ylabel("NSKD",{'fontname':'Times New Roman'},fontweight='light',color='k', fontsize=24)
        plt.show()
    else:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1,1,1)
        for errors in error_lists:
            ax.plot(np.array(errors),'-o')
        ax.grid()
        ax.set_xlabel('Iteration',{'fontname':'Times New Roman'},fontweight='light',color='k', fontsize=24)
        ax.set_ylabel("NSKD",{'fontname':'Times New Roman'},fontweight='light',color='k', fontsize=24)
        plt.show()

def save_holo(filename, holo):
    cv2.imwrite(filename,holo)

def show (class_obj, informative_img_zone = True, holo = False):
    pic_box = plt.figure(figsize=(12, 6))
    if holo:
        ax = pic_box.add_subplot(1,1,1)
        ax.set_title("Holo")
        ax.imshow(class_obj.holo, cmap = 'gray')
    else:   
        if informative_img_zone:
            if class_obj.name == 'Bipolar_intensity':
                plt.suptitle("NSTD = " + str(class_obj.error_list[-1]))
            else:
                plt.suptitle("NSTD = " + str(class_obj.error_list[-1][-1]))

            ax1 = pic_box.add_subplot(1,2,1)
            ax1.set_title("Real image")
            ax1.imshow(class_obj.reshape_img, cmap = 'gray')

            ax2 = pic_box.add_subplot(1,2,2)
            ax2.set_title('Reconstructed image')
            ax2.imshow(class_obj.informative_img_zone,cmap = 'gray')
            
        else:
            if class_obj.name == 'Bipolar_intensity':
                plt.suptitle("NSTD = " + str(class_obj.error_list[-1]))
            else:
                plt.suptitle("NSTD = " + str(class_obj.error_list[-1][-1]))

            ax1 = pic_box.add_subplot(1,2,1)
            ax1.set_title("Real image")
            ax1.imshow(class_obj.new_img, cmap = 'gray')

            ax2 = pic_box.add_subplot(1,2,2)
            ax2.set_title('Reconstructed image')
            ax2.imshow(class_obj.recovery_img, cmap = 'gray')
           
        
    plt.show()
    
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
    first_inv_frenel_factor = np.exp(
                    np.array([
                        [
                            -1j * np.pi * ((i- int(N_h / 2))**2 /( N_h * scale_h) 
                            + (j - int(N_w / 2))**2 / (N_w * scale_w))
                            for j in range(N_w)
                        ]
                        for i in range(N_h)
                    ])
                ) 
    second_inv_frenel_factor = np.exp(
                        np.array([
                            [
                                -1j * np.pi * (scale_h * (i- int(N_h / 2))**2 / N_h  
                                + scale_w * (j - int(N_w / 2))**2 / N_w)
                                for j in range(N_w)
                            ]
                            for i in range(N_h)
                        ])
                    )    

    inv_fourier = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(input_matrix * first_inv_frenel_factor)))
    return inv_fourier * second_inv_frenel_factor

