import numpy as np 
import cv2
import matplotlib.pyplot as plt
import time 
import copy
from Gerchberg_Sexton_Fresnel import GS_Fresnel_synthesis
from Direct_search_with_random_trajectory import DSWRT

class GS_Fresnel_ws_synthesis(GS_Fresnel_synthesis):
    def __init__(
                    self, 
                    holo_pixel_size,
                    distance,
                    source_distance,
                    wavelength, 
                    holo_size,
                    error_dif,
                    iter_limit, 
                    holo_type,
                    dynamic_range,
                    del_area,
                    random_tr = False,
                    save_errors = False,
                    alpha = None,
                    epochs = None,
                ):
        self.save_errors = save_errors
        self.random_tr = random_tr
        self.alpha = alpha 
        self.epochs = epochs
        self.source_distance = source_distance
        self._source_amplitude = None
        self._source_phase = None
        super().__init__(
                    del_area,
                    holo_pixel_size,
                    distance, 
                    wavelength, 
                    holo_size,
                    error_dif,
                    iter_limit, 
                    holo_type,
                    dynamic_range,
                )
    
    @property
    def source_phase (self):
        if self._source_phase is None:
            h_index = int(self.holo_size[0] / 2)
            w_index = int(self.holo_size[1] / 2)
            I,J = np.meshgrid(np.arange(- h_index,  h_index), np.arange(-w_index, w_index))
            K = I**2 + J**2
            self._source_phase = np.angle(
                                            np.exp(
                                                    -np.pi * 1j / (self.source_distance * self.wavelength)
                                                    *(self.holo_pixel_size[0]**2)*(K)
                                                )
                                        )
        return self._source_phase

    @property   
    def source_amplitude(self):
        if self._source_amplitude is None:
            h_index = int(self.holo_size[0] / 2)
            w_index = int(self.holo_size[1] / 2)
            self._source_amplitude = np.ones(self.holo_size)
            for i in range(self.holo_size[0]):
                for j in range(self.holo_size[1]):
                    self._source_amplitude[i,j]=(
                                            np.sqrt(    
                                                    self.source_distance**2/(self.source_distance**2 + (self.holo_pixel_size[0]**2)*
                                                    ((i- h_index)**2+(j-w_index)**2))
                                            )
                                        )

        return  self._source_amplitude

    def change_inf_zone(self, input_matrix, position, reshaped_img_position_coord_h_w):
        if position == "centre":
            index = (int((self.holo_size[0] - self.restored_img_size[0]) / 2), int((self.holo_size[1] - self.restored_img_size[1]) / 2))
            input_matrix[
                index[0]:index[0] + self.restored_img_size[0], index[1]:index[1] + self.restored_img_size[1]
                ] = self.reshape_img**0.5 * np.mean(self.informative_zone(input_matrix,position, reshaped_img_position_coord_h_w)) / np.mean(self.reshape_img**0.5)
        elif position == "free":
            input_matrix[
                reshaped_img_position_coord_h_w[0]:reshaped_img_position_coord_h_w[0] + self.restored_img_size[0], 
                reshaped_img_position_coord_h_w[1]:reshaped_img_position_coord_h_w[1] + self.restored_img_size[1],
                ] = self.reshape_img**0.5 * np.mean(self.informative_zone(input_matrix,position, reshaped_img_position_coord_h_w)) / np.mean(self.reshape_img**0.5)
        else:
            print("Incorre input, the argument 'position' must be equal center/free, but not ", position)
            exit()
        return input_matrix

    def matrix_normalization(self,input_matrix):
        if self.holo_type == 'phase':
            input_matrix = (input_matrix + np.pi) / (2*np.pi)
            if self.dynamic_range == 'bin':
                n = 2
                input_matrix = np.uint8(np.round((input_matrix-1/(2*n))*n)*127)
            elif self.dynamic_range == 'gray':
                n = 256
                input_matrix = np.uint8(np.round((input_matrix-1/(2*n))*n))
            else:
                print("Incorre input, the argument 'dynamic_range' must be equal gray/bin, but not ", self.dynamic_range)
                exit()
        elif self.holo_type == 'amplitude':
            if self.dynamic_range == 'bin':
                input_matrix = cv2.threshold(input_matrix, 0, 127, cv2.THRESH_OTSU)[1]
            elif self.dynamic_range == 'gray':
                input_matrix = input_matrix/input_matrix.max() * 255
            else:
                print("Incorre input, the argument 'dynamic_range' must be equal gray/bin, but not ", self.dynamic_range)
                exit()
        else:
            print("Incorre input, the argument 'position' must be equal phase/amplitude, but not  ", self.holo_type)
            exit()
        return  input_matrix

    def img_recovery(self, holo, position, reshaped_img_position_coord_h_w):
        if self.holo_type == "phase":
            rec_img = abs(self.fresnel_transform(self.prepare_for_fresnel(holo)))
        elif self.holo_type == "amplitude":
            rec_img = abs(self.fresnel_transform(holo))
            rec_img = self.del_central_zone(rec_img)
        else: 
            print("Incorre input, the argument 'holo_type' must be equal phase/amplitude, but not ", self.holo_type)
            exit()

        rec_img  = rec_img**2   
        self.rec_img = rec_img
        plt.imshow(rec_img, cmap = "gray")
        plt.show()
        self.informative_img_zone = self.informative_zone(rec_img, position, reshaped_img_position_coord_h_w)
        plt.imshow(self.informative_img_zone, cmap = "gray")
        plt.show()


    def __call__(
                        self,
                        input_matrix,  
                        control=False, 
                        position=None, 
                        restored_img_size = None,  
                        reshaped_img_position_coord_h_w = (None,None)

                    ):
            self.reshaped_img_coord_h, self.reshaped_img_coord_w = reshaped_img_position_coord_h_w
            error_list = []
            start_time = time.time()
            holo = self.source_amplitude *  self.initial_approx() * np.exp(1j * self.source_phase)
            img = self.reshape_img_for_syn(input_matrix, position, restored_img_size, reshaped_img_position_coord_h_w )
            if control:
                plt.imshow(img, cmap="gray")
                plt.show()
            error_dif = float('inf')
            last_error_dif = 0
            i = 0
            while error_dif > self.error_dif and i < self.iter_limit:
                ref_img = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(holo)))
                ref_img_phase = np.angle(ref_img)
                ref_img_amplitude = abs(ref_img)  
                ref_img_amplitude = self.change_inf_zone(ref_img_amplitude, position, reshaped_img_position_coord_h_w )
                ref_img = ref_img_amplitude * np.exp(1j * ref_img_phase)
                holo = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(ref_img)))
                holo = holo * np.exp(-1j * self.source_phase)
                holo_phase = np.angle(holo)
                holo_abs = abs(holo)

                if self.holo_type == 'phase':
                    holo_final = self.matrix_normalization(holo_phase)
                    holo =  self.source_amplitude *  np.exp(1j * holo_phase) * np.exp(1j * self.source_phase)
                    #error 
                    Ans = self.source_amplitude * self.prepare_for_fresnel(holo_final) *  np.exp(1j * self.source_phase)

                elif self.holo_type == 'amplitude':
                    holo_final = self.matrix_normalization(holo_abs)
                    holo =  self.source_amplitude *  holo_abs * np.exp(1j * self.source_phase)
                    #error 
                    Ans = self.source_amplitude * holo_final *  np.exp(1j * self.source_phase)
                else:
                    print("Incorre input, the argument 'position' must be equal phase/amplitude, but not  ", self.holo_type)
                    exit()
                
                Ans = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Ans)))
                Ans = abs(Ans)**2
                self.recovery_img = Ans 
                Ans = self.informative_zone(Ans, position, reshaped_img_position_coord_h_w)
                self.inf_recovery_img = Ans
                
                error = self.calc_error(
                                        Ans,
                                        self.informative_zone(img, position, reshaped_img_position_coord_h_w)
                                        )  
                if i == self.iter_limit - 1:
                    plt.imshow(Ans, cmap='gray')
                    plt.show()
                i += 1 
                error_list.append(error)
                print("Iteration ", i, "error(informative zone)", error) 
                
            self.iteration = i
            self.error_lists.append(error_list)
            # if control:
            #     self.img_recovery(holo,position, reshaped_img_position_coord_h_w)
            self.holo = holo_final
            if self.random_tr:
                search = DSWRT(
                        perfect_img = self.new_img,
                        epochs=self.epochs, 
                        alpha=self.alpha, 
                        holo_type= self.holo_type,
                        position= position, 
                        restored_img_size=restored_img_size, 
                        source_amplitude = self.source_amplitude,
                        source_phase=self.source_phase,
                        holo_size = self.holo_size,
                        save_errors = self.save_errors,
                        reshaped_img_position_coord_h_w = None,)

                holo_final, self.recovery_img, self.inf_recovery_img = search(holo_final)      
            print("Finish --- %s seconds ---" % (time.time() - start_time))
            return holo_final
transform = GS_Fresnel_ws_synthesis(
    holo_pixel_size = (8e-6, 8e-6),
    distance = 0.3,
    wavelength = 532e-9,
    holo_size = (1024,1024),
    error_dif = 1e-4,
    holo_type = 'phase',
    dynamic_range = 'gray',
    iter_limit = 40,
    del_area = 0,
    source_distance = 0.3,
    random_tr = False,
    save_errors = True,
    alpha = 0.5,
    epochs = 3

)
img = cv2.imread("C:\\Users\\minik\\Desktop\\I_gray.jpg", cv2.IMREAD_GRAYSCALE)
holo = transform(
                    img, 
                    position='centre', 
                    reshaped_img_position_coord_h_w = (100,100), 
                    restored_img_size = (int(img.shape[0]/5), int(img.shape[1]/5)),
                    control=True,
                )
name_holo = "holo_I_gray_gray_phase_size_div 5_1024x1024.bmp"
name_restored = "rest_I_gray_gray_phase_1024x1024_dist_03.bmp"
name_restored_inf = "rest_inf_I_gray_gray_phase_1024x1024_dist_03.bmp"
cv2.imwrite("C:\\Users\\minik\\Desktop\\" + name_holo, holo)
plt.imsave("C:\\Users\\minik\\Desktop\\" + name_restored, transform.recovery_img, cmap='gray')
plt.imsave("C:\\Users\\minik\\Desktop\\" + name_restored_inf, transform.inf_recovery_img, cmap='gray')
    
        
        