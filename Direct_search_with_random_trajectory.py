import torch 
import time 
import copy
import random
import matplotlib.pyplot as plt
class DSWRT(object):
    def __init__(
                    self, 
                    perfect_img,
                    epochs, 
                    alpha, 
                    holo_type,
                    position, 
                    restored_img_size, 
                    source_amplitude,
                    source_phase,
                    holo_size,
                    reshaped_img_position_coord_h_w = None,
                ):
        self.NSTD_errors = []
        self.DE_errors = []
        self.TFs = []
        self.change_iterations = []
        self.best_DEs = []
        self.best_NSTDs = []
        self.best_TFs=[] 
        self.best_DE = None
        self.best_NSTD = None
        self.best_TF=None
        
        self.source_amplitude = torch.from_numpy(source_amplitude)
        self.source_phase = torch.from_numpy(source_phase)
        self.position = position
        self.restored_img_size = restored_img_size
        self.reshaped_img_position_coord_h_w = reshaped_img_position_coord_h_w
        self.holo_size = holo_size
        self.epochs = epochs
        self.alpha = alpha
        self.holo_type = holo_type
        self.perfect_informative_img =  self.informative_zone(torch.from_numpy(perfect_img))
        self.epochs_generator(holo_size, epochs)
        # trajectories_len = img_size[0]*img_size[0]
        # trajectories = torch.ones(epochs,trajectories_len)
        # # for epoch in range(epochs):
        # #     trajectory = torch.randperm(trajectories_len)
        # #     trajectories[epoch] = trajectory

    def epochs_generator(self, holo_size, epochs):
        trajectories = []
        trajectory = []

        for i in range(holo_size[0]):
            for j in range (holo_size[1]):
                    trajectory.append([i,j])

        for _ in range(epochs):
            trajectory = list(trajectory)
            random.shuffle(trajectory)
            trajectories.append(trajectory)

        self.trajectories = trajectories
        iter_in_one = len(trajectories[0])
        print("Очередь из {} эпох по {} итераций сгенерирована". format(str(self.epochs),str(iter_in_one) ))

    def informative_zone (self, input_matrix):

        if self.position == "centre":
            index = (int((self.holo_size[0] - self.restored_img_size[0]) / 2), int((self.holo_size[1] - self.restored_img_size[1]) / 2))
            res = input_matrix[index[0]:index[0] + self.restored_img_size[0], index[1]:index[1] + self.restored_img_size[1]]
        elif self.position == "free":
            res = input_matrix[
                self.reshaped_img_position_coord_h_w[0]:self.reshaped_img_position_coord_h_w[0] + self.restored_img_size[0], 
                self.reshaped_img_position_coord_h_w[1]:self.reshaped_img_position_coord_h_w[1] + self.restored_img_size[1],
                ]
        else:
            print("Incorre input, the argument 'position' must be equal center/free, but not ", self.position)
            exit()
        return res

    def NSTD(self, input_matrix, real_img):
        # input_matrix = torch.float64(abs(input_matrix))
        # real_img = torch.float64(abs(real_img))
        a = torch.sum(input_matrix**2)
        b = torch.sum(real_img**2)
        ab = torch.sum(input_matrix*real_img)
        NSTD = torch.sqrt(1-(ab*ab)/(a*b))
        device = torch.device('cpu')
        NSTD = NSTD.to(device)
        return float(NSTD)
    
    def DE(self, input_matrix):
        DE = torch.sum(self.informative_zone(input_matrix)) / torch.sum(input_matrix) 
        device = torch.device('cpu')
        DE = DE.to(device)
        return float(DE)    

    def __call__(self, holo):
        print("DSWRT start")
        start_time = time.time()
        holo = torch.from_numpy(holo)
        self.start_holo = holo
        device = torch.device(  
                              'cuda:0' 
                              if torch.cuda.is_available() 
                              else 'cpu'
                            )
        holo = holo.to(device)
        self.perfect_informative_img = self.perfect_informative_img.to(device)
        self.source_phase = self.source_phase.to(device)
        self.source_amplitude = self.source_amplitude.to(device)

        if self.holo_type == 'phase':
            new_holo = self.source_amplitude * torch.exp(1j* holo*2*torch.pi / 255)*torch.exp(1j * self.source_phase)
        elif self.holo_type == 'amplitude':
            new_holo = self.source_amplitude * holo * torch.exp(1j * self.source_phase)
        restored_img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(new_holo)))
        restored_img = abs(restored_img) **2
        self.start_restored_img = copy.copy(restored_img)
        self.best_DE = self.DE(restored_img)
        restored_img = self.informative_zone(restored_img)
        self.best_NSTD = self.NSTD(restored_img, self.perfect_informative_img)
        self.best_TF = self.alpha * self.best_NSTD + (1 - self.alpha)*(1 - self.best_DE)
        iteration = 0
        print("Iteration", iteration, "best NSTD =", self.best_NSTD, "best DE = ", self.best_DE, "best TF = ", self.best_TF)
        for epoch in range(self.epochs):
            for i, j in self.trajectories[epoch]:

                
                if holo[i,j] == 127:
                    holo[i,j] = 0
                else: 
                    holo[i,j]  = 127
                
                if self.holo_type == 'phase':
                    new_holo = self.source_amplitude * torch.exp(1j* holo*2*torch.pi / 255)*torch.exp(1j * self.source_phase)
                elif self.holo_type == 'amplitude':
                    new_holo = self.source_amplitude * holo * torch.exp(1j * self.source_phase)

                restored_img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(new_holo)))
                restored_img = abs(restored_img) **2
                DE = self.DE(restored_img)
                restored_img = self.informative_zone(restored_img)
                NSTD = self.NSTD(restored_img, self.perfect_informative_img)
                TF = self.alpha * NSTD + (1 - self.alpha) * (1- DE) 
                if TF >= self.best_TF:
                    if holo[i,j] == 127:
                        holo[i,j] = 0
                    else: 
                        holo[i,j]  = 127
                else:
                    self.best_TF = TF
                    self.best_NSTD = NSTD
                    self.best_DE = DE 
                    self.best_DEs.append(DE)
                    self.best_NSTDs.append(NSTD)
                    self.best_TFs.append(TF)
                self.NSTD_errors.append(NSTD)
                self.DE_errors.append(DE)
                self.TFs.append(TF)
                self.change_iterations.append(iteration)
                iteration += 1
                # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                # print("Iteration", iteration, "NSTD =", NSTD, "DE = ", DE, "TF = ", TF)
                # print("Iteration", iteration, "best NSTD =", self.best_NSTD, "best DE = ", self.best_DE, "best TF = ", self.best_TF)
                if iteration %1000 == 0:
                    print("Iteration", iteration, "NSTD =", self.best_NSTD, "DE = ", self.best_DE, "TF = ", self.best_TF)
        device = torch.device('cpu')
        holo =  holo.to(device)
        holo = holo.numpy()
        restored_img = restored_img.to(device)
        restored_img = restored_img.numpy()
        plt.imshow(restored_img, cmap='gray')
        plt.show()
        plt.imshow(holo, cmap='gray')
        plt.show()
        print("DSWRT finish --- %s seconds ---" % (time.time() - start_time))
        return holo

