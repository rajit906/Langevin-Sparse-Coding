import torch
import torch.nn
from utils import phi
import numpy as np

class LangevinSparseCoding_L0():
    def __init__(self,
                 n_features:int,
                 n_latent:int,
                 sparsity_penalty:float,
                 temperature:float,
                 dt:float,
                 cost_function,
                 cost_function_grad,
                 momentum:float=0.,
                 s0:float=0
                 ):
        self.n_features = n_features
        self.n_latent = n_latent
        self.sparsity_penalty = sparsity_penalty
        self.temperature = torch.FloatTensor([temperature])
        self.dt = torch.FloatTensor([dt])
        self.momentum = momentum
        self.A = torch.tensor([[1., 0.], [0., 1.]])#torch.FloatTensor(n_features, n_latent).normal_()
        self.normalize_dictionary()
        self.s0 = torch.FloatTensor([s0])
        self.cost_function = cost_function
        self.cost_function_grad = cost_function_grad

    def set_temperature(self,temperature:float):
        self.temperature = torch.FloatTensor([temperature])

    def set_dt(self,dt:float):
        self.dt = torch.FloatTensor([dt])

    def energy(self,x,s):
        return 0.5 * ((self.A @ phi(s.clone(), self.s0) - x)**2).sum() + self.sparsity_penalty*self.cost_function(s) 

    def p(self,x,s): #Does this need to be modified?
        return torch.exp(-self.energy(x,s)/self.temperature)
    

    def energy_grad(self,x,s):
        return self.A.T@(x-self.A @ phi(s.clone(), self.s0)) - self.sparsity_penalty*self.cost_function_grad(s)


    def first_order_langevin_update(self,x,s):
        return self.dt*(self.energy_grad(x,s)) + torch.sqrt(2*self.temperature*self.dt)*torch.FloatTensor(s.shape).normal_()


    def second_order_langevin_update(self,x,s,v):
        ds = self.dt/self.momentum*v
        dv = -ds + self.first_order_langevin_update(x,s) 
        return ds, dv

    def normalize_dictionary(self):
        with torch.no_grad():
            self.A.div_(torch.norm(self.A,dim=1,keepdim=True))