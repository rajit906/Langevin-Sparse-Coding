import torch

class LangevinSparseCoding():
    def __init__(self,
                 n_features:int,
                 n_latent:int,
                 sparsity_penaly:float,
                 temperature:float,
                 dt:float,
                 cost_function,
                 cost_function_grad,
                 mass:float=0.,
                 gauss_std:float=1.,
                ):
        self.n_features = n_features
        self.n_latent = n_latent
        self.sparsity_penaly = sparsity_penaly
        self.temperature = torch.FloatTensor([temperature])
        self.dt = torch.FloatTensor([dt])
        self.mass = mass
        self.cost_function = cost_function
        self.cost_function_grad = cost_function_grad
        self.gauss_std = gauss_std
        self.A = torch.FloatTensor(n_features, n_latent).normal_()
        self.normalize_dictionary()

    def set_temperature(self,temperature:float):
        self.temperature = torch.FloatTensor([temperature])


    def set_dt(self,dt:float):
        self.dt = torch.FloatTensor([dt])


    def energy(self,x,s):
        """
        Parameters
        ----------
        x,s : shape=[batch,n_features],[batch,n_latent]

        Returns
        -------
        energy at s : [batch]
        """
        return 0.5/self.gauss_std**2 * (((self.A @ s.T).T - x)**2).sum(dim=1) + self.sparsity_penaly*self.cost_function(s) 
    

    def p(self,x,s):
        """
        Parameters
        ----------
        x,s : shape=[batch,n_features],[batch,n_latent]

        Returns
        -------
        probability at s : [batch]
        """
        return torch.exp(-self.energy(x,s)/self.temperature)
    

    def energy_grad(self,x,s):
        """
        Parameters
        ----------
        x,s : shape=[batch,n_features],[batch,n_latent]

        Returns
        -------
        delta s : [batch,n_latent]
        """
        return 1/self.gauss_std*(self.A.T@((self.A@s.T).T-x).T).T + self.sparsity_penaly*self.cost_function_grad(s)

    def dictionary_grad(self,x,s):
        """
        Parameters
        ----------
        x,s : shape=[batch,n_features],[batch,n_latent]

        Returns
        -------
        grad A : [n_features, n_latent]
        """
        return (-((self.A@s.T) - x.T) @ s)


    def first_order_langevin_update(self,x,s):
        """
        Parameters
        ----------
        x,s : shape=[batch,n_features],[batch,n_latent]

        Returns
        -------
        delta s : [batch,n_latent]
        """
        return -self.energy_grad(x,s)*self.dt + torch.sqrt(2*self.temperature*self.dt)*torch.FloatTensor(s.shape).normal_()


    def second_order_langevin_update(self,x,s,v):
        """
        Parameters
        ----------
        x,s,v : shape=[batch,n_features],[batch,n_latent],[batch,n_latent]

        Returns
        -------
        delta s,delta v : shape=[batch,n_latent],[batch,n_latent]
        
        """
        ds = self.dt/self.mass*v 
        dv = -ds + self.first_order_langevin_update(x,s) 
        return ds, dv


    def normalize_dictionary(self):
        with torch.no_grad():
            self.A.div_(torch.norm(self.A,dim=0,keepdim=True))
            