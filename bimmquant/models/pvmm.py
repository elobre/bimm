import torch
import math
import numpy as np
import scipy.special
import torch.nn.functional as F



class PVMM(torch.nn.Module):
    '''Partial volume mixture model with number of components: n_phases'''

    def __init__(self, n_phases):
        super(PVMM, self).__init__()

        self.n_phases = n_phases
        self.n_interfaces = int(0.5*(self.n_phases-1)*self.n_phases) #all different interface types

        self.I = torch.nn.Parameter(torch.Tensor(self.n_phases))
        self.sigma_n = torch.nn.Parameter(torch.Tensor(1))
        self.W = torch.nn.Parameter(torch.Tensor(self.n_interfaces+self.n_phases)) #weights before normalization

        #Set all parameters
        self.reset_params()


    def reset_params(self):
        '''Initialize model with component means (I) evenly spaced on [0, 1], same std for all components (sigma_n = 0.1) and equal weights'''
        self.I.data = torch.linspace(0, 1, self.n_phases)
        self.sigma_n.data = torch.tensor(0.1)
        self.W.data = torch.zeros(self.n_interfaces+self.n_phases)

    def set_params(self, params_dict):
        '''Initialize model with values specified in the dictionary params_dict '''
        self.I.data = params_dict['I']
        self.sigma_n.data = params_dict['sigma_n']
        self.W.data = torch.tensor([math.log(params_dict['w'][i]/params_dict['w'][0]) for i in range(len(params_dict['w']))])




    #Interior components
    @staticmethod
    def log_p_u_interior(u, I, sigma_n):
        return - torch.log(sigma_n) - 0.5*math.log(2*math.pi) -0.5*((u-I)/sigma_n)**2

    @staticmethod
    def log_p_u_cond_I(u, I, sigma_n):
        return -torch.log(sigma_n) -0.5*math.log(2*math.pi) -0.5*((u-I)/sigma_n)**2

    #Interface components
    def log_p_u_interface(self, u, N, Ia, Ib, sigma_n):
        '''Integral using MC sampling of uniform p(I) on the interval [Ia, Ib]'''

        #draw N samples form Uniform dist
        uniform_eps = torch.distributions.Uniform(torch.zeros(N), torch.ones(N)).sample()
        In = uniform_eps*(Ib-Ia)+Ia

        return   -math.log(N) \
                    + torch.logsumexp( self.log_p_u_cond_I(u, In.unsqueeze(-1), sigma_n), dim = 0)



    # #Interface components do have an analytical solution but log-implementation not possible. PROBLEM: log(zero)
    #### def log_p_u_interface(self, u, Ia, Ib, sigma_n):
    ####     return -torch.log(2*(Ib-Ia)) + torch.log( torch.erf( (u-Ia)/(math.sqrt(2)*sigma_n ))  - torch.log( torch.erf( (u-Ib)/(math.sqrt(2)*sigma_n )) )


    def log_p_u(self, u, n_MC_components):
        '''Combination of all model components'''

        log_p_list = []

        #Interior components
        log_p_list.append(self.log_p_u_interior(u, self.I.unsqueeze(-1), self.sigma_n))

        #Interface components
        for Ia in self.I:
            for Ib in self.I:
                if Ia<Ib: #to get each interface comp. only once
                    log_p_list.append(self.log_p_u_interface(u, n_MC_components, Ia, Ib, self.sigma_n).reshape(1,-1))

        log_p = torch.cat(log_p_list, 0)

        return torch.logsumexp( self.log_w_sm.unsqueeze(-1) + log_p, dim = 0)   #logsumexp for numerical stability


    @property
    def log_w_sm(self):
      return F.log_softmax(self.W, dim = 0)

    @property
    def w(self):
      return torch.exp(self.log_w_sm)

    def forward(self, u, n_MC_components):
        ''' Loss: sum over all datapoints $u_m$, $m = 1, 2, ..., M$: $$ L = - \frac{1}{M} \sum_m^M log(p(u_m)) $$ '''
        return - torch.sum (self.log_p_u( u, n_MC_components))/len(u)
