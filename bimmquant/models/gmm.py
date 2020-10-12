import torch
import math
import numpy as np
import scipy.special
import torch.nn.functional as F



class GMM(torch.nn.Module):
    '''Gaussian mixture model with number of components: n_phases'''

    def __init__(self, n_phases):
        super(GMM, self).__init__()

        self.n_phases=n_phases

        self.I=torch.nn.Parameter(torch.Tensor(self.n_phases))
        self.sigma_n=torch.nn.Parameter(torch.Tensor(self.n_phases))
        self.W=torch.nn.Parameter(torch.Tensor(self.n_phases)) #weights before normalization

        #Set all parameters
        self.reset_params()


    def reset_params(self):
        '''Initialize model with component means (I) evenly spaced on [0, 1], same std for all components (sigma_n = 0.1) and equal weights'''
        self.I.data = torch.linspace(0, 1, self.n_phases)
        self.sigma_n.data = torch.ones(self.n_phases)*0.1
        self.W.data = torch.zeros(self.n_phases)


    def set_params(self, params_dict):
        '''Initialize model with values specified in the dictionary params_dict '''
        self.I.data = params_dict['I']
        self.W.data = torch.tensor([math.log(params_dict['w'][i]/params_dict['w'][0]) for i in range(len(params_dict['w']))])

        if params_dict['sigma_n'].shape==torch.Size([self.n_phases]):
            self.sigma_n.data = params_dict['sigma_n']
        elif params_dict['sigma_n'].shape==torch.Size([]): #only one number
            self.sigma_n.data = torch.tensor([params_dict['sigma_n'] for _ in range(self.n_phases)])



    #Interior components (the GMM does not have interface components - only one interior component for each material phase)
    @staticmethod
    def log_p_u_interior(u, I, sigma_n):
        return - torch.log(sigma_n) - 0.5*math.log(2*math.pi) -0.5*((u-I)/sigma_n)**2

    def log_p_u(self, u):
        '''Combination of all model components'''

        log_p_list=[]

        #Interior components
        log_p_list.append(self.log_p_u_interior(u, self.I.unsqueeze(-1), self.sigma_n.unsqueeze(-1)))

        log_p = torch.cat(log_p_list, 0)

        return torch.logsumexp( self.log_w_sm.unsqueeze(-1) + log_p, dim=0)   #logsumexp for numerical stability


    @property
    def log_w_sm(self):
      return F.log_softmax(self.W, dim=0)

    @property
    def w(self):
      return torch.exp(self.log_w_sm)

    def forward(self, u):
        ''' Loss: sum over all datapoints $u_m$, $m = 1, 2, ..., M$: $$ L = - \frac{1}{M} \sum_m^M log(p(u_m)) $$ '''
        return - torch.sum (self.log_p_u(u))/len(u)
