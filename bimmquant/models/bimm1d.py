import torch
import math
import numpy as np
import scipy.special
import torch.nn.functional as F



class BIMM1D(torch.nn.Module): #inherits from Module class
    '''Arc model for material with n_phases'''

    def __init__(self, n_phases):
        super(BIMM1D, self).__init__() #initializes superclass, does set-up

        self.n_phases = n_phases
        self.n_interfaces = int(0.5*(self.n_phases-1)*self.n_phases)#all different interface types

        self.I = torch.nn.Parameter(torch.Tensor(self.n_phases))
        self.sigma_b = torch.nn.Parameter(torch.Tensor(1))
        self.sigma_n = torch.nn.Parameter(torch.Tensor(1))
        self.d = torch.nn.Parameter(torch.Tensor(1))
        self.W = torch.nn.Parameter(torch.Tensor(self.n_interfaces+self.n_phases)) #weights before normalization




        #Set all parameters
        self.reset_params()


    def reset_params(self):
        '''Initialize model with component means (I) evenly spaced on [0, 1], same std for all components (sigma_n = 0.1), equal weights, d = 2 and aigms_b = 2'''
        self.I.data = torch.linspace(0, 1, self.n_phases)
        self.sigma_b.data = torch.tensor(2.) #muligens fjerne seinare? 1D avh. ikkje av sigma_b, men kanskje tydelegare likningar slik...
        self.sigma_n.data = torch.tensor(0.1)
        self.d.data = torch.tensor(2.)
        self.W.data = torch.zeros(self.n_interfaces+self.n_phases)


    def set_params(self, params_dict):
        '''Initialize model with values specified in the dictionary params_dict '''
        self.I.data = params_dict['I']
        self.sigma_b.data = params_dict['sigma_b'] #muligens fjerne seinare? 1D avh. ikkje av sigma_b, men kanskje tydelegare likningar slik...
        self.sigma_n.data = params_dict['sigma_n']
        self.d.data = params_dict['d']
        self.W.data = torch.tensor([math.log(params_dict['w'][i]/params_dict['w'][0]) for i in range(len(params_dict['w']))])




    #Interior components
    @staticmethod
    def log_p_u_interior(u, I, sigma_n):
        return - torch.log(sigma_n) - 0.5*math.log(2*math.pi) -0.5*((u-I)/sigma_n)**2

    @staticmethod
    def log_p_I(I, d, Ia, Ib):
        return -torch.log( 2*d*(Ib-Ia)) + 0.5*math.log(2*math.pi) + ( torch.erfinv( 2*(I-Ia)/(Ib-Ia) -1 ) )**2

    @staticmethod
    def log_p_u_cond_I(u, I, sigma_n):
        return -torch.log(sigma_n) -0.5*math.log(2*math.pi) -0.5*((u-I)/sigma_n)**2

    #Interface components
    def log_p_u_interface(self, u, N, d, Ia, Ib, sigma_b, sigma_n):
        '''Integral using MC sampling from p(I)'''

        #draw N samples form Uniform dist
        uniform_eps = torch.distributions.Uniform(torch.zeros(N), torch.ones(N)).sample()
        uniform_x = uniform_eps*2*d*sigma_b -d*sigma_b

        #transform from u to p(I) distribution: using inverse p(I)
        In = (torch.erf(uniform_x/(math.sqrt(2)*sigma_b)) + 1)*0.5*(Ib-Ia) + Ia # =I(x) !!  #sigma_b can be removed - but kept for readability

        #!!!! In practice, this funciton is INDEP. of sigma_b!!
        #In = (torch.erf( d*(2*uniform_eps-1)/math.sqrt(2)) + 1)*0.5*(Ib-Ia) + Ia # =I(uniform_eps) !!


        return   -math.log(N) \
                    + torch.logsumexp( self.log_p_u_cond_I(u, In.unsqueeze(-1), sigma_n), dim = 0)


    def log_p_u(self, u, n_MC_components):
        '''Combination of all model components'''

        log_p_list = []

        #Interior components
        log_p_list.append(self.log_p_u_interior(u, self.I.unsqueeze(-1), self.sigma_n))

        #Interface components
        for Ia in self.I:
            for Ib in self.I:
                if Ia<Ib: #to get each interface comp. only once
                    log_p_list.append(self.log_p_u_interface(u, n_MC_components, self.d, Ia, Ib, self.sigma_b, self.sigma_n).reshape(1,-1))

        log_p = torch.cat(log_p_list, 0)

        return torch.logsumexp( self.log_w_sm.unsqueeze(-1) + log_p, dim = 0)  #logsumexp for numerical stability


    @property
    def log_w_sm(self):
      return F.log_softmax(self.W, dim = 0)

    @property
    def w(self):
      return torch.exp(self.log_w_sm)

    def forward(self, u, n_MC_components):
        ''' Loss: sum over all datapoints $u_m$, $m = 1, 2, ..., M$: $$ L = - \frac{1}{M} \sum_m^M log(p(u_m)) $$ '''
        return - torch.sum (self.log_p_u( u, n_MC_components))/len(u)
