import torch
import math
import numpy as np
import scipy.special
import torch.nn.functional as F



def logsinh(x):
    '''Numerically stable implementation of log(sinh(x)) (which otherwise fails at x>~90)'''
    s = torch.sign(x) * x
    p = -torch.exp(-2 * s)
    return - math.log(2) + s + torch.log1p(p)

def log_bessel_05(x):
    '''I_0.5(x) = sqrt(2/(pi x)) sinh(x)'''
    return 0.5*math.log(2) - 0.5*torch.log(math.pi*x) + logsinh(x)
    

def G(I, Ia, Ib, sigma_b):
    return ((Ib-Ia)/torch.sqrt(2*math.pi*sigma_b**2))*torch.exp(-( torch.erfinv( (  2*(I-Ia)/(Ib-Ia) -1 ) )**2))




class BIMM2D(torch.nn.Module): #inherits from Module class
    '''Arc model for material with n_phases'''

    def __init__(self, n_phases):
        super(BIMM2D, self).__init__() #initializes superclass, does set-up

        self.n_phases = n_phases
        self.n_interfaces = int(scipy.special.comb(self.n_phases, 2))#all different interface types

        self.I = torch.nn.Parameter(torch.Tensor(self.n_phases))
        self.sigma_b = torch.nn.Parameter(torch.Tensor(1))
        self.sigma_n = torch.nn.Parameter(torch.Tensor(1))
        self.d = torch.nn.Parameter(torch.Tensor(1))
        self.W = torch.nn.Parameter(torch.Tensor(self.n_interfaces+self.n_phases)) #weights before normalization
        self.r = torch.nn.Parameter(torch.Tensor(1)) #correlation before normalization to interval [-1, 1]

        #Set all parameters
        self.reset_params()


    def reset_params(self):
        '''Initialize model with component means (I) evenly spaced on [0, 1], same std for all components (sigma_n = 0.1), equal weights, sigma_b = 2.0, d = 2.0, no correlation (r = 0)'''
        self.I.data = torch.linspace(0, 1, self.n_phases)
        self.W.data = torch.zeros(self.n_interfaces+self.n_phases)
        self.sigma_b.data = torch.tensor(2.)
        self.sigma_n.data = torch.tensor(0.1)
        self.d.data = torch.tensor(2.)
        self.r.data = torch.tensor(0.)


    def set_params(self, params_dict):
        '''Initialize model with values specified in the dictionary params_dict '''
        self.I.data = params_dict['I']
        self.sigma_b.data = params_dict['sigma_b']
        self.sigma_n.data = params_dict['sigma_n']
        self.d.data = params_dict['d']
        self.W.data = torch.tensor([math.log(params_dict['w'][i]/params_dict['w'][0]) for i in range(len(params_dict['w']))])

        rho = params_dict['rho']
        assert torch.min(rho) >= -1 and torch.max(rho) <= 1, 'z must be in [-1,1]'
        self.r.data = 0.5 * torch.log((1 + rho) / (1 - rho)) #arctanh


    @property
    def log_w_sm(self):
      return F.log_softmax(self.W, dim=0)

    @property
    def w(self):
      return torch.exp(self.log_w_sm)

    @property
    def rho(self):
      return torch.tanh(self.r)


    #Interior components
    @staticmethod
    def log_p_u_v_interior(u, v, I, sigma_n, rho):
        return math.log(2) + 2*torch.log(v) - math.log(math.gamma(3/2)) - 3*torch.log(sigma_n*torch.sqrt(1-rho)) - (v/(sigma_n*torch.sqrt(1-rho)))**2 \
        - torch.log(sigma_n) - 0.5*math.log(2*math.pi) -0.5*((u-I)/sigma_n)**2

    @staticmethod
    def log_p_I(I, d, Ia, Ib):
        return -torch.log( 2*d*(Ib-Ia)) + 0.5*math.log(2*math.pi) + ( torch.erfinv( 2*(I-Ia)/(Ib-Ia) -1 ) )**2

    @staticmethod
    def log_p_u_cond_I(u, I, sigma_n):
        return -torch.log(sigma_n) -0.5*math.log(2*math.pi) -0.5*((u-I)/sigma_n)**2

    @staticmethod
    def log_p_v_cond_I(v, I, Ia, Ib, sigma_b, sigma_n, rho):

        G_ = G(I, Ia, Ib, sigma_b)
        log_bessel_term = log_bessel_05(2*v*G_/(sigma_n**2*(1-rho)))

        return math.log(2) -2*torch.log(sigma_n*torch.sqrt(1-rho)) + (3/2)*torch.log(v) - 0.5*torch.log(G_) \
        + log_bessel_term  -(v**2+G_**2)/(sigma_n**2*(1-rho))


    #Interface components
    def log_p_u_v_interface(self, u, v, N, d, Ia, Ib, sigma_b, sigma_n, rho):
        '''Integral using MC sampling from p(I)'''

        #draw N samples form Uniform dist
        uniform_eps = torch.distributions.Uniform(torch.zeros(N), torch.ones(N)).sample()
        uniform_x = uniform_eps*2*d*sigma_b -d*sigma_b

        #transform from u to p(I) distribution: using inverse p(I)
        In = (torch.erf(uniform_x/(math.sqrt(2)*sigma_b)) + 1)*0.5*(Ib-Ia) + Ia    #sigma_b can be removed - but kept for readability

        return   -math.log(N) \
                    + torch.logsumexp( self.log_p_u_cond_I(u, In.unsqueeze(-1), sigma_n) \
                    + self.log_p_v_cond_I(v, In.unsqueeze(-1), Ia, Ib, sigma_b, sigma_n, rho), dim=0)


    def log_p_u_v(self, u, v, n_MC_components):
        '''Combination of all model components'''

        log_p_list = []

        #Interior components
        log_p_list.append(self.log_p_u_v_interior(u, v, self.I.unsqueeze(-1), self.sigma_n, self.rho))

        #Interface components
        for Ia in self.I:
            for Ib in self.I:
                if Ia<Ib: #to get each interface comp. only once
                    log_p_list.append(self.log_p_u_v_interface(u, v, n_MC_components, self.d, Ia, Ib, self.sigma_b, self.sigma_n, self.rho).reshape(1,-1))

        log_p = torch.cat(log_p_list, 0)

        return torch.logsumexp( self.log_w_sm.unsqueeze(-1) + log_p, dim=0) #logsumexp for numerical stability
        

    def forward(self, u, v, n_MC_components):
        ''' Loss: sum over all datapoints $u_m$, $v_m$, $m = 1, 2, ..., M$: $$ L = - \frac{1}{M} \sum_m^M log(p(u_m, v_m)) $$ '''
        return - torch.sum (self.log_p_u_v( u, v, n_MC_components))/len(u)
