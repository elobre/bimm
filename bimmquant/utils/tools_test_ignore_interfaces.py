import numpy as np
import copy
import torch
import matplotlib.pyplot as plt


import os
from skimage import io
import matplotlib as mpl
import scipy



def get_log_dict(model):
    '''Prepares dict containing all parameters that should be logged/averaged
    for the BIMM-2D model'''

    log_dict = {}

    n_phases=len(model.I)

    if n_phases==2:

        log_dict['params/I2'] = model.I[1].data
        log_dict['weights/w2'] = model.w[1].data
        try:
            log_dict['weights/w12'] = model.w[2].data
        except:
            pass

    elif n_phases==3:
        log_dict['params/I2'] = model.I[1].data
        log_dict['params/I3'] = model.I[2].data
        log_dict['weights/w2'] = model.w[1].data
        log_dict['weights/w3'] = model.w[2].data
        try:
            log_dict['weights/w12'] = model.w[3].data
        except:
            pass
        try:
            log_dict['weights/w13'] = model.w[4].data
        except:
            pass
        try:
            log_dict['weights/w23'] = model.w[5].data
        except:
            pass


    else:
        if n_phases!=1:
            print('Logging only implemented for 1, 2 and 3 phases! Default: logg 1 phase')

    log_dict['params/I1'] = model.I[0].data
    log_dict['weights/w1'] = model.w[0].data

    log_dict['params/d'] = model.d.data
    log_dict['params/sigma_b'] = model.sigma_b.data
    log_dict['params/rho'] = model.rho.data
    log_dict['params/sigma_n'] = model.sigma_n.data

    return log_dict




def evaluate_log_prob(model, u_in, v_in, nMC, n_labels = 0):
    '''evaluate log probability for each model component - interior or interface'''

    shape_in = u_in.shape
    u = u_in.flatten()
    v = v_in.flatten()

    log_p_list = []

    #interior components
    log_p_list.append(model.log_p_u_v_interior(u, v, model.I.unsqueeze(-1), model.sigma_n, model.rho))

    #Interface components
    interface_ind = 0
    for Ii in model.I:
        for Ij in model.I:
            if Ii < Ij:
                if interface_ind not in model.ignore_interfaces:
                    log_pij = model.log_p_u_v_interface(u, v, nMC, model.d, Ii, Ij, model.sigma_b, model.sigma_n, model.rho)
                    log_p_list.append(log_pij.reshape(1,-1))
                interface_ind += 1

    log_p = torch.cat(log_p_list, 0)
    max_prob_labels = torch.argmax(log_p, dim=0)

    labels_unique = np.unique(max_prob_labels)
    interface_labels = labels_unique[len(model.I):]


    u = u.numpy()
    max_prob_labels=max_prob_labels.numpy()

    if  n_labels == np.max(max_prob_labels)+1: #want interface labels as well
        return max_prob_labels.reshape(shape_in)

    elif  n_labels == len(model.I): #split interface voxels into interiors using intensity threshold

        max_prob_labels_out =  copy.copy(max_prob_labels)

        interface_label_ind = 0
        for i in range(len(model.I)):
            for j in range(len(model.I)):
                if i < j:
                    if interface_label_ind not in model.ignore_interfaces:

                        threshold = np.mean([ (model.I[i]).detach().numpy(), (model.I[j]).detach().numpy()])

                        interface_label = interface_labels[interface_label_ind]

                        max_prob_labels_out[(max_prob_labels == interface_label) * (u< threshold)] = i
                        max_prob_labels_out[(max_prob_labels == interface_label) * (u >= threshold) ]= j

                    interface_label_ind += 1

        return max_prob_labels_out.reshape(shape_in)
    else:
        print('n_labels must be equal to number of phases OR number of components!')
