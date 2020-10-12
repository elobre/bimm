import numpy as np
import copy
import torch
import matplotlib.pyplot as plt


import os
from skimage import io
import matplotlib as mpl
import scipy


def normalize(arr, amin=0, amax=1):
    arr01=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
    return arr01*(amax-amin)+amin

def plot_1D_histogram(data, bins=100, fig_external=[], title='', ylabel='Frequency', xlabel='', alpha=0.8, figsize=(10,6), color='silver', edgecolor='silver', density=False):
    if len(fig_external)==0:
        fig,ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = fig_external[0]
        ax = fig_external[1]
    ax.set_title(title)
    data_hist =np.histogram(data, bins=bins, density=density)
    ax.hist(data, bins=bins, alpha=alpha, color=color, edgecolor=edgecolor)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return data_hist

def plot_2D_histogram(intensity_toPlot, gradient_toPlot, bins=1000,  norm_hist=False, bin_arrays=[],
                    title='', cbar_label='', xlabel='Intensity', ylabel='Gradient magnitude', fig_external=[], colorbar=True, figsize=(8,8), cmap_gamma=0.2, cmap='viridis', density=False):
    '''bins is int or [array, array] '''

    if len(bin_arrays)==2:
        hist_bins=bin_arrays
    else:
        hist_bins=bins

    hist=np.histogram2d(intensity_toPlot, gradient_toPlot, bins=hist_bins, density=density)
    zzz=hist[0]

    dx=hist[1][1]-hist[1][0]
    dy=hist[2][1]-hist[2][0]
    xxx=hist[1][1:]-dx
    yyy=hist[2][1:]-dy
    X,Y=np.meshgrid(xxx,yyy)


    if norm_hist:
        bin_area = (hist[1][1]-hist[1][0])*(hist[2][1]-hist[2][0])
        #bin_area = (xxx[1]-xxx[0])*(yyy[1]-yyy[0])
        zzz/=(np.sum(zzz))*bin_area

    if len(fig_external)==0:
        fig,ax = plt.subplots(1, figsize=figsize)
    else:
        fig = fig_external[0]
        ax = fig_external[1]

    im=ax.pcolormesh(X,Y, zzz.T, cmap=cmap, norm=mpl.colors.PowerNorm(gamma=cmap_gamma), shading='nearest')

    ax.set_title(title)
    if colorbar:
        cbar=fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()

    return zzz, hist


def plot_2D_scatter_plot(intensity_toPlot, gradient_toPlot, colors=[], n_samples=0, xlim = False, ylim = False,
                    title='', cbar_label='', xlabel='Intensity', ylabel='Gradient magnitude', fig_external=[], set_colorbar=False, figsize=(8,8), alpha=0.2):
    '''Plots 2D scatter plot with colors according to e.g. segmentation labels'''

    if n_samples==0:
        inds = np.arange(len(intensity_toPlot))
    else:
        inds = np.random.randint(0, len(intensity_toPlot), n_samples)

    if len(colors)==0:
        colors = 'k'
    else:
        colors = colors[inds]


    if len(fig_external)==0:
        fig,ax = plt.subplots(1, figsize=figsize)
    else:
        fig = fig_external[0]
        ax = fig_external[1]

    im = ax.scatter(intensity_toPlot[inds], gradient_toPlot[inds],
                c = colors, alpha = 0.1 )

    ax.set_title(title)
    if set_colorbar:
        cbar=fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim:
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    plt.tight_layout()



def log_to_params_dict(log_dict):
    '''Prepares dict containing all parameters from log format tomodel initialization format'''

    params_dict = {}
    # I_keys = [key for key in log_dict if 'params/I' in key]
    # I_keys_sorted = [I_keys[i] for i in np.argsort([int(''.join([i for i in key if i.isdigit()])) for key in I_keys])]
    # params_dict['I']=torch.tensor([log_dict[key] for key in I_keys_sorted])
    I_keys = [key for key in log_dict if 'params/I' in key]
    params_dict['I']=torch.tensor([log_dict[key] for key in np.sort(I_keys)])  #fungerar for I1, I2, I3 men ikkje w1 w12 osv!!



    w_keys = [key for key in log_dict if 'weights/w' in key]
    w_keys_sorted = [w_keys[i] for i in np.argsort([int(''.join([i for i in key if i.isdigit()])) for key in w_keys])]
    params_dict['w']=torch.tensor([log_dict[key] for key in w_keys_sorted])
    #params_dict['w']=torch.tensor([log_dict[key] for key in np.sort(w_keys)])

    n_keys = [key for key in log_dict if 'params/sigma_n' in key]
    #n_keys_sorted = [n_keys[i] for i in np.argsort([int(''.join([i for i in key if i.isdigit()])) for key in n_keys])]
    #sigma_n=torch.tensor([log_dict[key] for key in n_keys_sorted])
    sigma_n=torch.tensor([log_dict[key] for key in np.sort(n_keys)])

    if sigma_n.shape==torch.Size([1]): #not GMM
        params_dict['sigma_n']=sigma_n[0]
    else:
        params_dict['sigma_n']=sigma_n #GMM

    try: #ARC2D ARC1D
        params_dict['d'] = torch.tensor(log_dict['params/d'])
    except:
        pass


    try: #ARC2D (og ARC1D foreløpig, skal fjernast)
        params_dict['sigma_b'] = torch.tensor(log_dict['params/sigma_b'])
    except:
        pass

    try: #ARC2D
        params_dict['rho'] = torch.tensor(log_dict['params/rho'])
    except:
        pass

    return params_dict

def res1090(sigma_b):
    return sigma_b * 2*np.sqrt(2)*scipy.special.erfinv(0.8)

def get_V(resdict):
    #resdict=resdict_to_last_params_mean(filename)
    try:
        V1=resdict['weights/w1']
        V2=resdict['weights/w2']
        n_phases=2
    except:
        pass

    try:
        V1=resdict['weights/w1']+0.5*resdict['weights/w12']
        V2=resdict['weights/w2']+0.5*resdict['weights/w12']
        n_phases=2
    except:
        pass


    try:
        V1=resdict['w'][0]
        V2=resdict['w'][1]
        n_phases=2
    except:
        pass
    try:
        V1=resdict['w'][0]+0.5*resdict['w'][2]
        V2=resdict['w'][1]+0.5*resdict['w'][2]
        n_phases=2
    except:
        pass

    try:
        V3=resdict['weights/w3']
        V1=resdict['weights/w1']
        V2=resdict['weights/w2']
        n_phases=3
    except:
        pass

    try:
        V3=resdict['weights/w3']+0.5*resdict['weights/w13']+0.5*resdict['weights/w23']
        V1=resdict['weights/w1']+0.5*resdict['weights/w12']+0.5*resdict['weights/w13']
        V2=resdict['weights/w2']+0.5*resdict['weights/w12']+0.5*resdict['weights/w23']
        n_phases=3
    except:
        pass


    try:
        V3=resdict['w'][2]
        V1=resdict['w'][0]
        V2=resdict['w'][1]
        n_phases=3
    except:
        pass
    try:
        V3=resdict['w'][2]+0.5*resdict['w'][5]+0.5*resdict['w'][4]
        V1=resdict['w'][0]+0.5*resdict['w'][3]+0.5*resdict['w'][4]
        V2=resdict['w'][1]+0.5*resdict['w'][3]+0.5*resdict['w'][5]
        n_phases=3
    except:
        pass


    if n_phases == 3:
        return V1, V2, V3
    elif n_phases == 2:
        return V1, V2



def get_A(resdict, voxelsize):
    #resdict=resdict_to_last_params_mean(filename)
    try:
        A12 = (resdict['w'][2]/(2*resdict['d']*resdict['sigma_b']))/voxelsize
        n_phases=2
    except:
        pass

    try:
        A12 = (resdict['weights/w12']/(2*resdict['params/d']*resdict['params/sigma_b']))/voxelsize
        n_phases=2
    except:
        pass

    try:
        A12 = (resdict['w'][3]/(2*resdict['d']*resdict['sigma_b']))/voxelsize
        A13 = (resdict['w'][4]/(2*resdict['d']*resdict['sigma_b']))/voxelsize
        A23 = (resdict['w'][5]/(2*resdict['d']*resdict['sigma_b']))/voxelsize
        n_phases=3
    except:
        pass

    try:
        A12 = (resdict['weights/w12']/(2*resdict['params/d']*resdict['params/sigma_b']))/voxelsize
        A13 = (resdict['weights/w13']/(2*resdict['params/d']*resdict['params/sigma_b']))/voxelsize
        A23 = (resdict['weights/w23']/(2*resdict['params/d']*resdict['params/sigma_b']))/voxelsize
        n_phases=3
    except:
        pass

    if n_phases == 3:
        return A12, A13, A23
    elif n_phases == 2:
        return A12



from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def tabulate_events_onerun(summary_iterator):

    tags = summary_iterator.Tags()['scalars']

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterator.Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in [summary_iterator]]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps

def get_mean_lastvals(runpath, nvals_mean):
    summary_iterator = EventAccumulator(runpath).Reload()

    out, steps = tabulate_events_onerun(summary_iterator)

    mean_dict={}
    std_dict={}
    for key in out.keys():
        mean_dict[key]=np.mean(out[key][-nvals_mean:])
        std_dict[key]=np.std(out[key][-nvals_mean:])

    return mean_dict, std_dict


def plot_2D_image(image_array, title='', cmap='Greys_r', axis='on', fig_external=[], xvals=[], yvals=[], figsize=(5,5), colorbar=False,xlabel='', ylabel='', cbar_label='', vmin=None, vmax=None, cmap_gamma=1.):
    '''send in fig_external=[fig,ax] if figure should be plotted in subplot'''

    if len(fig_external)==0:
        fig,ax = plt.subplots(1, figsize=figsize)
    else:
        fig = fig_external[0]
        ax = fig_external[1]

    if len(xvals) == 0:
        xxx=np.arange(0, np.shape(image_array)[1])
        yyy=np.arange(0, np.shape(image_array)[0])
    else:
        xxx=yvals
        yyy=xvals

    X,Y=np.meshgrid(xxx,yyy)

    ax.axis(axis)
    im=ax.pcolormesh(Y,X, image_array, cmap=cmap, vmin=vmin, vmax=vmax, norm=mpl.colors.PowerNorm(gamma=cmap_gamma), shading='nearest')
    ax.set_title(title)

    if colorbar:
        cbar=fig.colorbar(im)
        cbar.set_label(cbar_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()



def plot_2D_gridplot(model, Ivals, Gvals, n_MC_components=1000, figsize=(10,10), fig_external = [], cmap='viridis',cmap_gamma=0.2,
                            title='Model evaluated on grid', xlabel='', ylabel='', cbar_label='', set_colorbar=True, axis='on'):
    '''will plot the model on a nbins*nbins grid
    Ivals, Gvals should be linspace arrays determining the grid'''

    xxx=Ivals
    yyy=Gvals
    X,Y=np.meshgrid(xxx,yyy)

    x_grid=torch.tensor(X.ravel()).float()
    y_grid=torch.tensor(Y.ravel()).float()

    #evaluate model in grid values, then reshape to grid dimensions
    model_2D=torch.exp(model.log_p_u_v(x_grid, y_grid, n_MC_components))

    model_2D_image= np.reshape(model_2D.detach().numpy(), (len(Ivals), len(Gvals))).T

    if len(fig_external)==0:
        fig,ax = plt.subplots(1, figsize=figsize)
    else:
        fig = fig_external[0]
        ax = fig_external[1]

    ax.axis(axis)
    im=ax.pcolormesh(X,Y, model_2D_image.T, cmap=cmap, norm=mpl.colors.PowerNorm(gamma=cmap_gamma), shading='nearest')
    ax.set_title(title)

    if set_colorbar:
        cbar=fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()

    #plt.show()
    return model_2D_image.T





def evaluate_log_prob(model, u_in, v_in, nMC, n_labels = 0):
    '''evaluate log probability for each model component - interior or interface'''

    shape_in = u_in.shape
    u = u_in.flatten()
    v = v_in.flatten()

    log_p_list = []

    #interior components
    log_p_list.append(model.log_p_u_v_interior(u, v, model.I.unsqueeze(-1), model.sigma_n, model.rho))

    #Interface components
    for Ii in model.I:
        for Ij in model.I:
            if Ii < Ij:
                log_pij = model.log_p_u_v_interface(u, v, nMC, model.d, Ii, Ij, model.sigma_b, model.sigma_n, model.rho)
                log_p_list.append(log_pij.reshape(1,-1))


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
                    threshold = np.mean([ (model.I[i]).detach().numpy(), (model.I[j]).detach().numpy()])

                    interface_label = interface_labels[interface_label_ind]

                    max_prob_labels_out[(max_prob_labels == interface_label) * (u< threshold)] = i
                    max_prob_labels_out[(max_prob_labels == interface_label) * (u >= threshold) ]= j

                    interface_label_ind += 1

        return max_prob_labels_out.reshape(shape_in)
    else:
        print('n_labels must be equal to number of phases OR number of components!')





def plot_center_slices(volume, title='', fig_external=[],figsize=(15,5), cmap='Greys_r', colorbar=False, vmin=None, vmax=None):
        shape=np.shape(volume)

        if len(fig_external)==0:
            fig,ax = plt.subplots(1,3, figsize=figsize)
        else:
            fig = fig_external[0]
            ax = fig_external[1]

        fig.suptitle(title)
        im=ax[0].imshow(volume[:,:, int(shape[2]/2)], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        ax[0].set_title('Center z slice')
        ax[0].set_xlabel('y')
        ax[0].set_ylabel('x')
        ax[1].imshow(volume[:,int(shape[1]/2),:], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        ax[1].set_title('Center y slice')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('x')
        ax[2].imshow(volume[int(shape[0]/2),:,:], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        ax[2].set_title('Center x slice')
        ax[2].set_xlabel('z')
        ax[2].set_ylabel('y')

        if colorbar:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)



#
def gradientMagnitude3D(vol):
    ''' computes 3D gradient magnitude through central difference'''
    gradx, grady, gradz = np.gradient(vol)
    return np.sqrt(gradx**2 + grady**2+ gradz**2)


# USED IN EXPERIMENTS:
def tb_log(writer, current_log_dict, it):
    for key, value in current_log_dict.items():
        writer.add_scalar(key, value, it)



def get_log_dict(model):
    '''Prepares dict containing all parameters that should be logged/averaged
    for the BIMM-2D model'''

    log_dict = {}

    n_phases=len(model.I)

    if n_phases==2:

        log_dict['params/I2'] = model.I[1].data
        log_dict['weights/w2'] = model.w[1].data
        log_dict['weights/w12'] = model.w[2].data

    elif n_phases==3:
        log_dict['params/I2'] = model.I[1].data
        log_dict['params/I3'] = model.I[2].data
        log_dict['weights/w2'] = model.w[1].data
        log_dict['weights/w3'] = model.w[2].data
        log_dict['weights/w12'] = model.w[3].data
        log_dict['weights/w13'] = model.w[4].data
        log_dict['weights/w23'] = model.w[5].data


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




def plot_model_with_density_1D_hist(model, Modelname, u_data, u_plot, n_MC_components=10000, nbins=500, linewidth=2, figsize=(16,8),fig_external=[], legend=False, individual_components=False,
                                            xlabel='', ylabel='', yticks=True, ytickslabels=True,
                                            title='Intensity histogram with 1D model p(u)',
                                   Modellabel=''):

    if len(fig_external)==0:
        fig,ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = fig_external[0]
        ax = fig_external[1]

    #Histogram
    _=plt.hist(u_data, density=True, bins=nbins,  color='silver',linewidth=0.1,  edgecolor='silver', alpha=0.8)


    n_phases = len(model.I)
    interface_ind = n_phases-1

    if individual_components:

        if Modelname == 'GMM':
            for Ii, sigma_ni, wi in zip(model.I, model.sigma_n, model.w):
                plt.plot(u_plot, wi.detach()*torch.exp(model.log_p_u_interior(u_plot, Ii, sigma_ni).detach()), linewidth=linewidth, label=Modellabel)


        elif Modelname in ['PVGMM', 'BIMM1D']:


            #Interior components
            for Ib, wi in zip(model.I, model.w[:n_phases]): #will get the len(I) first weights
                plt.plot(u_plot, wi.detach()*torch.exp(model.log_p_u_interior(u_plot, Ib, model.sigma_n).detach()), linewidth=linewidth, label=Modellabel)

                #Interface components
                for Ia in model.I:
                    if Ia<Ib:
                        interface_ind+=1
                        wij = model.w[interface_ind]

                        if Modelname == 'BIMM1D':
                            plt.plot(u_plot, wij.detach()*torch.exp(model.log_p_u_interface(u_plot, n_MC_components, model.d, Ia, Ib, model.sigma_b, model.sigma_n).detach()), linewidth=linewidth, label='')

                        if Modelname == 'PVGMM':
                            plt.plot(u_plot, wij.detach()*torch.exp(model.log_p_u_interface(u_plot, n_MC_components, Ia, Ib, model.sigma_n).detach()), linewidth=linewidth, label='')


    #total model
    if Modelname == 'GMM':
        plt.plot(u_plot, torch.exp(model.log_p_u(u_plot).detach()), 'm', alpha=0.8, linewidth=2*linewidth)
    elif Modelname in ['PVGMM', 'BIMM1D']:
        plt.plot(u_plot, torch.exp(model.log_p_u(u_plot, n_MC_components).detach()), 'm', alpha=0.8, linewidth=2*linewidth)

    if legend:
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=yticks,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft=ytickslabels)
    plt.title(title)
