# A Physical Model for Microstructural Characterization and Segmentation for 3D Tomography Data

This repository contains the code for our [paper](http://arxiv.org/abs/2009.07218) [1], including a Pytorch implementation of the Blurred Interface Mixture Model (BIMM) for analysis of 3D tomography data.

Examples are provided of how to utilize the BIMM for direct quantification and segmentation on artificial and experimental data.

* **Direct quantification**: Determine material parameters directly from the raw image data, without segmentation; volume fractions, interface areas, phase intensities, image resolution and noise levels.

* **Segmentation**: Perform a maximum probability segmentation on the data based on the fitted model.



## Contents

* `/bimmquant/`: Code for the **bimmquant** library. See description below.
* `/examples/`: Runnable notebooks with examples of how to use the BIMM for structural quantification and segmentation.
    * `/art_data_generation.ipynb`: Code for generating an artificial dataset used for model verification.
    * `/art_data_quantification_and_segmentation.ipynb`: Quantification on artificial data, comparison with ground truth. Segmentation.
    * `/exp_data_quantification_and_segmentation.ipynb`: Quantification and segmentation of experimental data.
    * `/example_data/`: Data **not** included in repo - must be *downloaded* or *generate* according to instructions in the example notebooks.
      * `/SOCPristineTiffs/`: Folder with 500 *.tif* slices of the pristine fuel cell electrode. NB! Must be downloaded according to instructions in *examples/exp_data_quantification_and_segmentation.ipynb*.  
      * `art_data_2phases.npy`: Artificial dataset with two phases. NB! Must be generated with the provided notebook *examples/art_data_generation.ipynb*.
    * `/example_log/`: Directory where tensorboard logs will be saved after model fitting.

### The `bimmquant` library

* `/models/`: The different models considered in the paper: The 1D and 2D versions of the BIMM
* `/data/`: code for importing data for 1D (intensity) or 2D (intensity-gradient) model fitting (`bimm1d.py`, `bimm2d.py`), the Gaussian mixture model (GMM) `gmm.py`, and the Partial volume mixture model (PVMM) `pvmm.py`.
* `/utils/`: `tools.py` for plotting, logging, segmentation etc.



### Installation
In the folder containing `setup.py`, run

    pip install .


This code was tested on Ubuntu 18.04.4 LTS using Pytorch version 1.3.1.


### Reference

If you find this code useful, please consider citing:

[1] Brenne, E. O., Dahl, V. A., & Jørgensen, P. S. (2020). *A Physical Model for Microstructural Characterization and Segmentation of 3D Tomography Data*. [arXiv:2009.07218](http://arxiv.org/abs/2009.0n7218)
