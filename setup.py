import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='bimmquant',
      version='0.1',
      description='Package for fitting statistical models to 3D tomography data for quantification of material parameters and segmentation. Models include the blurred interface mixture model (BIMM), the Gaussian mixture model (GMM) and the partial volume mixture model (PVMM).',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/elobre/bimm',
      #zip_safe=False,
      #license='GPLv3',
      #scripts=['scripts/initialize_notebook_data_to_npy'],
      packages=setuptools.find_packages(),
      #install_requires=[#jupyter notebook?? python?
      #     'nbformat',
      # ],
      author='Elise Otterlei Brenne')
