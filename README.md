# TLGP: Transfer Learning for Genomic Prediction

# Software Description

Here are the software scripts implementing our machine learning
methods for multi-ancestral clinico-genomic prediction of diseases and
reproducing the results described in the manuscript.

  ---------------------------------------------------------------------------------
  **Entity**    **Path/location**                      **Note**
  ------------- -------------------------------------- ----------------------------
  Data          ./data/instructions.txt                The path of the real and
                                                       synthetic datasets

  Features      ./data/Features.xlsx                   The features used for each
                                                       study

  Model         ./\*.py/build_model                    The interface used to build
  generator                                            a deep learning model

  Mixture       ./scripts/\*.py/mixture_learning       The mixture learning scheme
                                                       for each study.

  Independent   ./scripts/\*.py/independent_learning   The independent learning
                                                       scheme for each study

  Naïve         ./scripts/\*.py/naive_transfer         The Naïve transfer learning
  Transfer                                             scheme for each study

  Transfer      ./scripts/\*.py/super_transfer         The transfer learning scheme
                                                       for each study

  TL_PRS        ./script/LR/TL_PRS.py                  The implementation of the
                                                       TL_PRS method
                                                       
  ---------------------------------------------------------------------------------

# System Requirements

## Software dependency

The system relies on the following software, reagent, or resources.

## Software version

Our software has been tested on the following software version.

  -----------------------------------------------------------------------------------------------------------------
  **Software and         **SOURCE**        **IDENTIFIER**
  Hardware**                               
  ---------------------- ----------------- ------------------------------------------------------------------------
  REAGENT or RESOURCE    SOURCE            IDENTIFIER

  Python 3.7             Python Software   <https://www.python.org/download/releases/2.7/>
                         Foundation               

  Numpy 1.15.4           Tidelift, Inc     <https://libraries.io/pypi/numpy/1.15.4>

  Numpydoc 0.9.1         Tidelift, Inc     <https://libraries.io/pypi/numpydoc>

  Scipy 1.2.1            The SciPy         <https://docs.scipy.org/doc/scipy-1.2.1/reference/>
                         community         

  Sklearn 0.0            The Python        <https://pypi.org/project/sklearn/>
                         community         

  Keras 2.2.4            GitHub, Inc.      <https://github.com/keras-team/keras/releases/tag/2.2.4>

  Keras-Applications     GitHub, Inc.      <https://github.com/keras-team/keras-applications>
  1.0.8                                    

  Keras-Preprocessing    GitHub, Inc.      <https://github.com/keras-team/keras-preprocessing/releases/tag/1.1.0>
  1.1.0                                    

  Tensorboard 1.13.1     GitHub, Inc.      <https://github.com/tensorflow/tensorboard/releases/tag/1.13.1>

  Tensorflow 1.13.1      tensorflow.org    <https://www.tensorflow.org/install/pip>

  Tensorflow-estimator   The Python        <https://pypi.org/project/tensorflow-estimator/>
  1.13.1                 community         

  Statsmodels 0.9.0      Statsmodels.org   <https://www.statsmodels.org/stable/release/version0.9.html>

  Xlrd 1.2.0             The Python        <https://pypi.org/project/xlrd/>
                         community         

  XlsxWriter 1.1.8       The Python        <https://pypi.org/project/XlsxWriter/>
                         community         

  Xlwings 0.15.8         The Python        <https://pypi.org/project/xlwings/>
                         community         
  Xlwt 1.3.0             The Python        <https://pypi.org/project/xlwt/>
                         community

  -----------------------------------------------------------------------------------------------------------------

## Hardware requirements

We recommend using a GPU (V100) for optimal software performance.

# Installation Guide

This package contains the source code and instructions to reproduce the
results represented in our paper. Our software would run on Windows and
Ubuntu, but we suggest using Linux system which is easier for
environment configuration.

Conda --install requirements.txt

Requirements.txt

numpy==1.15.4

numpydoc==0.9.1

scipy==1.2.1

seaborn==0.9.0

sklearn==0.0

skrebate==0.6

torch==1.7.1

Keras==2.2.4

Keras-Applications==1.0.8

Keras-Preprocessing==1.1.0

tensorboard==1.13.1

tensorflow==1.13.1

tensorflow-estimator==1.13.0

statsmodels==0.9.0

lifelines==0.16.3

Optunity==1.1.1

xlrd==1.2.0-

XlsxWriter==1.1.8

xlwings==0.15.8

xlwt==1.3.0



## Authors

Yan Gao and Yan Cui, ({ygao45, ycui2}@uthsc.edu).

## License

This project is covered under the GNU General Public License (GPL).
