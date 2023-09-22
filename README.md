# Variational Unsupervised Domain Translation
This code allows to replicate the experiments presented in the article <span style="color: blue"> *Unsupervised Multiple Domain Translation through Controlled Disentaglement in Variational Autoencoder* </span> (available soon). The present are explained below.


## To begin with
To create the working environment to run the different experiments you must execute the following chunk of code:
```
python3 -m venv venv
source ./venv/bin/activate
python setup.py develop
pip install -r requirements.txt
```


## Launch experiments
The following explains how to launch the experiments proposed in the paper. To do so, each of the scripts found in the **bin** folder is used. 
In all of them, some of their input arguments are of the form *config_x*. In this case, the different values that *config_x* can take are those found in the **configs/x** directory.
The results of all experiments are stored in the **results** folder which will be created in case it do not exist.
We do not provide the trained models or the results since the goal of the paper is not to provide a big pretrained model but present a method. The aim of this code is to help to understand this method.


### Train, Test and Translate VAE
All the results associated with this section are obtained with the **run_all.py** script located at **bin**. It is used as follows:
```
python run_all.py config_data config_model config_training
```
For example, in case we want to run the experiment for the VAE proposed in the paper for MNIST, we would have:
```
python run_all.py mnist small-vae adam-128-1e-3-100
```
The results of this training are the following:
* The models at some epochs, which will be in **results/.../models**.
* Some reconstructions obtained during training in some epochs to control that the reconstruction is converging can be found in **results/.../images**.
* Predictions and translations after the training is done, which can be found in **results/.../preds**.

(where **...** means **config_data/config_model/config_training**)

### Classification to verify disentanglement
To verify the disentanglement of the two variables, we propose an experiment that uses the two latent variables to train a classifier. To launch this experiment, we must use **classification.py** in a similar manner to **run_all.py**. We must specify with the **--freeze_encoder** flag that we do not want to finetune the encoder. Otherwise, it will finetune the encoder parameters and the accuracy will end up being high for the two latent variables. This script is used as follows:
```
python classification.py config_data config_model config_training --freeze_encoder
```
Here config_training refers to the configuration used to train the VAE and not the one that is gone to be used to train the classifier. The latter is trained in all cases with the configuration found in **configs/training/sgd-128-1e-3.yaml**.
An example of how to launch this experiment with for MNIST is given below.
```
python classification.py mnist small-vae adam-128-1e-3-100 --freze_encoder
```