<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/><img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" /><img src="https://img.shields.io/badge/node.js%20-%2343853D.svg?&style=for-the-badge&logo=node.js&logoColor=white"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/3ZadeSSG/Automatic-Image-Captioning/blob/main/LICENSE)

# Automatic Image Captioning

## 1. Project Overview

This project focuses on Computer Vision methods to generate Image Captions.
* It uses transfer learning on a "Convolutional Neural Network (CNN)" and uses it as encoder for LSTM based RNN
* CNN instead of generating class scrores for image, has been modified to generate only feature maps by removing linear layer used in prediction.
* A LSTM based RNN is used as decoder which then takes captions from training data and feature map from CNN to train in generating captions.
* `Automatic Image Captioning using Encoder CNN & Decoder RNN.ipynb` Notebook contains all code for training and prediction.
* `Node.js` is used to create API for serving model for Web Apps. A Single page web app is located in `Node.js_Server`
* Application takes images and shows proper caption for it.

### Example
##### Output example with Web Application which uses a model which was trained for only 2 epochs

"a laptop computer sitting on top of a desk" | "a plate of food with a fork and fork"
--- | ---
<img src="https://github.com/3ZadeSSG/Automatic-Image-Captioning/blob/main/images/Screenshot2.PNG"> | <img src="https://github.com/3ZadeSSG/Automatic-Image-Captioning/blob/main/images/Screenshot1.PNG">



## 2. About "Original Training Notebooks"

Detailed data visualization, training and inference notebooks are as follows:-

__Notebook 1__ : Testing COCO dataset API and running a visualization on sample images

__Notebook 2__ : Implementing and testing data loaders and tokenizers

__Notebook 3__ : Training Encoder-Decoder model

__Notebook 4__ : Testing trained model on test dataset and input images

To run the notebooks properly move them to project's root directory.

### Setup

Install required python packages from requirements file using:-
```
pip install -r requirements.txt
```


### Data & Checkpoints

1. Training, Testing and Validation datasets are over 24 GB, hence are needed to be downloaded from source. (https://cocodataset.org/#download)

        a. Training Dataset: http://images.cocodataset.org/zips/train2014.zip

        b. Testing Dataset: http://images.cocodataset.org/zips/test2014.zip

        c. Validation Dataset: http://images.cocodataset.org/zips/val2014.zip

        d. Annotations:
                        http://images.cocodataset.org/annotations/image_info_test2014.zip
                        http://images.cocodataset.org/annotations/annotations_trainval2014.zip

2. Trained Model's checkpoint (only upto 2 epochs) is located in `model_checkpoints` as well as `Node.js_Server/python_models/saved_models`

***To setup download datasets and annotations and extract everything in `Data` directory.***

### Training Hardware
With batch size = 32 and model as per in training notebooks:-
1. It takes 2.30 Hrs (Approx) to run a single epoch on following hardware resources available on Google Colab

        
        Intel(R) Xeon(R) CPU @ 2.20GHz [Core(s) per socket:  1 | Thread(s) per core:  2 ]
        Tesla T4 [CUDA Version: 10.1]
        

It takes 8 Hrs (Approx. as per back calculation basd on time taken for 100 steps) to run single epoch on following local hardware:-

        
        Intel(R) Core(TM) i3-2120 CPU @ 3.20GHz [Core(s) :  2 | Thread(s) per core:  2 ]
        GTX 1060 [CUDA Version: 10.1]
        

## 3. Running Node.js Application

  Implemented Node.js Application works by creating a python child process for generating captions for images.

  1. Navigate to `Node.js_Server`
  2. Place the trained pytorch model's checkpoint file `checkpoint.pth` to `python_models/saved_models` folder (in case new model was trained).
  4. In case new vocab.pkl file was genereated, or new model's checkpoint filename is different, change following variables in `Node.js_Server/python_models/model.py` to appropriate names as per requirement.
  
        ```
        ENCODER_CNN_CHECKPOINT = "python_models/saved_models/encoderEpoch_2.pth"
        DECODER_LSTM_RNN_CHECKPOINT = "python_models/saved_models/decoderEpoch_2.pth"
        VOCAB_FILE = "python_models/saved_models/vocab.pkl"
        ```
        
  5. Run following commands
     ```
     npm install                      (This installs Node.js dependencies)
     pip install -r requirements.txt  (If python packages haven't been already installed from project root)
     npm start
     ```
  6. To test it on other devices on local network
     ```
     node app.js <your_ip>:8000
     ```
     Example
     ```
     node app.js 192.168.32.134:8000
     ```

 LICENSE: This project is licensed under the terms of the MIT license.


 #### Source Repo:
 The project's root repo is: https://github.com/udacity/CVND---Image-Captioning-Project
 Due to limits of not being able to make a forked repo private, it has to follow these stpes: https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/duplicating-a-repository





