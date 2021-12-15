# CommaSpeedChallenge - Predicting speed of car from dashcam point-of-view.

The contribution of this work is an attempt of the 2017 Comma Speed Challenge. The goal of the challenge is to predict the speed of a car from the view of its own dash camera. In this work, I demonstrate how to integrate recurrent all-pairs field transforms for optical flow alongside an OpenPilot-style network architecture to train. Experimental results indicate this approach performs fairly well. More information regarding the challenge can be found in the CommaAI GitHub repository https://github.com/commaai/speedchallenge.


## Outline
1. [Methodology](#Methodology)
2. [Experimental Setup]
3. [Results]

## Methodology
Being that we are trying to detect the intensity of motion within an image, it makes sense to utilize optical flow as a feature in our training pipeline. In short, optical flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene. A better definition in the context of this project is that optical flow is the distribution of apparent velocities of movement of brightness pattern in an image. Algorithmically this can be solved using differentiation between two images and approximations of corresponding pixels, however, this method results in two unknowns and thus is unsolvable without additional constraints. While there are numerous mathematical approaches to fix this issue, recent approaches have found success in using deep learning with the most popular project known as Recurrent All-Pairs Field Transforms (RAFT).

Since RAFT is open-sourced, I downloaded the pretrained model and converted all video frames into their corresponding optical-flow representation before being inputted into any downstream neural architecture. In this manner, we maintain the edges and shapes within images and use the RGB channels to pass optical flow information.

For the learnable layers, I use a neural architecture style based off of OpenPilot, a shippable level-two self-driving system. The developers of OpenPilot, inspired by value prediction networks in MuZero, use a three level architecture consisting of a feature extraction layer, dynamics layer, and prediction layer. More specifically, they use an efficientnet-b2 for feature extraction, bidirectional-gated recurrent unit for dynamics, and a few dense layers for prediction. 

The architecture I personally used for training consists of four stacked convolutional layers for feature extraction of the optical flow images from RAFT. Batch normalization and max-pooling are also applied after each convolutional layer. The output of the convolutional layers is then fed into three stacked bidirectional LSTMs to extract the temporal features. Finally, the last hidden states of each cell are flattened before being fed into four fully connected layers for prediction. Before each dense layer, a dropout of 0.2 was applied to aid in generalization. To meet the requirements of Machine Learning 6140 I train two different networks, one with a recurrent layer and one without, to demonstrate the utility of temporal dimensions in computer-vision for driving tasks. 

## [Experimental Setup]
For both models, 10,000 test cases were randomly sampled from the challenge dataset with a validation set size of 0.1. Both models were trained for 100 epochs with a batch size of 32 using an Adam optimizer. Learning rate was set to 0.02 alongside an exponential decay scheduler with gamma set to 0.99 and stepped every epoch. For the model which used a recurrent layer, a sequence of length 5 was used which amounts to 0.25 seconds of video. This is a regression task, so predictions of both models were evaluated using the mean squared error between model output and actual speed in MPH. The MSE of the training set and the validation set was logged on every epoch. GPU training was utilized on the Northeastern Discovery compute cluster with a NVIDIA Telsa P100 12GB. Average time to completion of training was around three hours for both models.

## [Results]
The recurrent model achieves a minimum validation MSE of 4.623 (within 2 MPH) while the non-recurrent model achieves 6.844 (within 2.6 MPH). The Comma Speed Challenge repository states that a mean squared error less than 10 is good, while less than 5 is better and less than 3 is heart (<3, get it?). By this grading scale I conclude that my model achieved above average performance on the validation set. 
