# CommaSpeedChallenge - Can we predict vehicle speed using only dashcam video and computer vision?

This repository contains the work for my attempt at the CommaSpeedChallenge (about four years to late, I know). The goal of the challenge is to use machine learning to predict the speed of a car from a dashcam video. The challenge provides a labeled dataset of video shot at 20 fps totaling to 20400 frames and their corresponding speeds. The challenge also contains an unlabeled test video which you can submit to the company for grading. 


## Outline
1. [Related Work]
   * Optical Flow
   * Convelutional Nueral Networks
   * Reccurent Nueral Networks
2. [Methodology]
3. 

## [Related Work](#Related-Work)
I am entering this challenge with minimal to no computer vision experience. All of my prior projects are in natural language processing, so related work research was a necessary task for this endevour. CommaAI seems to attact very smart people, so finding prior attempts at this challenge on GitHub was often accompanied by well-organized and relevant research already conducted for me. This [page](https://github.com/ryanchesler/comma-speed-challenge/blob/master/README.md) was particulary useful. Prior attempts seemed to focus on three main methods, optical flow, convelutional neural networks (duh), and reccurent nueral networks.

### [Optical Flow]
Optical flow seemed to be a fan favorite in this challenge, most if not every attempt I found used some version of optical flow. [Optical flow](https://en.wikipedia.org/wiki/Optical_flow) is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene. A better definition in the context of this project is that optical flow is the distribution of apparent velocities of movement of brightness pattern in an image. Algorithmically this can be solved using differentiation between two images and approximations of corresponding pixels, however, this method results in two unknowns and thus is unsolvable without aditional constraints. While there are numerous mathematical approaches to fix this issue, recent approaches have found success in using deep learning (shocker) with the most popular project known as [Recurrent All-Pairs Field Transforms](https://arxiv.org/abs/2003.12039) (RAFT).

### [Convelutional Neural Networks]
I was somewhat familiar with convelutional nueral networks before this project, mainly the idea of sliding a kernal across an image making learnable transformations. I recently read a paper called [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946), and while the paper itself did not discuss the foundations of convelutional operations, it offered a framework to scale up very large deep learning models for image processing. Long story short, the authors got SOTA results with magnitudes less paramters then prior deep learning models. Even better, they open-sourced 8 different models (EF-0 - EF8). Even BETTER was that a kid at Harvard across the street generously made a PyTorch implementation that can be found [here](https://github.com/lukemelas/EfficientNet-PyTorch). I did not do much CNN research because I decided prior I wanted to use the feature extraction of these models for the convelutional aspect of this project. 

### [Reccurent Nueral Networks]
I am decently familiar with reccurent neural networks, having some prior experience in industry with alarm prediction as well as a prior [publication](https://ieeexplore.ieee.org/document/9529451) using LSTMs to detect vulnerabilities in source code. My intuition suggests that incorperating a lookback period would be beneficial, as it would allow the model to make inferences based off previous images if it was unsure about the current one. This would hopefully somewhat keep MSE low for difficult examples. Ideally we have the LSTM after the CNN to learn features over a predefined lookback period, but after a few google searches it doesnt look training end-to-end is possible (let me know if this is incorrect). Instead, you train your CNN to extract features and then afterwards learn your LSTM. 

## [Methodology]
My approach mimics that of Comma's approach to learning self driving. They use three layers which they describe as representation, dynamics, and prediction. A representation layer takes in the sensor data and transforms the manifold to something more learnable, referred to as the “hidden state”. It could also discard non task relevant information. A dynamics layer deals with long term temporal correspondences. It functions as both a summarizer of the past and a predictor of the future. A prediction layer tells you how to act in a state. It can do this by directly outputting an action or by outputting a value of this state allowing you to search. You can read more about this architecture [here](https://geohot.github.io/blog/jekyll/update/2021/10/29/an-architecture-for-life.html). Taking from this paradigm, I developed my most complex network to be an EfficientNet-B0 feed into an LSTM before a few dense prediction layers. In this model the EfficientNet is a CNN which extracts the learnable features, the LSTM is the dynamics model which captures temproal features, and the linear layers at the end learn to make the speed prediction.

## [Data Preprocessing]
