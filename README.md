# CommaSpeedChallenge - Can we predict vehicle speed using only dashcam video and computer vision?

This repository contains the work for my attempt at the CommaSpeedChallenge (about four years to late, I know). The goal of the challenge is to use machine learning to predict the speed of a car from a dashcam video. The challenge provides a labeled dataset of video shot at 20 fps totaling to 20400 frames and their corresponding speeds. The challenge also contains an unlabeled test video which you can submit to the company for grading. 


## Outline
1. [Related Work](https://github.com/garrett-partenza-us/CommaSpeedChallenge/blob/master/README.md#Background-Research)
   * [Previous Attempts](https://github.com/garrett-partenza-us/CommaSpeedChallenge/blob/master/README.md#Previous-Attempts)

## [Related Work](#Related-Work)
I am entering this challenge with minimal to no computer vision experience. All of my prior projects are in natural language processing, so related work research was a necessary task for this endevour. CommaAI seems to attact very smart people, so finding prior attempts at this challenge on GitHub was often accompanied by well-organized and relevant research already conducted for me. This [page](https://github.com/ryanchesler/comma-speed-challenge/blob/master/README.md) was particulary useful. Prior attempts seemed to focus on three main methods, optical flow, convelutional neural networks (duh), and reccurent nueral networks.

### [Optical Flow](https://github.com/garrett-partenza-us/CommaSpeedChallenge/blob/master/README.md#Optical-Flow)
Optical flow seemed to be a fan favorite in this challenge, most if not every attempt I found used some version of optical flow. [Optical flow](https://en.wikipedia.org/wiki/Optical_flow) is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene. A better definition in the context of this project is that optical flow is the distribution of apparent velocities of movement of brightness pattern in an image. Algorithmically this can be solved using differentiation between two images and approximations of corresponding pixels, however, this method results in two unknowns and thus is unsolvable without aditional constraints. While there are numerous mathematical approaches to fix this issue, recent approaches have found success in using deep learning (shocker) with the most popular project known as [Recurrent All-Pairs Field Transforms](https://arxiv.org/abs/2003.12039) (RAFT)
