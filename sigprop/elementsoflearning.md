---
title: "The Two Elements of Learning"
parent: "Forward Learning Framework"
nav_order: 2
---

Table of Contents
2. [The Two Elements of Learning](#2-the-two-elements-of-learning)

## 2. The Two Elements of Learning

How does learning function in neural networks?\
The short answer: Spatial and Temporal Credit Assignment

There are two primary forms of data: individual inputs, and multiple connected inputs which are sequentially or temporally connected. An image of a dog is an individual input as the network makes a prediction based solely on that image. In this case, the network is given a single image to predict if the image is of a dog or turtle.

A video of a turtle walking is multiple connected inputs as videos are made up of multiple images, and the network makes a prediction after seeing all of these images. In this case, the network is given multiple images to predict if the turtle is walking or hiding.

Backpropagation (BP) is used for individual inputs; backpropagation through time (BPT) is used for multiple connected inputs.

BP provides learning for:
- Every neuron (spatial credit assignment)

BPT provides learning for:
- Every neuron (spatial credit assignment)
- Multiple connected inputs (temporal credit assignment)

Providing learning for every neuron is known as the spatial credit assignment problem. Spatial credit assignment refers to the placement of neurons in the network, such as organized into layers of neurons. For example, in a five layer network, the backpropagation learning signal travels from the fifth layer sequentially all the way down to the first layer of neurons. In section 3, I will show how the signal propagation learning signal travels from the first layer to the fifth layer, the same as inference.

Providing learning for multiple connected inputs is known as the temporal credit assignment problem. Temporal credit assignment refers to moving through the multiple connected inputs. For example, each image in the video is fed into the network, producing a new response from the same neurons. Each neuron response is specific to each of the images/inputs. So, the backpropagation learning signal travels through each of these neuron responses, starting from the neuron response for the last image in the video to the neuron response for the first image. In section 3, it will become clear that the signal propagation learning signal travels from the neuron response to the first image to the neuron response for the last image, the same as inference.

Note, the inner problem of temporal credit assignment is spatial credit assignment. Temporal credit assignment takes the learning signal through each of the images making up the video. For each image, spatial credit assignment takes the learning signal to each neuron. Signal propagation gracefully addresses the outer problem by addressing the inner problem - a forward pass, by construction of the inference network, traverses through both problems.

BP does spatial credit assignment. BPT extends BP to do both spatial and temporal credit assignment. (Refer to Section 6 for a complete reading on spatial and temporal credit assignment.)