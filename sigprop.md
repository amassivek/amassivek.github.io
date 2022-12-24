---
title: "The Framework for Learning and Inference in a Forward Pass - Signal Propagation"
permalink: /sigprop
---

In this post, I present the framework for inference and learning in a forward pass, called the Signal Propagation framework. This is a framework for using only forward passes to learn any kind of data and on any kind of network. I demonstrate it works well for discrete networks, continuous networks, and spiking networks, all without modification to the network architecture. In other words, the version of network used for inference is the same as the version used for learning. In contrast, backpropagation and previous works have additional structure and algorithm elements for the training version of the network than for the inference version of the network, which are referred to as learning constraints.

Signal Propagation is a least constrained method for learning, and yet has better performance, efficiency, and compatibility than previous alternatives to backpropagation. It also has better efficiency and compatibility than backpropagation. This framework is introduced in https://arxiv.org/abs/2204.01723 (2022) by Adam Kohan, Ed Rietman, and Hava Siegelmann. Hava Siegelmann and Ed Rietman are my advisors. The origin of forward learning is in our work https://arxiv.org/abs/1808.03357 (2018).

This page is a concise tutorial for learning in a forward pass. By the end of the tutorial, you will understand the concept, and know how to apply this form of learning in your work. The tutorial provides explanations for beginners, and detailed steps for experts. Use the table of contents to go where you want. If you have a work on forward learning, [add your work](#4i-add-your-work) to this page.

(Link to article)

Table of Contents
1. [Introduction](#1-introduction)\
  1.1. [Previous Approaches to Learning](#11-previous-approaches-to-learning)\
  1.2. [A New Framework for Learning](#12-a-new-framework-for-learning)\
  1.3. [The Problem with Learning Constraints](#13-the-problem-with-learning-constraints)
2. [The Two Elements of Learning](#2-the-two-elements-of-learning)
3. [Learning in a Forward Pass](#3-learning-in-a-forward-pass)\
  3.1. [The Approach to Learn](#31-the-approach-to-learn)\
  3.2. [The Steps to Learn](#32-the-steps-to-learn)\
  3.3. [Overview of Complete Procedure](#33-overview-of-complete-procedure)\
  3.4. [Spiking Networks](#34-spiking-networks)
4. [Works on Forward Learning](#4-works-on-forward-learning)\
  4.i. [Add Your Work](#4i-add-your-work)\
  4.1. [Error Forward Propagation (2018)](#41-error-forward-propagation-2018)\
  4.2. [Forward Forward (2022)](#42-forward-forward-2022)
5. [Reading Material](#5-reading-material)
6. [Appendix: Reading on Credit Assignment](#6-appendix-reading-on-credit-assignment)\
  6.1. [Spatial Credit Assignment](#61-spatial-credit-assignment)\
  6.2. [Temporal Credit Assignment](#62-temporal-credit-assignment)

## 1. Introduction

### 1.1. Previous Approaches to Learning
Learning is the active ingredient in making artificial neural networks work. Backpropagation is recognized as the best performing learning algorithm, powering the success of artificial neural networks. However, it is a highly constrained learning algorithm. And, it is these constraints that are seen as necessary for its high performance. It is well accepted that reducing even some of these constraints lowers performance. However, due to these same constraints, backpropagation has problems with efficiency and compatibility. It is not efficient with time, memory, and energy. It has low compatibility with biological models of learning, neuromorphic chips, and edge devices. So, one may think to address this problem by reducing different subsets of constraints in an attempt to increase efficiency and compatibility without heavily lowering performance.

For example, two constraints of backpropagation on the training network are: (1) the addition of feedback weights that are symmetric with the feedforward weights; and (2) the requirement of having these feedback weights for every neuron. The inference network never uses the feedback weights, that is why we refer to them as learning constraints. Subsets of these constraints include: not adding any feedback weights, only adding feedback weights for one or two layers in a five layer network, not having the feedback weights be symmetric, or any combination of these. This means constraints can be added or removed in part or entirely to form subsets of constraints to reduce. One may keep trying to reduce different subsets of these constraints, in an attempt to increase efficiency and compatibility, and hope to not heavily impact performance.

Previous alternative learning algorithms to backpropagation have attempted relaxing constraints, without success. They reduce subsets of constraints on learning to improve efficiency and compatibility. They keep other constraints, with the expectation of retaining performance similar to the performance found by keeping all the constraints (which is backpropagation). So, this implies there is a spectrum for learning constraints, from highly constrained, such as backpropagation, to no constraints, such as Signal Propagation, the framework I am introducing here.

### 1.2. A New Framework for Learning
Now, I demonstrate a shift away from previous works. The results presented here provide support that the least constrained learning method, Signal Propagation, has better performance, efficiency, and compatibility than alternatives to backpropagation that selectively reduce constraints on learning. This includes well established and highly impactful methods such as random feedback alignment, direct feedback alignment, and local learning (all without backpropagation). This is a fascinating insight into learning across fields from neuroscience to computer science. It benefits areas from biological learning (e.g. in the brain) to artificial learning (e.g. in neural networks, hardware, neuromorphic chips).

Signal Propagation also significantly informs the direction of future research in learning algorithms where backpropagation is the standard of comparison. On the spectrum of learning constraints, contrary to the highly constrained backpropagation, Signal Propagation is the least constrained method to compare with and to start from for developing learning algorithms. With only backpropagation as a best performing comparison, learning algorithms did not have a starting point, only an end goal. Now, I am introducing Signal Propagation as the new baseline for learning algorithms to assess their efficiency, compatibility, and performance.

### 1.3. The Problem with Learning Constraints
 
What are the constraints found under backpropagation?\
Why are they an issue?

Learning constraints under backpropagation are difficult to reconcile with learning in the brain. Below, I provide the main constraints: 
- A complete forward pass through the network is required before sequentially delivering feedback in reverse order during a backward pass. 
- The training network needs the addition of comprehensive feedback connectivity for every neuron. 
- There are two different computations for learning and for inference. In other words, the feedback algorithm is a distinct type of computation, separate from feedforward activity.
- The feedback weights need to be symmetric with the feedforward weights.

These constraints also hinder efficient implementations of learning algorithms on hardware for the following reasons: 
- weight symmetry is incompatible with elementary computing units which are not bidirectional.
- transportation of non local weight and error information requires special communication channels.

These learning constraints prohibit parallelization of computations during learning and increase memory and compute for the following reasons:
- The forward pass needs to complete before the backward pass can begin (Time, Sequential)
- Activations of hidden layers need to be stored during the forward pass for the backward pass (Memory)
- Backward pass requires special feedback connectivity (Structure)
- Parameters are updated in reverse order of the forward pass (Time, Synchronous)

<picture>
 <img alt="motivation" src="./sigprop/Slide2.PNG">
</picture>

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

## 3. Learning in a Forward Pass

### The Signal Propagation Framework (SP)
I present here, the Framework for Learning and Inference in a Forward Pass, called Signal Propagation (SP). It is a satisfyingly straightforward solution to temporal and spatial credit assignment. SP is a least constrained method for learning, and yet has better performance, efficiency, and compatibility than previous alternatives to backpropagation. It also has better efficiency and compatibility than backpropagation. SP provides a reasonable performance tradeoff for efficiency and compatibility. This is particularly appealing, considering it's compatibility for target based deep learning (e.g. supervised and reinforcement) with new hardware and long-standing biological models, whereas previous works are not. (In general, backpropagation is the best performing algorithm.)

SP is free of constraints for learning to take place, with:
- only a forward pass, no backward pass
- no feedback connectivity or symmetric weights
- only one type of computation for learning and inference.
- a learning signal which travels with the input in the forward pass
- updates to parameters once the neuron/layer is reached by the forward pass

An interesting insight, SP provides an explanation for how neurons in the brain without error feedback connections receive global learning signals.

As a result, Signal Propagation is:
- Compatible with models of learning in the brain and in hardware.
- More efficient in learning, with lower time and memory, and no additional structure.
- A low complexity algorithm for learning.

### 3.1. The Approach to Learn

Signal Propagation treats the target as an additional input (figure below). With this approach, SP feeds the target forward through the network, as if it were an input.
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide17.PNG">
</picture>

SP moves forward through the network (figure below), bringing the target and the input closer and closer together, starting from the first layer (top left) all the way to the last layer (bottom right). Notice that by the last step/layer, the image of the dog is close to its target [1,0,0], and the image of the frog is close to its target [0, 1, 0]. However, the image and target of the dog is far away from the frog. This operation takes place in the representational space of the neurons at each layer. For instance, the neurons at layer 1 take in the dog picture (the input x) and dog label (the target c) and output activations h_1_dog and t_1_dog, respectively. The same happens for the frog producing h_1_frog and t_1_frog. In this activation space of the neurons, SP trains the network to bring an input and its target closer together, but farther away from other inputs and their respective targets.
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide18.PNG">
</picture>
Layer by layer, bring the target and its respective input closer together, but farther from the other input and target. Images of animals from Upsplash.

### 3.2. The Steps to Learn
Below is the total picture for an example three layer network. Each layer has its own loss function, which is used to update weights in the network. So, SP executes the loss function and updates the weights as soon as the target and label reach a layer. Since SP feeds the target and input together (alternating), layer/neuron weights are updated immediately. For spatial credit assignment, SP updates weights without waiting for the input to reach the last layer from the first layer. For temporal credit assignment, SP provides a learning signal for (each time step) each of the multiple connected inputs (e.g. images in a video), without waiting for the last input to be fed into the network.
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide20.PNG">
</picture>
This is a three layer network. The forward pass for learning and inference will proceed in three steps. Each layer has its own loss; a total of three losses. The inputs are x, and the targets are c, both fed in through the front of the network.

<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide21.PNG">
</picture>
The overall algorithm for learning and inference in a forward pass. The inference and learning phases run in parallel, such that each layer's weights are updated immediately. Notes: For the network shown (left), N = 3, the number of layers. The biases (b and d) are left out for clarity. There are many choices for a loss L (e.g. gradient, Hebbian) and optimizer (e.g. SGD, Momentum, ADAM). The output(), y, is detailed in step 4 below.

Below, we will go step by step, layer by layer, doing learning and inference (i.e. producing an answer/prediction) in forward passes. Note, in the guide below, the target and input are batch concatenated into a forward pass, making it easier to follow.

#### Step 1) Layer 1
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide22.PNG">
</picture>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide23.PNG">
</picture>	
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide24.PNG">
</picture>	

#### Step 2) Layer 2
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide25.PNG">
</picture>	
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide26.PNG">
</picture>	
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide27.PNG">
</picture>	

#### Step 3) Layer 3
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide28.PNG">
</picture>	

#### Step 4) Prediction
At the output layer, there are three choices for outputting a prediction. The first and second options provide more flexibility and follow naturally from the procedure to train using a forward pass. The first option is to take an h_3 for a class and compare it with each t_3 for every class. For example, SP inputs the image of a dog and gets h_3_dog , then inputs the labels for all the classes and gets t_3_i = { t_3_dog, t_3_frog, and t_3_horse}, finally it compares h_3_dog with each of the t_3_i; the closes t_3_i is the correct class.

The second option is an adaptive version of the first option. It is adaptive since SP no longer compares h_3_dog with every t_3_i, instead finds a subset of closest t_3_i. For example, we maintain a tree where t_3_frog is closer in the tree to t_3_dog than t_3_horse. So, we first compare h_3_dog to t_3_frog, then to t_3_dog, and stop. We never compare with t_3_horse as it is too far away and not in our subset of closest t_3_i.

The third option: the classical and intuitive choice is to train a prediction output layer. This option is also more straightforward for regression and generative tasks. For example, a classification layer, which has has one output per class. So, layer 3 would be a classification layer. Note, that during inference t_3 is not longer used. In addition, notice that t_3_i is equivalent to having a classification layer. To see this, simply concatenate t_3_i together to form the weight matrix of a classification (prediction) layer that is taken with h_3 (e.g. h_3_dog, h_3_horse, …). This means that  this third option is a special case of the first option, and can be a special case of the second option.

<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide29.PNG">
</picture>	

### 3.3. Overview of Complete Procedure

<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide30.PNG">
</picture>	
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide31.PNG">
</picture>	


### 3.4. Spiking Networks

Spiking neural networks are similar to biological neural networks. They are used in models of learning in the brain. They are also used for neuromorphic chips. There are two problems for learning in spiking neural networks. First, the learning constraints under backpropagation are difficult to reconcile with learning in the brain, and hinders efficient implementations of learning algorithms on hardware (discussed above). Second, training spiking networks results in the dead neuron problem (see below).

A reference figure is provided below. The neurons in these networks respond to inputs by either activating (spiking) to convey information to another neuron or by doing nothing (top-left figure). Commonly, these networks have a problem where neurons never activate, which means they never spike (bottom-left figure). Thereby, regardless of the input, the neurons response is to always do nothing. This is called the dead neuron problem.

The most popular approach to resolve this problem uses a surrogate function to replace the spiking behavior of the neurons. The network uses the surrogate only during learning, when the learning signal is sent to the neurons. The surrogate function (blue) provides a value for the neuron even when it does not spike (top-right figure). So, the neuron learns even when it does not spike to convey information to another neuron (bottom-right figure). This helps stop the neuron from dying. However, surrogates are difficult to implement for learning in hardware, such as neuromorphic chips. Furthermore, surrogates do not fit models of learning in the brain.

Signal Propagation provides two solutions that are compatible with models of learning in the brain and in hardware.

<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide32.PNG">
</picture>	

Below is a visualization of the learning signal (colored in red) going through a spiking neuron (shown as S), past the voltage or membrane potential (U), to update the weights (W). Backpropagation, with the dead neuron problem, is on the left. Backpropagation, with a surrogate function (f), is second from the left. The learning signal for backpropagation is global (L_G) and comes from the last layer of the network; the dotted boxes are upper neurons/layers.

The other images on the right show the two solutions Signal Propagation (SP) provides. First, SP can use a surrogate as well, but the learning signal does not go through the spiking equation (S). Instead, the learning signal is before the spiking equation (S), directly attached to the surrogate function (f). As a result, SP is more compatible with learning in the brain, such as in a multi compartment model of a biological neuron. Second, SP can learn using only the voltage or membrane potential (U). In this case, the learning signal is directly attached to U. This requires no surrogate or change to the neuron. Thereby, SP provides compatibility with learning in hardware.
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide33.PNG">
</picture>	


## 4. Works on Forward Learning

A list of works on forward learning, using the forward pass for learning. Works are ordered by date.

### 4.i. Add your work
Contact me or [submit a pull request](https://github.com/amassivek/amassivek.github.io) to add a paragraph and slide on your work. The content is at your discretion. I may provide minor edits (e.g. grammar and positioning).

### 4.1. Error Forward Propagation (2018)

Error Forward-Propagation: Reusing Feedforward Connections to Propagate Errors in Deep Learning\
https://arxiv.org/abs/1808.03357

### 4.2. Forward Forward (2022)

The forward forward algorithm is an implementation of the signal propagation framework for learning and inference in a forward pass (figure below). Under signal propagation, S is the transform of the context c, which for supervised learning is the target. In forward forward, S is a concatenation of the target c with the input x, as shown in the figure below.

<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide38.PNG">
</picture>	

Forward Forward Algorithm\
https://www.cs.toronto.edu/~hinton/FFA13.pdf


## 5. Reading Material

Works are ordered by date.

Error Forward-Propagation: Reusing Feedforward Connections to Propagate Errors in Deep Learning\
https://arxiv.org/abs/1808.03357 (2018)

Forward Forward Algorithm\
https://www.cs.toronto.edu/~hinton/FFA13.pdf (2022)

Signal Propagation: A Framework for Learning and Inference In a Forward Pass\
https://arxiv.org/abs/2204.01723 (2022)


### 5.1 Other Material

A well written guide on spatial and temporal credit assignment. I used it to help write this post.\
Training Spiking Neural Networks using lessons from deep learning\
https://arxiv.org/abs/2109.12894


With Thanks to: Alexandra Marmarinos for her editing work and guidance. (Add additional editors).


## Citations
\[1] Image of 7 from MNIST dataset, http://yann.lecun.com/exdb/mnist/ \
\[2] Images of dogs, horses, and frogs from CIFAR Dataset, https://www.cs.toronto.edu/~kriz/cifar.html; or from Upsplash



## 6. Appendix: Reading on Credit Assignment

### 6.1. Spatial Credit Assignment
Spatial Locality of Credit Assignment is the question: How does the learning signal reach every neuron?

On the left of the figure below, is a three layer network. In general, learning takes place over two phases: the inference phase and the learning phase. In the first phase, called the inference phase, the input is fed through the network from the first layer up to the last layer. Since the input is fed forward through the network, the inference phase takes place during the "forward pass" through the network. In the second phase, called the learning phase, the learning signal (colored in red) needs to reach every neuron in this network.

Different learning algorithms have different solutions to the learning phase. In backpropagation, the learning signal goes backward through the network, so the learning phase takes place during the "backward pass" through the network. As we will see with Signal Propagation, learning can take place during the forward pass as well.

Broadly, there are two approaches to the learning phase. The first approach computes a global learning signal (left middle figure) and then sends this learning signal to every neuron. The second approach computes a local learning signal (right figure) at each neuron (or layer). The first approach has the problem of having to coordinate sending this signal to every neuron in a precise way. This is costly in time, memory, and compatibility. The second approach does not encounter this problem, but has worse performance.

<picture>
 <img alt="spatial-credit-assignment" src="./sigprop/Slide5.PNG">
</picture>

### 6.2. Temporal Credit Assignment

Temporal Locality of Credit Assignment is the question: How does the global learning signal reach multiple connected inputs (aka every time step)?

A single image requires only that the learning signal reach every neuron. However, a video is a series of connected images. So, now the learning signal needs to travel through multiple connected inputs (aka time), starting from the last image in the video all the way to the first image in the video. This concept applies to any sequential or time series data. So, how does the global learning signal reach every time step? There are two popular methods to answer this question: Backpropagation through time, and forward mode differentiation.

#### 6.2.1. Backpropagation Through Time (BPT)
The primary answer to the question posed above follows, and takes place in two phases. First, input all the images that make up the video, one by one, into the network. This is the inference phase where the multiple connected inputs are sent forward through the network (a forward pass). Second, go backwards from the last image to the first image propagating the learning signal. This is the learning phase where the learning signal goes backward (a backward pass) through the multiple connected inputs (aka time); thus the name backpropagation through time.

##### Step 1: Inference
In the figure below, BPT feeds each image X[i] (e.g. of the turtle walking), which makes up the video, through the network. BPT starts with the 1st image X[0] (bottom left of the first figure), which is time step 1 (time is shown at the top of the figure). Next, BPT feeds in image X[1], which is time step 2. Finally, we end with the last image X[2] at time step 3 - this demonstration is for a very short video, or gif. Every time BPT feeds an image to the network notice that the middle layer in the network connects each image to the next image through time. 
<table>
<tr>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide7.PNG">
</picture>
</td>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide8.PNG">
</picture>
</td>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide9.PNG">
</picture>
</td>
</tr>
</table>

##### Step 2: Learning Backward through time
BPT feeds the learning signal, colored in red, backward through the images (time), making up the video of the turtle walking. The learning signal is formed from the loss function (top right of figure). It travels in the opposite direction of how we fed in the images X[i]. First a gradient/update is calculated for image X[2] at time 3, then image X[1] at time 2, and finally image X[0] at time 1. This is why it is called backpropagation through time. Again, notice that the middle layer in the network connects the learning signal from the last image X[2] to the first image X[0].
<table>
<tr>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide10.PNG">
</picture>
</td>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide11.PNG">
</picture>
</td>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide12.PNG">
</picture>
</td>
</tr>
</table>

#### 6.2.2. Forward Mode Differentiation (FMD)
Under FMD, the behavior of the inference (step 1) and learning (step 2) phases are similar to each other. As a result, FMD does step 1 (inference) and step 2 (learning) together (alternating). How? In step 2, FMD propagates the learning signal forward through the images (time), much the same as inference does in step 1. So, the learning signal no longer needs to travel from the last image X[3] in the video back to the first X[0]. The result: FMD has a learning signal that starts with X[0], instead of having to wait for X[3]. 

Why FMD vs BPT? Above, I discussed the learning constraints under backpropagation and the problems it has with efficiency and compatibility. FMD attempts to improve efficiency. Particularly, BPT feeds all of the images, making up the video, into the network before learning. FMD does not, so it is more efficient in time than BPT. However, FMD is significantly more costly than BPT, particularly in memory and computation. Note that FMD addresses time. However, it does not help with the learning constraints on spatial credit assignment found under backpropagation, which exist in FMD as well.

<table>
<tr>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide13.PNG">
</picture>
</td>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide14.PNG">
</picture>
</td>
<td>
<picture>
 <img alt="temporal-credit-assignment" src="./sigprop/Slide15.PNG">
</picture>
</td>	
</tr>
</table>
