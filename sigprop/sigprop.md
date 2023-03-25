---
title: "Forward Learning - SigProp"
permalink: /sigprop
has_children: true
---

# The Framework for Learning and Inference in a Forward Pass

This page is an introduction to and concise tutorial for learning in a forward pass. By the end of the tutorial, you will understand the concept, and know how to apply this form of learning in your work. The tutorial provides explanations for beginners, and detailed steps for experts. Use the table of contents to go where you want. If you have a work on forward learning, [add your work](#4i-add-your-work) to this page.

In this post, I present the framework for inference and learning in a forward pass, called the Signal Propagation framework. This is a framework for using only forward passes to learn any kind of data and on any kind of network. I demonstrate it works well for discrete networks, continuous networks, and spiking networks, all without modification to the network architecture. In other words, the version of network used for inference is the same as the version used for learning. In contrast, backpropagation and previous works have additional structure and algorithm elements for the training version of the network than for the inference version of the network, which are referred to as learning constraints.

Signal Propagation is a least constrained method for learning, and yet has better performance, efficiency, and compatibility than previous alternatives to backpropagation. It also has better efficiency and compatibility than backpropagation. This framework is introduced in https://arxiv.org/abs/2204.01723 (2022) by Adam Kohan, Ed Rietman, and Hava Siegelmann. Hava Siegelmann and Ed Rietman are my advisors. The origin of forward learning is in our work https://arxiv.org/abs/1808.03357 (2018). The library is available at https://github.com/amassivek/signalpropagation .

(Link to article)

Table of Contents
1. [Introduction](introduction.md#1-introduction)\
  1.1. [Previous Approaches to Learning](introduction.md#11-previous-approaches-to-learning)\
  1.2. [A New Framework for Learning](introduction.md#12-a-new-framework-for-learning)\
  1.3. [The Problem with Learning Constraints](introduction.md#13-the-problem-with-learning-constraints)
2. [The Two Elements of Learning](elementsoflearning.md#2-the-two-elements-of-learning)
3. [Learning in a Forward Pass](learninginaforwardpass.md#3-learning-in-a-forward-pass)\
  3.1. [The Approach to Learn](learninginaforwardpass.md#31-the-approach-to-learn)\
  3.2. [The Steps to Learn](learninginaforwardpass.md#32-the-steps-to-learn)\
  3.3. [Overview of Complete Procedure](learninginaforwardpass.md#33-overview-of-complete-procedure)\
  3.4. [Spiking Networks](learninginaforwardpass.md#34-spiking-networks)
4. [Works on Forward Learning](worksonforwardlearning.md#4-works-on-forward-learning)\
  4.i. [Add Your Work](worksonforwardlearning.md#4i-add-your-work)\
  4.1. [Error Forward Propagation (2018)](worksonforwardlearning.md#41-error-forward-propagation-2018)\
  4.2. [Forward Forward (2022)](worksonforwardlearning.md#42-forward-forward-2022)\
  4.3. [Predictive Forward Forward (2022)](worksonforwardlearning.md#43-predictive-forward-forward-2022)
5. [Reading Material](readingmaterial.md#5-reading-material)
6. [Appendix: Reading on Credit Assignment](creditassignment.md#6-appendix-reading-on-credit-assignment)\
  6.1. [Spatial Credit Assignment](creditassignment.md#61-spatial-credit-assignment)\
  6.2. [Temporal Credit Assignment](creditassignment.md#62-temporal-credit-assignment)