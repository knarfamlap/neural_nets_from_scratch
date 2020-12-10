# Neural Nets From Scratch :sun: :clouds:


## Table of Contents

* [About the Project](#about)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Roadmap](#roadmap)
<!-- * [Contributing](#contributing) -->
<!-- * [License](#license) -->
<!-- * [Contact](#contact) -->
* [Acknowledgements](#acknowledgements)
  

# About

  Neural Nets from Scratch is a project intended to build neural networks from the bottom up.
  The goal of this project is to learn how to implement basic neural network layers to gain 
  further understaning on how they work. This project also is intended for those who are curious
  about how libraries such as Tensorflow, or Pytorch work under the hood. 

## Built With
    
  The only library used in this project is numpy. 

# Getting Started
  
  The first thing you'll need to do is clone this repo, and change into into the neural-nets-from-scratch directory

  ```bash
  $ git clone https://github.com/knarfamlap/neural_nets_from_scratch.git
  $ cd neural_nets_from_scratch
  ```

  Once you're inside the directory, I recommended using a virtual enviroment. More on virtual enviroments [here]().
  Once you grabbed a virtual env, you can install the dependencies with the following command

  ```bash
  $ pip3 install -r requirements.txt
  ```

# Usage
  
  This project is organinzed the following way

### [Examples](/examples)

  Contains examples that put the implementation into practice. I hope to implement at least one example for every layer
  that I implement.

### [Layers](/layers)

  Contains all the implementation for the layers. Currently I have implemented 2D Convolutional, Dense, RNN, LSTM, and Activation layers.
  The file [layers.py](/layers/layers.py) is the abstract class that extends all implemented layers.

### [Model](/model)

  Contains the implementation for the model class. The model class groups layers into an object with training features.
  My goal with this class was to mimic the way Keras allows you to train neural nets.

### [Utils](/utils)

  Contains file with useful functions. For now, the only file in utils is [loss.py](/utils/loss.py). This file contains
  all implementations regarding loss(cost) functions.

# Roadmap

So far I plan to implemnt the following layers

- [x] Dense
- [x] Activation
- [x] Conv2D
- [x] RNN
- [] LSTM
- [] GRU


