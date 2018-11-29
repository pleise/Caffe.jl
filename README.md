# Caffe interface for the [Julia] language

### THIS PROJECT WILL NOT BE FURTHER MAINTAINED.

[![Build Status](https://travis-ci.org/pleise/Caffe.jl.svg?branch=master)](https://travis-ci.org/pleise/Caffe.jl)

This package provides an interface for Julia to the [Caffe] deep learning framework and a modelling language to model a neural network in Julia. Most of the functionality of Caffe is supported by this interface. But it is work in progress right now, so additional functionalities will be added in the future. Right now it is possible to define a arbitrary network and the SGD or Adam Solver settings and call Caffe with automatically created .prototxt files for the network and the solver. Furthermore the output can be saved in a log file for further processing. 

## Installation
To use this package you have to install Caffe first and make sure, that the Caffe binary is in your PATH. Afterwards you can install the package with the following command:
```julia
Pkg.clone("git://github.com/pleise/Caffe.jl.git")
```

## Basic usage

To load the Package use:
```julia
using Caffe
```

## Example 
The mnist example can be found in the example folder.



[Caffe]: http://caffe.berkeleyvision.org/ "Caffe"
[Julia]: http://julialang.org "Julia"



