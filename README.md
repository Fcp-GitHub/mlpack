# `mlpack`: a Python project for machine learning students

# Contents
- [Overview](#overview)
- [Installation](#installation)
  - [Installation using `pip`](#installation-using-pip)
  - [Installation using `conda`](#installation-using-conda)
- [Examples and Notes](#examples-and-notes)
- [List of Features](#list-of-features)

# Overview
This project features a variety of models, each with different levels of verbosity and visualization options, in order to make every algorithm as clear as possible.
Furthermore, it contains notes on Machine Learning concepts.

The project is organized into different sections:
- mlpack: the actual Python package.
- examples:  some examples on how to use the package.
- notes:  notes on different Machine Learning techniques.

# Installation
## Installation using `pip`
In order to install and use `mlpack`, perform the following (OS-dependent) operations:
- **Linux and MacOS**: 
  ```Shell
  pip install -rU requirements/requirements.txt
  ```
- **Windows**: 
  ```PowerShell
  pip install -rU requirements/requirements_win.txt
  ```

## Installation using `conda`
In order to install packages based on the specified `requirements`, Conda users can either install `pip` locally using `conda install pip` or create an environment with `pip` installed in it with the following:
```Shell
conda create -n python=<python_version> <environment_name> pip
```
Of course, `python=<python_version>` is entirely optional and can be added if a specific Python version is required (this project doesn't require a specific Python version).
Once `pip` is installed, one can proceed by reading the [Installation using `pip`](#installation-using-pip) section.

# Examples and Notes
The `examples` folder contains a list of introductory programs for the new user. <br>
Additionally, one can find notes on Machine Learning concepts in the `notes` folder. Such files can be read using Jupyter Notebook (contained in the `requirements` files) with the following command:
```Shell
jupyter notebook notes
```
The command has to be issued from the root folder of the directory. The notes present further the features of `mlpack` as well as additional concepts that might be interesting for the user.

# List of Features
- **Linear classifiers**: (Single-Layer) Perceptron, Logistic Regression, Softmax Regression, Linear Support Vector Machines.
- **Nonlinear classifiers**: Support Vector Machines.
- **Neural Networks**: Multi-Layer Perceptron.
- **Strategies**: One-Versus-All (OVA).
- **Visualization**: Using class methods and the `visualize` and `presentation` sub-packages.
