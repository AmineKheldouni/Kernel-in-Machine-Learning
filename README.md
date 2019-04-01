# Kernels in Machine Learning: Protein sequences classification
This data challenge was part of the Kernels in Machine Learning course for MSc Students (MVA) at l'ENS Paris Saclay.
The goal of the data challenge was to learn how to implement Kernel-based Machine Learning algorithms to classify sequences by predicting whether a DNA sequence region is binding site to a specific transcription factor or not.

### Requirements
A specificity of this data challenge was that every algorithm and approach had to be coded from scratch without using any external machine learning library. Therefore the only libraries used were:
- numpy, scipy (algebra)
- cvxopt (optimization)
- Levenshtein. Even though we implemented our own Levenshtein distance computer, we used this library for faster computations (`pip install python-Levenshtein`).

### Data
This challenge contained three datasets. For each dataset, we had to predict separately the labels of DNA sequences. Each dataset contained 2000 examples for train sets, and 1000 examples for test sets.
For development purposes, we chose to split the train sets by taking 200-300 validation examples for parameter tuning.
We also implemented a k-fold cross validation for regularization tuning and averaging the validation accuracy (splits taken randomly).

### Getting Started
To use our system, you can refer to `start.py` as an example illustrating the following steps:
- Load the data
- Split the data into train/validation
- Call for the sought Kernel
- Perform an SVM optimization based on the Training Gram Matrix
- Provide the train accuracy and the validation accuracy if data was splitted
- Write the submission file `Yte.csv`

If you want to build other types of kernels, please refer to the `./kernels/kernel.py` abstract class to learn how to develop similar kernels that would connect successfully to the rest of the code.

### Kernel models
We implemented several kernels that handle string data and help classify the DNA sequences by learning the TF structures:
1. Linear Kernel
2. Gaussian Kernel
3. Exponential Kernel with Levenshtein distance
4. Spectrum Kernel
5. Spectrum Kernel with Mismatch
6. Local Alignment (LA) Kernel
7. Fischer Kernel (with HMM modelling)

Abstract class Kernel allows for Normalizing kernels.
We also built a `SumKernel` class which concatenates implicitely the features of the given kernels.
Finally, we made an attempt to implement Multiple Kernel Learning approach (MKL) where we optimize a weighted version of the `SumKernel` (`WeightedSumKernel`), optimizing the weights of the sum.

### Results and comments
Below are our results for the considered `SumKernel` in `start.py`, that achieved the best accuracy in validation:

Train accuracy  | Validation accuracy | Public test accuracy | Final test accuracy
------------- | ------------- | ------------- | -------------  
100% | 72% | **71.066% (22/76)** | **67.533% (37/76)**

We managed to build an end-to-end module for DNA sequences classification making use of several kernel methods.
However, we noticed that our ranking (according to our test accuracy) dropped from the public dataset to the final private test set which is unfortunate.
This is certainly due to some overfitting, which was unexpected since we considered regularization tuning with validation set. We provide hypothesis in the report.

### Report and submission script
The link for our report can be found [here](). To produce our final submission, you can run the `start.py` script which uses our best-found combination of kernels and our tuned parameters (regularization and kernels parameters).

Command : `python3 start.py`

### Authors (alphabetic order) :
Charles Auguste
Yonatan Deloro
Amine Kheldouni