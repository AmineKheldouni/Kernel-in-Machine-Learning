# Kernels in Machine Learning: Protein sequences classification
This data challenge was part of the Kernels in Machine Learning course for MSc Students (MVA) at l'ENS Paris Saclay.
The goal of the data challenge was to learn how to implement Machine Learning algorithms and implement Kernel methods to classify sequences by predicting whether a DNA sequence region is binding site to a specific transcription factor or not.

### Requirements
A specificity of this data challenge was that every algorithm and approach had to be coded from scratch without using any pre-implemented library. Therefore the only libraries used were:
- NumPy
- tqdm (progress bar)
- Levenshtein, even though we implemented our own Levenshtein distance, we used this library for faster computations (`pip install python-Levenshtein`).
- cvxopt for optimization problems.

### Data
This challenge contained three datasets. For each dataset, we had to predict separately the labels of DNA sequences. Each dataset contained 2000 examples for train sets, and 1000 examples for test sets.
For development purposes, we chose to split the train sets by taking 200-300 validation examples for parameter tuning.
We also implemented a k-fold cross validation for regularization tuning and averaging the validation accuracy (since the splits where taking randomly).

### Getting Started
To use our system, you can refer to `start.py` as an example illustrating the following steps:
- Load the data
- Split the data into train/validation
- Call for the sought Kernels
- Perform an SVM optimization to train the Kernel
- Provide the train and validation accuracy
- Write the submission file `Yte.csv`

If you want to build other types of kernels, please refer to the `./kernels/kernel.py` abstract class to learn how to develop similar kernels that would connect successfully to the rest of the code.

### Kernel models
We implemented several kernels that handle string datasets and help classify the DNA sequences by learning the TF structures:
1. Linear Kernel
2. Exponential Linear Kernel
3. Levenshtein Kernel
4. Spectrum Kernel
5. Spectrum Kernel with Mismatch
6. Local Alignment (LA) Kernel
7. Fischer Kernel

We also built a `SumKernel` class that concatenates the features of the given kernels to improve the score.
Finally, we made a considerable attempt to construct a Multiple Kernel Learning approach (MKL) where we optimize a weighted version of the `SumKernel` (`WeightedSumKernel`) by optimizing the weights of the sum.

### Results and comments
Below are our results for the considered `SumKernel` that achieved the best accuracy in validation:

Train accuracy  | Validation accuracy | Public test accuracy | Final test accuracy
------------- | ------------- | ------------- | -------------  
100% | 72% | **71.066% (22/76)** | **67.533% (37/76)**

We managed to build an end-to-end module for DNA sequences classification making use of several kernel methods. However, we noticed that our ranking (according to our test accuracy) dropped from the public dataset to the final private test set which is unfortunate. This proves that there is some overfitting in the chosen parameters, which is unexpected since we considered regularization and did not have enough time (due to submissions daily limit) to perfectly tune the parameters regarding the test set.
### Report and submission script
The link for our report can be found [here](). To produce our final submission, you can run the `start.py` script which makes use of our best-found combination of kernels as well as some parameter tuning in terms of regularization (as well as kernels parameters).
