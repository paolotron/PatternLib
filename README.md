## Machine Learning and Pattern Recognition Progect
Authors: Paolo Rabino, Matteo Ferrenti

### PatternLib Structure
1. Blueprint: Abstract classes to guarantee model compatibility:
2. Classifier: models for classification:
   1. Perceptron
   2. Gaussian classifier
   3. NaiveBayes classifier
   4. Logistic Regression classifier
   5. Kernel-SVM
   6. Gaussian Mixture classifier
3. Preproc: preprocessing methods proposed in class and other simple utility methods:
   1. PCA
   2. LDA
   3. Poly Features
4. Probability: Various functions that are used for probability calculations and miscellaneous tasks 
5. Validation: Functions for computing scores and plots of results 
6. Pipeline: Data Pipeline and jointer supermodels to build the final model using the costruction blocks defined in the other modules

#### Development and Dependencies
The whole project was built using only:
* matplotlib
* numpy
* standard python library   

The PatternLib was built during the course in order to solve the
laboratories, it was then reworked to be used for the project.

