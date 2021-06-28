## Machine Learning and Pattern Recognition Progect
Authors: Paolo Rabino, Matteo Ferrenti

### Project Structure
* Tiblib library:  
    * Blueprint: Abstract classes to guarantee model compatibility
    * Classifier: models for classification, all models proposed in class + a protype for peceptron classifier
    * Preproc: preprocessing methods proposed in class and other simple utility methods
    * Probability: Various functions that are used for probability calculations and miscellaneous tasks
    * Validation: Functions for computing scores and plots of results
    * Pipeline: Data Pipeline and jointer supermodels to build the final model using the costruction blocks defined in the other modules
* DataExploration script: functions used for evaluating initial data and making the plots in the first sections of the report
* ModelEval script: functions used for computing scores of various models that are then used in the ScoreEval script
* JointEval script: functions used to compute scores of joint models that are then used in the ScoreEval script
* ScoreEval script: functions used to evaluate the minDCF for the scores obtained in the ModelEval and JointEval script

### Final Model
The final model (effectively built in the jointeval script) is made using the
pipeline and jointer classes defined in pipeline.py to combine both preprocessing and various models
in a single supermodel.

#### Development
The whole project was built using only:
* matplotlib
* numpy
* scipy 
* standard python libraries    

The TiblibLibrary was built during the course in order to solve the
laboratories, it was then reworked to be used for the project.
A smaller and less complete version can be found on github from my profile, that was my initial version that I used to solve the laboratories.

#### Numerical instability

For some reason to me unknown the results varies slightly from machine to machine (probably numpy or scipy libraries implement architecture dependent methods),
the results in the report were obtained from running these scripts
on https://jupyter.polito.it/ with a student account.
The variations are very small and are consistent between the
various models, this means that even though the overall scores could be slightly
higher or lower the decisions made are still consistent with the results.