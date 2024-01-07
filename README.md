# SupportVectorMachine_with_Diff_KernelsAndParameters
This repository includes two Python scripts, SVM_EX-1 and SVM_EX-2, demonstrating the use of Support Vector Machines (SVM) for breast cancer classification. The scripts perform data loading, preprocessing, exploration, model training, and evaluation.

# Exersice-1:Breast Cancer Classification
This Python script demonstrates the application of Support Vector Machines (SVM) for breast cancer classification. 
The code performs data preprocessing, exploratory data analysis, SVM model training with different kernels, and model evaluation.

- Data Loading:
Loads breast cancer data into a pandas DataFrame.
- Initial Data Investigation:
Displays column names, data types, missing values, statistics, and other details.
- Data Cleaning:
Replaces '?' in the 'bare' column with NaN and converts the column to a float type.
Fills missing data with the median of the column.
Drops the 'ID' column.
- Data Visualization:
Generates 3-5 plots using Pandas, Matplotlib, and Seaborn.
- Feature and Class Separation:
Separates features and classes from the dataset.
- Data Splitting:
Splits data into 80% training and 20% testing using the last two digits of the student number.
- Linear SVM:
Trains an SVM classifier with a linear kernel and C=0.1.
Prints accuracy scores on both training and testing sets.
Generates the accuracy matrix.
- RBF, Poly, and Sigmoid SVMs:
Repeats steps 7 for SVMs with RBF, Poly, and Sigmoid kernels.
Compares accuracy scores and confusion matrices for different kernels.


# Exercise-2: Breast Cancer Grid Search
This script extends the SVM example by incorporating a pipeline and grid search for hyperparameter tuning. 
It uses the breast cancer dataset to create a pipeline with imputation, standardization, and SVM classification. 
Grid search is employed to find the best combination of hyperparameters.
 
- Data Loading and Initial Investigations:
Loads breast cancer data into a Pandas DataFrame.
Conducts initial investigations on the data.
- Data Preprocessing:
Replaces '?' in the 'bare' column with NaN and converts the column to a float type.
Drops the 'ID' column.
- Feature Extraction and Train-Test Split:
Separates features from the target class.
Splits the data into 80% training and 20% testing sets.
- Pipeline Creation:
Creates a pipeline that includes imputation, scaling, and an SVM classifier.
Grid Search:
- Defines grid search parameters.
Creates a grid search object and fits it to the training data.
- Best Model Selection:
Prints the best parameters and the best estimator.
Evaluates the test data using the best model.
Prints the accuracy score.
- Model and Pipeline Saving:
Saves the best model and the full pipeline using Joblib.

# Usage:

Run SVM_EX-1 and SVM_EX-2 to perform breast cancer classification using SVM.
Review data exploration, preprocessing, and model evaluation steps.
Experiment with different SVM kernels and parameters.

