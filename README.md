# credit-risk-classification
This project uses various techniques to train and evaluate a model based on loan risk. The dataset contains data of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## Overview of the Analysis

Use Logistic Regression to train and evaluate a model based on loan risk. Dataset of historical lending activity from a peer-to-peer lending services company to be used to build a model that can identify the creditworthiness of borrowers.
As part of the machine learning process, following steps were followed:

1. Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
2. Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
3. Split the data into training and testing datasets by using train_test_split.
4. Fit a logistic regression model by using the training data (X_train and y_train).
5. Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.
6. Evaluate the model’s performance by doing the following:
7. Generate a confusion matrix.
8. Print the classification report.

 ## The Results:
 
![Screenshot 2025-03-24 225924](https://github.com/user-attachments/assets/e6661b35-487b-4cd6-85dd-b9f6653bc8cf)
 
The logistic regression model performs very well in predicting both healthy loans (0) and high-risk loans (1), but there are some differences in performance between the two classes.

# Performance Analysis:

The logistic regression model achieves an accuracy of 99%, which is very high. However, since the dataset is highly imbalanced (18,765 healthy loans vs. 619 high-risk loans), accuracy alone can be misleading. Class breakdown below.

Healthy Loans (Class 0):
Precision: 1.00 (No false positives—when the model predicts a healthy loan, it's almost always correct)
Recall: 0.99 (Some false negatives—rarely misclassifies healthy loans as high-risk)
F1-score: 1.00 (Perfect balance between precision and recall)
Samples: 18,765
# The model is highly effective in identifying healthy loans.

High-Risk Loans (Class 1):
Precision: 0.84 (Some false positives—misclassifies some healthy loans as high-risk)
Recall: 0.94 (Few false negatives—captures most high-risk loans)
F1-score: 0.89 (Strong performance but slightly lower than Class 0)
Samples: 619
# The model does well but is slightly weaker at identifying high-risk loans.

Key Observations for Class 1:
The model recalls 94% of high-risk loans, which is great.
However, its precision is lower (84%), meaning it sometimes misclassifies healthy loans as high-risk.

## Summary:
Macro Average (0.92 Precision, 0.97 Recall, 0.94 F1-score)
Shows the model is well-balanced across classes.
Weighted Average (0.99 Precision, 0.99 Recall, 0.99 F1-score)
Skewed by the large number of Class 0 samples, making it appear better than it might be for Class 1.

## Potential Recommendations:
If identifying high-risk loans correctly is the priority, then we can adjust the Decision Threshold: Lower the threshold to classify more loans as high-risk, increasing recall.
We could also try Alternative Models, like Random Forest which may capture high-risk loans better.
