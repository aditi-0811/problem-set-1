'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=5):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def calibrate():
    #Read in df_arrests_test
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')
    
    #Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5.
    calibration_plot(df_arrests_test['y'], df_arrests_test['pred_lr_prob'], n_bins=5)

    #Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5.
    calibration_plot(df_arrests_test['y'], df_arrests_test['pred_dt_prob'], n_bins=5)

    # Which model is more calibrated? Print this question and your answer.
    #Used .mean() to compare predicted probabilities of both models and output the answer
    print("Which model is more calibrated?")

    #Set up mean probability variables to answer question:
    #Tells absolute value of difference between true label and predicted probability, this gives the mean prediction error
    #Lower mean prediction error indicates better calibration

    lr_close_to_prob = np.abs(df_arrests_test['y'] - df_arrests_test['pred_lr_prob']).mean()
    dt_close_to_prob = np.abs(df_arrests_test['y'] - df_arrests_test['pred_dt_prob']).mean()

    if lr_close_to_prob < dt_close_to_prob: #lr model has less error than dt model
        print("The logistic regression model is more calibrated than the decision tree model.")
    else: #dt model has less error than lr model
        print("The decision tree model is more calibrated than the logistic regression model.")


    #EXTRA CREDIT: 
    
    #PPV
    #Get top 50 using .head(50), use pred_lr_prob to rank the predicted probabilities
    head50_lr = df_arrests_test.sort_values('pred_lr_prob', ascending=False).head(50) #logical regression
    head50_dt = df_arrests_test.sort_values('pred_dt_prob', ascending=False).head(50) #decision tree

    #Compute  PPV for the logistic regression model for arrestees
    ppv_lr = head50_lr['y'].mean() #Mean predicted probability
    #Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
    ppv_dt = head50_dt['y'].mean()

    #Print the PPV results
    print(f"PPV for the logistic regression model (top 50): {ppv_lr:.2f}")
    print(f"PPV for the decision tree model (top 50): {ppv_dt:.2f}")    

    #AUC
    #Compute AUC for the logistic regression model
    auc_lr = roc_auc_score(df_arrests_test['y'], df_arrests_test['pred_lr_prob'])

    #Compute AUC for the decision tree model
    auc_dt = roc_auc_score(df_arrests_test['y'], df_arrests_test['pred_dt_prob'])

    #Print the AUC results
    print(f"AUC for the logistic regression model: {auc_lr:.2f}")
    print(f"AUC for the decision tree model: {auc_dt:.2f}")

    #Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
    print("Do both metrics agree that one model is more accurate than the other?")
    if (ppv_lr > ppv_dt and auc_lr > auc_dt):
        print("Yes, logistic regression is more accurate.")
    elif (ppv_lr < ppv_dt and auc_lr < auc_dt):
        print("Yes, decision tree is more accurate.")
    else:
        print("No, the metrics disagree.")
if __name__ == "__main__":
    calibrate()
    