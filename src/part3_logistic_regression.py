'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here

def logistic_regression():
    # Read in `df_arrests`
    df_arrests = pd.read_csv('data/df_arrests.csv')

    # Use train_test_split to create two dataframes from `df_arrests`
    df_arrests_train, df_arrests_test = train_test_split(
        # Set test_size to 0.3, shuffle to be True. Stratify by the outcome
        df_arrests,
        test_size=0.3,
        shuffle=True,
        stratify=df_arrests['y'],
    )
    # Create a list called `features`
    #Contains two feature names: num_fel_arrests_last_year, current_charge_felony
    features = ['num_fel_arrests_last_year', 'current_charge_felony']

    # Create a parameter grid called `param_grid`
    #Containing three values for the C hyperparameter. (Note C has to be greater than zero)
    param_grid = {'C': [0.01, 0.1, 1]}

    # Initialize the Logistic Regression model with a variable called `lr_model`
    lr_model = lr(solver='liblinear')

    # Initialize the GridSearchCV using logistic regression model and parameter grid with 5 fold crossvalidation
    gs_cv = GridSearchCV(lr_model, param_grid, cv=5)

    # Run the model via .fit()
    gs_cv.fit(df_arrests_train[features], df_arrests_train['y'])

    # Question 1: What was the optimal value for C?
    optimal_C = gs_cv.best_params_['C']
    print(f"Optimal value for C: {optimal_C}")
    # Question 2: Did it have the most or least regularization? Or in the middle?
    if optimal_C < 0.1:
        print("Optimal C has the most regularization.")
    elif optimal_C > 0.1 and optimal_C < 1:
        print("Optimal C is in the middle of regularization.")
    else:
        print("Optimal C has the least regularization.")

    # Now predict for the test set. Name this column `pred_lr`
    df_arrests_test['pred_lr'] = gs_cv.predict(df_arrests_test[features])

    #Predict probabilities for classification in part 5
    df_arrests_test['pred_lr_prob'] = gs_cv.predict_proba(df_arrests_test[features])[:, 1]

    #Save test and train - had to do this earlier because the run was not working
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)
    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    
    # Return dataframe(s) for use in main.py for PART 4 and PART 5
    return df_arrests_test, df_arrests_train, gs_cv #To get optimal C and predictions

if __name__ == "__main__":
    logistic_regression()
    

