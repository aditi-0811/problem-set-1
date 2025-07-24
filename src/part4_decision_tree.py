'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

# Your code here
def decision_tree():
    #Read in the dataframe(s) from PART 3
    df_test = pd.read_csv('data/df_arrests_test.csv')
    df_train = pd.read_csv('data/df_arrests_train.csv')

    #Define features and target lists from part 3 to run model:
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    target = 'y'

    #Create a parameter grid called `param_grid_dt` containing three values for tree depth. 
    #(Note C has to be greater than zero)
    param_grid_dt = {'max_depth': [2, 4, 6]}

    #Initialize the Decision Tree model. Assign this to a variable called `dt_model`.
    dt_model = DTC()

    #Initialize the GridSearchCV using the decision tree model and parameter grid with 5 fold crossvalidation
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)

    #Run the model via .fit()
    #Used features and train which were defined above
    gs_cv_dt.fit(df_train[features], df_train[target])

    #Question 1: What was the optimal value for max_depth?
    optimal_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"Optimal value for max_depth: {optimal_max_depth}")

    #Question 2: Did it have the most or least regularization? Or in the middle?
    if optimal_max_depth == 2:
        print("Optimal max_depth has the most regularization.")
    elif optimal_max_depth == 4:
        print("Optimal max_depth is in the middle of regularization.")
    else:
        print("Optimal max_depth has the least regularization.")   

    #Now predict for the test set. Name this column `pred_dt`
    df_test['pred_dt'] = gs_cv_dt.predict(df_test[features])

    #Predict probabilities for test for part 5
    df_test['pred_dt_prob'] = gs_cv_dt.predict_proba(df_test[features])[:, 1]
    
    #Save the predictions to CSV for PART 5
    df_test.to_csv('data/df_arrests_test.csv', index=False)

    #Return dataframe(s) for use in main.py for PART 5
    return df_test, df_train, gs_cv_dt

if __name__ == "__main__": 
    decision_tree()
