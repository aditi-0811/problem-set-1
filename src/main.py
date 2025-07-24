'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd


# Call functions / instanciate objects from the .py files
import part1_etl
import part2_preprocessing 
import part3_logistic_regression 
import part4_decision_tree 
import part5_calibration_plot 

def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    part1_etl.etl()
    # PART 2: Call functions/instanciate objects from preprocessing
    part2_preprocessing.preprocessing()
    # PART 3: Call functions/instanciate objects from logistic_regression
    part3_logistic_regression.logistic_regression
    # PART 4: Call functions/instanciate objects from decision_tree
    part4_decision_tree.decision_tree()
    # PART 5: Call functions/instanciate objects from calibration_plot
    part5_calibration_plot.calibrate()
    pass


if __name__ == "__main__":
    main()