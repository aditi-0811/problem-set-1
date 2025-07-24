'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd


# Your code here
def preprocessing():
    # Load the datasets
    pred_universe = pd.read_csv('data/pred_universe_raw.csv')
    arrest_events = pd.read_csv('data/arrest_events_raw.csv')

    # Perform full outer join on 'person_id'
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

    # Convert date columns to datetime
    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ'])
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])

    arrest_events['arrest_date_event'] = pd.to_datetime(arrest_events['arrest_date_event'])

    # Create 'y' column: 1 if rearrested for felony in the next year, else 0
    y_values = []
    for index, row in df_arrests.iterrows():
        person_id = row['person_id']
        arrest_date = row['arrest_date_univ']

        rearrested_felony = arrest_events[
            (arrest_events['person_id'] == person_id) &
            (arrest_events['arrest_date_event'] > arrest_date) &
            (arrest_events['arrest_date_event'] <= arrest_date + pd.Timedelta(days=365)) &
            (arrest_events['charge_degree'] == 'felony')
        ]

        y_values.append(1 if rearrested_felony.shape[0] > 0 else 0)

    df_arrests['y'] = y_values

    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? {df_arrests['y'].mean():.2%}")

    # Create 'current_charge_felony' column using lambda function
    df_arrests['current_charge_felony'] = df_arrests['charge_degree'].apply(lambda x: 1 if x == 'felony' else 0)
    print(f"What share of current charges are felonies? {df_arrests['current_charge_felony'].mean():.2%}")

    # Create 'num_fel_arrests_last_year' column
    num_fel_arrests_last_year = []
    for index, row in df_arrests.iterrows():
        person_id = row['person_id']
        arrest_date = row['arrest_date_univ']

        last_year_arrests = arrest_events[
            (arrest_events['person_id'] == person_id) &
            (arrest_events['arrest_date_event'] < arrest_date) &
            (arrest_events['arrest_date_event'] >= arrest_date - pd.Timedelta(days=365)) &
            (arrest_events['charge_degree'] == 'felony')
        ]

        num_fel_arrests_last_year.append(last_year_arrests.shape[0])

    df_arrests['num_fel_arrests_last_year'] = num_fel_arrests_last_year

    

    print(f"What is the average number of felony arrests in the last year? {df_arrests['num_fel_arrests_last_year'].mean():.2f}")

    print(df_arrests['num_fel_arrests_last_year'].mean())
    print(df_arrests.head())

    #Save to CSV for PART 3
    df_arrests.to_csv('data/df_arrests.csv', index=False)
    
    return df_arrests


if __name__ == "__main__":
    preprocessing()
