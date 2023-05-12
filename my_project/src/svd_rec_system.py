# Рекомендательная система на основе SVD и фильтрации обучающих данных

# Import libraries

import numpy as np
import pandas as pd
import pickle
import os
import surprise
from surprise import SVD


# Define constants

# Default column names
DEFAULT_USER_COL = "UID"
DEFAULT_ITEM_COL = "JID"
DEFAULT_RATING_COL = "Rating"
DEFAULT_PREDICTION_COL = "Prediction"

# Filtering variables
DEFAULT_K = 10
DEFAULT_THRESHOLD = 10
MOST_POPULAR_DISLIKED_JOKES = [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]

#Other
SEED = 42



    # Define functions to predict and rank recommendations

    # Borrowed from Microsoft recommenders tutorial. 

def predict(
    algo,
    data,
    usercol=DEFAULT_USER_COL,
    itemcol=DEFAULT_ITEM_COL,
    predcol=DEFAULT_PREDICTION_COL
):
    """Computes predictions of an algorithm from Surprise on the data. Can be used for computing rating metrics like RMSE.

    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pandas.DataFrame): the data on which to predict
        usercol (str): name of the user column
        itemcol (str): name of the item column

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    predictions = [
        algo.predict(getattr(row, usercol), getattr(row, itemcol))
        for row in data.itertuples()
    ]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(
        index=str, columns={"uid": usercol, "iid": itemcol, "est": predcol}
    )
    return predictions.drop(["details", "r_ui"], axis="columns")


def compute_ranking_predictions(
    algo,
    data,
    usercol=DEFAULT_USER_COL,
    itemcol=DEFAULT_ITEM_COL,
    predcol=DEFAULT_PREDICTION_COL,
    remove_seen=False,
):
    """Computes predictions of an algorithm from Surprise on all users and items in data.

    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pandas.DataFrame): the data from which to get the users and items
        usercol (str): name of the user column
        itemcol (str): name of the item column
        remove_seen (bool): flag to remove (user, item) pairs seen in the training data

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    preds_lst = []
    users = data[usercol].unique()
    items = data[itemcol].unique()

    for user in users:
        for item in items:
            preds_lst.append([user, item, algo.predict(user, item).est])

    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])

    if remove_seen:
        tempdf = pd.concat(
            [
                data[[usercol, itemcol]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions
    
def get_top_k_items(
    dataframe, col_rating=DEFAULT_PREDICTION_COL, col_user=DEFAULT_USER_COL, k=DEFAULT_K
):
    """Get the input customer-item-rating tuple in the format of Pandas
    DataFrame, output a Pandas DataFrame in the dense format of top k items
    for each user.

    Note:
        If it is implicit rating, just append a column of constants to be
        ratings.

    Args:
        dataframe (pandas.DataFrame): DataFrame of rating data (in the format
        customerID-itemID-rating)
        col_user (str): column name for user
        col_rating (str): column name for rating
        k (int or None): number of items for each user; None means that the input has already been
        filtered out top k items and sorted by ratings and there is no need to do that again.

    Returns:
        pandas.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    """
    # Sort dataframe by col_user and (top k) col_rating
    if k is None:
        top_k_items = dataframe
    else:
        top_k_items = (
            dataframe.sort_values([col_user, col_rating], ascending=[True, False])
            .groupby(col_user, as_index=False)
            .head(k)
            .reset_index(drop=True)
        )
    # Add ranks
    top_k_items["Rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items    

# Prediction


def run_model(data_folder):
    # Load test data
    input = pd.read_csv(os.path.join(
                data_folder, 
                'inputdata.csv'
                ))
    #input.head()

    #f"Количество уникальных пользователей в тестовых данных: {input[DEFAULT_USER_COL].nunique()}"

    # Load model

    with open(os.path.join(
                data_folder, 
                'svd_model.pickle'
                ), "rb") as f:
        svd_model = pickle.load(f)
    #print('Model loaded:', svd_model)
    predictions = compute_ranking_predictions(svd_model, input, usercol=DEFAULT_USER_COL, itemcol=DEFAULT_ITEM_COL, remove_seen=True)

    #f"Количество уникальных пользователей в данных c предиктом: {predictions[DEFAULT_USER_COL].nunique()}"

# Recommendations

    df_output = get_top_k_items(predictions)
    #df_output.head()

# Create the format of data required by Bootcamp

# Group by UID and create a dictionary of top joke ID and its score
    top_joke_dict = df_output[df_output['Rank'] == 1].groupby('UID').apply(
        lambda x: {x['JID'].iloc[0]: x['Prediction'].iloc[0]}
    ).reset_index().rename(columns={0: 'top_joke_dict'})

    # Create a list of top 10 joke IDs
    top_joke_list = df_output.groupby('UID')['JID'].apply(list).reset_index(name='top_joke_list')

    # Merge the two dataframes on UID
    result_df = pd.merge(top_joke_dict, top_joke_list, on='UID')[['UID', 'top_joke_dict', 'top_joke_list']]

    # Create the 'Recommendations' column as a list of dictionaries and lists
    result_df['Recommendations'] = result_df.apply(
        lambda x: [x['top_joke_dict'], x['top_joke_list']], axis=1
    )

    # Drop the unnecessary columns
    result_df = result_df.drop(['top_joke_dict', 'top_joke_list'], axis=1)

    #result_df.head()

    # Save data

    result_df.to_csv(os.path.join(
                data_folder, 
                'outputdata.csv'
                ), sep=';', index=False)

    return 'outputdata.csv'
                

#run_model(data_folder='D:/my_project/data/')