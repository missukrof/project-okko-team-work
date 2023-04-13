from configs.config import logging

import pandas as pd
from typing import Dict


def prepare_data_for_train_lfm(
        paths_config: Dict[str, str],
        user_ids: str = 'user_id',
        item_ids: str = 'item_id',
        redundant_filter: str = 'total_dur',
        redundant_filter_val: int = 300,
        datetime_filter: str = 'last_watch_dt',
        datetime_filter_interval: int = 14,
        known_items_filter: bool = False
):
    """
    function to prepare data to train lightfm classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py

        paths_config: dict, where key is path name and value is the path to data

    # ПОМЕНЯТЬ ОПИСАНИЕ !!!!!
    """
    import datetime as dt
    from utils.utils import read_csv_from_gdrive

    INTERACTIONS_PATH = paths_config.get('INTERACTIONS_PATH')
    interactions = read_csv_from_gdrive(INTERACTIONS_PATH)
    logging.info("Interactions set downloaded.")

    # remove redundant data points
    interactions_filtered = interactions.loc[
        interactions[redundant_filter] > redundant_filter_val
    ].reset_index(drop=True)
    interactions_filtered[datetime_filter] = pd.to_datetime(interactions_filtered[datetime_filter])

    # set dates params for filter
    MAX_DATE = interactions_filtered[datetime_filter].max()
    MIN_DATE = interactions_filtered[datetime_filter].min()
    TEST_INTERVAL_DAYS = datetime_filter_interval

    TEST_MAX_DATE = MAX_DATE - dt.timedelta(days=TEST_INTERVAL_DAYS)

    # define global train and test
    global_train = interactions_filtered.loc[interactions_filtered[datetime_filter] < TEST_MAX_DATE]
    global_test = interactions_filtered.loc[interactions_filtered[datetime_filter] >= TEST_MAX_DATE]

    # now, we define "local" train and test to use some part of the global train for ranker
    local_train_thresh = global_train[datetime_filter].quantile(q=.7, interpolation='nearest')

    global_train = global_train.dropna().reset_index(drop=True)

    local_train = global_train.loc[global_train[datetime_filter] < local_train_thresh]
    local_test = global_train.loc[global_train[datetime_filter] >= local_train_thresh]

    # finally, we will focus on warm start -- remove cold start users
    local_test = local_test.loc[local_test[user_ids].isin(local_train[user_ids].unique())]

    known_items = local_train.groupby(user_ids)[item_ids].apply(list).to_dict()

    if known_items_filter is False:
        logging.info("Local train / test sets formed.")
        return local_train, local_test
    elif known_items_filter is True:
        logging.info("User's already watched items formed.")
        return known_items