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
    Function to prepare data to train LightFM classifier.
    :param paths_config: Dict[str, str], dictionary with the location of the necessary files:
    interactions, user features, item features; the name of the location as keys, links to files as values
    :param user_ids: str, name of the user ids column in data
    :param item_ids: str, name of the item ids column in data
    :param redundant_filter: str, name of the redundant filter column in data
    :param redundant_filter_val:  int, value of the redundant filter
    :param datetime_filter: str, name of the datetime filter column in data
    :param datetime_filter_interval: int, value of the datetime filter interval
    :param known_items_filter: bool, flag of the need for the formation of items known to users
    :return: local train / test sets if known items are not necessary,
    local train / test sets / known items if known items are necessary
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

    # define "local" train and test to use some part of the global train for ranker
    local_train_thresh = global_train[datetime_filter].quantile(q=.7, interpolation='nearest')

    global_train = global_train.dropna().reset_index(drop=True)

    local_train = global_train.loc[global_train[datetime_filter] < local_train_thresh]
    local_test = global_train.loc[global_train[datetime_filter] >= local_train_thresh]

    # focus on warm start - remove cold start users
    local_test = local_test.loc[local_test[user_ids].isin(local_train[user_ids].unique())]

    if known_items_filter is False:
        logging.info("Local train / test sets formed.")
        return local_train, local_test
    elif known_items_filter is True:
        # form known items
        known_items = local_train.groupby(user_ids)[item_ids].apply(list).to_dict()
        logging.info("Local train / test sets  and user's already watched items formed.")
        return local_train, local_test, known_items