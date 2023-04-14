from configs.config import logging

import pandas as pd
from typing import Any, Dict, List


def prepare_data_for_train(
        paths_config: Dict[str, str],
        local_test: pd.DataFrame,
        user_features: List[str],
        item_features: List[str],
        drop_features: List[str],
        user_ids: str = 'user_id',
        item_ids: str = 'item_id',
        sampling_feature: str = 'watched_pct'
):
    """
    Function to prepare data to train CatBoostClassifier
    :param paths_config: Dict[str, str], dictionary with the location of the necessary files:
    interactions, user features, item features; the name of the location as keys, links to files as values
    :param local_test: pd.DataFrame, local test set formed for LightFM model
    :param user_features: List[str], list of user features for ranker training
    :param item_features: List[str], list of item features for ranker training
    :param drop_features: List[str], list of features that need to be removed before training a ranker
    :param user_ids: str, name of the user ids column in data
    :param item_ids: str, name of the item ids column in data
    :param sampling_feature: str, name of the sampling feature column in data
    :return: local X_train / X_test / y_train / y_test sets for ranker (CatBoostClassifier) training
    """
    from models.lfm import LFMModel
    from utils.utils import read_csv_from_gdrive

    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split

    ITEMS_METADATA_PATH = paths_config.get('ITEMS_METADATA_PATH')
    items_metadata = read_csv_from_gdrive(ITEMS_METADATA_PATH)
    logging.info("Items features set downloaded.")

    USERS_DATA_PATH = paths_config.get('USERS_DATA_PATH')
    users_data = read_csv_from_gdrive(USERS_DATA_PATH)
    logging.info("Users features set downloaded.")

    USER_IDS = user_ids
    ITEM_IDS = item_ids

    # let's make predictions for all users in test
    lfm_model = LFMModel()

    users_preds = []
    items_preds = []

    for ind, id in enumerate(local_test[USER_IDS].unique()):
        candidates = lfm_model.infer(
            user_id=id,
            enable_logging=False
        )
        users_preds.append(id)
        items_preds.append(candidates.keys())

    test_preds = pd.DataFrame({USER_IDS: users_preds, ITEM_IDS: items_preds})
    test_preds = test_preds.explode(ITEM_IDS)
    test_preds['rank'] = test_preds.groupby(USER_IDS).cumcount() + 1

    # Need to create 0/1 as indication of interaction
    # positive event - 1, if watch_pct is not null
    positive_preds = pd.merge(test_preds, local_test, how='inner', on=[USER_IDS, ITEM_IDS])
    positive_preds['target'] = 1

    # negative evnet - 0 otherwise
    negative_preds = pd.merge(test_preds, local_test, how='left', on=[USER_IDS, ITEM_IDS])
    negative_preds = negative_preds.loc[negative_preds[sampling_feature].isnull()].sample(frac=.2)
    negative_preds['target'] = 0

    # random split to train ranker
    train_users, test_users = train_test_split(
        local_test[USER_IDS].unique(),
        test_size=.2,
        random_state=13
        )

    cbm_train_set = shuffle(
        pd.concat(
            [positive_preds.loc[positive_preds[USER_IDS].isin(train_users)],
             negative_preds.loc[negative_preds[USER_IDS].isin(train_users)]]
        )
    )

    cbm_test_set = shuffle(
        pd.concat(
            [positive_preds.loc[positive_preds[USER_IDS].isin(test_users)],
             negative_preds.loc[negative_preds[USER_IDS].isin(test_users)]]
        )
    )

    USER_FEATURES = user_features
    ITEM_FEATURES = item_features

    # joins user features
    cbm_train_set = pd.merge(cbm_train_set, users_data[[USER_IDS] + USER_FEATURES],
                             how='left', on=[USER_IDS])
    cbm_test_set = pd.merge(cbm_test_set, users_data[[USER_IDS] + USER_FEATURES],
                            how='left', on=[USER_IDS])

    # joins item features
    cbm_train_set = pd.merge(cbm_train_set, items_metadata[[ITEM_IDS] + ITEM_FEATURES],
                             how='left', on=[ITEM_IDS])
    cbm_test_set = pd.merge(cbm_test_set, items_metadata[[ITEM_IDS] + ITEM_FEATURES],
                            how='left', on=[ITEM_IDS])

    ID_COLS = [USER_IDS] + [ITEM_IDS]
    TARGET = ['target']
    DROP_COLS = drop_features

    X_train, y_train = cbm_train_set.drop(ID_COLS + DROP_COLS + TARGET, axis=1), cbm_train_set[TARGET]
    X_test, y_test = cbm_test_set.drop(ID_COLS + DROP_COLS + TARGET, axis=1), cbm_test_set[TARGET]

    X_train = X_train.fillna(X_train.mode().iloc[0])
    X_test = X_test.fillna(X_test.mode().iloc[0])

    logging.info("Train / test set for ranker formed.")
    return X_train, X_test, y_train, y_test


def get_items_features(
        items_metadata_path: str,
        item_ids: List[str],
        item_ids_col: str,
        item_cols: List[str]
) -> Dict[int, Any]:
    """
    Function to get items features from available data
    that we used in training (for all candidates)
    :param items_metadata_path: str, the location (link) of the item features file
    :param item_ids: List[str], item ids to filter by
    :param item_ids_col: str, name of the item ids column in data
    :param item_cols: List[str], feature cols are need for inference
    :return: Dict[int, Any], dictionary in which: item (movie) id as key,
    dictionary (with structure feature name: value) as values

    EXAMPLE OUTPUT
    {
    9169: {
    'content_type': 'film',
    'release_year': 2020,
    'for_kids': None,
    'age_rating': 16
        },

    10440: {
    'content_type': 'series',
    'release_year': 2021,
    'for_kids': None,
    'age_rating': 18
        }
    }

    """
    from utils.utils import read_csv_from_gdrive

    items_metadata = read_csv_from_gdrive(items_metadata_path)

    item_features = items_metadata[items_metadata[item_ids_col].isin(item_ids)]\
        .set_index(item_ids_col)[item_cols]\
        .apply(lambda x: x.to_dict(), axis=1)\
        .to_dict()

    logging.info('Items features formed.')
    return item_features


def get_user_features(
        users_data_path: str,
        user_id: int,
        user_ids_col: str,
        user_cols: List[str]
) -> Dict[str, Any]:
    """
    Function to get user features from available data
    that we used in training
    :param users_data_path: str, the location (link) of the user features file
    :param user_id: int, user id to filter by
    :param user_ids_col: str, name of the user ids column in data
    :param user_cols: List[str], feature cols are need for inference
    :return: Dict[int, Any], dictionary with structure user feature names as keys, user feature values as values

    EXAMPLE OUTPUT
    {
        'age': None,
        'income': None,
        'sex': None,
        'kids_flg': None
    }
    """
    from utils.utils import read_csv_from_gdrive

    users_data = read_csv_from_gdrive(users_data_path)

    if users_data[users_data[user_ids_col] == user_id].shape[0] == 0:
        user_features = dict(zip(user_cols, ['#N/A'] * len(user_cols)))
    else:
        user_features = users_data[users_data[user_ids_col] == user_id][user_cols]\
            .apply(lambda x: x.to_dict(), axis=1)\
            .values[0]

    logging.info('User features formed.')
    return user_features


def prepare_ranker_input(
    candidates: Dict[int, int],
    item_features: Dict[int, Any],
    user_features: Dict[int, Any],
    ranker_features_order,
) -> List:
    """
    A function for preparing data for prediction on a ranker
    :param candidates: Dict[int, int], dictionary containing recommendations ordered by rank:
    item (movie) id as keys, rank as values
    :param item_features: Dict[int, Any], dictionary in which: item (movie) id as key,
    dictionary (with structure feature name: value) as values
    :param user_features: Dict[int, Any], dictionary with structure
    user feature names as keys, user feature values as values
    :param ranker_features_order: required order of user / item features arrangement
    :return: List, two-dimensional list with user / item (in accordance with LightFM) features for ranker input
    """
    ranker_input = []
    for k in item_features.keys():
        item_features[k].update(user_features)
        item_features[k]['rank'] = candidates[k]
        item_features[k] = {
            feature: item_features[k][feature] for feature in ranker_features_order
        }
        ranker_input.append(list(item_features[k].values()))

    logging.info('Ranker input formed.')

    return ranker_input
