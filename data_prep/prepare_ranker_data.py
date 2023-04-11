# TODO
# WRITE PIPELINE FOR DATA PREPARATION IN HERE TO USE FOR RANKER TRAININIG PIPELINE
import pandas as pd
from typing import Any, Dict, List


def prepare_data_for_train(paths_config: Dict[str, str]):
    """
    function to prepare data to train catboost classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py

        paths_config: dict, where key is path name and value is the path to data
    """
    import datetime as dt
    from models.lfm import LFMModel
    from utils.utils import read_csv_from_gdrive

    for k, v in paths_config.items():
        globals()[k.lower()[:-5]] = read_csv_from_gdrive(v)

    # remove redundant data points
    interactions_filtered = interactions.loc[interactions['total_dur'] > 300].reset_index(drop=True)
    interactions_filtered['last_watch_dt'] = pd.to_datetime(interactions_filtered['last_watch_dt'])

    # set dates params for filter
    MAX_DATE = interactions_filtered['last_watch_dt'].max()
    MIN_DATE = interactions_filtered['last_watch_dt'].min()
    TEST_INTERVAL_DAYS = 14

    TEST_MAX_DATE = MAX_DATE - dt.timedelta(days=TEST_INTERVAL_DAYS)

    # define global train and test
    global_train = interactions_filtered.loc[interactions_filtered['last_watch_dt'] < TEST_MAX_DATE]
    global_test = interactions_filtered.loc[interactions_filtered['last_watch_dt'] >= TEST_MAX_DATE]

    # now, we define "local" train and test to use some part of the global train for ranker
    local_train_thresh = global_train['last_watch_dt'].quantile(q=.7, interpolation='nearest')

    global_train = global_train.dropna().reset_index(drop=True)

    local_train = global_train.loc[global_train['last_watch_dt'] < local_train_thresh]
    local_test = global_train.loc[global_train['last_watch_dt'] >= local_train_thresh]

    # finally, we will focus on warm start -- remove cold start users
    local_test = local_test.loc[local_test['user_id'].isin(local_train['user_id'].unique())]
    
    lfm_model = LightFM()


    # Now, we need to creat 0/1 as indication of interaction
    # positive event -- 1, if watch_pct is not null
    positive_preds = pd.merge(test_preds, local_test, how = 'inner', on = ['user_id', 'item_id'])
    positive_preds['target'] = 1


    # Нихера не поняла
    pass


def get_items_features(df: pd.DataFrame, item_ids: str, item_cols: List[str]) -> Dict[int, Any]:
    """
    function to get items features from our available data
    that we used in training (for all candidates)
        :item_ids:  item ids to filter by
        :item_cols: feature cols we need for inference

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
    item_features = df.set_index(item_ids)[item_cols].apply(lambda x: x.to_dict(), axis=1).to_dict()

    return item_features


def get_user_features(df: pd.DataFrame, user_id: int, user_ids: str, user_cols: List[str]) -> Dict[str, Any]:
    """
    function to get user features from our available data
    that we used in training
        :user_id: user id to filter by
        :user_cols: feature cols we need for inference

    EXAMPLE OUTPUT
    {
        'age': None,
        'income': None,
        'sex': None,
        'kids_flg': None
    }
    """
    user_features = df[df[user_ids] == user_id][user_cols].apply(lambda x: x.to_dict(), axis=1).values[0]

    return user_features


def prepare_ranker_input(
    candidates: Dict[int, int],
    item_features: Dict[int, Any],
    user_features: Dict[int, Any],
    ranker_features_order,
):
    ranker_input = []
    for k in item_features.keys():
        item_features[k].update(user_features)
        item_features[k]["rank"] = candidates[k]
        item_features[k] = {
            feature: item_features[k][feature] for feature in ranker_features_order
        }
        ranker_input.append(list(item_features[k].values()))

    return ranker_input
