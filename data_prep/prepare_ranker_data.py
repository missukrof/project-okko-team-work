# TODO
# WRITE PIPELINE FOR DATA PREPARATION IN HERE TO USE FOR RANKER TRAININIG PIPELINE
import pandas as pd
from typing import Any, Dict, List
from configs.config import settings


def prepare_data_for_train(paths_config: Dict[str, str]):
    """
    function to prepare data to train catboost classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py

        paths_config: dict, where key is path name and value is the path to data
    """
    from models.lfm import LFMModel
    from data_prep.prepare_lfm_data import prepare_data_for_train_lfm
    from utils.utils import read_csv_from_gdrive
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from models.lfm import LFMModel
    import dill

    local_train, local_test = prepare_data_for_train_lfm(paths_config.DATA)
    users_data = read_csv_from_gdrive(settings.DATA.USERS_DATA_PATH)
    movies_metadata = read_csv_from_gdrive(settings.DATA.ITEMS_METADATA_PATH)

    # Вот отсюда
    test_preds = pd.DataFrame({
        'user_id': local_test['user_id'].unique()
    })

    model_path = "./artefacts/lfm_model.dill" 

    with open("./artefacts/lfm_mapper.dill", "rb") as mapper_file:
                dataset = dill.load(mapper_file)
    mapperr = dataset.mapping()

    all_cols = list(mapperr[2].values())
    item_inv_mapper = {v: k for k, v in mapperr[2].items()}
    item_name_mapper = dict(zip(movies_metadata['item_id'], movies_metadata['title']))

    with open(model_path, "rb") as model_file:
        lfm_model = dill.load(model_file)


    mapper = LFMModel.generate_lightfm_recs_mapper(
        model=lfm_model,
        item_ids=all_cols,
        known_items = dict(),
        N = 10,
        user_features = None, 
        item_features = None,
        user_mapping = mapperr[0],
        item_inv_mapping = item_inv_mapper,
        num_threads = 4,
    #model_path= "./artefacts/lfm_model.dill"
    )
    # До сюда нужно переделать код в более приятный и без бесконечного количества аргументов в этом файле и в соответствующем методе lfm
    
    test_preds['item_id'] = test_preds['user_id'].map(mapper)

    test_preds = test_preds.explode('item_id')
    test_preds['rank'] = test_preds.groupby('user_id').cumcount() + 1 

    test_preds['item_name'] = test_preds['item_id'].map(item_name_mapper)

    positive_preds = pd.merge(test_preds, local_test, how = 'inner', on = ['user_id', 'item_id'])
    positive_preds['target'] = 1

    negative_preds = pd.merge(test_preds, local_test, how = 'left', on = ['user_id', 'item_id'])
    negative_preds = negative_preds.loc[negative_preds['watched_pct'].isnull()].sample(frac = .2)
    negative_preds['target'] = 0

    train_users, test_users = train_test_split(
    local_test['user_id'].unique(),
    test_size = .2,
    random_state = 13
    )

    cbm_train_set = shuffle(
    pd.concat(
    [positive_preds.loc[positive_preds['user_id'].isin(train_users)],
    negative_preds.loc[negative_preds['user_id'].isin(train_users)]]
    ))

    cbm_test_set = shuffle(
    pd.concat(
    [positive_preds.loc[positive_preds['user_id'].isin(test_users)],
    negative_preds.loc[negative_preds['user_id'].isin(test_users)]]
    ))

    # USER_FEATURES = settings.ITEMS.ITEM_FEATURES
    # ITEM_FEATURES = settings.USERS.USER_FEATURES

    USER_FEATURES = ['age', 'income', 'sex', 'kids_flg']
    ITEM_FEATURES = ['content_type', 'release_year', 'for_kids', 'age_rating']  


    cbm_train_set = pd.merge(cbm_train_set, users_data[['user_id'] + USER_FEATURES],
                         how = 'left', on = ['user_id'])
    cbm_test_set = pd.merge(cbm_test_set, users_data[['user_id'] + USER_FEATURES],
                        how = 'left', on = ['user_id'])
    
    cbm_train_set = pd.merge(cbm_train_set, movies_metadata[['item_id'] + ITEM_FEATURES],
                         how = 'left', on = ['item_id'])
    cbm_test_set = pd.merge(cbm_test_set, movies_metadata[['item_id'] + ITEM_FEATURES],
                        how = 'left', on = ['item_id'])
    
    # в томл
    ID_COLS = ['user_id', 'item_id']
    TARGET = ['target']
    CATEGORICAL_COLS = ['age', 'income', 'sex', 'content_type']
    DROP_COLS = ['item_name', 'last_watch_dt', 'watched_pct', 'total_dur']

    X_train, y_train = cbm_train_set.drop(ID_COLS + DROP_COLS + TARGET, axis = 1), cbm_train_set[TARGET]
    X_test, y_test = cbm_test_set.drop(ID_COLS + DROP_COLS + TARGET, axis = 1), cbm_test_set[TARGET]

    X_train = X_train.fillna(X_train.mode().iloc[0])
    X_test = X_test.fillna(X_test.mode().iloc[0])

    X_train['age'] = X_train['age'].astype(object)
    X_test['age'] = X_test['age'].astype(object)
    y_test['age'] = y_test['age'].astype(object)
    y_train['age'] = y_train['age'].astype(object)
    
    return X_train, y_train, X_test, y_test

def get_items_features(items_metadata_path: str, item_ids: List[str], item_ids_col: str, item_cols: List[str]) -> Dict[int, Any]:
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
    from utils.utils import read_csv_from_gdrive

    items_metadata = read_csv_from_gdrive(items_metadata_path)

    return items_metadata[items_metadata[item_ids_col].isin(item_ids)].set_index(item_ids_col)[item_cols].apply(lambda x: x.to_dict(), axis=1).to_dict()


def get_user_features(users_data_path: str, user_id: int, user_ids_col: str, user_cols: List[str]) -> Dict[str, Any]:
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
    from utils.utils import read_csv_from_gdrive

    users_data = read_csv_from_gdrive(users_data_path)

    if users_data[users_data[user_ids_col] == user_id].shape[0] == 0:
        user_features = dict(zip(user_cols, [None] * len(user_cols)))
    else:
        user_features = users_data[users_data[user_ids_col] == user_id][user_cols].apply(lambda x: x.to_dict(), axis=1).values[0]
    
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
