from models.lfm import LFMModel
from models.ranker import Ranker

from configs.config import settings

from data_prep.prepare_ranker_data import (
    get_user_features,
    get_items_features,
    prepare_ranker_input
    )

import logging


def get_recommendations(user_id: int):
    """
    function to get recommendation for a given user id
    """

    lfm_model = LFMModel()
    ranker = Ranker()
    print(user_id)
    logging.info('getting 1st level candidates')
    candidates = lfm_model.infer(user_id = user_id) # FIXME: remove comment when feaature collection is done
    # print(candidates)
    logging.info('getting features...')
    user_features = get_user_features(
        users_data_path = settings.DATA.USERS_DATA_PATH,
        user_id = user_id,
        user_ids_col = settings.USERS.USER_IDS,
        user_cols = settings.USERS.USER_FEATURES
        )
    # print(user_features)

    item_features = get_items_features(
        items_metadata_path = settings.DATA.ITEMS_METADATA_PATH,
        item_ids = list(candidates.keys()),
        item_ids_col = settings.ITEMS.ITEM_IDS,
        item_cols = settings.ITEMS.ITEM_FEATURES
        )
    # print(item_features)

    # TODO - TMP hardcode, need to use the output of the 36-37 lines
    # candidates = {9169: 5, 10440: 1}
    # item_features = {
    #         9169: {
    #         'content_type': 'film',
    #         'release_year': 2020,
    #         'for_kids': 0,
    #         'age_rating': 16
    #             },

    #         10440: {
    #         'content_type': 'series',
    #         'release_year': 2021,
    #         'for_kids': None,
    #         'age_rating': 18
    #             }
    #         }

    # user_features = {
    #         'age': 'age_55_64',
    #         'income': 'income_20_40',
    #         'sex': 'M',
    #         'kids_flg': 0
    #     }

    ranker_input = prepare_ranker_input(
        candidates = candidates,
        item_features = item_features,
        user_features = user_features,
        ranker_features_order = ranker.ranker.feature_names_
        )
    # print(ranker_input)
    preds = ranker.infer(ranker_input = ranker_input)
    # print(preds)
    output = dict(zip(candidates.keys(), preds))
    # print(output)
    return output


if __name__ == '__main__':
    print(get_recommendations(215229))
