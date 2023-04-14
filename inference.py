from configs.config import logging
from configs.config import settings
from typing import Dict

from models.lfm import LFMModel
from models.ranker import Ranker

from data_prep.prepare_lfm_data import prepare_data_for_train_lfm
from data_prep.prepare_ranker_data import (
    get_user_features,
    get_items_features,
    prepare_ranker_input
    )


def get_recommendations(
        user_id: int,
        known_items_filter: bool = False
) -> Dict[str, float]:
    """
    Function to get recommendation for a given user id
    :param user_id: int, user id in dataset
    :param known_items_filter: bool, identification of the need to filter the items known to the user
    :return: Dict[str, float], containing recommendations ordered by LightFM rank:
    item (movie) id as keys, CatBoost probability as values
    """

    lfm_model = LFMModel()
    ranker = Ranker()

    logging.info('Getting 1st level candidates...')
    if known_items_filter is False:
        candidates = lfm_model.infer(
            user_id=user_id
        )
    elif known_items_filter is True:
        _, _, known_items = prepare_data_for_train_lfm(settings.DATA, known_items_filter=True)
        candidates = lfm_model.infer(
            user_id=user_id,
            known_items=known_items
        )

    logging.info('Getting user features...')
    user_features = get_user_features(
        users_data_path=settings.DATA.USERS_DATA_PATH,
        user_id=user_id,
        user_ids_col=settings.USERS.USER_IDS,
        user_cols=settings.USERS.USER_FEATURES
        )

    logging.info('Getting items features...')
    item_features = get_items_features(
        items_metadata_path=settings.DATA.ITEMS_METADATA_PATH,
        item_ids=list(candidates.keys()),
        item_ids_col=settings.ITEMS.ITEM_IDS,
        item_cols=settings.ITEMS.ITEM_FEATURES
        )

    logging.info('Getting ranker input...')
    ranker_input = prepare_ranker_input(
        candidates=candidates,
        item_features=item_features,
        user_features=user_features,
        ranker_features_order=ranker.ranker.feature_names_
        )

    preds = ranker.infer(ranker_input=ranker_input)

    output = dict(zip(candidates.keys(), preds))
    logging.info(f'The final result of the prediction for user {user_id} is ready.')

    return output


if __name__ == '__main__':
    print(get_recommendations(settings.TARGET_USER.TARGET_USER_ID, known_items_filter=True))
