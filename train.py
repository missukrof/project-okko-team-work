from configs.config import settings

from data_prep.prepare_lfm_data import prepare_data_for_train_lfm
from data_prep.prepare_ranker_data import prepare_data_for_train

from models.lfm import LFMModel
from models.ranker import Ranker

local_train, local_test = prepare_data_for_train_lfm(
    paths_config=settings.DATA
)

# X_train, X_test, y_train, y_test = prepare_data_for_train(
#     paths_config=settings.DATA,
#     local_test=local_test,
#     user_features=settings.USERS.USER_FEATURES,
#     item_features=settings.ITEMS.ITEM_FEATURES,
#     drop_features=settings.PREPROCESS.DROP_FEATURES
# )

lfm_model = LFMModel()
lfm_model.fit(
    local_train,
    user_col=settings.USERS.USER_IDS,
    item_col=settings.ITEMS.ITEM_IDS
)

# ranker = Ranker()
# ranker.fit(
#     X_train=X_train,
#     y_train=y_train,
#     X_test=X_test,
#     y_test=y_test,
#     categorical_cols=settings.USERS.USER_FEATURES
# )
