from configs.config import settings
from data_prep.prepare_lfm_data import prepare_data_for_train_lfm
from data_prep.prepare_ranker_data import prepare_data_for_train

from models.lfm import LFMModel
from models.ranker import Ranker


# # print(settings.DATA)
# # path_dict = settings.DATA
# # print(path_dict.get('INTERACTIONS_PATH'))

local_train, local_test = prepare_data_for_train_lfm(settings.DATA)

lfm_model = LFMModel()
lfm_model.fit(
    local_train, 
    user_col=settings.USERS.USER_IDS, 
    item_col=settings.ITEMS.ITEM_IDS
    )


X_train, y_train, X_test, y_test = prepare_data_for_train(settings)
ranker_params = {
        "loss_function": "CrossEntropy",
        "iterations": 5000,
        "lr": 0.1,
        "depth": 6,
        "random_state": 1234,
        "verbose": True}


ranker_model = Ranker()
ranker_model.fit(
    X_train,
    y_train,
    X_test,
    y_test, 
    ranker_params=ranker_params
)

