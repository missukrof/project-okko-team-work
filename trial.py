from configs.config import settings
from data_prep.prepare_lfm_data import prepare_data_for_train_lfm

from models.lfm import LFMModel


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

