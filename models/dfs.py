from models.lfm import LFMModel
import pandas as pd
from utils.utils import read_csv_from_gdrive
from data_prep.prepare_lfm_data import prepare_data_for_train_lfm
from configs.config import settings
import dill


train, test = prepare_data_for_train_lfm(settings.DATA)

test_preds = pd.DataFrame({
    'user_id': test['user_id'].unique()
})

model_path = "./artefacts/lfm_model.dill" 

with open("./artefacts/lfm_mapper.dill", "rb") as mapper_file:
            dataset = dill.load(mapper_file)
mapperr = dataset.mapping()

all_cols = list(mapperr[2].values())
item_inv_mapper = {v: k for k, v in mapperr[2].items()}

with open(model_path, "rb") as model_file:
    lfm_model = dill.load(model_file)

# print(mapperr)
mapper = LFMModel.generate_lightfm_recs_mapper(
    lfm_model,
    item_ids=all_cols,
    known_items = dict(),
    N = 10,
    user_features = None, 
    item_features = None,
    user_mapping = mapperr[0],
    item_inv_mapping = item_inv_mapper,
    num_threads = 4,
)

test_preds['item_id'] = test_preds['user_id'].map(mapper)

test_preds = test_preds.explode('item_id')
print(test_preds.shape)




