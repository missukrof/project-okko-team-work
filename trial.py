from configs.config import settings
from data_prep.prepare_ranker_data import get_items_features
from utils.utils import read_csv_from_gdrive

INTERACTIONS_PATH = 'https://drive.google.com/file/d/1i6kmeJIUNJjqAa0vAvztFSDN3BFz9x1i/view?usp=share_link'
ITEMS_METADATA_PATH = 'https://drive.google.com/file/d/12a80lS3vXQOl6i6ENgz-WqWw3Wms0nqB/view?usp=share_link'
USERS_DATA_PATH = 'https://drive.google.com/file/d/1MwPaye0cRi53czLqCnH0bOuvIhOeNlAx/view?usp=share_link'


# interactions = read_csv_from_gdrive(INTERACTIONS_PATH)
movies_metadata = read_csv_from_gdrive(ITEMS_METADATA_PATH)
# users_data = read_csv_from_gdrive(USERS_DATA_PATH)

movies_metadata.set_index('item_id')[settings.item_features.ITEMS.ITEM_FEATURES].apply(lambda x: x.to_dict(), axis=1).to_dict()
