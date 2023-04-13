from configs.config import logging
from typing import Any, Dict, List

import dill
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset


class LFMModel:
    def __init__(
            self,
            model_path: str = "./artefacts/lfm_model.dill",
            mapper_path: str = "./artefacts/lfm_mapper.dill"
    ):
        logging.info('Loading the LightFM model...')
        with open(model_path, "rb") as model_file:
            self.lfm_model = dill.load(model_file)

        logging.info('Loading the LightFM mapper...')
        with open(mapper_path, "rb") as mapper_file:
            self.dataset = dill.load(mapper_file)

    @staticmethod
    def df_to_tuple_iterator(
            data: pd.DataFrame
    ):
        """
        special iterator to use with lightfm
        :df: pd.DataFrame, interactions dataframe
        returs iterator
        """
        return zip(*data.values.T)

    def fit(
            self,
            data: pd.DataFrame,
            user_col: str,
            item_col: str,
            model_params: Dict[str, Any] = {}
    ) -> None:
        """
        Trains and saves model with mapper for further inference
        """

        # init class and fit tuple of user and movie interactions
        dataset = Dataset()
        dataset.fit(data[user_col].unique(), data[item_col].unique())

        # defining train set on the whole interactions dataset
        logging.info('Creating training matrix...')
        train_mat, train_mat_weights = dataset.build_interactions(
            self.df_to_tuple_iterator(data[[user_col, item_col]])
        )

        # save mappers
        with open(f"./artefacts/lfm_mapper.dill", "wb") as mapper_file:
            dill.dump(dataset, mapper_file)

        # init model
        logging.info(f'LightFM model initialization with params {model_params}...')
        epochs = model_params.get("epochs", 10)
        lfm_model = LightFM(
            no_components=model_params.get("no_components", 10),
            learning_rate=model_params.get("learning_rate", 0.05),
            loss=model_params.get("loss", "logistic"),
            max_sampled=model_params.get("max_sampled", 10),
            random_state=model_params.get("random_state", 42),
        )

        # execute training
        logging.info('Fitting LightFM model...')
        for i in range(epochs):
            logging.info(f"Epoch num {i + 1} in LFM model training")
            lfm_model.fit_partial(train_mat, num_threads=4)

        # save model
        with open(
            f"{model_params.get('model_path', './artefacts/lfm_model.dill')}", "wb"
        ) as model_file:
            dill.dump(lfm_model, model_file)

    def infer(
        self,
        user_id: int,
        top_n: int = 20,
        known_items: Dict[int, List] = {},
        enable_logging: bool = True
    ) -> Dict[str, int]:
        """
        method to make recommendations for a single user id
        :user_id: str, user id
        :model_path: str, relative path for the model
        """
        if enable_logging is False:
            logging.disable(level=logging.INFO)

        logging.info('LightFM mapping initialization...')
        mapper = self.dataset.mapping()

        # set params
        user_row_id = mapper[0][user_id]
        all_items_list = list(mapper[2].values())

        logging.info('Making LightFM predictions...')
        preds = self.lfm_model.predict(user_row_id, all_items_list)

        # make final predictions
        item_inv_mapper = {v: k for k, v in mapper[2].items()}

        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = top_n + additional_N
        top_preds = np.argpartition(preds, -np.arange(total_N))[-total_N:][::-1]

        item_pred_ids = [item_inv_mapper[item] for item in top_preds]

        if additional_N > 0:
            filter_items = known_items[user_id]
            item_pred_ids = [item for item in item_pred_ids if item not in filter_items]

        # item_pred_ids = []
        # for item in top_preds:
        #     item_pred_ids.append(item_inv_mapper[item])

        final_preds = {v: k + 1 for k, v in enumerate(item_pred_ids)}
        logging.info('Final LightFM predictions formed.')

        if enable_logging is False:
            logging.disable(level=logging.NOTSET)

        return final_preds
