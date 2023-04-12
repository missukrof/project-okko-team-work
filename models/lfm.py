import logging
from typing import Any, Dict

import dill
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset


class LFMModel:
    def __init__(self):
        pass

    @staticmethod
    def df_to_tuple_iterator(data: pd.DataFrame):
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
        model_params: Dict[str, Any] = {},
    ) -> None:
        """
        Trains and saves model with mapper for further inference
        """

        # init class and fit tuple of user and movie interactions
        dataset = Dataset()
        dataset.fit(data[user_col].unique(), data[item_col].unique())

        # defining train set on the whole interactions dataset
        logging.info("Creating training matrix")
        train_mat, train_mat_weights = dataset.build_interactions(
            self.df_to_tuple_iterator(data[[user_col, item_col]])
        )

        # save mappers
        with open(f"./artefacts/lfm_mapper.dill", "wb") as mapper_file:
            dill.dump(dataset, mapper_file)

        # init model
        epochs = model_params.get("epochs", 10)
        lfm_model = LightFM(
            no_components=model_params.get("no_components", 10),
            learning_rate=model_params.get("learning_rate", 0.05),
            loss=model_params.get("loss", "logistic"),
            max_sampled=model_params.get("max_sampled", 10),
            random_state=model_params.get("random_state", 42),
        )

        # execute training
        for i in range(epochs):
            logging.info(f"Epoch num {i} in LFM model training")
            lfm_model.fit_partial(train_mat, num_threads=4)

        # save model
        with open(
            f"{model_params.get('model_path', './artefacts/lfm_model.dill')}", "wb"
        ) as model_file:
            dill.dump(lfm_model, model_file)

    @staticmethod
    def infer(
        user_id: int, top_n: int = 20, model_path: str = "./artefacts/lfm_model.dill"
    ) -> Dict[str, int]:
        """
        method to make recommendations for a single user id
        :user_id: str, user id
        :model_path: str, relative path for the model
        """
        with open(model_path, "rb") as model_file:
            lfm_model = dill.load(model_file)

        with open("./artefacts/lfm_mapper.dill", "rb") as mapper_file:
            dataset = dill.load(mapper_file)
        mapper = dataset.mapping()

        # set params
        user_row_id = mapper[0][user_id]
        all_items_list = list(mapper[2].values())

        preds = lfm_model.predict(user_row_id, all_items_list)

        # make final predictions
        item_inv_mapper = {v: k for k, v in mapper[2].items()}
        top_preds = np.argpartition(preds, -np.arange(top_n))[-top_n:][::-1]
        item_pred_ids = []
        for item in top_preds:
            item_pred_ids.append(item_inv_mapper[item])
        final_preds = {v: k + 1 for k, v in enumerate(item_pred_ids)}

        return final_preds
    
    @staticmethod
    def generate_lightfm_recs_mapper(
        model: object,
        item_ids: list,
        known_items: dict,
        user_features: list,
        item_features: list,
        user_mapping,
        item_inv_mapping,
        N: int,
        num_threads: int = 4,
        # model_path: str = "./artefacts/lfm_model.dill"
        ):
        def _recs_mapper(user):

            # with open(model_path, "rb") as model_file:
            #     model = dill.load(model_file)
            
            # with open("./artefacts/lfm_mapper.dill", "rb") as mapper_file:
            #     dataset = dill.load(mapper_file)

            # mapper = dataset.mapping()

            # user_mapping = mapper[0]
            # item_inv_mapping = {v: k for k, v in mapper[2].items()}
            # item_ids = list(mapper[2].values())

            user_id = user_mapping[user]
            recs = model.predict(
                user_id,
                item_ids,
                user_features = user_features,
                item_features = item_features,
                num_threads = num_threads)
        
            additional_N = len(known_items[user_id]) if user_id in known_items else 0
            total_N = N + additional_N
            top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]
        
            final_recs = [item_inv_mapping[item] for item in top_cols]
            if additional_N > 0:
                filter_items = known_items[user_id]
                final_recs = [item for item in final_recs if item not in filter_items]
            return final_recs[:N]
        return _recs_mapper