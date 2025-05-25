import numpy as np
from hyperopt import tpe, fmin, Trials, STATUS_OK
from src.models.base_model import BaseModelClass
from src.utils.config import MODEL_REGISTRY, GLOBAL_RANDOM_STATE, HYPEROPT_SPACE, HYPEROPT_CONVERSION

class HyperoptModel(BaseModelClass):

    def __init__(self, model_type, target, max_evals=50):
        super().__init__(model_type, target)
        self.max_evals = max_evals

    def _convert_params(self, params):
        converted = params.copy()

        if self.model_type in HYPEROPT_CONVERSION:
            conversion_config = HYPEROPT_CONVERSION[self.model_type]

            for param_name, options in conversion_config.items():
                if param_name in params:
                    converted[param_name] = options[params[param_name]]

        return converted

    def train(self, data_path):
        self._load_and_split_data(data_path)

        def objective(params):
            classifier = MODEL_REGISTRY[self.model_type](**params)
            pipeline = self._build_pipeline(classifier)
            score = self._evaluate_model(pipeline)
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()

        best = fmin(
            fn=objective,
            space=HYPEROPT_SPACE[self.model_type],
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(GLOBAL_RANDOM_STATE)
        )

        self.best_params = self._convert_params(best)
        return self._fit_best_model(self.best_params)