from skopt import gp_minimize
from skopt.utils import use_named_args
from src.models.base_model import BaseModelClass
from src.utils.config import MODEL_REGISTRY, GLOBAL_RANDOM_STATE, SKOPT_SPACE

class SkoptModel(BaseModelClass):

    def __init__(self, model_type, target, n_calls=50):
        super().__init__(model_type, target)
        self.n_calls = n_calls

    def train(self, data_path):
        self._load_and_split_data(data_path)

        @use_named_args(SKOPT_SPACE[self.model_type])
        def objective(**params):
            classifier = MODEL_REGISTRY[self.model_type](**params)
            pipeline = self._build_pipeline(classifier)
            score = self._evaluate_model(pipeline)
            return -score

        result = gp_minimize(
            func=objective,
            dimensions=SKOPT_SPACE[self.model_type],
            n_calls=self.n_calls,
            random_state=GLOBAL_RANDOM_STATE
        )

        best_params = {}
        for i, dim in enumerate(SKOPT_SPACE[self.model_type]):
            best_params[dim.name] = result.x[i]

        self.best_params = best_params
        return self._fit_best_model(self.best_params)