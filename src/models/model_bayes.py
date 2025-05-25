from bayes_opt import BayesianOptimization
from src.models.base_model import BaseModelClass
from src.utils.config import MODEL_REGISTRY, GLOBAL_RANDOM_STATE, BAYESOPT_SPACE, BAYESOPT_CONVERSION

class BayesOptModel(BaseModelClass):

    def __init__(self, model_type, target, n_iter=50):
        super().__init__(model_type, target)
        self.n_iter = n_iter

    def _create_objective(self):
        def objective(**kwargs):
            params = self._convert_params(kwargs)
            classifier = MODEL_REGISTRY[self.model_type](**params)
            pipeline = self._build_pipeline(classifier)
            return self._evaluate_model(pipeline)
        return objective

    def _convert_params(self, kwargs):
        params = {}
        conversion_config = BAYESOPT_CONVERSION[self.model_type]

        for param_name, conversion in conversion_config.items():
            if param_name == 'random_state':
                params[param_name] = conversion
            elif conversion == 'int':
                params[param_name] = int(kwargs[param_name])
            elif conversion == 'float':
                params[param_name] = kwargs[param_name]
            elif isinstance(conversion, tuple):
                options, _ = conversion
                params[param_name] = options[int(kwargs[param_name])]

        return params

    def train(self, data_path):
        self._load_and_split_data(data_path)

        objective = self._create_objective()
        bounds = BAYESOPT_SPACE[self.model_type]

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=bounds,
            random_state=GLOBAL_RANDOM_STATE,
            verbose=0
        )

        optimizer.maximize(init_points=10, n_iter=self.n_iter)

        best_params_raw = optimizer.max['params']
        self.best_params = self._convert_params(best_params_raw)
        return self._fit_best_model(self.best_params)