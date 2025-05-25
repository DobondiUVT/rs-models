from openbox import Optimizer
from src.models.base_model import BaseModelClass
from src.utils.config import MODEL_REGISTRY, GLOBAL_RANDOM_STATE, OPENBOX_SPACE

class OpenboxModel(BaseModelClass):

    def __init__(self, model_type, target, max_runs=50):
        super().__init__(model_type, target)
        self.max_runs = max_runs

    def _create_objective(self):
        def objective(config):
            params = dict(config)
            classifier = MODEL_REGISTRY[self.model_type](**params)
            pipeline = self._build_pipeline(classifier)
            score = self._evaluate_model(pipeline)
            return {'objectives': [-score]}
        return objective

    def train(self, data_path):
        self._load_and_split_data(data_path)

        objective = self._create_objective()
        space = OPENBOX_SPACE[self.model_type]

        optimizer = Optimizer(
            objective,
            space,
            max_runs=self.max_runs,
            random_state=GLOBAL_RANDOM_STATE,
            task_id=f'{self.model_type}_optimization'
        )

        history = optimizer.run()

        best_observation = history.get_incumbents()[0]
        self.best_params = dict(best_observation.config)
        return self._fit_best_model(self.best_params)