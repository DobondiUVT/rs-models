import optuna
from src.models.base_model import BaseModelClass
from src.utils.config import MODEL_REGISTRY, GLOBAL_RANDOM_STATE, OPTUNA_PARAMS

class OptunaModel(BaseModelClass):

    def __init__(self, model_type, target, n_trials=50):
        super().__init__(model_type, target)
        self.n_trials = n_trials

    def _suggest_params(self, trial):
        params = {}
        param_config = OPTUNA_PARAMS[self.model_type]

        for param_name, param_def in param_config.items():
            if param_name == 'random_state':
                params[param_name] = param_def
            elif param_def[0] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_def[1], param_def[2])
            elif param_def[0] == 'float':
                params[param_name] = trial.suggest_float(param_name, param_def[1], param_def[2])
            elif param_def[0] == 'float_log':
                params[param_name] = trial.suggest_float(param_name, param_def[1], param_def[2], log=True)
            elif param_def[0] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_def[1])

        return params

    def train(self, data_path):
        self._load_and_split_data(data_path)

        def objective(trial):
            params = self._suggest_params(trial)
            classifier = MODEL_REGISTRY[self.model_type](**params)
            pipeline = self._build_pipeline(classifier)
            return self._evaluate_model(pipeline)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_STATE)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        return self._fit_best_model(self.best_params)