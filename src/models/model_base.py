from src.models.base_model import BaseModelClass
from src.utils.config import MODEL_REGISTRY, MODEL_DEFAULT_PARAMS

class BaseModel(BaseModelClass):

    def train(self, data_path):
        self._load_and_split_data(data_path)

        params = MODEL_DEFAULT_PARAMS[self.model_type].copy()
        classifier = MODEL_REGISTRY[self.model_type](**params)

        self.best_model = self._build_pipeline(classifier)
        self.best_model.fit(self.X_train, self.y_train)

        return self.best_model.score(self.X_test, self.y_test)