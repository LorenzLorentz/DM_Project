from model import BaseModel

class DINModel(BaseModel):
    def __init__(self, lr = None, num_epochs = None, load_path = None, save_path = None, random_seed = None):
        super().__init__(lr, num_epochs, load_path, save_path, random_seed)

    def train(self, feature, label, eval, eval_metrics, verbose):
        return super().train(feature, label, eval, eval_metrics, verbose)
    
    def predict(self, feature):
        return super().predict(feature)