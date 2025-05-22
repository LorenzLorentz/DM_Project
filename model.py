import pandas as pd

class BaseModel:
    def __init__(self, lr:float=None, num_epochs:int=None,load_path:str=None, save_path:str=None, random_seed:int=None,):
        pass

    def train(self, feature:pd.DataFrame, label:pd.Series, eval:list[tuple[pd.DataFrame, pd.Series]], eval_metrics:str, verbose:int):
        pass

    def predict(self, feature:pd.DataFrame) -> pd.Series:
        pass