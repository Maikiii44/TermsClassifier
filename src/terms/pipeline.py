import pandas as pd
from terms.model.module import TermsModule
from terms.preprocess import preprocess


class Pipeline:
    def __init__(self, pretrained_model_name):
        self.pretrained_model_name = pretrained_model_name

        self.model = ""


    def preprocess(self, data : str | pd.DataFrame):
        data_preprocess = preprocess(data=data)
        return data_preprocess
    
    def postprocess(self):
        return
    
    def inference(self, input_ids, attention):
        return
    
    def run(self, data):

        return self.postprocess(self.inference(self.preprocess(data = data)))
