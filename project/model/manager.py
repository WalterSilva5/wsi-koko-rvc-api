import os
from TTS.vc.models.openvoice import OpenVoice # type: ignore
from .factory import ModelFactory# type: ignore

class ModelManager:
    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.model: OpenVoice = None
        self.checkpoint_path = os.path.join(model_base_path, "config.json")
        self.model_path = os.path.join(model_base_path, "model.pth")
        self.factory = ModelFactory()
        
    def get_model(self) -> OpenVoice:
        if self.model is None:
            self.model = self.factory.create_model(self.checkpoint_path)
        return self.model
