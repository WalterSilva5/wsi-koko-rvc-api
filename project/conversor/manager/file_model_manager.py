from project.conversor.wrapper.model_wrapper import VoiceConverterModelWrapper
from project.observers.observable import Observable
from typing import Optional


class FileModelManager(Observable):
    """
    Singleton class that manages the voice converter models.
    """
    _instance = None
    model: VoiceConverterModelWrapper

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = VoiceConverterModelWrapper()
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            super().__init__()
            self.initialized = True
            
    def load_model(self, model_path: str):
        """Load model and notify observers"""
        self.model.load_model(model_path)
        self.notify_observers({})

    @classmethod
    def get_instance(cls) -> "FileModelManager":
        if cls._instance is None:
            cls._instance = FileModelManager()
        return cls._instance
