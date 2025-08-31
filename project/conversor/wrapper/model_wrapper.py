from project.model.factory import ModelFactory
import os
import torch
from project.core.application import Application
from TTS.vc.models.openvoice import OpenVoice# type: ignore

class VoiceConverterModelWrapper:
    """Wrapper class for the voice converter model"""
    
    def __init__(self):
        self.app = Application()
        self.model: OpenVoice | None = None
        self.factory = ModelFactory()
        
    def load_model(self, model_path: str):
        """Load the model from the given path"""
        print(f"[ModelWrapper] Loading model from {model_path}")
        checkpoint_path = os.path.join(model_path, "model.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"[ModelWrapper] Config file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Config file not found: {checkpoint_path}")

        try:
            self.model = self.factory.create_model(checkpoint_path)
            print("[ModelWrapper] Model loaded successfully")
            return self.model
        except Exception as e:
            print(f"[ModelWrapper] Error loading model: {str(e)}")
            raise
    
    def extract_se(self, src):
        """Extract speaker embedding"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model first.")
        return self.model.extract_se(src)
    
    def inference(self, src_spec, aux_input):
        """Run inference on the model"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model first.")
        return self.model.inference(src_spec, aux_input)

    @property
    def config(self):
        """Get model config"""
        return self.model.config if self.model else None

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
