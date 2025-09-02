from abc import ABC, abstractmethod
from dataclasses import dataclass
from TTS.vc.configs.openvoice_config import OpenVoiceConfig  # type: ignore
from TTS.vc.models.openvoice import OpenVoice  # type: ignore
from typing import Type, Any, Tuple
import torch
import os


@dataclass
class ModelConfig:
    """Base configuration for all models"""

    config_path: str
    model_path: str


class VoiceModel(ABC):
    """Abstract base class for voice models"""

    @abstractmethod
    def __init__(self, config: ModelConfig):
        pass

    @abstractmethod
    def load_checkpoint(self, config: ModelConfig) -> None:
        pass

    @abstractmethod
    def to_cuda(self) -> None:
        pass

    @abstractmethod
    def to_cpu(self) -> None:
        pass

    @abstractmethod
    def extract_se(self, src: str) -> Tuple[torch.Tensor, Any]:
        """Extract speaker embedding from audio file"""
        pass

    @abstractmethod
    def inference(self, src_spec: torch.Tensor, aux_input: Any) -> torch.Tensor:
        """Run inference on source spectrogram with auxiliary input"""
        pass


class OpenVoiceModelAdapter(VoiceModel):
    """Adapter for OpenVoice model to work with our interface"""

    def __init__(self, config: ModelConfig):
        self.config = OpenVoiceConfig(config.config_path)
        self.model = OpenVoice(self.config)

    def load_checkpoint(self, config: ModelConfig) -> None:
        self.model.load_checkpoint(self.config, config.model_path, eval=True)

    def to_cuda(self) -> None:
        self.model.cuda()

    def to_cpu(self) -> None:
        self.model.cpu()

    def extract_se(self, src: str) -> Tuple[torch.Tensor, Any]:
        """Extract speaker embedding from audio file using OpenVoice model"""
        return self.model.extract_se(src)

    def inference(self, src_spec: torch.Tensor, aux_input: Any) -> torch.Tensor:
        """Run inference using OpenVoice model"""
        return self.model.inference(src_spec, aux_input)


class ModelFactory:
    """Factory for creating voice models with better testability"""

    def __init__(self, model_class: Type[VoiceModel] = OpenVoiceModelAdapter):
        self.model_class = model_class

    def create_model(self, model_path: str) -> VoiceModel:
        config_path = os.path.join(model_path, "config.json")
        config = ModelConfig(config_path=config_path, model_path=model_path)
        model = self.model_class(config)
        model.load_checkpoint(config)

        if torch.cuda.is_available():
            model.to_cuda()

        return model
