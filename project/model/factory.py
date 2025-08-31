from abc import ABC, abstractmethod
from dataclasses import dataclass
from TTS.vc.configs.openvoice_config import OpenVoiceConfig# type: ignore
from TTS.vc.models.openvoice import OpenVoice# type: ignore
from typing import Type,Any, Tuple
from glob import glob
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
        
    def extract_se(self, src: Any) -> Tuple[torch.Tensor, Any]:
        """Extract speaker embedding.

        Backwards compatible behaviour:
        - if `src` is a str pointing to a file -> delegate to the wrapped model (unchanged)
        - if `src` is a str pointing to a directory -> find all WAV files and compute per-file embeddings, returning the mean
        - if `src` is a list/tuple of file paths -> compute per-file embeddings and return the mean

        Returns a tuple (se, spec) where `se` is the speaker embedding tensor and `spec` is the
        spectrogram for the first processed file (or None if unavailable).
        """
        # Single path: file or directory
        if isinstance(src, str):
            # directory: aggregate all wavs inside
            if os.path.isdir(src):
                wavs = sorted(glob(os.path.join(src, "*.wav")))
                if len(wavs) == 0:
                    raise FileNotFoundError(f"No wav files found in directory: {src}")
                ses = []
                specs = []
                for wav in wavs:
                    g, spec = self.model.extract_se(wav)
                    # move to CPU for safe stacking and normalize along channel dim
                    g_cpu = g.to(torch.device("cpu")) if g.device != torch.device("cpu") else g
                    # g shape may be [1, C, 1] -> squeeze last dim, normalize, keep batch dim
                    g_vec = g_cpu.squeeze(-1)
                    g_vec = torch.nn.functional.normalize(g_vec, p=2, dim=1)
                    ses.append(g_vec)
                    specs.append(spec)
                # choose aggregation mode via env var SE_AGGREGATION: 'mean' (default), 'first', 'max_norm'
                agg = os.environ.get('SE_AGGREGATION', 'mean').lower()
                ses_cat = torch.cat(ses, dim=0)  # [N, C]
                if agg == 'first':
                    se_vec = ses_cat[0:1]
                elif agg == 'max_norm':
                    norms = torch.norm(ses_cat, dim=1)
                    idx = int(torch.argmax(norms).item())
                    se_vec = ses_cat[idx:idx+1]
                else:
                    # mean
                    se_vec = ses_cat.mean(dim=0, keepdim=True)

                se_vec = torch.nn.functional.normalize(se_vec, p=2, dim=1)
                # restore shape [1, C, 1] and move to model device
                se = se_vec.unsqueeze(-1).to(self.model.device)

                # log aggregation mode for visibility
                try:
                    print(f"[ModelFactory] SE aggregation='{agg}', num_segments={len(ses)}")
                except Exception:
                    pass

                return se, specs[0] if len(specs) > 0 else None
            # file: keep original behaviour
            return self.model.extract_se(src)

        # Sequence of paths: compute per-file and average
        if isinstance(src, (list, tuple)):
            if len(src) == 0:
                raise ValueError("Empty list passed to extract_se")
            ses = []
            specs = []
            for wav in src:
                g, spec = self.model.extract_se(wav)
                g_cpu = g.to(torch.device("cpu")) if g.device != torch.device("cpu") else g
                g_vec = g_cpu.squeeze(-1)
                g_vec = torch.nn.functional.normalize(g_vec, p=2, dim=1)
                ses.append(g_vec)
                specs.append(spec)
            ses_cat = torch.cat(ses, dim=0)

            agg = os.environ.get('SE_AGGREGATION', 'mean').lower()
            if agg == 'first':
                se_vec = ses_cat[0:1]
            elif agg == 'max_norm':
                norms = torch.norm(ses_cat, dim=1)
                idx = int(torch.argmax(norms).item())
                se_vec = ses_cat[idx:idx+1]
            else:
                se_vec = ses_cat.mean(dim=0, keepdim=True)

            se_vec = torch.nn.functional.normalize(se_vec, p=2, dim=1)
            se = se_vec.unsqueeze(-1).to(self.model.device)
            try:
                print(f"[ModelFactory] SE aggregation='{agg}', num_segments={len(ses)}")
            except Exception:
                pass
            return se, specs[0] if len(specs) > 0 else None

        # Fallback: delegate to wrapped model (keeps retro-compatibility for other types)
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
