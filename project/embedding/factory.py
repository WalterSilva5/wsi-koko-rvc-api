from TTS.vc.models.openvoice import OpenVoice# type: ignore
import torch

class EmbeddingFactory:
    def __init__(self, model: OpenVoice):
        self.model = model

    def create_embedding(self, wav_path: str) -> torch.Tensor:
        print(f"Creating embedding for {wav_path}")
        se, _ = self.model.extract_se(wav_path)
        return se
