import os
from typing import Dict, Any
import torch
from project.embedding.factory import EmbeddingFactory
from project.core.application import Application

app = Application()


class EmbeddingManager:
    def __init__(self, factory: EmbeddingFactory, speakers_path: str):
        self.factory = factory
        self.speakers_path = speakers_path
        print(f"Speakers path: {speakers_path}")
        print(f"Speakers path: {speakers_path}")
        self.embeddings: Dict[str, torch.Tensor] = {}

        self.load_all_speakers()
        print(f"Initialized EmbeddingManager with speakers path: {speakers_path}")
        print(f"Initialized EmbeddingManager with speakers path: {speakers_path}")

    def load_all_speakers(self) -> None:
        print("Loading all speakers")
        print("Loading all speakers")
        for speaker_name in os.listdir(self.speakers_path):
            print(f"Loading speaker: {speaker_name}")
            if speaker_name.endswith(".wav"):
                speaker_name = speaker_name[:-4]
                self.load_speaker(speaker_name)

    def load_speaker(self, speaker_name: str) -> None:
        print(f"Loading speaker: {speaker_name}")
        wav_path = f"{self.speakers_path}/{speaker_name}.wav"
        if not os.path.exists(wav_path):
            app.logger.error(f"Speaker file not found: {wav_path}")
            raise FileNotFoundError(f"Speaker file not found: {wav_path}")
        self.embeddings[speaker_name] = self.factory.create_embedding(wav_path)
        app.logger.debug(f"Successfully loaded embedding for speaker: {speaker_name}")
        print(f"Successfully loaded embedding for speaker: {speaker_name}")

    def get_embedding(self, speaker_name: str) -> torch.Tensor:
        app.logger.debug(f"Getting embedding for speaker: {speaker_name}")
        app.logger.debug(f"Available speakers: {list(self.embeddings.keys())}")

        if speaker_name not in self.embeddings:
            print(f"Speaker {speaker_name} not loaded, loading now...")
            print(f"Speaker {speaker_name} not loaded, loading now...")
            self.load_speaker(speaker_name)

        embedding = self.embeddings[speaker_name]
        app.logger.debug(
            f"Retrieved embedding for {speaker_name} with shape: {embedding.shape}"
        )
        return embedding

    def get_all_embeddings_names(self) -> list:
        app.logger.debug("Getting all embedding names")
        return list(self.embeddings.keys())

    def get_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Retorna uma matriz de similaridade entre todos os speakers"""
        similarity_matrix = {}
        speakers = list(self.embeddings.keys())

        for speaker1 in speakers:
            similarity_matrix[speaker1] = {}
            for speaker2 in speakers:
                if speaker1 == speaker2:
                    similarity_matrix[speaker1][speaker2] = 1.0
                else:
                    embedding1 = self.embeddings[speaker1]
                    embedding2 = self.embeddings[speaker2]

                    similarity = torch.cosine_similarity(
                        embedding1.flatten(), embedding2.flatten(), dim=0
                    ).item()

                    similarity_matrix[speaker1][speaker2] = similarity

        return similarity_matrix

    def check_speaker_compatibility(
        self, speaker1: str, speaker2: str
    ) -> Dict[str, Any]:
        """Verifica a compatibilidade entre dois speakers para conversão"""
        if speaker1 not in self.embeddings or speaker2 not in self.embeddings:
            raise ValueError(
                f"Um dos speakers não está carregado: {speaker1}, {speaker2}"
            )

        embedding1 = self.embeddings[speaker1]
        embedding2 = self.embeddings[speaker2]

        similarity = torch.cosine_similarity(
            embedding1.flatten(), embedding2.flatten(), dim=0
        ).item()

        if similarity > 0.95:
            quality = "CRÍTICO"
            recommendation = "Não recomendado - embeddings quase idênticos"
        elif similarity > 0.85:
            quality = "RUIM"
            recommendation = "Conversão será muito sutil"
        elif similarity > 0.7:
            quality = "MODERADO"
            recommendation = "Conversão pode ser sutil mas perceptível"
        else:
            quality = "BOM"
            recommendation = "Boa diferença para conversão efetiva"

        return {
            "similarity": similarity,
            "quality": quality,
            "recommendation": recommendation,
            "speaker1": speaker1,
            "speaker2": speaker2,
        }
