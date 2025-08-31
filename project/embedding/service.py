# filepath: project/embedding/service.py
import numpy as np
import time
from typing import List
from project.embedding.factory import EmbeddingFactory
from project.embedding.manager import EmbeddingManager
from project.conversor.manager.file_model_manager import FileModelManager # Assumindo que o ModelManager é necessário aqui
from project.core.application import Application

class EmbeddingService:
    def __init__(self):
        self.app = Application()
        self.app.logger.info("Inicializando EmbeddingService")
        # Carregar modelo e configurar dependências de embedding
        self.model_manager = FileModelManager.get_instance()
        model_base_path = self.app.envs.MODELS_DIR_PATH
        speakers_path = self.app.envs.SPEAKERS_DIR_PATH
        if not self.model_manager.model:
            self.app.logger.info(f"Carregando modelo de {model_base_path}")
            self.model_manager.load_model(model_base_path)

        self.embedding_factory = EmbeddingFactory(self.model_manager.model)
        self.embedding_manager = EmbeddingManager(self.embedding_factory, speakers_path)
        self.app.logger.info("EmbeddingService inicializado com sucesso")

    def get_embedding(self, speaker_name: str) -> np.ndarray:
        """Busca o embedding para um determinado locutor."""
        self.app.logger.info(f"[Embedding] Buscando embedding para o locutor: {speaker_name}")
        embedding_start = time.time()
        try:
            target_embedding = self.embedding_manager.get_embedding(speaker_name)
            embedding_time = time.time() - embedding_start
            self.app.logger.info(f"[Embedding] Embedding do locutor obtido em {embedding_time:.2f} segundos")
            return target_embedding
        except Exception as e:
            self.app.logger.error(f"[Embedding] Erro ao buscar embedding para o locutor {speaker_name}: {e}", exc_info=True)
            raise # Propaga a exceção

    def get_all_speaker_names(self) -> List[str]:
        """Busca uma lista com os nomes de todos os locutores disponíveis."""
        self.app.logger.info("[Embedding] Buscando locutores disponíveis")
        try:
            speakers = self.embedding_manager.get_all_embeddings_names()
            self.app.logger.info(f"[Embedding] Locutores encontrados: {speakers}")
            return speakers
        except Exception as e:
            self.app.logger.error(f"[Embedding] Erro ao buscar nomes dos locutores: {e}", exc_info=True)
            return [] # Retorna lista vazia em caso de erro
