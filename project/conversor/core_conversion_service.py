# filepath: src/conversor/core_conversion_service.py
import numpy as np
import time
import os
from project.conversor.processor import VoiceConverterProcessor
from project.conversor.manager.file_model_manager import FileModelManager
from project.embedding.factory import EmbeddingFactory
from project.embedding.manager import EmbeddingManager
from project.core.application import Application

class CoreConversionService:
    def __init__(self):
        self.app = Application()
        self.app.logger.info("Initializing CoreConversionService")
        self.model_manager = FileModelManager.get_instance()
        model_base_path = self.app.envs.MODELS_DIR_PATH
        speakers_path = self.app.envs.SPEAKERS_DIR_PATH
        
        if not os.path.exists(model_base_path):
            self.app.logger.error(f"Model base path does not exist: {model_base_path}")
            raise FileNotFoundError(f"Model base path does not exist: {model_base_path}")
        
        self.app.logger.info(f"Loading model from {model_base_path}")
        try:
            self.model_manager.load_model(model_base_path)
            if not self.model_manager.model.is_loaded():
                raise RuntimeError("Model failed to load")
        except Exception as e:
            self.app.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
            
        self.model = self.model_manager.model
        self.embedding_factory = EmbeddingFactory(self.model)
        self.embedding_manager = EmbeddingManager(self.embedding_factory, speakers_path)
        self.voice_converter = VoiceConverterProcessor(self.model)
        self.app.logger.info("CoreConversionService initialized successfully")
    
    async def get_speakers(self) -> list[str]:
        """Get list of available speakers"""
        self.app.logger.info("Retrieving available speakers")
        return self.embedding_manager.get_all_embeddings_names()
    
    def get_speaker_embedding(self, speaker: str) -> np.ndarray:
        """Get embedding for a specific speaker"""
        self.app.logger.info(f"[Audio] Getting embedding for speaker: {speaker}")
        embedding_start = time.time()
        target_embedding = self.embedding_manager.get_embedding(speaker)
        embedding_time = time.time() - embedding_start
        self.app.logger.info(f"[Audio] Speaker embedding obtained in {embedding_time:.2f} seconds")
        return target_embedding

    async def convert_voice(self, audio_array: np.ndarray, target_embedding: np.ndarray) -> np.ndarray:
        """Execute core voice conversion"""
        self.app.logger.info("[Audio] Processing voice conversion")
        conversion_start = time.time()
        try:
            output_buffer = await self.voice_converter.convert_voice(audio_array, target_embedding)
            if output_buffer is None or len(output_buffer) == 0:
                self.app.logger.error("[Audio] Empty audio buffer after voice conversion")
                raise ValueError("Empty audio buffer after voice conversion")
                
            conversion_time = time.time() - conversion_start
            self.app.logger.info(f"[Audio] Voice conversion completed in {conversion_time:.2f} seconds")
            return output_buffer
        except Exception as e:
            conversion_time = time.time() - conversion_start
            self.app.logger.error(f"[Audio] Error during voice conversion after {conversion_time:.2f} seconds: {str(e)}", exc_info=True)
            raise
