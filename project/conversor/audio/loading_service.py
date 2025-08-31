import librosa
import numpy as np
import io
import tempfile
import os
import time
from fastapi import UploadFile
from project.core.application import Application

class AudioLoadingService:
    def __init__(self):
        self.app = Application()
        self.sample_rate = 24000

    async def create_temp_file(self, audio_file: UploadFile) -> str:
        print("[Audio] Creating temporary file")
        contents = await audio_file.read()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(contents)
            return temp_file.name

    async def load_audio_file(self, file_path: str) -> np.ndarray:
        print("[Audio] Loading audio file and extracting features")
        load_start = time.time()
        audio_array, _ = librosa.load(file_path, sr=self.sample_rate)
        load_time = time.time() - load_start
        print(f"[Audio] Audio loaded in {load_time:.2f} seconds. Shape: {audio_array.shape}")
        return audio_array

    async def load_from_upload_file(self, audio_file: UploadFile) -> tuple[np.ndarray, str]:
        """Carrega áudio de UploadFile, retorna array numpy e caminho do arquivo temporário."""
        print(f"[AudioLoad] Lendo conteúdo do arquivo {audio_file.filename}")
        contents = await audio_file.read()
        if not contents:
            self.app.logger.error("[AudioLoad] Arquivo de áudio de entrada está vazio")
            raise ValueError("Arquivo de áudio de entrada está vazio")

        print(f"[AudioLoad] Tamanho do conteúdo lido: {len(contents)} bytes")
        print("[AudioLoad] Criando arquivo temporário")
        # Usar um sufixo pode ajudar o librosa
        temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_f.write(contents)
        temp_file_path = temp_f.name
        temp_f.close() # Fechar o manipulador de arquivo

        try:
            print(f"[AudioLoad] Carregando array de áudio de {temp_file_path}")
            print(f"[AudioLoad] Arquivo existe: {os.path.exists(temp_file_path)}, Tamanho: {os.path.getsize(temp_file_path)} bytes")
            load_start = time.time()
            
            # Tentar carregar com diferentes configurações de taxa de amostragem
            try:
                audio_array, sr = librosa.load(temp_file_path, sr=self.sample_rate, mono=True)
            except Exception as load_error:
                self.app.logger.warning(f"[AudioLoad] Erro na primeira tentativa: {load_error}")
                # Tentar carregar com a taxa de amostragem nativa do arquivo
                audio_array, sr = librosa.load(temp_file_path, sr=None, mono=True)
            
            print(f"[AudioLoad] Taxa de amostragem original: {sr}Hz")
            if audio_array is None:
                self.app.logger.error("[AudioLoad] audio_array é None após o carregamento")
                raise ValueError("Falha ao carregar o áudio")
                
            if sr != self.sample_rate:
                self.app.logger.warning(f"[AudioLoad] Taxa de amostragem original {sr} difere do alvo {self.sample_rate}. Reamostrando.")
            load_time = time.time() - load_start
            print(f"[AudioLoad] Áudio carregado em {load_time:.2f} segundos. Shape: {audio_array.shape}")
            return audio_array, temp_file_path
        except Exception as e:
            self.cleanup_temp_file(temp_file_path) # Limpa se o carregamento falhar
            self.app.logger.error(f"[AudioLoad] Erro ao carregar áudio do arquivo temporário {temp_file_path}: {e}", exc_info=True)
            raise

    async def load_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Carrega áudio de bytes, retorna array numpy."""
        if not audio_bytes:
            self.app.logger.error("[AudioLoad] Bytes de áudio de entrada estão vazios")
            raise ValueError("Bytes de áudio de entrada estão vazios")
        try:
            print("[AudioLoad] Carregando array de áudio de bytes")
            load_start = time.time()
            audio_data = io.BytesIO(audio_bytes)
            audio_array, sr = librosa.load(audio_data, sr=self.sample_rate)
            if sr != self.sample_rate:
                 self.app.logger.warning(f"[AudioLoad] Taxa de amostragem original {sr} difere do alvo {self.sample_rate}. Reamostrando.")
            load_time = time.time() - load_start
            print(f"[AudioLoad] Áudio carregado de bytes em {load_time:.2f} segundos. Shape: {audio_array.shape}")
            return audio_array
        except Exception as e:
            self.app.logger.error(f"[AudioLoad] Erro ao carregar áudio de bytes: {e}", exc_info=True)
            raise

    def cleanup_temp_file(self, temp_file_path: str):
        """Exclui um arquivo temporário."""
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                self.app.logger.debug(f"[Audio] Temporary file deleted: {temp_file_path}")
        except Exception as e:
            self.app.logger.error(f"[Audio] Error deleting temporary file: {e}", exc_info=True)
