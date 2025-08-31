from fastapi import UploadFile
from project.conversor.audio.loading_service import AudioLoadingService
from project.conversor.core_conversion_service import CoreConversionService
from project.core.application import Application
from project.dto.tts_dto import RvcDTO
import time


class ConversorService:
    def __init__(self):
        self.app = Application()
        print("Initializing ConversorService")
        self.core_service = CoreConversionService()
        self.audio_loading_service = AudioLoadingService()
        
    async def get_speakers(self) -> list[str]:
        print("Retrieving available speakers")
        speakers = await self.core_service.get_speakers()
        print("speakers", speakers) 
        return speakers
    
    async def convert_voice_for_file(self, dto: RvcDTO, audio_file: UploadFile) -> bytes:
        print(f"[Audio] Starting voice conversion for {audio_file.filename}")
        try:
            audio_array, temp_file_path = await self.audio_loading_service.load_from_upload_file(audio_file)
            
            target_embedding = self.core_service.get_speaker_embedding(dto.target_voice)
            output_buffer = await self.core_service.convert_voice(audio_array, target_embedding)
            
            self.audio_loading_service.cleanup_temp_file(temp_file_path)
            return output_buffer
            
        except Exception as e:
            self.app.logger.error(f"[Audio] Error converting voice: {str(e)}", exc_info=True)
            print(f"[Audio] Error converting voice: {str(e)}")
            raise

    async def get_converted_audio(self, dto: RvcDTO, audio_file: UploadFile):
        print("Processing audio conversion")
        try:
            print("Loading audio file...")
            audio_array, temp_file_path = await self.audio_loading_service.load_from_upload_file(audio_file)
            if audio_array is None or len(audio_array) == 0:
                print("Audio array is empty after loading. Check the input file.")
                raise ValueError("Audio array is empty after loading.")
            print(f"Audio array loaded. Shape: {audio_array.shape}, Temp file path: {temp_file_path}")

            print("Getting target speaker embedding...")
            target_embedding = self.core_service.get_speaker_embedding(dto.target_voice)
            if target_embedding is None:
                print("Failed to retrieve target speaker embedding.")
                raise ValueError("Target speaker embedding is None.")
            print(f"Target embedding shape: {target_embedding.shape}")

            print("Converting voice...")
            output_buffer = await self.core_service.convert_voice(audio_array, target_embedding)
            if output_buffer is None or len(output_buffer) == 0:
                print("Output buffer is empty after voice conversion.")
                raise ValueError("Output buffer is empty after voice conversion.")
            print(f"Output buffer type: {type(output_buffer)}, Length: {len(output_buffer)}")

            self.audio_loading_service.cleanup_temp_file(temp_file_path)
            print("Temporary file cleaned up.")

            return output_buffer
            
        except Exception as e:
            self.app.logger.error(f"Error during audio conversion: {str(e)}", exc_info=True)
            print(f"Error during audio conversion: {str(e)}")
            raise