import io
import base64
from project.core.application import Application
from project.dto.tts_dto import RvcDTO
from project.conversor.service import ConversorService
from typing import BinaryIO, Dict, Union
from project.conversor.audio.format.service import AudioFormatConversor
class StreamService:
    def __init__(self):
        self.app = Application()
        self.conversor_service = ConversorService()
        self.format = AudioFormatConversor()

    async def process_audio_stream(
        self,
        dto: RvcDTO,
        audio_file: BinaryIO
    ) -> Dict[str, Union[str, str]]:
        return await self.handle_get_stream_audio(dto, audio_file)

    async def handle_get_stream_audio(self, dto, audio_file):
        self.app.logger.info("Converting audio...")
        audio_buffer = await self.conversor_service.get_converted_audio(dto, audio_file)
        self.app.logger.info("Audio conversion completed")
        
        if isinstance(audio_buffer, io.BytesIO):
            self.app.logger.debug("Converting BytesIO to bytes")
            audio_bytes = audio_buffer.getvalue()
        else:
            audio_bytes = audio_buffer
            
        self.app.logger.debug(f"Audio size: {len(audio_bytes)} bytes")
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        self.app.logger.info("Audio successfully encoded to base64")

        return {
            "status": "success",
            "audio": encoded_audio,
            "audio_format": dto.audio_format
        }
