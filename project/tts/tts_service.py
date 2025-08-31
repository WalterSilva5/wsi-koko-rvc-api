import requests
from project.core.application import Application
from project.dto.tts_dto import KokoroTtsDto, RvcTtsDTO
from project.tts.tts_provider import TtsProvider

class SynthesizerService:
    def __init__(self):
        self.app = Application()
        self.tts_provider = TtsProvider()

    async def synthesize_audio(self, dto: RvcTtsDTO) -> bytes:
        self.app.logger.info("Calling KokoroTTS Provider...")
        try:
            result = await self.tts_provider.synthesize(
                text=dto.text,
            )

            if result.get("success"):
                self.app.logger.info("KokoroTTS Provider call successful.")
                return result.get("audio")
            else:
                self.app.logger.error("KokoroTTS Provider returned an error.")
                raise Exception("Failed to synthesize audio.")
        except Exception as e:
            self.app.logger.error(f"Error calling KokoroTTS Provider: {str(e)}")
            raise
