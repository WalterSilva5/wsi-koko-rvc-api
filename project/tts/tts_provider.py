import aiohttp
import asyncio
import logging
from aiohttp import ClientTimeout


class TtsProvider:
    def __init__(self, url="http://localhost:8880/v1"):
        self.url = url
        self.logger = logging.getLogger(self.__class__.__name__)

    async def synthesize(self, text: str, options: dict = None) -> dict:
        if options is None:
            options = {}

        # TODO cadastrar vozes e mapear o cadastro
        # "af_kore" = voz masculina
        # "af_alloy" = voz feminina
        payload = {
            "model": "tts-1-hd",
            "input": text,
            "voice": "af_kore",
            "response_format": options.get("response_format", "mp3"),
            "download_format": options.get("download_format", "mp3"),
            "speed": options.get("speed", 1),
            "stream": options.get("stream", False),
            "return_download_link": options.get("return_download_link", False),
            "lang_code": options.get("lang_code", "a"),
            "volume_multiplier": options.get("volume_multiplier", 1),
            "normalization_options": options.get(
                "normalization_options", {"normalize": True}
            ),
        }

        endpoint = f"{self.url}/audio/speech"
        timeout = ClientTimeout(total=60)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Ensure payload is a dictionary
                if hasattr(payload, "dict"):
                    payload = payload.dict()

                async with session.post(endpoint, json=payload) as response:
                    content_type = response.headers.get("Content-Type", "")
                    self.logger.info(
                        f"Received response with content-type: {content_type}"
                    )

                    if content_type.startswith("application/json"):
                        json_response = await response.json()
                        self.logger.info(f"Response JSON: {json_response}")
                        return {"success": True, **json_response}

                    audio_data = await response.read()
                    self.logger.info("Returning audio buffer")
                    return {
                        "success": True,
                        "audio": audio_data,
                        "content_type": content_type,
                    }

        except aiohttp.ClientError as e:
            self.logger.error(f"Error synthesizing speech: {str(e)}")
            raise Exception(f"Error synthesizing speech: {str(e)}")


# Example usage
# async def main():
#     provider = TtsProvider()
#     result = await provider.synthesize("Hello world", "voice")
#     print(result)

# asyncio.run(main())
