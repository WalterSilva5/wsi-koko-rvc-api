from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from project.conversor.service import ConversorService
from project.core.application import Application
from project.dto.tts_dto import RvcTtsDTO, RvcDTO
from project.tts.tts_service import SynthesizerService
import io
import os
import soundfile as sf
import tempfile

app = Application()
router = APIRouter()
conversor_service = ConversorService()
synthesizer_service = SynthesizerService()

@router.post("/rvc",
    summary="Convert voice from file and stream audio",
    description="Convert voice from file and return audio stream",
    response_class=JSONResponse,
)
async def apply_rvc(
    audio_file: UploadFile = File(..., description="Audio file to be converted"),
    speaker: str = Form("voice", description="Target speaker for voice conversion"),
):
    print(f"\n\n\nStarting voice conversion for file: {audio_file.filename}")
    try:
        dto = RvcDTO(
            target_voice=speaker
        )
        print(f"Created DTO: {dto}")
        
        print("Converting audio...")
        audio_buffer = await conversor_service.get_converted_audio(dto, audio_file)
        print("Audio conversion completed")
        
        if isinstance(audio_buffer, io.BytesIO):
            print("Converting BytesIO to bytes")
            audio_bytes = audio_buffer.getvalue()
        else:
            print("audio_buffer", type(audio_buffer))
            audio_bytes = audio_buffer
            
        print(f"Audio size: {len(audio_bytes)} bytes")

        # Inspecionar o tipo e o conteúdo do audio_buffer
        print(f"Type of audio_buffer: {type(audio_buffer)}")
        if isinstance(audio_buffer, io.BytesIO):
            print(f"BytesIO buffer size: {audio_buffer.getbuffer().nbytes} bytes")
        else:
            print(f"Buffer content type: {type(audio_buffer)}")

        if audio_bytes is None or len(audio_bytes) == 0:
            app.logger.error("Audio buffer is empty. Conversion might have failed.")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Audio buffer is empty. Conversion failed."},
            )

        print(f"Audio buffer size: {len(audio_bytes)} bytes")

        # Converter o numpy.ndarray para um arquivo de áudio válido
        temp_file_path = tempfile.mktemp(suffix=".wav")
        app.logger.debug("Saving numpy array as a valid WAV file...")
        try:
            sf.write(temp_file_path, audio_bytes, samplerate=24000)
            app.logger.info(f"Audio successfully saved as WAV file: {temp_file_path}")
        except Exception as e:
            app.logger.error(f"Failed to save audio as WAV file: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to save audio as WAV file."},
            )

        # Verificar se o arquivo foi salvo corretamente
        if not os.path.exists(temp_file_path):
            app.logger.error("Temporary file was not created.")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to create temporary file."},
            )

        file_size = os.path.getsize(temp_file_path)
        app.logger.info(f"Temporary file size: {file_size} bytes")

        if file_size == 0:
            app.logger.error("Temporary file is empty.")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Temporary file is empty."},
            )

        # Retornar o arquivo como resposta
        return FileResponse(temp_file_path, media_type="audio/wav", filename="converted_audio.wav")

    except Exception as e:
        app.logger.error(f"Error during voice conversion: {str(e)}", exc_info=True)
        raise


@router.post("/tts",
    summary="Synthesize text and apply voice conversion",
    description="Synthesize text using KokoroTTS and apply voice conversion",
    response_class=JSONResponse,
)
async def apply_rvc_in_tts(
    text: str = Form(..., description="Text to synthesize"),
    speaker: str = Form("voice", description="Target speaker for voice conversion"),
):
    print(f"\n\n\nStarting TTS and voice conversion for text: {text}")
    try:
        # Step 1: Synthesize audio using KokoroTTS
        print("Synthesizing audio using KokoroTTS...")
        dto = RvcTtsDTO(text=text, voice=speaker)
        audio_data = await synthesizer_service.synthesize_audio(dto)
        print("Audio synthesis completed")

        # Validate synthesized audio_data
        if not audio_data:
            app.logger.error("Synthesized audio data is None or empty.")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Synthesized audio data is invalid."},
            )

        # Wrap audio_data into a file-like object with a filename
        audio_file = UploadFile(
            file=io.BytesIO(audio_data),
            filename="synthesized_audio.wav",
        )

        # Step 2: Apply voice conversion
        dto = RvcDTO(target_voice=speaker)
        print("Applying voice conversion...")
        try:
            audio_buffer = await conversor_service.get_converted_audio(dto, audio_file)
        except Exception as e:
            app.logger.error(f"Error during audio conversion: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Audio conversion failed."},
            )

        print("Voice conversion completed")

        # Validate audio_buffer
        if audio_buffer is None:
            app.logger.error("Audio buffer is None. Conversion might have failed.")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Audio buffer is None. Conversion failed."},
            )

        if isinstance(audio_buffer, io.BytesIO):
            print("Converting BytesIO to bytes")
            audio_bytes = audio_buffer.getvalue()
        else:
            print("audio_buffer", type(audio_buffer))
            audio_bytes = audio_buffer

        if audio_bytes is None or len(audio_bytes) == 0:
            app.logger.error("Audio bytes are empty. Conversion might have failed.")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Audio bytes are empty. Conversion failed."},
            )

        print(f"Audio size: {len(audio_bytes)} bytes")

        # Convert numpy.ndarray to a valid audio file
        temp_file_path = tempfile.mktemp(suffix=".wav")
        app.logger.debug("Saving numpy array as a valid WAV file...")
        try:
            sf.write(temp_file_path, audio_bytes, samplerate=24000)
            app.logger.info(f"Audio successfully saved as WAV file: {temp_file_path}")
        except Exception as e:
            app.logger.error(f"Failed to save audio as WAV file: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to save audio as WAV file."},
            )

        # Verificar se o arquivo foi salvo corretamente
        if not os.path.exists(temp_file_path):
            app.logger.error("Temporary file was not created.")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to create temporary file."},
            )

        file_size = os.path.getsize(temp_file_path)
        app.logger.info(f"Temporary file size: {file_size} bytes")

        if file_size == 0:
            app.logger.error("Temporary file is empty.")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Temporary file is empty."},
            )

        # Return the file as a response
        return FileResponse(temp_file_path, media_type="audio/wav", filename="converted_audio.wav")

    except Exception as e:
        app.logger.error(f"Error during TTS and voice conversion: {str(e)}", exc_info=True)
        raise