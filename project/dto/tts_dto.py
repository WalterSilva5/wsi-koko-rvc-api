from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel, Field


#TODO criar endpoint de tts com conversão com voz mapeada 
class RvcTtsDTO(BaseModel):
    voice: Optional[str] = None
    target_voice: Optional[str] = None
    text: str

class RvcDTO(BaseModel):
    target_voice: Optional[str] = None


class KokoroTtsDto(BaseModel):
    voice: Optional[str] = Field(None, description="Nome da voz")
    text: str = Field(..., description="Texto de entrada para TTS")
    response_format: Optional[str] = Field(None, description="Formato da resposta")
    download_format: Optional[str] = Field(None, description="Formato para download")
    speed: Optional[float] = Field(None, ge=0.5, le=2.0, description="Velocidade da fala")
    stream: Optional[bool] = Field(None, description="Se deve fazer streaming do áudio")
    return_download_link: Optional[bool] = Field(None, description="Retornar link para download")
    lang_code: Optional[str] = Field(None, description="Código do idioma")
    volume_multiplier: Optional[float] = Field(None, ge=0.1, le=2.0, description="Multiplicador de volume")