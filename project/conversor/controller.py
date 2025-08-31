from fastapi import APIRouter
from project.conversor.service import ConversorService
from project.core.application import Application
from project.conversor.stream.router import router as rvc_router

app = Application()
router = APIRouter()
conversor_service = ConversorService()

@router.get("/speakers/",
    summary="List all speakers",
    description="Get a list of all available speakers",
    response_description="Returns a list of speakers")
async def list_speakers():
    speakers = await conversor_service.get_speakers()
    return {"speakers": speakers}

@router.get("/speakers/compatibility/{speaker1}/{speaker2}",
    summary="Check speaker compatibility",
    description="Check the compatibility between two speakers for voice conversion",
    response_description="Returns compatibility analysis")
async def check_speaker_compatibility(speaker1: str, speaker2: str):
    try:
        compatibility = await conversor_service.check_speaker_compatibility(speaker1, speaker2)
        return compatibility
    except Exception as e:
        return {"error": str(e)}

@router.get("/speakers/similarity-matrix/",
    summary="Get similarity matrix",
    description="Get the similarity matrix between all speakers",
    response_description="Returns similarity matrix")
async def get_similarity_matrix():
    try:
        matrix = await conversor_service.get_similarity_matrix()
        return {"similarity_matrix": matrix}
    except Exception as e:
        return {"error": str(e)}

router.include_router(rvc_router)

