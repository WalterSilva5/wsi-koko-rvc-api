from fastapi import APIRouter
from project.core.application import Application
from project.router.rvc_router import router as rvc_router

app = Application()
router = APIRouter()

router.include_router(rvc_router)
