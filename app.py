from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from project.conversor.controller import router as conversor_router
from project.core.application import Application
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from logging_config import logger

app = Application()

server = FastAPI(
    title="wsi Voice Conversor API",
    description="API for wsi Voice Conversor",
    version="0.0.1",
    swagger_url="/docs",
    swagger_ui_parameters={
        "syntaxHighlight": {
            "activated": True
        }
    },
    debug=True,
)

class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.error("Unhandled exception: %s", str(exc), exc_info=True)
            raise

server.add_middleware(ExceptionLoggingMiddleware)

@server.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.info("Validation error: %s - Body: %s", exc.errors(), exc.body)
    print(f"Validation error: {exc.errors()} - Body: {exc.body}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@server.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.info("HTTP error: %s", exc.detail)
    print(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@server.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.info("Unhandled error: %s", str(exc), exc_info=True)
    print(f"Unhandled error: {str(exc)}")
    print(f"Request: {request}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

server.include_router(conversor_router, prefix="/api", tags=["Voice Conversion"])
