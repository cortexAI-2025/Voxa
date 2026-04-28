from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.logger import configure_logging, get_logger
from app.routes import voice, status as status_routes


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("voxa.startup")
    logger.info(
        "voxa_starting",
        model=settings.gemini_model,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )
    yield
    logger.info("voxa_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Voxa",
        description="AI Voice Agent Platform powered by Gemini 2.0 Flash",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(voice.router)
    app.include_router(status_routes.router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger = get_logger("voxa.error")
        logger.error("unhandled_exception", path=str(request.url), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_config=None,  # structlog handles logging
    )
