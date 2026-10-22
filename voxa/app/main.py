from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.routes import status as status_routes
from app.routes import voice
from core.config import get_settings
from core.logger import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # fails fast on invalid/missing configuration
    configure_logging(settings.log_level)
    get_logger("voxa.startup").info(
        "voxa_starting", model=settings.gemini_model, project=settings.gcp_project_id, location=settings.gcp_location
    )
    yield
    get_logger("voxa.startup").info("voxa_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Voxa",
        description="AI voice agent platform powered by Gemini on Vertex AI",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(voice.router)
    app.include_router(status_routes.router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        get_logger("voxa.error").error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal server error"})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.app_host, port=s.app_port, log_config=None)
