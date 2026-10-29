from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from servicedesk.api.dependencies import get_session_factory
from servicedesk.api.schemas import HealthResponse
from servicedesk.db.models import Chunk, Document

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    response: Response,
    session_factory: Annotated[sessionmaker[Session], Depends(get_session_factory)],
) -> HealthResponse:
    """Is the service able to answer? Load balancers and Docker health checks call this."""
    try:
        with session_factory() as session:
            documents = session.scalar(select(func.count()).select_from(Document))
            chunks = session.scalar(select(func.count()).select_from(Chunk))
    except OperationalError:
        # 503 tells a load balancer to stop sending traffic here until the database is back.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", database="unreachable")
    return HealthResponse(status="ok", database="ok", documents=documents, chunks=chunks)
