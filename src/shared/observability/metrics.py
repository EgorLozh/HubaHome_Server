from time import perf_counter

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

HTTP_REQUEST_COUNT = Counter(
    "hubahome_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_LATENCY = Histogram(
    "hubahome_http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)
WS_EVENT_COUNT = Counter(
    "hubahome_ws_events_total",
    "Total WebSocket events by event type",
    ["event_type"],
)
ERROR_COUNT = Counter(
    "hubahome_errors_total",
    "Total errors by source",
    ["source"],
)


class HttpMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = perf_counter()
        path = request.url.path
        method = request.method

        try:
            response = await call_next(request)
        except Exception:
            ERROR_COUNT.labels(source="http").inc()
            raise
        finally:
            elapsed = perf_counter() - started
            HTTP_REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)

        HTTP_REQUEST_COUNT.labels(
            method=method,
            path=path,
            status_code=str(response.status_code),
        ).inc()
        return response


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
