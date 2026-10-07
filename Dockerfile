FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TZ=UTC
WORKDIR /app

COPY pyproject.toml README.md ./
COPY heartless ./heartless
RUN pip install --upgrade pip && pip install ".[advisor]"

VOLUME ["/app/data"]
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=10s --retries=3 CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/',timeout=5).status in (200,401) else 1)" || exit 1

ENTRYPOINT ["heartless"]
CMD ["run"]
