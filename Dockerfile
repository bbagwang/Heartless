FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TZ=UTC
WORKDIR /app

COPY pyproject.toml README.md ./
COPY heartless ./heartless
RUN pip install --upgrade pip && pip install ".[advisor]"

VOLUME ["/app/data"]
EXPOSE 8080
# Liveness probe. An anonymous GET / is answered 401 (login page); http.client -- unlike urllib.request.urlopen, which raises
# on every 4xx -- hands 4xx back as a status, so 200/401 -> healthy and 5xx / refused / timeout -> unhealthy. The probe
# presents no token, so the dashboard's wrong-token throttle never counts it. WEB_PORT is honoured, a disabled dashboard
# (WEB_ENABLED=0) is always healthy since there is nothing to probe, and start-period covers the candle backfill that
# runs before the web server is started.
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
  CMD python -c "import http.client,os,sys;os.environ.get('WEB_ENABLED','1').strip().lower() in ('0','false','f','no','n','off') and sys.exit(0);c=http.client.HTTPConnection('127.0.0.1',int(os.environ.get('WEB_PORT') or 8080),timeout=5);c.request('GET','/');sys.exit(0 if c.getresponse().status in (200,401) else 1)" || exit 1

ENTRYPOINT ["heartless"]
CMD ["run"]
