# The fast notifier: a long-lived process instead of a cron slot.
#
# GitHub's scheduler floors out around five minutes and queues under load. A
# loop you control does not, so this image exists for the case where the alert
# latency matters more than the forecast: set TTF_POLL_SECONDS=30 and the only
# remaining delay is the mirror's own crawl.
#
#   docker build -t truthforecast .
#   docker run -d --restart=always \
#     -e TTF_NTFY_TOPIC=your-topic-name \
#     -e TTF_POLL_SECONDS=30 \
#     -v truthforecast-data:/app/data \
#     -p 8000:8000 truthforecast
#
# The volume matters: it holds the archive and the coverage watermark, so a
# restart resumes instead of re-walking 34,000 posts.

FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so a code change does not rebuild the model stack.
COPY pyproject.toml README.md ./
COPY truthforecast ./truthforecast
RUN pip install --no-cache-dir -e .

COPY daemon.py ./
COPY site ./site

ENV TTF_POLL_SECONDS=60 \
    PYTHONUNBUFFERED=1

VOLUME ["/app/data"]
EXPOSE 8000

# One pass on startup so a fresh volume backfills and the site has something to
# serve; then the loop, which polls, notifies, forecasts and re-ranks nightly.
CMD ["python", "daemon.py", "--port", "8000"]
