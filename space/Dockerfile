# ReMIX rating app — Hugging Face Space (Docker SDK).
#
# Docker, not the streamlit/gradio SDKs, because this serves a custom React SPA
# and a JSON API from one container.
#
# Build context is the repo root:
#   docker build -f space/Dockerfile -t remix-space .

FROM node:20-slim AS web

WORKDIR /build
COPY website/package.json website/package-lock.json ./
RUN npm ci

COPY website/ ./
# The rating workload lives outside public/ so it never ships to GitHub Pages;
# the Space is the only place it belongs, so it is copied in here.
RUN mkdir -p public/data && cp space-data/validation_tasks.json public/data/

# Served from the Space root, not /ReMIX/, and the API is same-origin.
# VITE_API_BASE is also what enables the /rate route at all.
ENV VITE_API_BASE=/api
RUN npx vite build --base=/

FROM python:3.11-slim

# HF Spaces run as uid 1000 and only $HOME and /data are writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DATA_DIR=/home/user/data/ratings \
    STATIC_DIR=/home/user/app/static \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user space/requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user space/app.py ./
COPY --from=web --chown=user /build/dist ./static

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
