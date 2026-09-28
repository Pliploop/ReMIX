# ReMIX human validation: the Streamlit app from src/jamendo_instruct/demo, as a Hugging Face Space.
# The same app runs on the cluster behind cloudflared (scripts/run_human_validation_cloudflared.slurm.sh);
# here the clips, the frozen sidecars and the ratings live in the private DATASET_REPO (see README).
FROM python:3.11-slim

# Spaces run as uid 1000; only $HOME is writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    REMIX_RATINGS_DIR=/home/user/data/ratings \
    REMIX_SIDECARS=/home/user/data/hub/sidecars/music4all.sidecar.json:/home/user/data/hub/sidecars/mtg_jamendo.sidecar.json

WORKDIR /home/user/app
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .

EXPOSE 7860
# ADMIN_PASSWORD is a Space secret; without it the admin pages get a random password nobody knows.
CMD python -m jamendo_instruct.demo.human_validation_app --host 0.0.0.0 --port 7860 \
    --admin-password "${ADMIN_PASSWORD:-$(python -c 'import secrets; print(secrets.token_hex(16))')}"
