FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/CATT-RL
COPY . .
RUN python -m pip install --upgrade pip && python -m pip install -e .

ENTRYPOINT ["catt-rl"]
CMD ["--help"]

