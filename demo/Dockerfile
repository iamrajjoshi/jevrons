# Jevrons demo, live mode. Build from the repo root (the data/ and runs/ files are gitignored or large, so
# deploy from a local checkout that has them):
#   fly launch --no-deploy --copy-config --config demo/fly.toml --dockerfile demo/Dockerfile
#   fly secrets set VERCEL_API_KEY=... --config demo/fly.toml
#   fly deploy --config demo/fly.toml --dockerfile demo/Dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "numpy>=2.2,<3"
COPY src/jevrons ./src/jevrons
COPY data/t10k-images-idx3-ubyte.gz data/t10k-labels-idx1-ubyte.gz ./data/
COPY runs/stage4/weights.npz ./runs/stage4/
COPY runs/stage5/weights-*.npz ./runs/stage5/
COPY runs/stage6/weights-seed0-*.npz ./runs/stage6/
COPY runs/stage7/weights-*.npz ./runs/stage7/
COPY runs/stage7/result-jev.json ./runs/stage7/
COPY runs/margin/summary.json ./runs/margin/
COPY demo ./demo
ENV PYTHONPATH=/app/src JEVRONS_BACKENDS=vercel PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "demo/server.py", "--live", "--host", "0.0.0.0", "--port", "8080", \
     "--daily-usd", "2", "--per-ip-hour", "20", "--max-live", "4"]
