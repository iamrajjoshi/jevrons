# Jevrons demo, live mode. Build from the repo root (the data/ and runs/ files are gitignored or large, so
# deploy from a local checkout that has them):
#   fly apps create jevrons-demo
#   fly secrets set VERCEL_API_KEY="$(pbpaste)" -a jevrons-demo
#   fly deploy . --config demo/fly.toml --dockerfile demo/Dockerfile --ha=false
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "numpy>=2.2,<3"
COPY src/jevrons ./src/jevrons
COPY data/t10k-images-idx3-ubyte.gz data/t10k-labels-idx1-ubyte.gz ./data/
COPY runs/stage8a/weights-*.npz ./runs/stage8a/
COPY runs/stage7/weights-*.npz ./runs/stage7/
COPY runs/margin/summary.json ./runs/margin/
COPY demo ./demo
ENV PYTHONPATH=/app/src JEVRONS_BACKENDS=vercel PYTHONUNBUFFERED=1
EXPOSE 8080
# No --journal: the hosted demo keeps nothing on disk but the spend counter. --max-live 16 bounds the threads a
# burst of live draws can start (about 74 each) on a 1 GB machine; --draws-per-min 30 keeps all visitors together at about 2,200 Jev calls a minute.
CMD ["python", "demo/server.py", "--live", "--host", "0.0.0.0", "--port", "8080", \
     "--daily-usd", "9", "--max-live", "16", "--draws-per-min", "30"]
