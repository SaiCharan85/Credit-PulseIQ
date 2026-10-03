# CreditPulse IQ — serving image.
#
# Deliberately slim. torch and transformers are declared optional in
# pyproject because they exist only for the self-hosted-model path, and
# installing them here would add roughly two gigabytes to an image that calls
# a hosted endpoint instead.
FROM python:3.11-slim

# libgomp is LightGBM's OpenMP runtime. Without it the ranker import fails at
# start-up rather than at first use, which reads as an application crash.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-resolve the whole tree.
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY . .

# The host supplies PORT; serve.py binds 0.0.0.0 whenever it is set.
ENV PORT=8000
EXPOSE 8000

# Fails the deploy loudly if the app cannot answer, instead of serving errors.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/api/health',timeout=5)"

CMD ["python", "serve.py"]
