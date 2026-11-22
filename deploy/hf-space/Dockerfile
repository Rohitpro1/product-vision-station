FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
# Same pins as deploy/vercel-api (runtime lock + pvs at a fixed commit); app.py fetches the SHA-256-pinned model bundle.
RUN pip install --no-cache-dir -r https://raw.githubusercontent.com/Rohitpro1/product-vision-station/5edc21f/deploy/vercel-api/requirements.txt
ADD https://raw.githubusercontent.com/Rohitpro1/product-vision-station/5edc21f/deploy/vercel-api/app.py app.py
RUN chown -R user /home/user
USER user
ENV PVS_ROOT=/tmp/pvs PVS_CORS_ORIGINS=https://pack-station-psi.vercel.app
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
