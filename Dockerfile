FROM python:3.11-slim

WORKDIR /app

# Force HTTPS for Debian package sources (some ISPs block/throttle plain HTTP
# to the Fastly CDN that deb.debian.org resolves to)
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list 2>/dev/null || true

# System deps for OpenCV + librosa
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# Install CPU-only PyTorch first (much smaller than the default CUDA build,
# ~180MB instead of 2-3GB, and we don't need GPU support inside the container)
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]