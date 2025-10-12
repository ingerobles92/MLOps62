FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Paquetes mínimos del sistema + SSH
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Evitar warning de Git por /work montado desde host
RUN git config --system --add safe.directory /work

# Directorio de trabajo
WORKDIR /work

# Instala dependencias de Python (sin Torch/CUDA)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt jupyterlab "dvc[s3]" boto3 s3fs && \
    rm -rf /root/.cache/pip
    
