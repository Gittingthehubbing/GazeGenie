# app/Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN mkdir ./results

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libcairo2-dev pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--browser.gatherUsageStats"," false"]
