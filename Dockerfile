FROM python:3.12-slim
LABEL authors="Dang"

WORKDIR /src

# Cài libomp + các công cụ biên dịch cần thiết cho wordcloud
RUN apt-get update && apt-get install -y \
    gcc \
    libomp-dev \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY W6/models/XGB_attack_all.pkl /src/XGB_attack_all.pkl
COPY server.py /src/server.py
COPY requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]
