# 1. Python 베이스 이미지 선택
FROM python:3.10-slim

# 2. 컨테이너 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (이미지 처리, 모델 다운로드 등에 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 5. 애플리케이션 소스 복사
COPY ./app /app

# 6. Gunicorn + Uvicorn 설정 (멀티 워커)
ENV WORKERS=4
ENV PORT=8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "app.main:app"]