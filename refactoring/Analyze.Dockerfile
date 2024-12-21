FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY . /app

RUN pip install numpy librosa

ENV RECS_DIR=/app/recordings
ENV PROCESSED_DIR=/app/processed
ENV LATITUDE=0
ENV LONGITUDE=0
ENV SENSITIVITY=1.0
ENV OVERLAP=0.0
ENV MIN_CONF=0.1

CMD ["python", "birdnet_analyzer.py"]