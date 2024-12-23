FROM python:3.8-slim

WORKDIR /app

# Copy only necessary files
COPY ./recording/analyze.py /app/

# Install minimal dependencies
RUN pip install --no-cache-dir \
    numpy \
    librosa \
    soundfile

# Environment variables
ENV RECS_DIR=/app/recordings
ENV PROCESSED_DIR=/app/processed
ENV LATITUDE=0
ENV LONGITUDE=0
ENV SENSITIVITY=1.0
ENV OVERLAP=0.0
ENV MIN_CONF=0.1

CMD ["python", "analyze.py"]