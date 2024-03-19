# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Install PortAudio and libsndfile. Needed for sounddevice and soundfile packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY ./recording/recording.py /app

# Install any needed packages specified in requirements.txt
RUN pip install sounddevice numpy soundfile

# Run recording.py when the container launches
CMD ["python", "recording.py"]