# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ../scripts/custom_recording.sh /app/scripts

# Install any needed packages specified in requirements.txt
RUN pip install sounddevice

# Make the shell script executable
RUN chmod +x scripts/custom_recording.sh

# Run custom_recording.sh when the container launches
CMD ["scripts/custom_recording.sh"]