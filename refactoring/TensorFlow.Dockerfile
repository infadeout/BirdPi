# Use a base image with TensorFlow pre-installed
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files to the working directory
COPY . /app

# Install any additional dependencies if needed
# RUN apt-get update && apt-get install -y <package-name>

# Set any environment variables if needed
# ENV VARIABLE_NAME=value

# Specify the command to run when the container starts
CMD [ "python", "app.py" ]