# Use an Alpine-based Python image
FROM python:3.8-alpine

# Set working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
# Note: Comments have been moved to avoid syntax errors
# build-base is often required for compiling Python packages
# libffi-dev may be required for certain Python packages
# musl-dev is equivalent to libc-dev in Debian/Ubuntu
# openssl-dev may be required for certain Python packages
RUN apk update && \
    apk add --no-cache \
    git \
    curl \
    build-base \
    libffi-dev \
    musl-dev \
    openssl-dev

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Clone the specific git repo
RUN git clone https://github.com/EckoTan0804/flying-guide-dog

# Copy the requirements file into the container and install Python dependencies
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

# Additional installation commands...
# Make sure to fix any syntax issues similar to the above correction

