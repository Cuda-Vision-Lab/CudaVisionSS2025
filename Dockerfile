# 1. Use an NVIDIA CUDA base image with Python 3
FROM python:3.11-slim-bullseye

# 2. Set environment variables to prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. Install system dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /workspace

# 5. Copy your requirements file first (for better Docker caching)
COPY requirements.txt .

# 6. Install Python dependencies
# Note: We specify the index-url for torch to ensure CUDA-enabled versions
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your repository
COPY . .

# 8. Expose the port for Jupyter Notebooks
EXPOSE 8888

# 9. Default command to start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]