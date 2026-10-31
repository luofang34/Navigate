FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir 'setuptools==80.9.0' 'rknn-toolkit2==2.3.2' 'opencv-python==4.11.0.86' 'numpy==1.26.4'
WORKDIR /work
