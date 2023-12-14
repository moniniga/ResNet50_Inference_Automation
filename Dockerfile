# Base Image:
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Update and Upgrade:
RUN apt-get update
RUN apt-get upgrade -y

# Install Python3.8:
RUN apt-get install python3 -y

# Install dependencies:
RUN apt-get install python3-pip -y
RUN apt-get install -y unzip
RUN pip install tqdm

# Install Pytorch and Torchvision:
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
CMD ["tail", "-f", "/dev/null"]
