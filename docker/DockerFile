FROM nsml/default_ml:tf-gpu-1.11.0torch-0.4.1opencv-3.4.3

EXPOSE 8097

RUN apt-get update && \
    apt-get install -y git

RUN pip install visdom

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install opencv-python

RUN git clone https://github.com/YBIGTA/pytorch-hair-segmentation.git

WORKDIR pytorch-hair-segmentation

RUN pip install -r requirements.txt

RUN sh data/figaro.sh