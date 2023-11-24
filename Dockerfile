FROM quay.io/jupyter/minimal-notebook:2023-11-19

RUN conda install -c conda-forge -c defaults -y \
    python=3.11.* \
    altair=5.1.2 \
    vegafusion \
    vegafusion-python-embed \
    vegafusion-jupyter \
    scipy \
    matplotlib \ 
    scikit-learn \
    requests \
    vl-convert-python=1.1.0

RUN pip install openmeteo-requests requests-cache retry-requests