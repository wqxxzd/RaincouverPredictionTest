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
    vl-convert-python \
    vega_datasets \
    graphviz \
    python-graphviz \
    eli5 \
    shap \
    jinja2 \
    selenium<4.3.0 \
    lightgbm

RUN pip install openmeteo-requests requests-cache retry-requests pytest joblib==1.3.2 mglearn psutil>=5.7.2
