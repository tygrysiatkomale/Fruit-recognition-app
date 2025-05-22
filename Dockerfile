FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG REPO_URL=https://github.com/tygrysiatkomale/Fruit-recognition-app.git
ARG REPO_BRANCH=master

WORKDIR /app

RUN git clone --depth 1 --branch ${REPO_BRANCH} ${REPO_URL} .

RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt ; \
    else \
        echo "Brak requirements.txt – instaluję domyślne pakiety" && \
        pip install --no-cache-dir streamlit onnxruntime pillow numpy ; \
    fi

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port", "8501", "--server.address", "0.0.0.0"]
