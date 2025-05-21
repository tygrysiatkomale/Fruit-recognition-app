# ---------- Dockerfile ----------
FROM python:3.10-slim

# 1. Niezbędne narzędzia
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# 2. Argumenty budowania
ARG REPO_URL=https://github.com/tygrysiatkomale/Fruit-recognition-app.git
ARG REPO_BRANCH=master

# 3. Katalog roboczy
WORKDIR /app

# 4. Klonujemy repo (depth=1 => szybciej, mniejszy obraz)
RUN git clone --depth 1 --branch ${REPO_BRANCH} ${REPO_URL} .

# 5. Instalujemy zależności (jeśli plik istnieje)
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt ; \
    else \
        echo "Brak requirements.txt – instaluję domyślne pakiety" && \
        pip install --no-cache-dir streamlit onnxruntime pillow numpy ; \
    fi

# 6. Domyślny port Streamlit
EXPOSE 8501

# 7. Domyślne polecenie
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port", "8501", "--server.address", "0.0.0.0"]
