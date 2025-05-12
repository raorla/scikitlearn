FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    fontconfig \ 
    libfreetype6 \ 
    fonts-dejavu-core \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Le reste de votre Dockerfile pour micromamba, python, et les libs pip
# Assurez-vous que cette section RUN est bien avant l'installation de micromamba ou python si possible,
# ou au moins avant l'installation de matplotlib si vous voulez être sûr.
# L'ordre actuel (le faire en premier après FROM) est bien.

RUN curl -L "https://micro.mamba.pm/api/micromamba/linux-64/2.0.5" \
    | tar -xj -C "/" "bin/micromamba"

ENV PATH=/root/.local/share/mamba/bin:$PATH
RUN micromamba install -q -y python=3.9.0

RUN pip3 install matplotlib scikit-learn

COPY ./src /app
ENTRYPOINT ["python3", "/app/app.py"]