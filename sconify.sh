#!/bin/bash

# Declare the app entrypoint
ENTRYPOINT="/root/.local/share/mamba/bin/python3 /app/app.py"

# Declare image related variables
IMG_NAME=tee-scikitlearn
IMG_FROM=raorla/scikitlearn:1.0.0
IMG_TO=raorla/${IMG_NAME}:1.0.0-debug

# Run the sconifier to build the TEE image based on the non-TEE image
docker run -it \
            -v /var/run/docker.sock:/var/run/docker.sock \
            registry.scontain.com/scone-production/iexec-sconify-image:5.9.0-v15 \
            sconify_iexec \
            --name=${IMG_NAME} \
            --from=${IMG_FROM} \
            --to=${IMG_TO} \
            --base=debian:bullseye-slim \
            --binary-fs \
            --fs-dir=/app \
            --host-path=/etc/hosts \
            --host-path=/etc/resolv.conf \
            --host-path=/tmp \
            --host-path=/etc/passwd \
            --host-path=/etc/group \
            --binary=/root/.local/share/mamba/bin/python3  \
            --heap=1G \
            --dlopen=1 \
            --no-color \
            --verbose \
            --env HOME=/root \
            --env TMPDIR=/tmp \
            --command="${ENTRYPOINT}" \
            && echo -e "\n------------------\n" \
            && echo "successfully built TEE docker image => ${IMG_TO}" \
            && echo "application mrenclave.fingerprint is $(docker run --rm -e SCONE_HASH=1 ${IMG_TO})"
