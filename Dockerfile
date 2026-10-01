# syntax=docker/dockerfile:1
ARG BASE_IMAGE=cr.us-central1.nebius.cloud/u00cw5vmkjqsegxtgd/base-image:v1
FROM ${BASE_IMAGE}

WORKDIR /app

# Inherit dependency layers rather than copying the virtualenv into a new image.
# Rebuild the base image when pyproject.toml or uv.lock changes.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
    && uv pip check --python /opt/venv/bin/python

# Kubeflow supplies the launcher command and arguments.
ENTRYPOINT []
CMD []