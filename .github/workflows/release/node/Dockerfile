FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git
# DO NOT build the Rust core through the pyproject.toml and install the deps
RUN python3 -m pip install pydantic pydantic_config requests web3==7.12.0 rsa python-gnupg aiohttp uvicorn fastapi
# Copy README and LICENSE files
COPY README.md /app/
COPY LICENSE /app/
# Copy the Python SDK files
COPY sdks/python /app/
# Show the directory structure
RUN ls -la
# Set environment variables
ENV PORT=8000
ENV LANG=en_US.utf8
ENV PYTHONPATH=/
# Expose the server port
EXPOSE 8000
# Run the node server
CMD ["python3", "-m", "alith.lazai.node.validator"]
