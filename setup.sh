#!/bin/bash

echo ">>> 1. Configuring Git & Network..."
export HF_ENDPOINT=https://hf-mirror.com
git config --global http.version HTTP/1.1
git config --global pull.rebase false

echo ">>> 2. Pulling latest code..."
git pull origin main

echo ">>> 3. Setting up Python Environment..."
# Check if mamba_ssm is installed
if python -c "import mamba_ssm" &> /dev/null; then
    echo "Mamba-SSM is already installed."
else
    echo "Installing Mamba-SSM (this may take time)..."
    export MAMBA_FORCE_BUILD=TRUE
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE
    pip install causal-conv1d>=1.2.0 mamba-ssm>=1.2.0 --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/
fi

echo ">>> 4. Checking other requirements..."
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

echo ">>> Done! You can now run: ./run_persistent.sh"
