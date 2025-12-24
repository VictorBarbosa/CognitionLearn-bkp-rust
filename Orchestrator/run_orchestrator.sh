#!/bin/bash
# CognitionLearn Orchestrator - Universal Launcher (Linux/macOS)

# 1. Basic configuration
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
OS="$(uname)"

echo "🌎 Operating System: $OS"

# 2. Check/Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment (.venv) not found!"
    read -p "Do you want to create the environment and install PyTorch now? (y/n): " confirm
    if [[ $confirm == [sS] || $confirm == [yY] ]]; then
        echo "🚀 Creating virtual environment..."
        python3 -m venv .venv || { echo "❌ Failed to create .venv. Please install python3-venv."; exit 1; }
        
        if [[ "$OS" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
            echo "⚡ NVIDIA GPU detected! Installing PyTorch with CUDA support..."
            .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
        else
            echo "💻 Installing standard PyTorch..."
            .venv/bin/pip install torch
        fi
    else
        echo "❌ Error: Orchestrator requires PyTorch to load AI libraries."
        exit 1
    fi
fi

# 3. Verify CUDA support if on Linux
if [[ "$OS" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
    HAS_CUDA=$(.venv/bin/python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [[ "$HAS_CUDA" == "False" ]]; then
        echo "⚠️  Warning: NVIDIA GPU found but the current .venv only supports CPU."
        read -p "Do you want to reinstall PyTorch with CUDA support? (y/n): " recnfirm
        if [[ $recnfirm == [sS] || $recnfirm == [yY] ]]; then
            .venv/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
        fi
    fi
fi

# 4. Dynamically find the torch lib path
TORCH_LIB_PATH=$(.venv/bin/python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)

if [ -z "$TORCH_LIB_PATH" ]; then
    echo "❌ Error: Could not locate Torch libraries within .venv."
    exit 1
fi

# 5. Export Library Paths based on OS
if [ "$OS" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH="$TORCH_LIB_PATH:$DYLD_LIBRARY_PATH"
    echo "🍎 macOS environment ready."
else
    # Crucial for Linux to see CUDA libs inside torch
    export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
    # Also add standard CUDA paths just in case
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    echo "Pytorch path: $TORCH_LIB_PATH"
    echo "🐧 Linux environment ready."
fi

# 6. Run the Orchestrator
cargo run --release
