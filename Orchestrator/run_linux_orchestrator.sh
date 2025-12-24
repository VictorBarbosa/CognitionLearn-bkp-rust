#!/bin/bash
# CognitionLearn Orchestrator - Universal Launcher (Linux/macOS)

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
OS="$(uname)"

echo "🌎 Operating System: $OS"

# Activate or create venv early
if [ ! -d ".venv" ]; then
    echo "⚠️ Virtual environment (.venv) not found!"
    read -p "Do you want to create the environment and install PyTorch now? (y/n): " confirm
    if [[ $confirm == [yYsS] ]]; then
        echo "🚀 Creating virtual environment..."
        python3 -m venv .venv || { echo "❌ Failed to create .venv."; exit 1; }
        
        source .venv/bin/activate || { echo "❌ Failed to activate .venv."; exit 1; }
        
        if [[ "$OS" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
            echo "⚡ NVIDIA GPU detected! Installing PyTorch 2.4.1 with CUDA support..."
            pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
        else
            echo "💻 Installing standard PyTorch 2.4.1..."
            pip install torch==2.4.1
        fi
    else
        echo "❌ Error: Orchestrator requires PyTorch."
        exit 1
    fi
else
    source .venv/bin/activate || { echo "❌ Failed to activate .venv."; exit 1; }
fi

# Compute TORCH_LIB_PATH early (requires venv activated)
TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)

if [ -z "$TORCH_LIB_PATH" ] || [ ! -d "$TORCH_LIB_PATH" ]; then
    echo "❌ Error: Could not locate Torch libraries in .venv (torch not installed?)."
    exit 1
fi

# Set LD_LIBRARY_PATH early for ALL subsequent python/torch calls
if [ "$OS" == "Linux" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_PATH:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

# Now verify CUDA (with paths set)
if [[ "$OS" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
    HAS_CUDA=$(python -c "import torch; print('True' if torch.cuda.is_available() else 'False')" 2>/dev/null)
    if [[ "$HAS_CUDA" != "True" ]]; then
        echo "⚠️ Warning: NVIDIA GPU detected, but torch.cuda.is_available() is $HAS_CUDA."
        echo "   This usually means the CUDA runtime libs aren't loading properly."
        read -p "Do you want to reinstall PyTorch 2.4.1 with CUDA support? (y/n): " recnfirm
        if [[ $recnfirm == [yYsS] ]]; then
            pip install --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
            # Re-compute path after reinstall
            TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
            export LD_LIBRARY_PATH="$TORCH_LIB_PATH:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        fi
    else
        echo "✅ CUDA support confirmed!"
    fi
fi

# Final paths (already set, but echo for debug)
if [ "$OS" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH="$TORCH_LIB_PATH:$DYLD_LIBRARY_PATH"
    echo "🍎 macOS environment ready."
else
    echo "Pytorch lib path: $TORCH_LIB_PATH"
    echo "🐧 Linux environment ready (LD_LIBRARY_PATH includes torch/lib)."
fi

# Run the Orchestrator
cargo run --release