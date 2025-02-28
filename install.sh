# Installation script for the fednnUNet

# Download the nnUNet repository added as a submodule
git submodule update --init

# Install nnUNet in editable mode (important for the next steps)
cd nnUNet
pip install -e .

# Install the required packages for the federated learning extension
cd ..
pip install -e .

echo "Installation complete. You can now use fednnUNet."
