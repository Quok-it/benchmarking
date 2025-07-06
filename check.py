import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")

# pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# need to install a pytorch version that supports cuda 
