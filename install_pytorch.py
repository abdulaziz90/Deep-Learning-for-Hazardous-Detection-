import subprocess
import os

def install_pytorch(cuda_version):
    base_command = "pip3.10 install"
    if cuda_version is not None:
        subprocess.run(f"{base_command} torch torchvision torchaudio -f https://download.pytorch.org/whl/cu{cuda_version}/torch_stable.html", shell=True)
    else:
        subprocess.run(f"{base_command} torch torchvision torchaudio", shell=True)

if __name__ == "__main__":
    cuda_version = os.environ.get('CUDA_VERSION')
    install_pytorch(cuda_version)
