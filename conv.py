
from safetensors.torch import save_file, load_file
import sys
import torch

# Now loading
loaded = load_file(sys.argv[1])

conv_tensors = {}
for name,tensor in loaded.items():
    print(name)
    conv_tensor = tensor.to(torch.float32)
    conv_tensors[name] = conv_tensor

save_file(conv_tensors, sys.argv[1])

