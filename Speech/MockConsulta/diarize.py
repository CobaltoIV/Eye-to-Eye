import speechbrain as sb
import torch

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
