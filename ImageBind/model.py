from imagebind.models.imagebind_model import ModalityType
from imagebind.models import imagebind_model
import torch

# loading the model
print("[INFO] Loading the model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
print("[INFO] Model loaded")