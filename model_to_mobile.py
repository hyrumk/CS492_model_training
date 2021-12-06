import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.resnet18(pretrained=True)

model.fc = nn.Sequential(nn.Linear(512, 256),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(256, 23),
                         nn.LogSoftmax(dim=1))
model.load_state_dict(torch.load('./saved_model/resnet18_dermnet.pt'))
model.eval()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("pruned_mobile_resnet18.ptl")
