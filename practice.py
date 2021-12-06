from torch.utils.mobile_optimizer import optimize_for_mobile
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset
from PIL import Image
from utils import evaluate_model, create_classification_report

train_loader, test_loader = torch.load('trainloader.pth'), torch.load('testloader.pth')
cuda_device = torch.device("cuda:0")

img = Image.open("paronychia.jpg")

train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   ])
ex = train_transforms(img=img)
ex = ex.unsqueeze(0)
print(ex.shape)

# for i,elem in enumerate(ex[0][1]):
#     print(i,": ",elem[0])
##########################################
# pruned_resnet18 = torchvision.models.mobilenet_v2(pretrained=True)
#
# pruned_resnet18.fc = nn.Sequential(nn.Linear(512, 256),
#                          nn.ReLU(),
#                          nn.Dropout(0.2),
#                          nn.Linear(256, 23),
#                          nn.LogSoftmax(dim=1))
# pruned_resnet18.load_state_dict(torch.load('best_mobilenet(2).pth'))
# pruned_resnet18.eval()
#
# start_time = time.time()
# _, eval_accuracy = evaluate_model(model=pruned_resnet18,
#                                   test_loader=test_loader,
#                                   device=cuda_device,
#                                   criterion=None)
# elapsed_time = time.time()-start_time
#
#
# print("MobileNet")
# print("Time Spent Testing: ",elapsed_time)
# print("Test Accuracy: {:.3f}".format(eval_accuracy))

##########################################
pruned_resnet18 = torchvision.models.resnet18(pretrained=True)

pruned_resnet18.fc = nn.Sequential(nn.Linear(512, 256),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(256, 23),
                         nn.LogSoftmax(dim=1))
pruned_resnet18.load_state_dict(torch.load('./saved_model/resnet18_dermnet.pt'))
pruned_resnet18.eval()

start_time = time.time()
_, eval_accuracy = evaluate_model(model=pruned_resnet18,
                                  test_loader=test_loader,
                                  device=cuda_device,
                                  criterion=None)
elapsed_time = time.time()-start_time

print("Pruned ResNet18")
print("Time Spent Testing: ",elapsed_time)
print("Test Accuracy: {:.3f}".format(eval_accuracy))

##########################################
pruned_resnet18 = torchvision.models.resnet18(pretrained=True)

pruned_resnet18.fc = nn.Sequential(nn.Linear(512, 256),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(256, 23),
                         nn.LogSoftmax(dim=1))
pruned_resnet18.load_state_dict(torch.load('best_resnet18_1.pth'))
pruned_resnet18.eval()

start_time = time.time()
_, eval_accuracy = evaluate_model(model=pruned_resnet18,
                                  test_loader=test_loader,
                                  device=cuda_device,
                                  criterion=None)
elapsed_time = time.time()-start_time

print("ResNet18")
print("Time Spent Testing: ",elapsed_time)
print("Test Accuracy: {:.3f}".format(eval_accuracy))

##########################################
import gc
gc.collect()
torch.cuda.empty_cache()

pruned_resnet18 = torchvision.models.resnet50(pretrained=True)

pruned_resnet18.fc = nn.Sequential(nn.Linear(2048, 256),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(256, 23),
                         nn.LogSoftmax(dim=1))
pruned_resnet18.load_state_dict(torch.load('best_resnet50.pth'))
pruned_resnet18.eval()

start_time = time.time()
_, eval_accuracy = evaluate_model(model=pruned_resnet18,
                                  test_loader=test_loader,
                                  device=cuda_device,
                                  criterion=None)
elapsed_time = time.time()-start_time

print("ResNet50")
print("Time Spent Testing: ",elapsed_time)
print("Test Accuracy: {:.3f}".format(eval_accuracy))





# model.fc = nn.Sequential(nn.Linear(2048, 512),
#                          nn.ReLU(),
#                          nn.Dropout(0.2),
#                          nn.Linear(512, 23),
#                          nn.LogSoftmax(dim=1))

# res = model(ex)
# print(res)
# pred = torch.argmax(res)
# print(pred)
# class_list = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']
# print(class_list[10])

