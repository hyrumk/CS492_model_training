from torch.utils.mobile_optimizer import optimize_for_mobile
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(2048, 23)

    def forward(self, x):
        return self.cnn(x)


class_list = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
              'Atopic Dermatitis Photos', 'Bullous Disease Photos',
              'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions',
              'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos',
              'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases',
              'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease',
              'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases',
              'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors',
              'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives',
              'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']

traindata = torch.load('traindata.pth')
testdata = torch.load('testdata.pth')
print(type(traindata))
print(len(testdata))
print(len(traindata))
trainloader = torch.load('trainloader1.pth')
testloader = torch.load('testloader1.pth')


device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")
print("Device: ", device)
model = torchvision.models.resnet50(pretrained=True)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam
# optimizer = optimizer(model.parameters(), lr=1e-4)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 23),
                         nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
model.to(device)

epochs = 500
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

min_loss = 10000

for epoch in range(epochs):
    #SASSSD("EPOCH {}".format(epoch))
    start_time = time.time()
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_time = time.time()-start_time

    # test start
    start_time = time.time()
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    # test end
    print("train time: {}, test time:{}".format(train_time, time.time()-start_time))

    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))
    print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / print_every:.3f}.. "
          f"Test loss: {test_loss / len(testloader):.3f}.. "
          f"Test accuracy: {accuracy / len(testloader):.3f}")
    running_loss = 0
    model.train()
    if min_loss > test_loss/len(testloader):
        min_loss = test_loss/len(testloader)
        torch.save(model, 'bestmodel_15.pth')
    #torch.save(model, 'tempmodel.pth')