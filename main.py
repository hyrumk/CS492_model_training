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
from torch.utils.data import Subset


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()

        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(1024,23)

    def forward(self,x):
        return self.cnn(x)


class_list = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']
data_dir = 'C://Users//hyrumk1//Documents//CS492_data//archive'
def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       ])
    train_transforms2 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomHorizontalFlip(p=1),
                                       transforms.ToTensor(),
                                       ])
    train_transforms3 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomRotation(90),
                                       transforms.ToTensor(),
                                       ])
    train_transforms4 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomRotation(180),
                                       transforms.ToTensor(),
                                       ])
    train_transforms5 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomRotation(270),
                                       transforms.ToTensor(),
                                       ])
    train_transforms6 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomVerticalFlip(p=1),
                                       transforms.ToTensor(),
                                       ])
    train_transforms7 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ColorJitter(brightness=0.5,hue=0.3),
                                       transforms.ToTensor(),
                                       ])
    train_transforms8 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
                                       transforms.ToTensor(),
                                       ])

    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       ])
    test_transforms2 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomHorizontalFlip(p=1),
                                       transforms.ToTensor(),
                                       ])
    test_transforms3 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomRotation(90),
                                       transforms.ToTensor(),
                                       ])
    test_transforms4 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomRotation(180),
                                       transforms.ToTensor(),
                                       ])
    test_transforms5 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomRotation(270),
                                       transforms.ToTensor(),
                                       ])
    test_transforms6 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomVerticalFlip(p=1),
                                       transforms.ToTensor(),
                                       ])
    test_transforms7 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ColorJitter(brightness=0.5,hue=0.3),
                                       transforms.ToTensor(),
                                       ])
    test_transforms8 = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
                                       transforms.ToTensor(),
                                       ])
    train_transform_list = [train_transforms,train_transforms2,train_transforms3,train_transforms4,train_transforms5,train_transforms6,train_transforms7,train_transforms8]
    test_transform_list = [test_transforms,test_transforms2,test_transforms3,test_transforms4,test_transforms5,test_transforms6,test_transforms7,test_transforms8]
    
    train_data_list = []
    test_data_list = []
    for t in train_transform_list:
        train_data_list.append(datasets.ImageFolder(datadir+'//train',
                    transform=t))
    for t in test_transform_list:
        test_data_list.append(datasets.ImageFolder(datadir+'//test',
                    transform=t))
    
    train_data = train_data_list[0]
    test_data =  test_data_list[0]

    # 53, 212
    train_idx, test_idx = [],[]
    train_number = [0]*23
    test_number = [0]*23


    for i in range(len(train_data)):
        data_label = train_data[i][1]
        if train_number[data_label] >= 404:
            continue
        train_number[data_label] += 1
        train_idx.append(i)
    for i in range(len(test_data)):
        data_label = test_data[i][1]
        if test_number[data_label] >= 101:
            continue
        test_number[data_label] += 1
        test_idx.append(i)
    
    train_subset_list = [Subset(t,train_idx) for t in train_data_list]
    test_subset_list = [Subset(t,test_idx) for t in test_data_list]

    train_data = torch.utils.data.ConcatDataset(train_subset_list)
    test_data = torch.utils.data.ConcatDataset(test_subset_list)

    print(type(train_data))

    num_train = len(train_data)
    print(num_train)

    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   shuffle=True, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   shuffle=True, batch_size=64)

    torch.save(train_data, './data/traindata1.pth')
    torch.save(test_data, './data/testdata1.pth')
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)
print(len(trainloader))
print(type(trainloader))

torch.save(trainloader,'./data/trainloader1.pth')
torch.save(testloader,'./data/testloader1.pth')



