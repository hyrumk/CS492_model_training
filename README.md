# # CS492_model_training

This code trains the model to use in android application for skin disease classification.
The application can be found on
https://github.com/junstar98/CS492

### Requirements

```pip install -U scikit-learn```

```pip install torchvision=0.11.1```

```pip install pytorch=0.10.0```

```pip install numpy=1.21.4```



### Step by Step Description 
execute the following files in a given order.

0. download kaggle dataset.(kaggle.com/shubhamgoel27/dermnet)
1. ```main.py```: creates trainloader of dermnet and goes through data augmentation. (optional, trainloader is already in data directory)
2. ```model_train3.py```: Trains ResNet18 using trainloader and testloader in data directory. The code currently saves the model as 'best_resnet18_1.pth' at the last line. Modify this according to your need.
3. ```pruning.py```: Iterative pruning using trained ResNet18 model from step 2. Choose the model you want to prune by adjusting the model name in line 238. (You don't need to change the name if you are using the model straight from step2)
4. After all these steps, the pruned model will be in ```./saved_model``` under the name of 'resnet18_pruned.pt'. 
5. Execute ```model_to_mobile.py``` to optimize .pt file to run on mobile environment. (MUST DO) Modify the name of the file name in line 15. 

After following all these steps, you will have ```pruned_mobile_resnet18.ptl``` in the main directory. Use this ptl file in android java file.


pruning code based on
https://leimao.github.io/blog/PyTorch-Pruning/


