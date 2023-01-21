# https://www.kaggle.com/code/michaelqq/tutorial-cifar10-resnet-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms, datasets
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import sys
from cifar_model import ResNet50
from mnist_utils import load_yaml
from mnist_model import IsometryReg, JacobianReg
from attacks_utils import TorchAttackGaussianNoise, TorchAttackFGSM, TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2


def initialize(param, device):
    trainset = datasets.CIFAR10('./data/cifar',
                                train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(size=32, padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Resize((224, 224)),  # original: 32 x 32
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2470, 0.2435, 0.2616))
                                ]))
    testset = datasets.CIFAR10('./data/cifar', train=False,
                               transform=transforms.Compose([
                                   transforms.Resize((224, 224)),  # original: 32 x 32
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2470, 0.2435, 0.2616))
                               ]))
    trainset = DataLoader(trainset, batch_size=param['batch_size'], shuffle=True)
    testset = DataLoader(testset, batch_size=param['batch_size'], shuffle=True)

    if param['load']:
        checkpoint = torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu')
        model = densenet121()
        model.classifier = nn.Linear(1024, 10)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()

    else:
        # model = ResNet50(img_channels=3, num_classes=10).to(device)
        # model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model = densenet121()
        model.classifier = nn.Linear(1024, 10)
        # torch.save(model.state_dict(), 'data/cifar/densenet121_imagenet.pt')
        model.load_state_dict(torch.load('data/cifar/densenet121_imagenet.pt'))
        model.to(device)

    reg_model = None
    if param['defense'] == 'isometry':
        reg_model = IsometryReg(param['epsilon'])
    elif param['defense'] == 'jacobian':
        reg_model = JacobianReg(param['epsilon'])

    attack = None
    if param['attack'] == 'fgsm':
        attack = TorchAttackFGSM(model=model,
                                 eps=param['budget'])

    elif param['attack'] == 'gn':
        attack = TorchAttackGaussianNoise(model=model,
                                          std=param['budget'])

    elif param['attack'] == 'pgd':
        if param['perturbation'] == 'linf':
            attack = TorchAttackPGD(model=model,
                                    eps=param['budget'],
                                    alpha=param['alpha'],
                                    steps=param['max_iter'],
                                    random_start=param['random_start'])
        elif param['perturbation'] == 'l2':
            attack = TorchAttackPGDL2(model=model,
                                      eps=param['budget'],
                                      alpha=param['alpha'],
                                      steps=param['max_iter'],
                                      random_start=param['random_start'])
        else:
            print("Invalid perturbation_type in config file, please use 'linf' or 'l2'")
            exit()

    elif param['attack_type'] == 'deep_fool':
        attack = TorchAttackDeepFool(model=model,
                                     max_iters=param['max_iter'])

    elif param['attack_type'] == 'cw':
        attack = TorchAttackCWL2(model=model,
                                 max_iters=param['max_iter'])

    for key in param:
        print(f'{key}: {param[key]}')

    return trainset, testset, model, reg_model, attack


def train(param, device, trainset, testset, model, reg_model, optimizer, epoch, attack, teacher):
    loss_list = []
    for idx, (x, label) in enumerate(trainset):
        x, label = x.to(device), label.to(device)

        if param['defense'] == 'adv_train':
            # Update attacker
            attack.model = model
            attack.set_attacker()

            # Generate attacks
            x = attack.perturb(x, label)

        # Ensure grad is on
        x.requires_grad = True

        # Forward pass
        logits = model(x)  # [b, 10]

        # Train teacher model for distillation
        if param['defense'] == 'teacher':
            loss = F.cross_entropy(logits / param['dist_temp'], label)

        # Train distilled model
        elif param['defense'] == 'distillation':
            soft_labels = F.softmax(teacher(x) / param["dist_temp"], -1)
            # numerical stability
            # c = soft_labels.shape[1]
            # soft_labels = soft_labels * (1 - c * 1e-6) + 1e-6
            loss = torch.sum(-soft_labels * torch.log_softmax(logits / param['dist_temp'], -1), -1).mean()

        elif param['defense'] == 'isometry':
            reg, _ = reg_model(x, label, device)
            loss = (1 - param['eta'])*F.cross_entropy(logits, label) + param['eta']*reg

        elif param['defense'] == 'jacobian':
            _, norm = reg_model(x, label, device)
            loss = (1 - param['eta'])*F.cross_entropy(logits, label) + param['eta']*norm

        elif param['defense'] == 'fir':
            # Compute regularization term and cross entropy loss
            # c           = label.shape[1]
            c = 10
            probs       = F.softmax(logits, dim=1) * (1 - c * 1e-6) + 1e-6  # for numerical stability
            max_eig_reg = torch.sum(1/probs, dim=1).mean()

            # Loss is only cross entropy
            loss = F.cross_entropy(logits, label) + param['eta']*max_eig_reg

        else:
            loss = F.cross_entropy(logits, label)  # label: [b]

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss)

        if idx % 300 == 0:
            print('Epoch {}: {}/{} ({:.0f}%)'.format(epoch, idx * len(x), len(trainset.dataset), 100. * idx / len(trainset)))

    model.eval()
    with torch.no_grad():
        tot_corr = 0
        tot_num = 0
        for x, label in testset:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)

            tot_corr += torch.eq(pred, label).float().sum().item()
            tot_num += x.size(0)
        acc = 100 * tot_corr / tot_num
        print('Epoch: {}, Loss: {:.6f}, Accuracy: {:.2f}%'.format(epoch, np.mean(loss_list), acc))


def training(param, device, trainset, testset, model, reg_model, attack):
    # optimizer = optim.Adam(model.parameters(), lr=param['lr'])
    optimizer = optim.SGD(model.parameters(), lr=param['lr'])
    if param['defense'] == 'distillation':
        checkpoint = torch.load(f'models/{param["name"]}/{param["teacher"]}', map_location='cpu')
        teacher = densenet121()
        teacher.classifier = nn.Linear(1024, 10)
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.to(device)
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher.eval()
    else:
        teacher = None
    print(f'Start training')
    tac = time.time()
    for epoch in range(1, param['epoch'] + 1):
        train(param, device, trainset, testset, model, reg_model, optimizer, epoch, attack, teacher)
        # checkpoint = {'model': ResNet50(img_channels=3, num_classes=10),
        #              'state_dict': model.state_dict(),
        #              'optimizer': optimizer.state_dict()}
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, f'models/{param["name"]}/epoch_{epoch:02d}.pt')
    print(f'Training time (s): {time.time() - tac}')


def one_run(param):
    # Set random seed
    torch.manual_seed(param['seed'])

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu_number']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialization
    trainset, testset, model, reg_model, attack = initialize(param, device)

    # Training
    training(param, device, trainset, testset, model, reg_model, attack)


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    prefix = 'cifar_conf/'
    conf_files = [
            'resnet',
            'baseline',
            'iso',
            'jac',
            'fir',
            'teacher',
            'dist',
            'adv_train'
    ]
    for conf_file in conf_files:
        print('=' * 101)
        param = load_yaml(prefix + conf_file + '_conf')
        if conf_file == 'distillation':
            os.system(
                f'cp ./models/cifar_icassp/distillation/epoch_50.pt ./models/cifar_icassp/distillation/teacher.pt')
        one_run(param)


if __name__ == '__main__':
    main()