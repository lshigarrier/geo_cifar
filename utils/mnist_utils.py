import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from utils.mnist_model import Lenet
from attack_defense.parseval import JacSoftmax, JacCoordChange
from attack_defense.regularizations import IsometryReg, IsometryRegRandom, IsometryRegNoBackprop
from attack_defense.regularizations import JacobianReg, EigenBound, RandomBound, AdaptiveTemp, RandomAdaptiveTemp
from attack_defense.attacks import TorchAttackGaussianNoise, TorchAttackFGSM
from attack_defense.attacks import TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2


def moving_average(array, window_size):
    i = 0
    moving_averages = []
    array = np.array(array)
    while i < len(array) - window_size + 1:
        geo_mean = np.exp(np.log(array[i: i + window_size]).mean())
        moving_averages.append(geo_mean)
        i += 1
    return moving_averages


def load_yaml(file_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    if args.config:
        yaml_file = f'config/{args.config}.yml'
    elif file_name:
        yaml_file = f'config/{file_name}.yml'
    else:
        raise RuntimeError
    with open(yaml_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    return param


def initialize_mnist(param, device):
    ## Create model and load dataset from torchvision
    # -------------------------------------------------------------- #
    if param['archi'] == 'lenet':
        model = Lenet(param)
        transform = transforms.ToTensor()
    elif param['archi'] == 'resnet':
        model = resnet18(weights='DEFAULT')
        model.fc = nn.Linear(512, 10)
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3,1,1))])
    else:
        raise NotImplementedError
    if param['dataset'] == 'mnist':
        trainset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
        testset  = datasets.MNIST('./data/mnist', train=False, transform=transform)
    elif param['dataset'] == 'fashion':
        trainset = datasets.FashionMNIST('./data/fashion', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST('./data/fashion', train=False, transform=transform)
    else:
        raise NotImplementedError

    # Small train set
    subset = torch.utils.data.Subset(trainset, range(1000))

    # Create data loaders
    train_loader       = DataLoader(trainset, batch_size=param['batch_size'],
                                    shuffle=False, pin_memory=True, num_workers=1)
    light_train_loader = DataLoader(subset, batch_size=param['batch_size'],
                                    shuffle=False, pin_memory=True, num_workers=1)
    test_loader        = DataLoader(testset, batch_size=param['batch_size'],
                                    shuffle=False, pin_memory=True, num_workers=1)

    ## Load model
    # -------------------------------------------------------------- #
    if param['load']:
        checkpoint = torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    model.to(device)

    ## Initialize regularization class
    # -------------------------------------------------------------- #
    reg_model = None
    if param['defense'] == 'isometry':
        reg_model = IsometryReg(param['epsilon'], norm=param['norm'])
    elif param['defense'] == 'isorandom' or param['defense'] == 'isogn':
        reg_model = IsometryRegRandom(param['epsilon'])
    elif param['defense'] == 'isonoback':
        reg_model = IsometryRegNoBackprop(param['epsilon'])
    elif param['defense'] == 'isolayer':
        reg_model = [JacSoftmax(), JacCoordChange()]
    elif param['defense'] == 'jacreg':
        reg_model = JacobianReg()
    elif param['defense'] == 'eigenbound':
        reg_model = EigenBound(param['epsilon'])
    elif param['defense'] == 'randombound':
        reg_model = RandomBound(param['epsilon'])
    elif param['defense'] == 'temperature':
        reg_model = AdaptiveTemp(param['epsilon'])
    elif param['defense'] == 'randomtemp':
        reg_model = RandomAdaptiveTemp(param['epsilon'])

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

    elif param['attack'] == 'deep_fool':
        attack = TorchAttackDeepFool(model=model,
                                     max_iters=param['max_iter'])

    elif param['attack'] == 'cw':
        attack = TorchAttackCWL2(model=model,
                                 max_iters=param['max_iter'])

    # Load teacher model
    if param['defense'] == 'distillation':
        # Initalize network class
        teacher_model = Lenet(param).to(device)
        checkpoint = torch.load(f'models/{param["name"]}/{param["teacher"]}', map_location='cpu')
        teacher_model.load_state_dict(checkpoint['state_dict'])

        # Make model deterministic and turn off gradient computations
        teacher_model.eval()

    else:
        teacher_model = None

    # Set optimizer
    if param['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    elif param['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    else:
        raise NotImplementedError

    for key in param:
        print(f'{key}: {param[key]}')

    return train_loader, light_train_loader, test_loader, model, reg_model, teacher_model, attack, optimizer
