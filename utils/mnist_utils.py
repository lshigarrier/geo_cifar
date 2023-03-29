import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from utils.mnist_model import Lenet
from attack_defense.parseval import JacSoftmax, JacCoordChange
from attack_defense.regularizations import IsometryReg, IsometryRegRandom, IsometryRegNoBackprop
from attack_defense.regularizations import JacobianReg, EigenBound, RandomBound, AdaptiveTemp
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


def fisher_distance(probs1, probs2):
    """
    :param probs1: torch.Tensor (batch size x nb of classes)
    :param probs2: torch.Tensor (batch size x nb of classes)
    :return: torch.Tensor (batch size)
    """
    dist = (torch.sqrt(probs1)*torch.sqrt(probs2)).sum(-1)
    return 2 * torch.acos(dist)


def softmax_stable(logits, num_stab):
    c = logits.shape[-1]
    return F.softmax(logits, dim=1)*(1 - c*num_stab) + num_stab


def coord_change(probs):
    m = probs.shape[-1]
    new_coord = torch.sqrt(probs)
    return 2*new_coord[:, :m] / (1 - new_coord[:, m].unsqueeze(1).repeat(1, m))


def inv_change(tau):
    """
    :param tau: torch.Tensor (batch size x m)
    :return: torch.Tensor (batch size x (m+1))
    """
    m = tau.shape[-1]
    theta = torch.zeros(tau.shape[0], m+1)
    tau_2 = torch.linalg.norm(tau/2, dim=1)**2
    theta[:, m] = ((tau_2 - 1)/(tau_2 + 1))**2
    theta[:, :m] = (tau/(1 + tau_2.unsqueeze(-1)))**2
    return theta


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
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    # optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])

    for key in param:
        print(f'{key}: {param[key]}')

    return train_loader, light_train_loader, test_loader, model, reg_model, teacher_model, attack, optimizer
