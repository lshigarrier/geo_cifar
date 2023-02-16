import torch
import torch.optim as optim
import yaml
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist.mnist_model import Lenet
from attack_defense.parseval import JacSoftmax, JacCoordChange
from attack_defense.regularizations import IsometryReg, IsometryRegRandom, IsometryRegNoBackprop
from attack_defense.attacks_utils import TorchAttackGaussianNoise, TorchAttackFGSM, TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2


def moving_average(array, window_size):
    i = 0
    moving_averages = []
    while i < len(array) - window_size + 1:
        this_window = array[i: i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def load_yaml(file_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_cifar')
    args = parser.parse_args()
    if file_name:
        yaml_file = f'config/{file_name}.yml'
    else:
        yaml_file = f'config/{args.yaml}_conf.yml'
    with open(yaml_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    return param


def initialize_mnist(param, device):
    ## Load dataset from torchvision
    # -------------------------------------------------------------- #
    # Train set
    trainset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.ToTensor())

    # Small train set
    subset = torch.utils.data.Subset(trainset, range(1000))

    # Test set
    testset = datasets.MNIST('./data/mnist', train=False, transform=transforms.ToTensor())

    # Create data loaders
    train_loader       = DataLoader(trainset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)
    light_train_loader = DataLoader(subset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)
    test_loader        = DataLoader(testset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)

    ## Load model
    # -------------------------------------------------------------- #
    model = Lenet(param).to(device)
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
        reg_model = IsometryReg(param['epsilon'])
    elif param['defense'] == 'isorandom':
        reg_model = IsometryRegRandom(param['epsilon'])
    elif param['defense'] == 'isonoback':
        reg_model = IsometryRegNoBackprop(param['epsilon'])
    elif param['defense'] == 'isolayer':
        reg_model = [JacSoftmax(), JacCoordChange()]

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
        teacher_model.load_state_dict(torch.load(f'models/{param["name"]}/{param["teacher"]}', map_location='cpu'))

        # Make model deterministic and turn off gradient computations
        teacher_model.eval()

    else:
        teacher_model = None

    # Set optimizer
    # optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])

    for key in param:
        print(f'{key}: {param[key]}')

    return train_loader, light_train_loader, test_loader, model, reg_model, teacher_model, attack, optimizer
