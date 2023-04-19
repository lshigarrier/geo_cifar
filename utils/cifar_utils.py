import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import densenet121, resnet18
from torch.utils.data import DataLoader
from attack_defense.parseval import JacSoftmax, JacCoordChange
from attack_defense.regularizations import IsometryReg, IsometryRegRandom, IsometryRegNoBackprop
from attack_defense.regularizations import JacobianReg, EigenBound, RandomBound, AdaptiveTemp, RandomAdaptiveTemp
from attack_defense.attacks import TorchAttackGaussianNoise, TorchAttackFGSM
from attack_defense.attacks import TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2


def initialize_cifar(param, device):
    # Load datasets and preprocessing
    # After ToTensor(): transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616))
    trainset = datasets.CIFAR10('./data/cifar',
                                train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(size=32, padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Resize((224, 224)),  # original: 32 x 32
                                    transforms.ToTensor()
                                ]))
    testset = datasets.CIFAR10('./data/cifar', train=False,
                               transform=transforms.Compose([
                                   transforms.Resize((224, 224)),  # original: 32 x 32
                                   transforms.ToTensor()
                               ]))
    trainset = DataLoader(trainset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)
    testset = DataLoader(testset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)

    # Create model
    if param['archi'] == 'densenet':
        model = densenet121(weights='DEFAULT')
        model.classifier = nn.Linear(1024, 10)
    elif param['archi'] == 'resnet':
        model = resnet18(weights='DEFAULT')
        model.fc = nn.Linear(512, 10)
    else:
        raise NotImplementedError

    # Save model architecture for visualization on Netron
    # batch = next(iter(trainset))
    # torch.onnx.export(model, batch[0], f'{param["archi"]}.onnx', input_names=['Image'], output_names=['Logits'])

    # Load weights
    if param['load']:
        checkpoint = torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    model.to(device)

    # Initialize regularization
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

    # Initialize attack
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

    # Initialize distillation
    if param['defense'] == 'distillation':
        checkpoint = torch.load(f'models/{param["name"]}/{param["teacher"]}', map_location='cpu')
        if param['archi'] == 'densenet':
            teacher = densenet121()
            teacher.classifier = nn.Linear(1024, 10)
        elif param['archi'] == 'resnet':
            teacher = resnet18()
            teacher.fc = nn.Linear(512, 10)
        else:
            raise NotImplementedError
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.to(device)
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher.eval()
    else:
        teacher = None

    # Set optimizer
    if param['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    elif param['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    else:
        raise NotImplementedError

    # Print hyperparameters
    for key in param:
        print(f'{key}: {param[key]}')

    return trainset, testset, model, reg_model, teacher, attack, optimizer
