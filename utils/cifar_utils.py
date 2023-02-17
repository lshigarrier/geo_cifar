import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import densenet121  # DenseNet121_Weights
from torch.utils.data import DataLoader
# from cifar_model import ResNet50
from attack_defense.parseval import JacSoftmax, JacCoordChange
from attack_defense.regularizations import IsometryReg, IsometryRegRandom, IsometryRegNoBackprop
from attack_defense.attacks import TorchAttackGaussianNoise, TorchAttackFGSM, TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2


def initialize_cifar(param, device):
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
    trainset = DataLoader(trainset, batch_size=param['batch_size'], shuffle=False, pin_memory=True, num_workers=1)
    testset = DataLoader(testset, batch_size=param['batch_size'], shuffle=False, pin_memory=True, num_workers=1)

    # model = ResNet50(img_channels=3, num_classes=10).to(device)
    # model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model = densenet121()
    if param['load']:
        checkpoint = torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu')
        model.classifier = nn.Linear(1024, 10)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()

    else:
        # torch.save(model.state_dict(), 'models/densenet121_imagenet.pt')
        model.load_state_dict(torch.load('models/densenet121_imagenet.pt'))
        model.classifier = nn.Linear(1024, 10)
        model.to(device)
    # torch.onnx.export(model, batch[0], 'densenet.onnx', input_names=['Image'], output_names=['Logits'])

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

    # Set optimizer
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    # optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])

    for key in param:
        print(f'{key}: {param[key]}')

    return trainset, testset, model, reg_model, teacher, attack, optimizer