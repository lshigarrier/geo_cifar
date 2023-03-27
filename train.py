# https://www.kaggle.com/code/michaelqq/tutorial-cifar10-resnet-pytorch

import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from utils.cifar_utils import initialize_cifar
from utils.mnist_utils import load_yaml, initialize_mnist, moving_average
from attack_defense.parseval import parseval_orthonormal_constraint
from utils.visualization import plot_curves


def train(param, device, trainset, testset, model, reg_model, teacher, attack, optimizer, epoch):
    loss_list = []
    entropy_val = []
    reg_val = []
    for idx, (x, label) in enumerate(trainset):
        optimizer.zero_grad()
        x, label = x.to(device), label.to(device)

        if param['defense'] == 'adv_train' or param['defense'] == 'isogn':
            # Update attacker
            attack.model = model
            attack.set_attacker()

            # Generate attacks
            x = attack.perturb(x, label)

        # Ensure grad is on
        x.requires_grad = True

        # Forward pass
        logits = model(x)

        # Cross entropy and regularization
        entropy = F.cross_entropy(logits, label)
        reg     = torch.tensor(0)

        # Train teacher model for distillation
        if param['defense'] == 'teacher':
            loss = F.cross_entropy(logits / param['dist_temp'], label)

        # Train distilled model
        elif param['defense'] == 'distillation':
            soft_labels = F.softmax(teacher(x) / param['dist_temp'], -1)
            loss        = torch.sum(-soft_labels * torch.log_softmax(logits / param['dist_temp'], -1), -1).mean()

        elif param['defense'] == 'isometry'\
                or param['defense'] == 'isorandom'\
                or param['defense'] == 'isonoback'\
                or param['defense'] == 'isogn'\
                or param['defense'] == 'eigenbound':
            reg  = reg_model(x, logits, device)
            loss = entropy + param['lambda']*reg

        elif param['defense'] == 'jacreg':
            reg  = reg_model(x, logits, device)
            loss = (1 - param['lambda'])*entropy + param['lambda']*reg

        elif param['defense'] == 'temperature':
            temp       = reg_model(x, logits, device).detach()  # or not detach()
            # new_logits = temp*logits
            new_logits = temp*F.softmax(logits, dim=1)
            loss       = F.cross_entropy(new_logits, label)

        elif param['defense'] == 'fir':
            # Compute regularization term and cross entropy loss
            c     = logits.shape[1]
            probs = F.softmax(logits, dim=1) * (1 - c * 1e-6) + 1e-6  # for numerical stability
            reg   = torch.sum(1/probs, dim=1).mean()
            loss  = entropy + param['lambda']*reg

        elif param['defense'] == 'isoapprox' or param['defense'] == 'isolayer':
            # loss = entropy + param['lambda']*isometry_reg_approx(model, device, x.shape[1:])
            raise NotImplementedError

        elif param['defense'] is None:
            loss = entropy

        else:
            raise NotImplementedError

        # Store cross entropy and regularization values
        entropy_val.append(entropy.item())
        reg_val.append(param['lambda']*reg.item())

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        if param['defense'] == 'parseval' or param['defense'] == 'isolayer':
            model = parseval_orthonormal_constraint(model, logits, device, reg_model,
                                                    defense=param['defense'], beta=param['beta'])

        loss_list.append(loss.item())

        if idx % int(len(trainset)/4) == 0:
            print('Epoch {}: {}/{} ({:.0f}%), Loss: {:.4f}, Entropy: {:.4f}, Reg: {:.4f}'.format(
                epoch, idx * len(x), len(trainset.dataset), 100. * idx / len(trainset),
                np.mean(loss_list), np.mean(entropy_val), np.mean(reg_val)))

    model.eval()
    with torch.no_grad():
        tot_corr = 0
        tot_num = 0
        for x, label in testset:
            x, label  = x.to(device), label.to(device)
            logits    = model(x)
            pred      = logits.argmax(dim=1)
            tot_corr += torch.eq(pred, label).float().sum().item()
            tot_num  += x.size(0)
        acc = 100 * tot_corr / tot_num
        print('Epoch: {}, Loss: {:.6f}, Accuracy: {:.2f}%'.format(epoch, np.mean(loss_list), acc))
    return entropy_val, reg_val


def training(param, device, trainset, testset, model, reg_model, teacher, attack, optimizer):
    print(f'Start training')
    tac = time.time()
    defense = param['defense']
    param['defense'] = None
    entropy_list = []
    reg_list = []
    for epoch in range(1, param['epochs'] + 1):
        if epoch == param['epoch_thr']:
            param['defense'] = defense
        print(f'Defense: {param["defense"]}')
        tic = time.time()
        entropy_val, reg_val = train(param, device, trainset, testset,
                                     model, reg_model, teacher, attack, optimizer, epoch)
        entropy_list += entropy_val
        reg_list += reg_val
        print(f'Epoch training time (s): {time.time() - tic}')
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, f'models/{param["name"]}/epoch_{param["epochs"]:02d}.pt')
    print(f'Training time (s): {time.time() - tac}')
    return entropy_list, reg_list


def one_run(param):
    # Deterministic
    torch.manual_seed(param['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu_number']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialization
    if param['dataset'] == 'cifar':
        trainset, testset, model, reg_model, teacher, attack, optimizer = initialize_cifar(param, device)
    elif param['dataset'] == 'mnist' or param['dataset'] == 'fashion':
        trainset, lightset, testset, model, reg_model, teacher, attack, optimizer = initialize_mnist(param, device)
    else:
        raise NotImplementedError

    # Training
    entropy_list, reg_list = training(param, device, trainset, testset, model, reg_model, teacher, attack, optimizer)

    # Figures
    entropy_list = moving_average(entropy_list, 1000)
    reg_list = moving_average(reg_list, 1000)
    fig = plot_curves([entropy_list, reg_list],
                      ["Cross entropy", "Regularization"],
                      "Cross-entropy and regularization during training",
                      "Batch",
                      "Value",
                      ylim=(0, 10))
    if param['save_plot']:
        fig.savefig(f'{param["dataset"]}_{param["archi"]}_{param["defense"]}_{param["norm"]}_{param["lambda"]}_{param["seed"]}.png')
    plt.close(fig)


def cifar_train_loop():
    prefix = 'cifar_conf/'
    conf_files = [
            'attack_base',
            'baseline',
            'fir',
            'iso',
            'jac',
            'teacher',
            'dist',
            'adv_train'
    ]
    for conf_file in conf_files:
        print('=' * 101)
        param = load_yaml(prefix + conf_file + '_conf')
        if conf_file == 'dist':
            os.system(
                f'cp ./models/cifar_icassp/distillation/epoch_03.pt ./models/cifar_icassp/distillation/teacher.pt')
        one_run(param)


def cifar_trace_plot_loop():
    from test import one_test_run
    param = load_yaml('train_conf')
    lambdas = np.linspace(5e-6, 6e-6, 11)

    for idx in range(len(lambdas)):
        print('=' * 101)
        param['eta'] = lambdas[idx]
        param['name'] = param['name'][:-1] + str(idx + 11)
        param['load'] = False
        one_run(param)
        print('-' * 101)
        print('Test')
        param['load'] = True
        one_test_run(param)


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    from test import one_test_run
    param         = load_yaml('baseline_cifar')
    name          = param['name']
    attack_type   = param['attack']
    attack_budget = param['budget']
    start_seed    = param['seed']
    budgets       = [4 / 255, 8 / 255, 16 / 255, 32 / 255]

    # one_run(param)
    # import sys
    # sys.exit(0)

    for seed in range(start_seed, start_seed+5):
        print('=' * 101)
        param['seed'] = seed
        param['name'] = name + '/seed_' + str(seed)
        param['attack'] = attack_type
        param['budget'] = attack_budget
        param['load'] = False
        one_run(param)
        print('-' * 101)
        print('Test')
        param['attack'] = 'fgsm' if param['dataset'] == 'cifar' else 'pgd'
        param['load'] = True
        for budget in budgets:
            param['budget'] = budget
            one_test_run(param)


if __name__ == '__main__':
    main()
