import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from train_cifar import initialize
from cifar_model import AttackDataset
from mnist_utils import load_yaml


def test(device, testset, model):
    with torch.no_grad():
        tot_corr = 0
        tot_num = 0
        for idx, (x, label) in enumerate(testset):
            # Push to device
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            tot_corr += torch.eq(pred, label).float().sum().item()
            tot_num += x.size(0)
            if idx % 100 == 0:
                print('Test: {}/{} ({:.0f}%)'.format(idx * len(x), len(testset.dataset), 100. * idx / len(testset)))
        acc = 100 * tot_corr / tot_num
        print(f'Accuracy: {tot_corr}/{tot_num} ({acc:.1f}%)')


def generate_adv(param, device, testset, attack):
    attack_list = []
    label_list = []
    with torch.no_grad():
        for idx, (x, label) in enumerate(testset):
            # Push to device
            x, label = x.to(device), label.to(device)

            # Generate attacks
            adv_x = attack.perturb(x, label)

            ## For testing purposes
            # assert not torch.isnan(adv_data).any()
            diff_tensor = adv_x.contiguous().view(adv_x.shape[0], -1) - x.contiguous().view(adv_x.shape[0], -1)
            min_norm = torch.max(torch.abs(diff_tensor), dim=1)[0].min()
            # print(f'Min Linf norm: {min_norm}')
            if param['attack'] != 'gn' and min_norm < 0.9 * param['budget']:
                print('PERTURBATION IS TOO SMALL!!!')

            attack_list.append(adv_x.cpu().numpy())
            label_list.append(label.cpu().numpy())

            if idx % 100 == 0:
                print('Test: {}/{} ({:.0f}%)'.format(idx * len(x), len(testset.dataset), 100. * idx / len(testset)))

    attack_array = np.concatenate(attack_list, axis=0)
    label_array = np.concatenate(label_list, axis=0)
    print('Saving')
    if param['attack'] == 'fgsm':
        np.save(f'data/cifar/attacks/{param["attack"]}_{param["budget"]}.npy', attack_array)
        np.save(f'data/cifar/attacks/{param["attack"]}_{param["budget"]}_label.npy', label_array)
    elif param['attack'] == 'pgd':
        np.save(f'data/cifar/attacks/{param["attack"]}_{param["budget"]}_{param["perturbation"]}.npy', attack_array)
        np.save(f'data/cifar/attacks/{param["attack"]}_{param["budget"]}_{param["perturbation"]}_label.npy', label_array)
    elif param['attack'] == 'deep_fool':
        np.save(f'data/cifar/attacks/{param["attack"]}.npy', attack_array)
        np.save(f'data/cifar/attacks/{param["attack"]}_label.npy', label_array)


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

    # Test
    if param['generate']:
        generate_adv(param, device, testset, attack)
    else:
        if param['clean']:
            test(device, testset, model)
        else:
            attackset = AttackDataset(param)
            attackset = DataLoader(attackset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)
            test(device, attackset, model)


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    param = load_yaml('test_cifar_conf')
    models = [
        'baseline',
        'iso',
        'jac',
        'fir',
        'distillation',
        'adv_train'
    ]
    budgets = [4/255, 8/255, 16/255]
    budgets_l2 = [1., 2., 3.]
    attacks = [
        'fgsm',
        'pgd',
        'deep_fool'
    ]

    print(f'Generate adversarial examples')
    param['generate'] = True
    for attack in attacks:
        param['attack'] = attack
        if attack != 'deep_fool':
            for idx in range(len(budgets)):
                param['budget'] = budgets[idx]
                if attack == 'pgd':
                    for perturb in ('l2', 'linf'):
                        param['perturbation'] = perturb
                        if perturb == 'l2':
                            param['budget'] = budgets_l2[idx]
                        else:
                            param['budget'] = budgets[idx]
                        print('-' * 101)
                        one_run(param)
                        break
                else:
                    print('-' * 101)
                    one_run(param)
        else:
            print('-' * 101)
            one_run(param)

    print('=' * 101)
    print(f'Test defenses')
    param['generate'] = False
    for name in models:
        param['name'] = 'cifar_icassp/' + name
        param['clean'] = True
        print('-' * 101)
        one_run(param)
        param['clean'] = False
        for attack in attacks:
            param['attack'] = attack
            if attack != 'deep_fool':
                for idx in range(len(budgets)):
                    param['budget'] = budgets[idx]
                    if attack == 'pgd':
                        for perturb in ('linf', 'l2'):
                            param['perturbation'] = perturb
                            if perturb == 'l2':
                                param['budget'] = budgets_l2[idx]
                            else:
                                param['budget'] = budgets[idx]
                            print('-' * 101)
                            one_run(param)
                    else:
                        print('-' * 101)
                        one_run(param)
            else:
                print('-' * 101)
                one_run(param)


if __name__ == '__main__':
    main()