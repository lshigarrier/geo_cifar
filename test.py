import torch
import os
from torch.utils.data import DataLoader
from utils.cifar_utils import initialize_cifar
from utils.cifar_model import AttackDataset
from utils.mnist_utils import load_yaml, initialize_mnist
from autoattack import AutoAttack


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
            if idx % int(len(testset)/4) == 0:
                print('Test: {}/{} ({:.0f}%)'.format(idx * len(x), len(testset.dataset), 100. * idx / len(testset)))
        acc = 100 * tot_corr / tot_num
        print(f'Accuracy: {tot_corr}/{tot_num} ({acc:.2f}%)')


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

            attack_list.append(adv_x.detach().cpu())
            label_list.append(label.detach().cpu())

            if idx % int(len(testset)/4) == 0:
                print('Test: {}/{} ({:.0f}%)'.format(idx * len(x), len(testset.dataset), 100. * idx / len(testset)))

    attack_tensor = torch.cat(attack_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)
    print('Saving')
    if param['attack'] == 'fgsm':
        torch.save(attack_tensor, f'data/{param["dataset"]}/attacks/{param["attack"]}_{param["budget"]}.pt')
        torch.save(label_tensor, f'data/{param["dataset"]}/attacks/{param["attack"]}_{param["budget"]}_label.pt')
    elif param['attack'] == 'pgd':
        torch.save(attack_tensor,
                   f'data/{param["dataset"]}/attacks/{param["attack"]}_{param["budget"]}_{param["perturbation"]}.pt')
        torch.save(label_tensor,
                f'data/{param["dataset"]}/attacks/{param["attack"]}_{param["budget"]}_{param["perturbation"]}_label.pt')
    elif param['attack'] == 'deep_fool':
        torch.save(attack_tensor, f'data/{param["dataset"]}/attacks/{param["attack"]}.pt')
        torch.save(label_tensor, f'data/{param["dataset"]}/attacks/{param["attack"]}_label.pt')


def one_test_run(param):
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

    # Test
    if param['generate']:
        generate_adv(param, device, testset, attack)
    else:
        if param['clean']:
            test(device, testset, model)
        else:
            attackset = AttackDataset(param)
            attackset = DataLoader(attackset, batch_size=param['batch_size'],
                                   shuffle=False, pin_memory=True, num_workers=1)
            test(device, attackset, model)


def one_auto_attack(param):
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

    # Load data and model
    if param['dataset'] == 'cifar':
        trainset, testset, model, reg_model, teacher, attack, optimizer = initialize_cifar(param, device)
    elif param['dataset'] == 'mnist' or param['dataset'] == 'fashion':
        trainset, lightset, testset, model, reg_model, teacher, attack, optimizer = initialize_mnist(param, device)
    else:
        raise NotImplementedError

    # Create adversary
    # log_path=param['save_dir']+'log_'+param['model'][:-2]+'txt',
    adversary = AutoAttack(model,
                           norm    =param['attack_norm'],
                           eps     =param['eps'],
                           log_path=param['log_path'],
                           version =param['version'])

    # Create images and labels
    lx = []
    ly = []
    for (x, y) in testset:
        lx.append(x)
        ly.append(y)
    x_test = torch.cat(lx, 0)
    y_test = torch.cat(ly, 0)

    # Run attack and save images
    with torch.no_grad():
        _ = adversary.run_standard_evaluation(x_test, y_test, bs=param['batch_size'])


def generate_loop(param, attacks, budgets, budgets_l2):
    print(f'Generate adversarial examples')
    for attack in attacks:
        param['attack'] = attack
        if attack != 'deep_fool':
            for idx in range(len(budgets)):
                param['budget'] = budgets[idx]
                if attack == 'pgd':
                    for perturb in ('linf',):  # 'l2'
                        param['perturbation'] = perturb
                        if perturb == 'l2':
                            param['budget'] = budgets_l2[idx]
                        else:
                            param['budget'] = budgets[idx]
                        print('-' * 101)
                        one_test_run(param)
                else:
                    print('-' * 101)
                    one_test_run(param)
        else:
            print('-' * 101)
            one_test_run(param)


def test_loop(param, models, attacks, budgets, budgets_l2):
    print('=' * 101)
    print(f'Test defenses')
    param['generate'] = False
    for name in models:
        param['name'] = 'cifar_icassp/' + name
        param['clean'] = True
        print('-' * 101)
        one_test_run(param)
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
                            one_test_run(param)
                    else:
                        print('-' * 101)
                        one_test_run(param)
            else:
                print('-' * 101)
                one_test_run(param)


def auto_attack_loop(param):
    name = param['name']
    for seed in range(param['seed'], param['seed'] + 5):
        print('=' * 101)
        param['seed'] = -seed
        param['name'] = name + '/seed_' + str(seed)
        one_auto_attack(param)


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    param = load_yaml('generate_cifar')
    models = [
        'baseline'
    ]
    # 'iso'
    # 'jac'
    # 'fir'
    # 'distillation'
    # 'adv_train'
    budgets = [4/255, 8/255, 16/255, 32/255]
    budgets_l2 = [2., 3., 4.]
    attacks = [
        'pgd'
    ]
    # 'fgsm'
    # 'pgd'
    # 'deep_fool'

    if param['generate']:
        generate_loop(param, attacks, budgets, budgets_l2)
    elif param['autoattack']:
        auto_attack_loop(param)
    else:
        # test_loop(param, models, attacks, budgets, budgets_l2)
        one_test_run(param)

if __name__ == '__main__':
    main()
