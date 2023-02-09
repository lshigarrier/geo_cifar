import torch
import torch.nn as nn
import numpy as np


def jac_soft(prob, device):
    b = prob.shape[0]
    c = prob.shape[1]
    prob = prob.unsqueeze(-1)
    rows = torch.transpose(prob, 1, 2)*torch.ones(b, c, c).to(device)
    columns = torch.ones(b, c, c).to(device)*prob
    batch_eye = torch.eye(c).unsqueeze(0).repeat(b, 1, 1).to(device)
    jac = columns*(batch_eye - rows)
    return jac[:, :-1, :].mean(dim=0)


def coord_change(prob):
    c = prob.shape[1]
    return 2*torch.sqrt(prob[:, :-1])/(1-torch.sqrt(prob[:, -1].unsqueeze(1).repeat(1, c-1)))


def jac_coord_change(prob, device):
    b = prob.shape[0]
    c = prob.shape[1]
    change = torch.ones(b, c-1, c-1).to(device)*coord_change(prob).unsqueeze(-1)
    columns = torch.ones(b, c-1, c-1).to(device)*prob[:, :-1].unsqueeze(-1)
    batch_eye = torch.eye(c-1).unsqueeze(0).repeat(b, 1, 1).to(device)
    jac = change/2*(batch_eye*torch.div(1, columns) - change*torch.div(1, torch.sqrt(columns))/2/torch.sqrt(prob[:,-1].reshape(b, 1, 1)))
    return jac.mean(dim=0)


def parseval_orthonormal_constraint(model, prob, device, defense=None, beta = 1e-4, percent_of_rows = 1):
    """
    TBD
    - How to handle the batch?
    - Add a factor for batch normalizations and residual connections?
    - What if dim(output) > dim(input)?
    """
    # From paper: https://arxiv.org/pdf/1704.08847.pdf
    with torch.no_grad():
        state_dict = model.state_dict()
        for name, param in model.named_parameters():
            module = getattr(model, name.split('.')[0])
            flag = True
            i = 1
            while flag:
                submodule = getattr(module, name.split('.')[i])
                i += 1
                if isinstance(submodule, nn.Module):
                    module = submodule
                else:
                    flag = False
            # print(module.__class__.__name__)

            # Scaling factor for 2D convs in https://www.duo.uio.no/bitstream/handle/10852/69487/master_mathialo.pdf?sequence=1
            if isinstance(module, nn.Conv2d):
                k = float(module.kernel_size[0])
                rescale_factor = np.sqrt(k)
            else:
                rescale_factor = 1.0

            # Constraint
            if name.split('.')[-1] == 'weight' and 'norm' not in name:
                # Flatten
                w = param.view(param.shape[0], -1) if 'conv' in name else param

                # Test
                max_w = torch.max(torch.abs(w))
                if max_w > 10:
                    print('Before')
                    print(name)
                    print(max_w)

                # Sample rows
                # S = torch.from_numpy(np.random.binomial(1, percent_of_rows, (w.size(0)))).bool()

                # Update
                if name.split('.')[0] == 'classifier' and defense=='isolayer':
                    J = torch.mm(jac_coord_change(prob, device), jac_soft(prob, device))
                    kappa = (4*(1 - torch.sqrt(prob[:,-1]))**2).mean()
                    JW = torch.mm(J, w)
                    JWWJ_I = torch.mm(JW, JW.T) - torch.eye(w.shape[0]-1).to(device)/kappa
                    # w[S,:] = w[S,:] - 4*beta*torch.mm(J[S,:].T, torch.mm(JWWJ_I[S,:], JW[S,:]))
                    w[:, :] = w - 4 * beta * torch.mm(J.T, torch.mm(JWWJ_I, JW))
                else:
                    # w[S,:] = ((1 + 4*beta)*w[S,:] - 4*beta*torch.mm(w[S,:], torch.mm(w[S,:].T, w[S,:])))/rescale_factor
                    w[:, :] = ((1 + 4 * beta) * w - 4 * beta * torch.mm(w, torch.mm(w.T, w))) / rescale_factor

                # Test
                max_w_bis = torch.max(torch.abs(w))
                if max_w_bis > 10 or max_w > 10:
                    print('After')
                    print(name)
                    print(max_w)
                    print(max_w_bis)

                # Set parameters
                state_dict[name] = w.view_as(param)

        model.load_state_dict(state_dict)

    return model


def main():
    prob = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    # print(jac_soft(prob, 'cpu'))
    print(jac_coord_change(prob, 'cpu'))


if __name__ == '__main__':
    main()