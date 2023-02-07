import torch
import torch.nn as nn
import numpy as np


def jac_soft(prob):
    b = prob.shape[0]
    c = prob.shape[1]
    prob = prob.unsqueeze(-1)
    rows = torch.transpose(prob, 1, 2)*torch.ones(b, c, c)
    columns = torch.ones(b, c, c) * prob
    jac = columns*(torch.eye(c) - rows)
    return jac[:, :-1, :].mean(dim=0)


def coord_change(prob):
    return 2*torch.sqrt(prob[:, :-1])/(1-torch.sqrt(prob[:, -1]))


def jac_coord_change(prob):
    b = prob.shape[0]
    c = prob.shape[1]
    change = torch.ones(b, c-1, c-1)*coord_change(prob)
    columns = torch.ones(b, c-1, c-1) * prob[:, :-1]
    jac = change/2*(torch.eye(c-1)*torch.div(1, columns) - change*torch.div(1, torch.sqrt(columns))/2/torch.sqrt(prob[:,-1]))
    return jac.mean(dim=0)


def parseval_orthonormal_constraint(model, prob, defense=None, beta = 1/8, percent_of_rows = 1):
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
            if name.split('.')[-1] == 'weight':
                # Flatten
                w = param.view(param.size(0), -1) if 'conv' in name else param

                # Sample rows
                S = torch.from_numpy(np.random.binomial(1, percent_of_rows, (w.size(0)))).bool()

                # Update
                if name.split('.')[0] == 'classifier' and defense=='isolayer':
                    # How to handle the batch?
                    J = torch.mm(jac_soft(prob), jac_coord_change(prob))
                    kappa = (4*(1 - torch.sqrt(prob[:,-1]))**2).mean()
                    JW = torch.mm(J, w)
                    JWWJ_I = torch.mm(JW, JW.T) - torch.eye(w.shape[0]-1)/kappa
                    w[S,:] = w[S,:] - 4*beta*torch.mm(J[S,:].T, torch.mm(JWWJ_I[S,:], JW[S,:]))
                else:
                    w[S,:] = ((1 + 4*beta)*w[S,:] - 4*beta*torch.mm(w[S,:], torch.mm(w[S,:].T, w[S,:])))/rescale_factor

                # Set parameters
                state_dict[name] = w.view_as(param)

        model.load_state_dict(state_dict)

    return model


def main():
    prob = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(jac_soft(prob))


if __name__ == '__main__':
    main()