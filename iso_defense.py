import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
import timeit


class JacSoftmax(nn.Module):

    def __init__(self):
        super(JacSoftmax, self).__init__()

    def forward(self, prob, device, mean=False):
        b = prob.shape[0]
        c = prob.shape[1]
        prob = prob.unsqueeze(-1)
        rows = torch.transpose(prob, 1, 2) * torch.ones(b, c, c).to(device)
        columns = torch.ones(b, c, c).to(device) * prob
        batch_eye = torch.eye(c).unsqueeze(0).repeat(b, 1, 1).to(device)
        jac = columns * (batch_eye - rows)
        if mean:
            return jac[:, :-1, :].mean(dim=0)
        else:
            return jac[:, :-1, :]


class CoordChange(nn.Module):

    def __init__(self):
        super(CoordChange, self).__init__()

    def forward(self, prob):
        c = prob.shape[1]
        return 2 * torch.sqrt(prob[:, :-1]) / (1 - torch.sqrt(prob[:, -1].unsqueeze(1).repeat(1, c - 1)))


class JacCoordChange(nn.Module):

    def __init__(self):
        super(JacCoordChange, self).__init__()
        self.coord_change = CoordChange()

    def forward(self, prob, device, mean=False):
        b = prob.shape[0]
        c = prob.shape[1]
        change = torch.ones(b, c-1, c-1).to(device)*self.coord_change(prob).unsqueeze(-1)
        columns = torch.ones(b, c-1, c-1).to(device)*prob[:, :-1].unsqueeze(-1)
        batch_eye = torch.eye(c-1).unsqueeze(0).repeat(b, 1, 1).to(device)
        jac = change/2*(batch_eye*torch.div(1, columns) - change*torch.div(1, torch.sqrt(columns))/2/torch.sqrt(prob[:,-1].reshape(b, 1, 1)))
        if mean:
            return jac.mean(dim=0)
        else:
            return jac


class IsometryRegNoBackprop(nn.Module):

    def __init__(self, epsilon, num_stab=1e-4):
        super(IsometryRegNoBackprop, self).__init__()
        self.epsilon = epsilon
        self.num_stab = num_stab
        self.jac_soft = JacSoftmax()
        self.jac_coord_change = JacCoordChange()

    def forward(self, data, logits, device):
        # Input dimension
        n = data.shape[1]*data.shape[2]*data.shape[3]
        # Number of classes
        c = logits.shape[1]
        m = c - 1

        # Numerical stability
        probs = F.softmax(logits, dim=1)*(1 - c*self.num_stab) + self.num_stab
        # assert torch.all(output>0)

        # Compute Jacobian matrix
        jac = torch.zeros(c, *data.shape).to(device)
        grad_output = torch.zeros(*logits.shape).to(device)
        for i in range(c):
            grad_output.zero_()
            grad_output[:, i] = 1
            jac[i] = torch.nan_to_num(torch.autograd.grad(logits, data, grad_outputs=grad_output, retain_graph=True)[0])
        jac = torch.transpose(jac, dim0=0, dim1=1)
        jac = jac.contiguous().view(jac.shape[0], jac.shape[1], -1)

        # Multiply by the Jacobian matrix of coordinate change
        jac = torch.bmm(self.jac_coord_change(probs, device), torch.bmm(self.jac_soft(probs, device), jac))

        # Gram matrix of Jacobian
        jac = torch.bmm(jac, torch.transpose(jac, 1, 2))

        # Compute the FIM coefficient in stereographic projection
        coeff = 4*(1 - torch.sqrt(probs[:, m]))**2

        # Distance from center of simplex
        delta = torch.sqrt(probs / c).sum(dim=1)
        delta = 2*torch.acos(delta)

        # Rescaled identity matrix
        factor = (delta**2/coeff/self.epsilon**2).view(-1, 1, 1)
        identity = factor*torch.eye(m).unsqueeze(0).repeat(logits.shape[0], 1, 1).to(device)

        # Compute regularization term (alpha in docs)
        reg = torch.linalg.norm((jac - identity).contiguous().view(len(data), -1), dim=1)/n

        # Return
        return reg.mean(), torch.tensor(0)


def parseval_orthonormal_constraint(model, logits, device, reg_model, defense=None, beta = 1e-4, num_stab=1e-4):
    """
    TBD
    - How to handle the batch?
    - Add a factor for batch normalizations and residual connections?
    - What if dim(output) > dim(input)?
    """
    # From paper: https://arxiv.org/pdf/1704.08847.pdf
    c = logits.shape[1]
    prob = F.softmax(logits, dim=1) * (1 - c * num_stab) + num_stab  # for numerical stability
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
                rescale_factor = float(module.kernel_size[0])
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
                    J = torch.mm(reg_model[0](prob, device, mean=True), reg_model[1](prob, device, mean=True))
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


'''
def convmatrix2d(kernel, image_shape, padding: int=0, stride: int=1):
    """
    kernel: (out_channels, in_channels, kernel_height, kernel_width, ...)
    image: (in_channels, image_height, image_width, ...)
    padding: assumes the image is padded with ZEROS of the given amount
    in every 2D dimension equally. The actual image is given with unpadded dimension.
    """

    # If we want to pad, request a bigger matrix as the kernel will convolve over a bigger image.
    if padding:
        old_shape = image_shape
        pads = (padding, padding, padding, padding)
        image_shape = (image_shape[0], image_shape[1] + padding*2, image_shape[2]
                       + padding*2)
    else:
        image_shape = tuple(image_shape)
    assert image_shape[0] == kernel.shape[1]
    assert len(image_shape[1:]) == len(kernel.shape[2:])
    # assert stride == 1

    kernel = kernel.to('cpu') # always perform the below work on cpu

    result_dims = torch.div(torch.tensor(image_shape[1:]) -
                   torch.tensor(kernel.shape[2:]), stride, rounding_mode='floor') + 1
    mat = torch.zeros((kernel.shape[0], *result_dims, *image_shape))
    for i in range(mat.shape[1]):
        for j in range(mat.shape[2]):
            mat[:,i,j,:,i:i+kernel.shape[2],j:j+kernel.shape[3]] = kernel
    mat = mat.flatten(0, len(kernel.shape[2:])).flatten(1)

    # Handle zero padding. Effectively, the zeros from padding do not
    # contribute to convolution output as the product at those elements is zero.
    # Hence the columns of the conv mat that are at the indices where the
    # padded flattened image would have zeros can be ignored. The number of
    # rows on the other hand must not be altered (with padding the output must
    # be larger than without). So..

    # We'll handle this the easy way and create a mask that accomplishes the
    # indexing
    if padding:
        mask = torch.nn.functional.pad(torch.ones(old_shape), pads).flatten()
        mask = mask.bool()
        mat = mat[:, mask]

    return mat
'''


def convmatrix2d(kernel, image_shape, padding: int=0, stride: int=1, device=None):
    """
    kernel: (out_channels, in_channels, kernel_height, kernel_width)
    image: (in_channels, image_height, image_width)
    padding: assumes the image is padded with ZEROS of the given amount
    in every 2D dimension equally. The actual image is given with unpadded dimension.
    """
    # assert image_shape[0] == kernel.shape[1]
    # assert len(image_shape[1:]) == len(kernel.shape[2:])
    # kernel = kernel.to('cpu')

    padded_shape = torch.tensor(image_shape).to(device)
    padded_shape[1:] += 2*padding
    result_dims = torch.div(padded_shape[1:] - torch.tensor(kernel.shape[2:]).to(device), stride, rounding_mode='floor') + 1
    mat = torch.zeros((kernel.shape[0], *result_dims, *padded_shape)).to(device)

    for i in range(mat.shape[1]):
        for j in range(mat.shape[2]):
            mat[:,i,j,:,i*stride:i*stride+kernel.shape[2],j*stride:j*stride+kernel.shape[3]] = kernel

    mat = mat[:,:,:,:,padding:image_shape[1]+padding,padding:image_shape[1]+padding]
    mat = mat.flatten(0, len(kernel.shape[2:])).flatten(1)
    return mat

def simple_iso_reg(model, device, input_shape):
    jacobian = torch.eye(input_shape[0]*input_shape[1]*input_shape[2])
    input_shape = torch.tensor(input_shape)
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

        if name.split('.')[-1] == 'weight':
            print(name)
            print(jacobian.shape)
            if 'conv' in name:
                jacobian = torch.mm(convmatrix2d(param, input_shape.tolist(), module.padding[0], module.stride[0], device), jacobian)
                input_shape[1:] = 1 + ((input_shape[1:] + 2 * module.padding[0] - torch.tensor(param.shape[2:])) / module.stride[0]).floor()
            else:
                jacobian = torch.mm(param, jacobian)

    return jacobian


def main():
    # prob = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    # print(jac_soft(prob, 'cpu'))
    # print(jac_coord_change(prob, 'cpu'))
    batch = 2
    d0 = 32
    cin = 3
    cout = 64
    k = 5
    s = 2
    p = 3
    d1 = int(1 + (d0 + 2*p - k)/s)
    print(f'd1={d1}')
    conv_layer = nn.Conv2d(in_channels=cin,
                           out_channels=cout,
                           kernel_size=(k,k),
                           stride=(s,s),
                           padding=p)
    inp = torch.rand(batch, cin, d0, d0)
    out = conv_layer(inp)
    kernel = conv_layer.weight
    print(f'input shape: {inp.shape}')
    print(f'kernel shape: {kernel.shape}')
    print(f'output shape: {out.shape}')
    matrix = convmatrix2d(kernel, inp.shape[1:], p, s)
    print(f'matrix shape: {matrix.shape}')
    print('Matrix multiplication execution time')
    t = timeit.Timer(lambda: inp.flatten(start_dim=1).mm(matrix.T) + conv_layer.bias.flatten().repeat_interleave(d1**2))
    print(t.timeit(100))
    print('Sparse multiplication execution time')
    t = timeit.Timer(lambda: torch.sparse.mm(matrix.to_sparse(), inp.flatten(start_dim=1).T).T + conv_layer.bias.flatten().repeat_interleave(d1**2))
    print(t.timeit(100))
    out_mat_sparse = torch.sparse.mm(matrix.to_sparse(), inp.flatten(start_dim=1).T).T + conv_layer.bias.flatten().repeat_interleave(d1**2)
    out_mat = inp.flatten(start_dim=1).mm(matrix.T) + conv_layer.bias.flatten().repeat_interleave(d1 ** 2)
    print(f'out mat shape: {out_mat.shape}')
    print(f'out mat sparse shape: {out_mat_sparse.shape}')
    print(f'out norm: {torch.norm(out)}')
    print(f'out mat norm: {torch.norm(out_mat)}')
    print(f'out mat sparse norm: {torch.norm(out_mat_sparse)}')
    print(f'diff: {torch.norm(out - out_mat.view_as(out))}')
    print(f'diff sparse: {torch.norm(out - out_mat_sparse.view_as(out))}')
    print(f'allclose: {torch.allclose(out, out_mat.view_as(out), atol=1e-6)}')
    print(f'allclose sparse: {torch.allclose(out, out_mat_sparse.view_as(out), atol=1e-6)}')


if __name__ == '__main__':
    main()