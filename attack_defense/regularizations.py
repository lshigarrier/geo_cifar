import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
from attack_defense.parseval import JacSoftmax, JacCoordChange


###################################### Isometry Regularization & variants ##############################################


class IsometryReg(nn.Module):

    def __init__(self, epsilon, norm='frobenius', num_stab=1e-7):
        super(IsometryReg, self).__init__()
        self.epsilon = epsilon
        self.num_stab = num_stab
        self.norm = norm

    def forward(self, data, logits, device):
        # Input dimension
        # n = data.shape[1]*data.shape[2]*data.shape[3]
        # Number of classes
        c = logits.shape[1]
        m = c - 1

        # Numerical stability
        probs = F.softmax(logits, dim=1)*(1 - c*self.num_stab) + self.num_stab

        # Coordinate change
        new_coord = torch.sqrt(probs)
        new_coord = 2 * new_coord[:, :m] / (1 - new_coord[:, m].unsqueeze(1).repeat(1, m))

        # Compute Jacobian matrix
        jac = torch.zeros(m, *data.shape).to(device)
        grad_output = torch.zeros(*new_coord.shape).to(device)
        for i in range(m):
            grad_output.zero_()
            grad_output[:, i] = 1
            jac[i] = torch.autograd.grad(new_coord, data, grad_outputs=grad_output, retain_graph=True)[0]
        jac = torch.transpose(jac, dim0=0, dim1=1)
        jac = jac.contiguous().view(jac.shape[0], jac.shape[1], -1)

        # Gram matrix of Jacobian
        jac = torch.bmm(jac, torch.transpose(jac, 1, 2))

        # Compute the FIM coefficient in stereographic projection
        coeff = 4 * (1 - torch.sqrt(probs[:, m])) ** 2

        # Distance from center of simplex
        delta = torch.sqrt(probs / c).sum(dim=1)
        delta = 2 * torch.acos(delta)

        # Rescaled identity matrix
        factor = (delta ** 2 / coeff / self.epsilon ** 2).view(-1, 1, 1)
        identity = factor * torch.eye(m).unsqueeze(0).repeat(logits.shape[0], 1, 1).to(device)

        # Compute regularization term
        if self.norm == 'frobenius':
            reg = torch.linalg.norm((jac - identity).contiguous().view(len(data), -1), dim=1)
        elif self.norm == 'holder':
            # Holder inequality
            abs_jac_id = torch.abs(jac - identity)
            norm_1 = torch.max(abs_jac_id.sum(dim=1, keepdim=True), dim=2)[0]
            norm_inf = torch.max(abs_jac_id.sum(dim=2, keepdim=True), dim=1)[0]
            reg = torch.sqrt(norm_1 * norm_inf)
        else:
            raise NotImplementedError

        # Return
        return reg.mean()/m**2


class IsometryRegRandom(nn.Module):

    def __init__(self, epsilon, num_stab=1e-7):
        super(IsometryRegRandom, self).__init__()
        self.epsilon = epsilon
        self.num_stab = num_stab

    def forward(self, data, logits, device):
        # Input dimension
        # n = data.shape[1]*data.shape[2]*data.shape[3]
        # Number of classes
        c = logits.shape[1]
        m = c - 1

        # Numerical stability
        probs = F.softmax(logits, dim=1)*(1 - c*self.num_stab) + self.num_stab

        # Coordinate change
        new_coord = torch.sqrt(probs)
        new_coord = 2 * new_coord[:, :m] / (1 - new_coord[:, m].unsqueeze(1).repeat(1, m))

        # Compute Jacobian matrix
        grad_output = torch.randn(*new_coord.shape).to(device)
        grad_output /= torch.norm(grad_output, dim=1).unsqueeze(-1)
        jac = torch.autograd.grad(new_coord, data, grad_outputs=grad_output, retain_graph=True)[0]
        jac = jac.contiguous().view(jac.shape[0], -1)

        # Estimation of the trace of JJ^T
        jac = torch.norm(jac, dim=1)

        # Distance from center of simplex
        delta = torch.sqrt(probs / c).sum(dim=1)
        delta = 2 * torch.acos(delta)

        # Compute regularization term
        reg = torch.norm(torch.add(jac,-delta/self.epsilon))

        # Return
        return reg.mean()/m**2


class IsometryRegNoBackprop(nn.Module):

    def __init__(self, epsilon, num_stab=1e-7):
        super(IsometryRegNoBackprop, self).__init__()
        self.epsilon = epsilon
        self.num_stab = num_stab
        self.jac_soft = JacSoftmax()
        self.jac_coord_change = JacCoordChange()

    def forward(self, data, logits, device):
        # Input dimension
        # n = data.shape[1]*data.shape[2]*data.shape[3]
        # Number of classes
        c = logits.shape[1]
        m = c - 1

        # Numerical stability
        probs = F.softmax(logits, dim=1)*(1 - c*self.num_stab) + self.num_stab

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

        # Compute regularization term
        reg = torch.linalg.norm((jac - identity).contiguous().view(len(data), -1), dim=1)

        # Return
        return reg.mean()/m**2


###################################### Jacobian Regularization & variants ##############################################


class JacobianReg(nn.Module):

    def __init__(self, epsilon, barrier='relu', num_stab=1e-7):
        super(JacobianReg, self).__init__()
        self.epsilon = epsilon
        self.num_stab = num_stab
        # Barrier function
        if barrier == 'relu':
            self.barrier = F.relu
        elif barrier == 'elu':
            self.barrier = F.elu
        elif barrier == 'exp':
            self.barrier = torch.exp
        else:
            raise NotImplementedError

    def forward(self, data, logits, device):
        # Input dimension
        # n = data.shape[1]*data.shape[2]*data.shape[3]
        # Number of classes
        c = logits.shape[1]
        m = c - 1

        # Numerical stability
        probs = F.softmax(logits, dim=1)*(1 - c*self.num_stab) + self.num_stab

        # Coordinate change
        new_coord = torch.sqrt(probs)
        new_coord = 2 * new_coord[:, :m] / (1 - new_coord[:, m].unsqueeze(1).repeat(1, m))

        # Compute Jacobian matrix
        jac = torch.zeros(m, *data.shape).to(device)
        grad_output = torch.zeros(*new_coord.shape).to(device)
        for i in range(m):
            grad_output.zero_()
            grad_output[:, i] = 1
            jac[i] = torch.autograd.grad(new_coord, data, grad_outputs=grad_output, retain_graph=True)[0]
        jac = torch.transpose(jac, dim0=0, dim1=1)
        jac = jac.contiguous().view(jac.shape[0], m, -1)

        # Compute delta and rho
        delta = torch.sqrt(probs/c).sum(dim=1)
        delta = 2*torch.acos(delta)
        rho = 1 - torch.sqrt(probs[:, m])
        bound = delta/(rho*self.epsilon)

        # Holder inequality
        abs_jac = torch.abs(jac)
        norm_1 = torch.max(abs_jac.sum(dim=1, keepdim=True), dim=2)[0]
        norm_inf = torch.max(abs_jac.sum(dim=2, keepdim=True), dim=1)[0]
        jac_norm_holder = torch.sqrt(norm_1 * norm_inf)

        # Compute regularization
        reg = self.barrier(jac_norm_holder - bound)
        return reg.mean()/m**2, jac_norm_holder.mean()/m**2


class AdaptiveTemp(nn.Module):

    def __init__(self, epsilon, num_stab=1e-7):
        super(AdaptiveTemp, self).__init__()
        self.epsilon = epsilon
        self.num_stab = num_stab

    def forward(self, data, logits, device):
        # Number of classes
        c = logits.shape[1]
        m = c - 1

        # Numerical stability
        probs = F.softmax(logits, dim=1)*(1 - c*self.num_stab) + self.num_stab

        # Coordinate change
        new_coord = torch.sqrt(probs)
        new_coord = 2 * new_coord[:, :m] / (1 - new_coord[:, m].unsqueeze(1).repeat(1, m))

        # Compute Jacobian matrix
        jac = torch.zeros(m, *data.shape).to(device)
        grad_output = torch.zeros(*new_coord.shape).to(device)
        for i in range(m):
            grad_output.zero_()
            grad_output[:, i] = 1
            jac[i] = torch.autograd.grad(new_coord, data, grad_outputs=grad_output, retain_graph=True)[0]
        jac = torch.transpose(jac, dim0=0, dim1=1)
        jac = jac.contiguous().view(jac.shape[0], m, -1)
        jac = torch.bmm(jac, torch.transpose(jac, dim0=1, dim1=2))

        # Compute delta and rho
        delta = torch.sqrt(probs/c).sum(dim=1)
        delta = 2*torch.acos(delta)
        rho = 1 - torch.sqrt(probs[:, m])
        jac = (rho**2).unsqueeze(-1).unsqueeze(-1)*jac

        # Holder inequality
        abs_jac = torch.abs(jac)
        norm_1 = torch.max(abs_jac.sum(dim=1, keepdim=True), dim=2)[0]
        norm_inf = torch.max(abs_jac.sum(dim=2, keepdim=True), dim=1)[0]
        jac_norm_holder = torch.sqrt(norm_1 * norm_inf)

        # Compute temperature
        temp = delta.unsqueeze(-1)/(self.epsilon*jac_norm_holder)
        return temp


################################ Product of weights matrices (not maintained) ##########################################


def convmatrix2d(kernel, image_shape, padding: int=0, stride: int=1, device=None):
    """
    kernel: (out_channels, in_channels, kernel_height, kernel_width)
    image: (in_channels, image_height, image_width)
    padding: assumes the image is padded with ZEROS of the given amount
    in every 2D dimension equally. The actual image is given with unpadded dimension.
    """
    padded_shape = torch.tensor(image_shape).to(device)
    padded_shape[1:] += 2*padding
    result_dims = torch.div(padded_shape[1:] - torch.tensor(kernel.shape[2:]).to(device),
                            stride,
                            rounding_mode='floor') + 1
    mat = torch.zeros((kernel.shape[0], *result_dims, *padded_shape)).to(device)

    for i in range(mat.shape[1]):
        for j in range(mat.shape[2]):
            mat[:,i,j,:,i*stride:i*stride+kernel.shape[2],j*stride:j*stride+kernel.shape[3]] = kernel

    mat = mat[:,:,:,:,padding:image_shape[1]+padding,padding:image_shape[1]+padding]
    mat = mat.flatten(0, len(kernel.shape[2:])).flatten(1)
    return mat


def isometry_reg_approx(model, device, input_shape):
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
                jacobian = torch.mm(
                    convmatrix2d(param, input_shape.tolist(), module.padding[0], module.stride[0], device), jacobian)
                input_shape[1:] = 1\
                                  + ((input_shape[1:] + 2 * module.padding[0] - torch.tensor(param.shape[2:]))
                                   /module.stride[0]).floor()
            else:
                jacobian = torch.mm(param, jacobian)

    return jacobian


###################################################### Main ############################################################


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
    t = timeit.Timer(lambda: torch.sparse.mm(matrix.to_sparse(), inp.flatten(start_dim=1).T).T
                             + conv_layer.bias.flatten().repeat_interleave(d1**2))
    print(t.timeit(100))
    out_mat_sparse = torch.sparse.mm(matrix.to_sparse(), inp.flatten(start_dim=1).T).T\
                     + conv_layer.bias.flatten().repeat_interleave(d1**2)
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
