import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import random
from attack_defense.parseval import JacSoftmax, JacCoordChange


###################################################### Utils ###########################################################


def fisher_distance(probs1, probs2):
    """
    :param probs1: torch.Tensor (batch size x nb of classes)
    :param probs2: torch.Tensor (batch size x nb of classes)
    :return: torch.Tensor (batch size)
    """
    dist = (torch.sqrt(probs1)*torch.sqrt(probs2)).sum(-1)
    return 2 * torch.acos(dist)


def softmax_stable(logits, num_stab):
    c = logits.shape[-1]
    return F.softmax(logits, dim=1)*(1 - c*num_stab) + num_stab


def coord_change(probs):
    m = probs.shape[-1] - 1
    new_coord = torch.sqrt(probs)
    return 2*new_coord[:, :m] / (1 - new_coord[:, m].unsqueeze(-1))


def diff_coord_change(probs, tau, device):
    m    = tau.shape[-1]
    jac  = torch.eye(m).unsqueeze(0).repeat(tau.shape[0], 1, 1).to(device)
    jac /= probs[:, :m].unsqueeze(-1)
    jac -= tau.unsqueeze(-1)/(2*torch.sqrt(probs[:, :m].unsqueeze(-1)*probs[:, m].unsqueeze(-1).unsqueeze(-1)))
    return tau.unsqueeze(-1)*jac/2


def inv_change(tau, device):
    """
    :param tau: torch.Tensor (batch size x m)
    :param device:
    :return: torch.Tensor (batch size x (m+1))
    """
    m = tau.shape[-1]
    theta = torch.zeros(tau.shape[0], m+1).to(device)
    tau_2 = torch.linalg.norm(tau/2, dim=1)**2
    theta[:, m] = ((tau_2 - 1)/(tau_2 + 1))**2
    theta[:, :m] = (tau/(1 + tau_2.unsqueeze(-1)))**2
    return theta


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

        # Compute the FIM coefficient in stereographic projection
        coeff = ((1 - torch.sqrt(probs[:, m]))**2).unsqueeze(-1).unsqueeze(-1)

        # Gram matrix of Jacobian
        jac = coeff*torch.bmm(jac, torch.transpose(jac, 1, 2))

        # Distance from center of simplex
        delta = torch.sqrt(probs / c).sum(dim=1)
        delta = 2 * torch.acos(delta)

        # Rescaled identity matrix
        factor = (delta ** 2 / self.epsilon ** 2).view(-1, 1, 1)
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

        # Compute gradient in two random directions
        vector1  = torch.randn(*new_coord.shape).to(device)
        vector2  = torch.randn(*new_coord.shape).to(device)
        vector1 /= torch.norm(vector1, dim=1).unsqueeze(-1)
        vector2 /= torch.norm(vector2, dim=1).unsqueeze(-1)
        grad1    = torch.autograd.grad(new_coord, data, grad_outputs=vector1, retain_graph=True)[0]
        grad2    = torch.autograd.grad(new_coord, data, grad_outputs=vector2, retain_graph=True)[0]
        grad1    = grad1.contiguous().view(grad1.shape[0], -1)
        grad2    = grad2.contiguous().view(grad2.shape[0], -1)

        # Distance from center of simplex
        delta = torch.sqrt(probs / c).sum(dim=1)
        delta = 2*torch.acos(delta).unsqueeze(-1)

        # Compute regularization term
        coeff = ((1 - torch.sqrt(probs[:, m]))**2).unsqueeze(-1)
        f_11 = coeff*((grad1*grad1).sum(-1).unsqueeze(-1)) - delta**2/self.epsilon**2*((vector1*vector1).sum(-1).unsqueeze(-1))
        f_22 = coeff*((grad2*grad2).sum(-1).unsqueeze(-1)) - delta**2/self.epsilon**2*((vector2*vector2).sum(-1).unsqueeze(-1))
        f_12 = coeff*((grad1*grad2).sum(-1).unsqueeze(-1)) - delta**2/self.epsilon**2*((vector1*vector2).sum(-1).unsqueeze(-1))
        reg = torch.abs(f_11) + torch.abs(f_22) + 2*torch.abs(f_12)

        # Return
        return reg.mean()/4


class IsometryRegNoBackprop(nn.Module):
    """
    Not maintained
    """

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

        # Compute the FIM coefficient in stereographic projection
        coeff = ((1 - torch.sqrt(probs[:, m]))**2).unsqueeze(-1).unsqueeze(-1)

        # Gram matrix of Jacobian
        jac = coeff*torch.bmm(jac, torch.transpose(jac, 1, 2))

        # Distance from center of simplex
        delta = torch.sqrt(probs / c).sum(dim=1)
        delta = 2*torch.acos(delta)

        # Rescaled identity matrix
        factor = (delta**2/self.epsilon**2).view(-1, 1, 1)
        identity = factor*torch.eye(m).unsqueeze(0).repeat(logits.shape[0], 1, 1).to(device)

        # Compute regularization term
        reg = torch.linalg.norm((jac - identity).contiguous().view(len(data), -1), dim=1)

        # Return
        return reg.mean()/m**2


############################################ Eigen Bound & variants ####################################################


class JacobianReg(nn.Module):
    """
    Paper: Hoffman et al.
    """

    def __init__(self):
        super(JacobianReg, self).__init__()

    def forward(self, data, logits, device):
        # Compute gradient in a random direction
        vector  = torch.randn(*logits.shape).to(device)
        vector /= torch.norm(vector, dim=1).unsqueeze(-1)
        grad   = torch.autograd.grad(logits, data, grad_outputs=vector, retain_graph=True)[0]
        grad   = grad.contiguous().view(grad.shape[0], -1)

        return torch.norm(grad, dim=1).mean()


class EigenBound(nn.Module):

    def __init__(self, epsilon, num_stab=1e-7):
        super(EigenBound, self).__init__()
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
        jac = jac.contiguous().view(jac.shape[0], jac.shape[1], -1)

        # Compute the FIM coefficient in stereographic projection
        coeff = ((1 - torch.sqrt(probs[:, m]))**2).unsqueeze(-1).unsqueeze(-1)

        # Gram matrix of Jacobian
        jac = coeff*torch.bmm(jac, torch.transpose(jac, 1, 2))

        # Distance from center of simplex
        delta = torch.sqrt(probs / c).sum(dim=1)
        delta = 2 * torch.acos(delta)

        # Rescaled identity matrix
        factor = (delta ** 2 / self.epsilon ** 2).view(-1, 1, 1)
        identity = factor * torch.eye(m).unsqueeze(0).repeat(logits.shape[0], 1, 1).to(device)

        # Compute regularization term
        jac_id = jac - identity
        reg = F.relu(jac_id.diagonal(dim1=-1, dim2=-2).sum(-1))

        # Return
        return reg.mean()/m


class RandomBound(nn.Module):

    def __init__(self, epsilon, num_stab=1e-7):
        super(RandomBound, self).__init__()
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

        # Compute gradient in two random directions
        vector1  = torch.randn(*new_coord.shape).to(device)
        vector2  = torch.randn(*new_coord.shape).to(device)
        vector1 /= torch.norm(vector1, dim=1).unsqueeze(-1)
        vector2 /= torch.norm(vector2, dim=1).unsqueeze(-1)
        grad1    = torch.autograd.grad(new_coord, data, grad_outputs=vector1, retain_graph=True)[0]
        grad2    = torch.autograd.grad(new_coord, data, grad_outputs=vector2, retain_graph=True)[0]
        grad1    = grad1.contiguous().view(grad1.shape[0], -1)
        grad2    = grad2.contiguous().view(grad2.shape[0], -1)

        # Distance from center of simplex
        delta = torch.sqrt(probs / c).sum(dim=1)
        delta = 2*torch.acos(delta).unsqueeze(-1)

        # Compute regularization term
        coeff = ((1 - torch.sqrt(probs[:, m]))**2).unsqueeze(-1)
        f_11 = coeff*((grad1*grad1).sum(-1).unsqueeze(-1)) - delta**2/self.epsilon**2*((vector1*vector1).sum(-1).unsqueeze(-1))
        f_22 = coeff*((grad2*grad2).sum(-1).unsqueeze(-1)) - delta**2/self.epsilon**2*((vector2*vector2).sum(-1).unsqueeze(-1))
        f_12 = coeff*((grad1*grad2).sum(-1).unsqueeze(-1)) - delta**2/self.epsilon**2*((vector1*vector2).sum(-1).unsqueeze(-1))
        reg = F.relu(f_11) + F.relu(f_22) + 2*F.relu(f_12)

        # Return
        return reg.mean()/4


class AdaptiveTemp(nn.Module):

    def __init__(self, epsilon, num_stab=1e-7):
        super(AdaptiveTemp, self).__init__()
        self.epsilon = epsilon
        self.num_stab = num_stab

    def forward(self, data, logits, device, model):
        # Dimensions
        batch = data.shape[0]
        d = data.shape[1]*data.shape[2]*data.shape[3]
        c = logits.shape[1]
        m = c - 1

        # Numerical stability
        probs = softmax_stable(logits, self.num_stab)

        # Sample m orthogonal directions and compute the vector with largest distance
        dirs = random.sample(range(d), m)
        vectors = torch.zeros(batch, c).to(device)
        distances = torch.zeros(batch).to(device)
        perturb = torch.zeros(batch, d).to(device)
        with torch.no_grad():
            for i in range(m):
                perturb.zero_()
                perturb[:, dirs[i]] = 1
                data_eps = data + self.epsilon*perturb.view_as(data)
                probs_eps = softmax_stable(model(data_eps), self.num_stab)
                distances = (fisher_distance(probs, probs_eps) > distances)
                vectors[distances] = (probs_eps - probs)[distances]

        # Coordinate change
        new_coord = coord_change(probs)
        new_vectors = torch.bmm(diff_coord_change(probs, new_coord, device), vectors[:, :m].unsqueeze(-1)).squeeze()

        # Estimate largest singular value
        new_vectors /= (torch.linalg.norm(new_vectors, dim=1).unsqueeze(-1) + self.num_stab)
        grad = torch.autograd.grad(new_coord, data, grad_outputs=new_vectors, retain_graph=True)[0]
        grad = grad.contiguous().view(batch, -1)
        coeff = (1 - torch.sqrt(probs[:, m]))**2
        grad_norm = coeff*torch.linalg.norm(grad, dim=-1)

        # Compute delta and coeff
        delta = torch.sqrt(probs/c).sum(dim=1)
        delta = 2*torch.acos(delta)

        # Compute temperature
        temp = (delta/(self.epsilon*grad_norm + self.num_stab)).unsqueeze(-1)
        return inv_change(temp.detach()*new_coord, device)


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
