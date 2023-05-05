# Rotated cINN Model
# ==================
# Defines the model class Rotated_cINN.

import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm


# Parameters
ndim_total = 28 * 28


# Support functions
    # one_hot() is used for every forward pass. This seems completely unnecessary. Just compute one_hot one time when creating the dataset.
def one_hot(labels, out=None):
    '''
    Convert LongTensor labels (contains labels 0-9), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    '''

    if out is None:
        out = torch.zeros(labels.shape[0], 10).to(labels.device)   # labels.shape[0] is the batch size
    else:
        out.zeros_()

    out.scatter_(dim=1, index=labels.view(-1,1), value=1.)
    return out


# Main class
class MNIST_cINN(nn.Module):
    
    def __init__(self, lr):
        super().__init__()

        # Build network
        self.cinn = self.build_inn()

        # Random initialization of parameters
        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)


    def build_inn(self):

        def subnet(ch_in, ch_out):
            '''the neural network inside the coupling block'''
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        nodes = []   # list of freia Nodes, analagous to the nn.Sequential list of pytorch Modules
        cond = Ff.ConditionNode(10)  # special node providing the external condition for the subnet


        # Input nodes
        nodes.append(Ff.InputNode(1, 28, 28))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))   # just flattens the input image

        # Creating an INN out of 20 coupling blocks
        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
                # In one coupling block only half of the dimensions get transformed. 
                # Fixed permutations between blocks ensures that all the dimensions get transformed similarly often.
                # Here 10 times on average.
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        # Output nodes
        nodes.extend([cond, Ff.OutputNode(nodes[-1])])

         # Full network
        return Ff.ReversibleGraphNet(nodes, verbose=False)



    def forward(self, x, l):
        #z = self.cinn(x, c=one_hot(l))
        #jac = self.cinn.log_jacobian(run_forward=False)
        z, jac = self.cinn.forward(x, c=one_hot(l))
        return z, jac


    def reverse_sample(self, z, l):
        return self.cinn(z, c=one_hot(l), rev=True)
