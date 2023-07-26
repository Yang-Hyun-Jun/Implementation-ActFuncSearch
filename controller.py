import torch
import torch.nn as nn

from typing import Tuple
from typing import List

class Controller(nn.Module):

    # Search Space Dim of Each Component
    INPUT_DIM = 7

    # RNN Hiden State Dimension
    HIDDEN_DIM = 128

    # Component Determination Iterations
    N_STEPS = 5

    def __init__(self):
        super().__init__()
        """
        Order 4 Polynomial Search
        Coefficient Space = {-3, -2, -1, 0, 1, 2, 3}
        Taget Function = (x-3)^2 * (x-1)
        """

        # Coffeficient Space
        self.coef = [-3, -2, -1, 0, 1, 2, 3]

        # Parameters (Wc, bc) for calculating a next input vector
        self.Wc = nn.Parameter(torch.rand((self.HIDDEN_DIM, self.INPUT_DIM)), True)
        self.bc = nn.Parameter(torch.rand((self.INPUT_DIM)), True)

        # Parameters (Wv, bv) for calculating a state value
        self.Wv = nn.Parameter(torch.rand((self.HIDDEN_DIM, 1)), True)
        self.bv = nn.Parameter(torch.rand((1)), True)

        self.cell = nn.RNNCell(self.INPUT_DIM, self.HIDDEN_DIM)
        self.softmax = nn.Softmax(1)
        
    def search(self, sampling:bool=True) -> \
        Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        RNN Search Function
        """
        pis = []
        indice = []

        greedy = torch.argmax
        sampling = torch.multinomial

        # First input is defined as an identity vector
        rnn_input = torch.ones((1, self.INPUT_DIM))

        for _ in range(self.N_STEPS):
            # Return the rnn hidden state
            rnn_hidden = self.cell(rnn_input)

            # Calculate the logits by multiplying a parameter matrix
            value_logit = torch.matmul(rnn_hidden, self.Wv) + self.bv
            pi_logit = torch.matmul(rnn_hidden, self.Wc) + self.bc

            # Calculate the state value and next input (autoregressively)
            value = value_logit
            rnn_input = self.softmax(pi_logit)

            # Select element of the multivariate action vector (sampling or greedy)  
            index = sampling(rnn_input, 1) if sampling else greedy(rnn_input, 1)
            pi = rnn_input.squeeze(0)[index]

            pis.append(pi)
            indice.append(index)

        pis = torch.cat(pis).swapaxes(0, 1)
        indice = torch.cat(indice).swapaxes(0, 1)

        return indice, pis, value 

    def generate(self, indice):
        """
        Return Search Polynomial Function
        """
        indice = indice.squeeze(0)
        order_4 = lambda x: self.coef[indice[0]] * (x**4)
        order_3 = lambda x: self.coef[indice[1]] * (x**3)
        order_2 = lambda x: self.coef[indice[2]] * (x**2)
        order_1 = lambda x: self.coef[indice[3]] * (x**1)
        order_0 = lambda x: self.coef[indice[4]] 

        search_funcion = lambda x: \
            order_4(x) + order_3(x) + \
            order_2(x) + order_1(x) + \
            order_0(x)

        return search_funcion 
    
    def result(self):
        indice, _, _ = self.search(sampling=False)
        indice = indice.squeeze(0)
        c4 = self.coef[indice[0]]
        c3 = self.coef[indice[1]]
        c2 = self.coef[indice[2]]
        c1 = self.coef[indice[3]]
        c0 = self.coef[indice[4]]
        print(f'{c4}x^4+{c3}x^3+{c2}x^2+{c1}x+{c0}')

        
if __name__ == '__main__':
    controller = Controller()

    indice, pis, value = controller.search()
    indice = indice.squeeze(0)

    print(controller.generate(indice)(3))
    print(controller.state_dict())
    