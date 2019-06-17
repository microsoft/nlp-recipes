# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A Gated Recurrent Unit (GRU) cell with peepholes."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGRU(nn.Module):
    """A Gated Recurrent Unit (GRU) cell with peepholes."""

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        """Initialize params.

        Args:
            input_dim: Dimension of the input vector.
            hidden_dim: Dimension of the hidden layer.
            dropout: Dropout of the network.
        """

        super(ConditionalGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_weights = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.hidden_weights = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.peep_weights = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Set params. """
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden, ctx):
        """Propogate input through the layer.

        Args:
            input: batch size x target sequence length  x embedding dimension.
            hidden: batch size x hidden dimension.
            ctx: batch size x source sequence length  x hidden dimension.

        Returns:
            output(torch.Tensor)  - batch size x target sequence length  x
            hidden dimension
            hidden(torch.Tensor)  - (batch size x hidden dimension, batch size x hidden
            dimension)

        """

        def recurrence(input, hidden, ctx):
            """Recurrence helper."""
            input_gate = self.input_weights(input)
            hidden_gate = self.hidden_weights(hidden)
            peep_gate = self.peep_weights(ctx)
            i_r, i_i, i_n = input_gate.chunk(3, 1)
            h_r, h_i, h_n = hidden_gate.chunk(3, 1)
            p_r, p_i, p_n = peep_gate.chunk(3, 1)
            resetgate = F.sigmoid(i_r + h_r + p_r)
            inputgate = F.sigmoid(i_i + h_i + p_i)
            newgate = F.tanh(i_n + resetgate * h_n + p_n)
            hy = newgate + inputgate * (hidden - newgate)

            return hy

        input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, ctx)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1)

        return output, hidden


# Original source: https://github.com/Maluuba/gensen
