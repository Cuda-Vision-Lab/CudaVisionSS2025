import torch
import torch.nn as nn
"""
batch_size = 2
input_size = 3
hidden_size = 5

# Step 1: Create the LSTMCell
lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

# Step 2: Create input and initial states
x = torch.randn(batch_size, input_size)       # (2, 3)
h_0 = torch.zeros(batch_size, hidden_size)    # (2, 5)
c_0 = torch.zeros(batch_size, hidden_size)    # (2, 5)

# Step 3: Forward pass through the LSTM cell
h_1, c_1 = lstm_cell(x, (h_0, c_0))            # h_1, c_1: (2, 5)

print("Input x shape: ", x.shape)
print("Hidden state h_1 shape: ", h_1.shape)
print("Cell state c_1 shape: ", c_1.shape)

print("x: ", x)
print("h_1: ", h_1)
print("c_1: ", c_1)
"""

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear layer to compute all gates at once
        # Note: 4 * hidden_size for i, f, o, g (input, forget, output, candidate)
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: Input tensor of shape (batch, input_size)
            h_prev: Hidden state of shape (batch, hidden_size)
            c_prev: Cell state of shape (batch, hidden_size)
        Returns:
            h_next, c_next: Updated hidden and cell states
        """
        # Concatenate input and hidden state
        combined = torch.cat((x, h_prev), dim=1)  # Shape: (batch, input_size + hidden_size)
        
        # Compute all gates in one go
        gates = self.linear(combined)  # Shape: (batch, 4 * hidden_size)
        
        # Split into input, forget, output, and candidate gates
        i, f, o, g = gates.chunk(4, dim=1)  # Each shape: (batch, hidden_size)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Candidate cell state
        
        # Update cell state
        c_next = f * c_prev + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
# Parameters
batch_size = 5
input_size = 12
hidden_size = 20

# Random input and hidden states
x = torch.randn(batch_size, input_size)
h_prev = torch.randn(batch_size, hidden_size)
c_prev = torch.randn(batch_size, hidden_size)

# Initialize LSTM cell
lstm_cell = LSTMCell(input_size, hidden_size)

# Forward pass
h_next, c_next = lstm_cell(x, h_prev, c_prev)

print("h_next shape:", h_next.shape)  # Should be (batch_size, hidden_size)
print("c_next shape:", c_next.shape)  # Should be (batch_size, hidden_size)