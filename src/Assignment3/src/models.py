import torch
import torch.nn as nn

class OwnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OwnLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier weight initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

        # Forget Gate
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        init_weights(self.forget_gate)

        # Input Gate
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        init_weights(self.input_gate)

        # Candidate Gate
        self.candidate_gate = nn.Linear(input_size + hidden_size, hidden_size)
        init_weights(self.candidate_gate)

        # Output Gate
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        init_weights(self.output_gate)

        # Final Layer
        self.final_layer = nn.Linear(hidden_size, output_size)
        init_weights(self.final_layer)

    def reset_hidden(self, batch_size, device):
        """Initialize hidden state and cell state to zeros"""
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))

    def forward(self, x):
        """
        Forward pass through the LSTM
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size] or [batch_size, input_size]
        Returns:
            outputs: Final output or all outputs depending on input shape
        """
        # Handle different input shapes
        if len(x.size()) == 2:  # [batch_size, input_size]
            batch_size, _ = x.size()
            seq_len = 1
            x = x.unsqueeze(1)  # Convert to [batch_size, 1, input_size]
        elif len(x.size()) == 3:  # [batch_size, seq_len, input_size]
            batch_size, seq_len, _ = x.size()
        else:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got {len(x.size())}")
            
        device = x.device
        
        # Initialize hidden state and cell state
        h_t, c_t = self.reset_hidden(batch_size, device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Concatenate input with previous hidden state
            combined = torch.cat((x_t, h_t), dim=1)
            
            f_t = torch.sigmoid(self.forget_gate(combined))
            i_t = torch.sigmoid(self.input_gate(combined))
            c_tilde = torch.tanh(self.candidate_gate(combined))
            o_t = torch.sigmoid(self.output_gate(combined))
            
            # Update cell state
            c_t = f_t * c_t + i_t * c_tilde
            
            # Update hidden state
            h_t = o_t * torch.tanh(c_t)
            
            output = self.final_layer(h_t)
            outputs.append(output)
        

        if len(x.size()) == 3 and seq_len == 1:
            return outputs[0]  # Return [batch_size, output_size]
            
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, output_size]
        
        return outputs
