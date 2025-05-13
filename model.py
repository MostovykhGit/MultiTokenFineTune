import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch import nn

class PredictorHead(torch.nn.Module):
    def __init__(self, config):
        super(PredictorHead, self).__init__()
        self.decoder = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            activation="gelu",
            batch_first=True
        )
        self.proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def generate_causal_mask(self, size, device):
        # Upper-triangular mask filled with -inf above the diagonal
        return torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)
    
    def forward(self, memory, tgt=None):
        if tgt is None:
            tgt = memory  # If no separate tgt, use memory for both

        seq_len = tgt.size(1)
        device = tgt.device
        causal_mask = self.generate_causal_mask(seq_len, device)

        # Apply decoder with causal mask
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        logits = self.proj(out)
        return logits

class TwoHeadModel(torch.nn.Module):
    def __init__(self, base_model, speculator_head):
        super(TwoHeadModel, self).__init__()
        self.base_model = base_model
        self.main_head = base_model.lm_head
        self.speculator_head = speculator_head

    def forward(self, x, attention_mask=None):
        # Get the output from the base model, including position embeddings (if available)
        hidden_states = self.base_model.model(x, attention_mask=attention_mask).last_hidden_state
        out1 = self.main_head(hidden_states)

        out2 = self.speculator_head(hidden_states)
        return out1, out2