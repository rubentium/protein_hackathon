import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PositionalEmbedding(torch.nn.Module):
    """Learned positional embeddings like in NanoGPT"""
    def __init__(self, max_seq_len, hidden_size):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        # Initialize like NanoGPT
        torch.nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, seq_len, device):
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        return self.pos_embed(pos)  # shape (seq_len, hidden_size)

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, sigma, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(sigma)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        # Split Q, K, V like in NanoGPT
        q, k, v = qkv.unbind(dim=-2)  # Each: [b, s, h, d]
        
        # Reshape for attention: [b, h, s, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)
        
        # Use PyTorch's scaled dot product attention (Flash Attention if available)
        with torch.cuda.amp.autocast(enabled=False):
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False
            )
        
        # Reshape back: [b, h, s, d] -> [b, s, h*d]
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        x = bias_dropout_scale_fn(self.attn_out(y), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x



class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SEDD(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, vocab_size):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        vocab_size = vocab_size + 1 # absorbing state

        self.vocab_embed = EmbeddingLayer(config['model']['hidden_size'], vocab_size)
        self.sigma_map = TimestepEmbedder(config['model']['cond_dim'])
        
        # Replace rotary embedding with positional embedding
        # Assuming max sequence length from config or default to 1024
        max_seq_len = config['model'].get('max_seq_len', 1024)
        self.pos_embed = PositionalEmbedding(max_seq_len, config['model']['hidden_size'])

        self.blocks = nn.ModuleList([
            DDiTBlock(config['model']['hidden_size'], config['model']['n_heads'], config['model']['cond_dim'], dropout=config['model']['dropout']) for _ in range(config['model']['n_blocks'])
        ])

        self.output_layer = DDitFinalLayer(config['model']['hidden_size'], vocab_size, config['model']['cond_dim'])

    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, indices, sigma):
        # Compute embeddings
        x = self.vocab_embed(indices)  # token embeddings
        
        if x.dim() == 4:  # [b, h, l, emb]
            seq_len = indices.shape[2]  # l dimension is at index 2
            pos_emb = self.pos_embed(seq_len, indices.device)  # [l, emb]
            # Broadcast to match 4D tensor: [1, 1, l, emb]
            pos_emb = pos_emb[None, None, :, :]
        else:  # [b, l, emb]
            seq_len = indices.shape[1]  # l dimension is at index 1
            pos_emb = self.pos_embed(seq_len, indices.device)  # [l, emb]
            # Broadcast to match 3D tensor: [1, l, emb]
            pos_emb = pos_emb[None, :, :]
        x = x + pos_emb  # Add positional embeddings to token embeddings
        
        c = F.silu(self.sigma_map(sigma))

        # Run transformer blocks (no more rotary embeddings needed)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, c, seqlens=None)

            x = self.output_layer(x, c)

        # Not sure exactly what this is doing...but has to do with scale_by_sigma and absorbing state
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]

        x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0

        # Put the absorbing state in the last position (zeros)
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        # Shape: [batch_size, seq_len, vocab_size]
        return x

def score_fn(model, x, sigma, train=True, sampling=False):
    sigma = sigma.reshape(-1)
    
    if train:
        model.train()
    else:
        model.eval()

    score = model(x, sigma)
    
    if sampling:
        # when sampling return true score (not log used for training)
        return score.exp()

    return score