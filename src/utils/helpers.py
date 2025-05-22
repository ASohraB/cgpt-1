import numpy as np
import torch
import math

import torch.nn as nn
#some functions borrowed from https://github.com/XinyuanLiao/ComplexNN, also https://github.com/wavefrontshaping/complexPyTorch

from torch.nn.functional import dropout, gelu, leaky_relu, elu, softmax
#from torch.nn import CosineSimilarity
from torch import relu, tanh, sigmoid


    
def cInp(inp):
    def cRandom():
        real = torch.randn(2, dtype=torch.float64)  # float64 for cdouble
        imag = torch.randn(2, dtype=torch.float64)
        cVectors = real + 1j * imag
        return cVectors / torch.norm(cVectors)
    return [(cRandom(), cRandom()) for _ in range(inp)]

# If class:
class complexInp(nn.Module):
    @staticmethod
    def forward(inp):
        return cInp(inp)
#not used
def ncInp(inp):
    def cRandom():
        real = torch.randn(2, dtype=torch.float64)  # float64 for cdouble
        imag = torch.randn(2, dtype=torch.float64)
        cVectors = real + 1j * imag
        cVectors = cVectors / torch.norm(cVectors)
        
        # Adding controlled noise
        noise_real = torch.normal(mean=0.0, std=0.2, size=(2,), dtype=torch.float64)
        noise_imag = torch.normal(mean=0.0, std=0.2, size=(2,), dtype=torch.float64)
        
        noisy_cVectors = cVectors + (noise_real + 1j * noise_imag)
        return noisy_cVectors / torch.norm(noisy_cVectors)  # Normalize again
    
    return [(cRandom(), cRandom()) for _ in range(inp)]
#not used
class noisycomplexInp(nn.Module):
    @staticmethod
    def forward(inp):
        return ncInp(inp)
    

def oncInp(tensor):
    def cRandom(vec):
        # Adding controlled noise
        noise_real = torch.normal(mean=0.0, std=0.2, size=vec.shape, dtype=torch.float64)
        noise_imag = torch.normal(mean=0.0, std=0.2, size=vec.shape, dtype=torch.float64)

        noisy_vec = vec + (noise_real + 1j * noise_imag)
        return noisy_vec / torch.norm(noisy_vec)  # Normalize again

    return cRandom(tensor)

class onenoisycomplexInp(nn.Module):
    @staticmethod
    def forward(inp):
        return oncInp(inp)
          
# Efficient implementation equivalent to the following:
def complex_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    """
    reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = complexSoftmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

####main

class cGelu(nn.Module):
    @staticmethod
    def forward(inp):
        return complexGelu(inp)

#not used
class cTanh(nn.Module):
    @staticmethod
    def forward(inp):
        return complexTanh(inp)

#not used
class cSigmoid(nn.Module):
    @staticmethod
    def forward(inp):
        return complexSigmoid(inp)
###

class cLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, elementwise_affine=False):
        super().__init__()
        self.real_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
        self.imag_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inp):
        real_input = torch.real(inp)
        imag_input = torch.imag(inp)

        real_output = self.real_norm(real_input)
        imag_output = self.imag_norm(imag_input)

        return torch.complex(real_output, imag_output)

#not used
class cDropout(nn.Module):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complexDropout(inp, self.p)
        else:
            return inp
        
class cLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, device=None, dtype=None):
        super().__init__()
        # Use torch.cdouble for double precision complex
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.cdouble))
        self.bias = nn.Parameter(torch.zeros(1, out_features, dtype=torch.cdouble), requires_grad=bias)

        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.xavier_uniform_(self.bias)
    def forward(self, inp):
        #if not inp.dtype == torch.cfloat:
            #inp = torch.complex(inp, torch.zeros_like(inp))
        if not torch.is_complex(inp):
            inp = torch.complex(inp, torch.zeros_like(inp))
        return torch.matmul(inp, self.weight.T) + self.bias


class cMLP(nn.Module):
    """
    Complex Multilayer Perceptron
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.input_layer = cLinear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([cLinear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = cLinear(hidden_size, output_size)
        self.activation = cGelu()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for i in range(self.num_layers - 1):
            x = self.activation(self.hidden_layers[i](x))
        output = self.output_layer(x)
        return output

#not used
def log(message):
    print(f"[LOG] {message}")

#not used
def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filepath)
    log(f"Checkpoint saved at {filepath}")

#not used
def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log(f"Checkpoint loaded from {filepath}, starting from epoch {epoch}")
    return epoch

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    log(f"Evaluation loss: {avg_loss}")
    return avg_loss

class cMultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.
    reference: https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html

    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = cLinear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = cLinear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = cLinear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = cLinear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = cLinear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:

        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    torch.matmul(query, q_weight.T) + q_bias,
                    torch.matmul(key, k_weight.T) + k_bias,
                    torch.matmul(value, v_weight.T) + v_bias,
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = complex_scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
    
    #not used
def complexDropout(inp, p=0.5, training=True):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp

#not used

def complex_relu(inp):
    return relu(inp.real).type(torch.complex64) + 1j * relu(inp.imag).type(
        torch.complex64
    )

#not used

def complex_sigmoid(inp):
    return sigmoid(inp.real).type(torch.complex64) + 1j * sigmoid(inp.imag).type(
        torch.complex64
    )

#not used

def complex_tanh(inp):
    return tanh(inp.real).type(torch.complex64) + 1j * tanh(inp.imag).type(
        torch.complex64
    )

#not used

def complex_opposite(inp):
    return -inp.real.type(torch.complex64) + 1j * (-inp.imag.type(torch.complex64))

#not used
def complexRelu(inp, **factory_kwargs):
    return torch.complex(relu(inp.real, **factory_kwargs), relu(inp.imag, **factory_kwargs))

#not used
def complexLeakyRelu(inp, **factory_kwargs):
    return torch.complex(leaky_relu(inp.real, **factory_kwargs), leaky_relu(inp.imag, **factory_kwargs))


def complexSoftmax(inp, **factory_kwargs):
    return torch.complex(softmax(inp.real, **factory_kwargs), softmax(inp.imag, **factory_kwargs))

#not used
def complexElu(inp, **factory_kwargs):
    return torch.complex(elu(inp.real, **factory_kwargs), elu(inp.imag, **factory_kwargs))


def complexGelu(inp, **factory_kwargs):
    return torch.complex(gelu(inp.real, **factory_kwargs), gelu(inp.imag, **factory_kwargs))

#not used
def complexTanh(inp, **factory_kwargs):
    return torch.complex(tanh(inp.real, **factory_kwargs), tanh(inp.imag, **factory_kwargs))

#not used
def complexSigmoid(inp, **factory_kwargs):
    return torch.complex(sigmoid(inp.real, **factory_kwargs), sigmoid(inp.imag, **factory_kwargs))
