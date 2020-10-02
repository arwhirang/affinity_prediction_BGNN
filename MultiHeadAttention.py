import torch
from torch import nn
import torch.nn.functional as F

cudaDevid = 2


def scaleDotProductAttention(q, k, v, mask = None):
    """
    :param q: Queries (256 batch, 8 d_k, 33 sequence, 64)
    :param k: Keys    (256, 8, 33, 64)
    :param v: Values  (256, 8, 33, 64)
    :param mask: mask (256, 1, 28) Source Mask
    :return: scaled dot attention: (256, 8, 33, 64)
    """
    dk = k.shape[-1]
    sqrt_dk = dk ** 0.5  # 8 = 64**0.5
    attn = (torch.matmul(q, k.transpose(-2, -1))) / sqrt_dk
    if mask is not None:
        mask = mask > 0#after torch 1.4, returns boolean type tensor
        attn = attn.masked_fill(~mask, -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    attn = F.softmax(attn, dim=-1) # (..., seq_len_q, seq_len_k)
    output = torch.matmul(attn, v)  #  (..., seq_len_q, depth_v) == (256, 8, 33, 64)
    return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % n_head == 0

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.dk = (embed_dim // n_head)#depth

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_f = nn.Linear(embed_dim, embed_dim, bias=False)  # Final linear layer

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.n_head, self.dk)
        k = self.linear_k(k).view(batch_size, -1, self.n_head, self.dk)
        v = self.linear_v(v).view(batch_size, -1, self.n_head, self.dk)
        q = q.transpose(1, 2)# (batch_size, num_heads, seq_len_q, depth)
        k = k.transpose(1, 2)# (batch_size, num_heads, seq_len_k, depth)
        v = v.transpose(1, 2)# (batch_size, num_heads, seq_len_v, depth)
        scaled_attention, attention_weights = scaleDotProductAttention(q, k, v, mask)
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # Final linear Layer
        output = self.linear_f(scaled_attention)
        return output, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model_, num_heads_, dff_, Drate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model_, num_heads_).cuda(cudaDevid)
        # self.ffn = point_wise_feed_forward_network(d_model_, dff_)
        # point_wise_feed_forward_network
        self.ffn = nn.Sequential(nn.Linear(d_model_, dff_),
                                 nn.LeakyReLU(),# (batch_size, seq_len, dff)
                                 nn.Linear(dff_, d_model_)# (batch_size, seq_len, d_model)
                                 )

        self.layernorm1 = nn.LayerNorm(d_model_)
        self.layernorm2 = nn.LayerNorm(d_model_)
        self.dropout1 = nn.Dropout(Drate)
        self.dropout2 = nn.Dropout(Drate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2
"""
class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff = 2048):
        super(PositionWiseFeedForward, self).__init__()


    def forward(self, x):
        #residual = x
        x = F.relu(self.w_1(x))
        #print(x.shape)
        x = self.w_2(x)
        return x# + residual
"""