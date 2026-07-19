"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    token_to_id = {spe:idx for idx,spe in enumerate(specials)}
    counter = len(specials)
    for sentence in sentences:
        words = sentence.split(' ')
        for word in words:
            if word not in token_to_id:
                token_to_id[word]=counter
                counter+=1
            else:
                continue
    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    id_to_token = {
    value:key  for key,value in token_to_id.items()
    }
    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    encodings = []
    if len(sentence)==0:
        return encodings
    words = sentence.split(' ')
    for word in words:
        if word not in token_to_id:
            encodings.append(token_to_id[unk_token])
        else:
            encodings.append(token_to_id[word])

    return encodings

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    tokens = []
    if not ids:
        return tokens 
    
    tokens = [id_to_token[idk] for idk in ids]
    return tokens

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    cur_len = len(ids)
    if max_len > cur_len:
        diff = max_len - cur_len 
        pad_el = [pad_id] * diff 
        ids.extend(pad_el)
        return ids

    elif max_len < cur_len:
        return ids[:max_len]

    elif max_len == cur_len:
        return ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    ten = torch.tensor(padded_sequences,dtype=torch.long)
    return ten

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return torch.mul(embeddings,d_model**0.5)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisor
    mid = ((d_model) // 2)
    i_s = [j for j in range(mid)]
    divisor = [10000**(-2*i/d_model) for i in i_s]
    divisor = torch.tensor(divisor,dtype=torch.float32)
    return divisor

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    position_idx = [[i] for i in range(max_len)]
    position_idx = torch.tensor(position_idx,dtype=torch.float32)
    return position_idx

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    cols = torch.arange(0,pe.size(1),2)
    pe[:,cols] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    cols = torch.arange(1,pe.size(1),2)
    pe[:,cols] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch


def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisor
    mid = ((d_model) // 2)
    i_s = [j for j in range(mid)]
    divisor = [10000**(-2*i/d_model) for i in i_s]
    divisor = torch.tensor(divisor,dtype=torch.float32)
    return divisor

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    position_idx = [[i] for i in range(max_len)]
    position_idx = torch.tensor(position_idx,dtype=torch.float32)
    return position_idx

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    cols = torch.arange(0,pe.size(1),2)
    pe[:,cols] = torch.sin(position * div_term)
    return pe

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    cols = torch.arange(1,pe.size(1),2)
    pe[:,cols] = torch.cos(position * div_term)
    return pe


def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    en_mat = torch.zeros((max_len,d_model),dtype=torch.float32)
    divisor = compute_positional_div_term(d_model)
    pidx = build_position_index_column(max_len)
     
    en_mat = fill_even_indices_with_sin(en_mat,pidx,divisor)
    en_mat = fill_odd_indices_with_cos(en_mat,pidx,divisor)
    return en_mat

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    B,L,d = embedded_batch.size()
    l_rows = positional_encoding[:L,:]
    return embedded_batch[:,:L,:] + l_rows

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    result = torch.isin(token_ids,pad_id,invert=True)
    result = result.unsqueeze(1).unsqueeze(1)
    return result

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    a = torch.ones((seq_len,seq_len))
    a = torch.tensor(a,dtype=torch.bool)
    a = torch.tril(a)
    a = a.unsqueeze(0).unsqueeze(0)
    return a

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).

    return causal_mask & padding_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    key = torch.transpose(key,-2,-1)
    prod = torch.matmul(query,key)
    return prod

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return torch.div(scores,torch.sqrt(torch.tensor(d_k)))

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill_(~mask,float("-inf"))

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf  
    all_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)  
    soft = F.softmax(masked_scores,dim=-1)
    soft = torch.where(all_masked,torch.zeros_like(soft),soft)
    return soft

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights,value)

# Step 22 - scaled_dot_product_attention
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    key = torch.transpose(key,-2,-1)
    prod = torch.matmul(query,key)
    return prod

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return torch.div(scores,torch.sqrt(torch.tensor(d_k)))

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill_(~mask,float("-inf"))

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf  
    all_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)  
    soft = F.softmax(masked_scores,dim=-1)
    soft = torch.where(all_masked,torch.zeros_like(soft),soft)
    return soft

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights,value)

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    d_k = key.size(-1)
    raw_scores = compute_raw_attention_scores(query,key)
    scaled_scores = scale_attention_scores(raw_scores,d_k)
    if mask is None:
        weights = softmax_attention_weights(scaled_scores)
    else:
        masked_scores = mask_attention_scores_with_neg_inf(scaled_scores,mask)
        weights = softmax_attention_weights(masked_scores)
    context = apply_attention_weights_to_values(weights,value)
    return (context,weights)

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B,L,d_model = tensor.size()
    return torch.reshape(tensor,(B,L,num_heads,d_model//num_heads))

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return torch.transpose(split_tensor,1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B,H,L,D = multi_head_tensor.size()
    tensor = torch.transpose(multi_head_tensor,2,1)
    tensor = torch.reshape(tensor,(B,L,H*D))
    return tensor

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is None:
        return x @ weight.T
    return x @ weight.T + bias

# Step 27 - project_to_query_key_value
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is None:
        return x @ weight.T
    return x @ weight.T + bias

def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    query  = apply_linear_projection(x,w_q,b_q)
    key  = apply_linear_projection(x,w_k,b_k)
    value = apply_linear_projection(x,w_v,b_v)
    return (query,key,value)

# Step 28 - split_qkv_into_heads
import torch


def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B,L,d_model = tensor.size()
    return torch.reshape(tensor,(B,L,num_heads,d_model//num_heads))

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return torch.transpose(split_tensor,1,2)


def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    query = transpose_heads_before_sequence(split_last_dim_into_heads(q,num_heads))
    key = transpose_heads_before_sequence(split_last_dim_into_heads(k,num_heads))
    value = transpose_heads_before_sequence(split_last_dim_into_heads(v,num_heads))
    return (query,key,value)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    key = torch.transpose(key,-2,-1)
    prod = torch.matmul(query,key)
    return prod

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return torch.div(scores,torch.sqrt(torch.tensor(d_k)))

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill_(~mask,float("-inf"))

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf  
    all_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)  
    soft = F.softmax(masked_scores,dim=-1)
    soft = torch.where(all_masked,torch.zeros_like(soft),soft)
    return soft

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights,value)

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    d_k = k_h.size(-1)
    raw_scores = compute_raw_attention_scores(q_h,k_h)
    scaled_scores = scale_attention_scores(raw_scores,d_k)
    if mask is None:
        weights = softmax_attention_weights(scaled_scores)
    else:
        masked_scores = mask_attention_scores_with_neg_inf(scaled_scores,mask)
        weights = softmax_attention_weights(masked_scores)
    context = apply_attention_weights_to_values(weights,v_h)
    return (context,weights)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B,H,L,D = multi_head_tensor.size()
    tensor = torch.transpose(multi_head_tensor,2,1)
    tensor = torch.reshape(tensor,(B,L,H*D))
    return tensor

def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is None:
        return x @ weight.T
    return x @ weight.T + bias

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    context = merge_heads_back_to_model_dim(context)
    context = apply_linear_projection(context,w_o,b_o)
    return context

# Step 31 - assemble_multi_head_attention_forward
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B,L,d_model = tensor.size()
    return torch.reshape(tensor,(B,L,num_heads,d_model//num_heads))

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return torch.transpose(split_tensor,1,2)


def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    query = transpose_heads_before_sequence(split_last_dim_into_heads(q,num_heads))
    key = transpose_heads_before_sequence(split_last_dim_into_heads(k,num_heads))
    value = transpose_heads_before_sequence(split_last_dim_into_heads(v,num_heads))
    return (query,key,value)


def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    key = torch.transpose(key,-2,-1)
    prod = torch.matmul(query,key)
    return prod

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return torch.div(scores,torch.sqrt(torch.tensor(d_k)))

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill_(~mask,float("-inf"))

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf  
    all_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)  
    soft = F.softmax(masked_scores,dim=-1)
    soft = torch.where(all_masked,torch.zeros_like(soft),soft)
    return soft

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights,value)

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    d_k = k_h.size(-1)
    raw_scores = compute_raw_attention_scores(q_h,k_h)
    scaled_scores = scale_attention_scores(raw_scores,d_k)
    if mask is None:
        weights = softmax_attention_weights(scaled_scores)
    else:
        masked_scores = mask_attention_scores_with_neg_inf(scaled_scores,mask)
        weights = softmax_attention_weights(masked_scores)
    context = apply_attention_weights_to_values(weights,v_h)
    return (context,weights)

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B,H,L,D = multi_head_tensor.size()
    tensor = torch.transpose(multi_head_tensor,2,1)
    tensor = torch.reshape(tensor,(B,L,H*D))
    return tensor

def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is None:
        return x @ weight.T
    return x @ weight.T + bias

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    context = merge_heads_back_to_model_dim(context)
    context = apply_linear_projection(context,w_o,b_o)
    return context

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q,k,v = split_qkv_into_heads(query,key,value,num_heads)
    context,_ = multi_head_scaled_dot_product_attention(q,k,v,mask)
    context = merge_heads_and_project_output(context,w_o,None)
    return context

# Step 32 - apply_ffn_first_linear_and_relu
import torch 
import torch.nn as nn 
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    temp_op = x @ w1 + b1
    relu = nn.ReLU()
    return relu(temp_op)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
import torch 
import torch.nn as nn 
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    relu = nn.ReLU()
    y1 = x@ w1 + b1 
    hidden = relu(y1)
    y2 = hidden @ w2 + b2
    return y2

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    D = x.size(-1)
    mean = torch.sum(x,dim=-1,keepdim=True) / D
    var =  torch.sum((x-mean)**2,dim=-1,keepdim=True) / D 
    return (mean,var)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def compute_layer_norm_mean_and_variance(x):
    var, mean = torch.var_mean(x, dim=-1, keepdim=True, unbiased=False)
    return mean, var

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, var = compute_layer_norm_mean_and_variance(x)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    y = gamma * x_hat + beta
    return y

# Step 37 - apply_residual_add_and_norm
import torch

def compute_layer_norm_mean_and_variance(x):
    var, mean = torch.var_mean(x, dim=-1, keepdim=True, unbiased=False)
    return mean, var

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, var = compute_layer_norm_mean_and_variance(x)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_hat + beta

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    residual_input = residual_input + sublayer_output 
    return normalize_and_scale_with_gamma_beta(residual_input,gamma,beta,eps)

# Step 38 - apply_dropout_with_keep_mask
import torch
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    masked = x * keep_mask.float()
    masked = masked * (1/keep_prob)
    return masked

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    context = assemble_multi_head_attention_forward(x,x,x,w_q,w_k,
    w_v,w_o,num_heads,src_mask)
    output = apply_residual_add_and_norm(
        x,context,gamma,beta,eps=1e-5
    )
    return output

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    out = position_wise_feed_forward_network(
        x,w1,b1,w2,b2
    )
    out = apply_residual_add_and_norm(
        x,out,gamma,beta,eps=1e-5
    )
    return out

# Step 41 - assemble_encoder_layer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    context = assemble_multi_head_attention_forward(x,x,x,w_q,w_k,
    w_v,w_o,num_heads,src_mask)
    output = apply_residual_add_and_norm(
        x,context,gamma,beta,eps=1e-5
    )
    return output

def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    out = position_wise_feed_forward_network(
        x,w1,b1,w2,b2
    )
    out = apply_residual_add_and_norm(
        x,out,gamma,beta,eps=1e-5
    )
    return out

def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    w_q = layer_params['w_q']
    w_k = layer_params['w_k']
    w_v = layer_params['w_v']
    w_o = layer_params['w_o']
    gamma = layer_params['attn_gamma']
    beta = layer_params['attn_beta']
    w1 = layer_params['w1']
    w2 = layer_params['w2']
    b1 = layer_params['b1']
    b2 = layer_params['b2']
    ffn_gamma = layer_params['ffn_gamma']
    ffn_beta = layer_params['ffn_beta']
    enc_out = encoder_layer_self_attention_sublayer(
        x,w_q,w_k,w_v,w_o,gamma,beta,num_heads,src_mask
    )

    out = encoder_layer_feed_forward_sublayer(
        enc_out,w1,b1,w2,b2,ffn_gamma,ffn_beta
    )
    return out

# Step 42 - stack_encoder_layers
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    context = assemble_multi_head_attention_forward(x,x,x,w_q,w_k,
    w_v,w_o,num_heads,src_mask)
    output = apply_residual_add_and_norm(
        x,context,gamma,beta,eps=1e-5
    )
    return output

def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    out = position_wise_feed_forward_network(
        x,w1,b1,w2,b2
    )
    out = apply_residual_add_and_norm(
        x,out,gamma,beta,eps=1e-5
    )
    return out

def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    w_q = layer_params['w_q']
    w_k = layer_params['w_k']
    w_v = layer_params['w_v']
    w_o = layer_params['w_o']
    gamma = layer_params['attn_gamma']
    beta = layer_params['attn_beta']
    w1 = layer_params['w1']
    w2 = layer_params['w2']
    b1 = layer_params['b1']
    b2 = layer_params['b2']
    ffn_gamma = layer_params['ffn_gamma']
    ffn_beta = layer_params['ffn_beta']
    enc_out = encoder_layer_self_attention_sublayer(
        x,w_q,w_k,w_v,w_o,gamma,beta,num_heads,src_mask
    )

    out = encoder_layer_feed_forward_sublayer(
        enc_out,w1,b1,w2,b2,ffn_gamma,ffn_beta
    )
    return out


def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    inp = x
    for layer_params in encoder_layer_params_list:
        h_i = assemble_encoder_layer(inp,layer_params,num_heads,src_mask)
        inp = h_i
    
    return inp

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    context = assemble_multi_head_attention_forward(
        y,y,y,w_q,w_k,w_v,w_o,num_heads,tgt_mask
    )
    output = apply_residual_add_and_norm(
        y,context,gamma,beta,eps=1e-5
    )
    return output

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None:
       src_mask= src_mask.unsqueeze(1).unsqueeze(2)
    context = assemble_multi_head_attention_forward(
        y,encoder_output,encoder_output,w_q,w_k,w_v,w_o,num_heads,src_mask
    )
    context = apply_residual_add_and_norm(y,context,gamma,beta,eps=1e-5)
    return context

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    y_out = position_wise_feed_forward_network(
        y,w1,b1,w2,b2
    )
    out = apply_residual_add_and_norm(
        y,y_out,gamma,beta,eps=1e-5
    )
    return out

# Step 46 - assemble_decoder_layer
def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    context = assemble_multi_head_attention_forward(
        y,y,y,w_q,w_k,w_v,w_o,num_heads,tgt_mask
    )
    output = apply_residual_add_and_norm(
        y,context,gamma,beta,eps=1e-5
    )
    return output

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k,
 w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None:
       src_mask= src_mask.unsqueeze(1).unsqueeze(2)
    context = assemble_multi_head_attention_forward(
        y,encoder_output,encoder_output,w_q,w_k,w_v,w_o,num_heads,src_mask
    )
    context = apply_residual_add_and_norm(y,context,gamma,beta,eps=1e-5)
    return context

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    y_out = position_wise_feed_forward_network(
        y,w1,b1,w2,b2
    )
    out = apply_residual_add_and_norm(
        y,y_out,gamma,beta,eps=1e-5
    )
    return out

def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # TODO: chain the three decoder sublayers using params from layer_params.
    w_q_self = layer_params['w_q_self'] 
    w_k_self = layer_params['w_k_self']
    w_v_self = layer_params['w_v_self']
    w_o_self = layer_params['w_o_self']
    gamma_self = layer_params['self_gamma']
    beta_self = layer_params['self_beta']
    w_q_cross = layer_params['w_q_cross']
    w_k_cross = layer_params['w_k_cross']
    w_v_cross = layer_params['w_v_cross']
    w_o_cross = layer_params['w_o_cross']
    gamma_cross = layer_params['cross_gamma']
    beta_cross = layer_params['cross_gamma']
    w1 = layer_params['w1']
    w2 = layer_params['w2']
    b1 = layer_params['b1']
    b2 = layer_params['b2']
    gamma = layer_params['ffn_gamma']
    beta = layer_params['ffn_beta']

    y = decoder_layer_cross_attention_sublayer(
        y,encoder_output,w_q_cross,w_k_cross,w_v_cross,w_o_cross,gamma_cross,beta_cross,
        num_heads,src_mask
    )


    y = decoder_layer_masked_self_attention_sublayer(
        y,w_q_self,w_k_self,w_v_self,w_o_self,gamma_self,beta_self,
        num_heads,tgt_mask
    )
    y = decoder_layer_feed_forward_sublayer(
        y,w1,b1,w2,b2,gamma,beta
    )
    return y

# Step 47 - stack_decoder_layers
def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    context = assemble_multi_head_attention_forward(
        y,y,y,w_q,w_k,w_v,w_o,num_heads,tgt_mask
    )
    output = apply_residual_add_and_norm(
        y,context,gamma,beta,eps=1e-5
    )
    return output

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k,
 w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None:
       src_mask= src_mask.unsqueeze(1).unsqueeze(2)
    context = assemble_multi_head_attention_forward(
        y,encoder_output,encoder_output,w_q,w_k,w_v,w_o,num_heads,src_mask
    )
    context = apply_residual_add_and_norm(y,context,gamma,beta,eps=1e-5)
    return context

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    y_out = position_wise_feed_forward_network(
        y,w1,b1,w2,b2
    )
    out = apply_residual_add_and_norm(
        y,y_out,gamma,beta,eps=1e-5
    )
    return out

def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # TODO: chain the three decoder sublayers using params from layer_params.
    w_q_self = layer_params['w_q_self'] 
    w_k_self = layer_params['w_k_self']
    w_v_self = layer_params['w_v_self']
    w_o_self = layer_params['w_o_self']
    gamma_self = layer_params['self_gamma']
    beta_self = layer_params['self_beta']
    w_q_cross = layer_params['w_q_cross']
    w_k_cross = layer_params['w_k_cross']
    w_v_cross = layer_params['w_v_cross']
    w_o_cross = layer_params['w_o_cross']
    gamma_cross = layer_params['cross_gamma']
    beta_cross = layer_params['cross_gamma']
    w1 = layer_params['w1']
    w2 = layer_params['w2']
    b1 = layer_params['b1']
    b2 = layer_params['b2']
    gamma = layer_params['ffn_gamma']
    beta = layer_params['ffn_beta']

    y = decoder_layer_cross_attention_sublayer(
        y,encoder_output,w_q_cross,w_k_cross,w_v_cross,w_o_cross,gamma_cross,beta_cross,
        num_heads,src_mask
    )


    y = decoder_layer_masked_self_attention_sublayer(
        y,w_q_self,w_k_self,w_v_self,w_o_self,gamma_self,beta_self,
        num_heads,tgt_mask
    )
    y = decoder_layer_feed_forward_sublayer(
        y,w1,b1,w2,b2,gamma,beta
    )
    return y

def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    
    for layer_param in decoder_layer_params_list:
        y = assemble_decoder_layer(y,encoder_output,layer_param,num_heads,src_mask,tgt_mask)
         
    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    out = apply_linear_projection(
        decoder_output,output_projection_weight,output_projection_bias
    )
    return out

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return torch.transpose(token_embedding_weight,1,0)

# Step 50 - apply_log_softmax_over_vocab
import torch 
import torch.nn as nn
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    logsoftmax = nn.LogSoftmax(dim=-1)
    return logsoftmax(logits)

# Step 51 - run_transformer_forward
def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    return torch.mul(embeddings,d_model**0.5)

def compute_positional_div_term(d_model):
    mid = ((d_model) // 2)
    i_s = [j for j in range(mid)]
    divisor = [10000**(-2*i/d_model) for i in i_s]
    divisor = torch.tensor(divisor,dtype=torch.float32)
    return divisor

def build_position_index_column(max_len):
    position_idx = [[i] for i in range(max_len)]
    position_idx = torch.tensor(position_idx,dtype=torch.float32)
    return position_idx

def fill_even_indices_with_sin(pe, position, div_term):
    cols = torch.arange(0,pe.size(1),2)
    pe[:,cols] = torch.sin(position * div_term)
    return pe

def fill_odd_indices_with_cos(pe, position, div_term):
    cols = torch.arange(1,pe.size(1),2)
    pe[:,cols] = torch.cos(position * div_term)
    return pe

def build_sinusoidal_positional_encoding(max_len, d_model):
    en_mat = torch.zeros((max_len,d_model),dtype=torch.float32)
    divisor = compute_positional_div_term(d_model)
    pidx = build_position_index_column(max_len)
    en_mat = fill_even_indices_with_sin(en_mat,pidx,divisor)
    en_mat = fill_odd_indices_with_cos(en_mat,pidx,divisor)
    return en_mat

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    B,L,d = embedded_batch.size()
    l_rows = positional_encoding[:L,:]
    return embedded_batch[:,:L,:] + l_rows

def build_padding_mask(token_ids, pad_id):
    result = torch.isin(token_ids,pad_id,invert=True)
    result = result.unsqueeze(1).unsqueeze(1)
    return result

def build_causal_mask(seq_len):
    a = torch.ones((seq_len,seq_len))
    a = torch.tensor(a,dtype=torch.bool)
    a = torch.tril(a)
    a = a.unsqueeze(0).unsqueeze(0)
    return a

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    return causal_mask & padding_mask

def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    out = apply_linear_projection(
        decoder_output,output_projection_weight,output_projection_bias
    )
    return out

def apply_log_softmax_over_vocab(logits):
    logsoftmax = nn.LogSoftmax(dim=-1)
    return logsoftmax(logits)

def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    token_embedding = model_params['token_embedding']
    encoder_layers = model_params['encoder_layers']
    decoder_layers = model_params['decoder_layers']
    output_projection = model_params['output_projection']
    #pe
    d_model = token_embedding.size(1)
    src_len = src_ids.size(1)
    tgt_len = tgt_ids.size(1)
    src_pe = build_sinusoidal_positional_encoding(src_len,d_model)
    tgt_pe = build_sinusoidal_positional_encoding(tgt_len,d_model)
    src_emb = scale_embeddings_by_sqrt_d_model(token_embedding[src_ids],d_model)
    tgt_emb = scale_embeddings_by_sqrt_d_model(token_embedding[tgt_ids],d_model)
    
    src_emb = add_positional_encoding_to_embeddings(src_emb,src_pe)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb,tgt_pe)
    #mask
    src_mask = build_padding_mask(src_ids,pad_id)
    tgt = build_padding_mask(tgt_ids,pad_id)
    causal_mask = build_causal_mask(tgt_emb.size(1))
    tgt_mask = combine_padding_and_causal_masks(tgt,causal_mask)
    #run encoder/decoder stacks

    encoder_output = stack_encoder_layers(
        src_emb,encoder_layers,num_heads,src_mask
    )
    decoder_output = stack_decoder_layers(
        tgt_emb,encoder_output,decoder_layers,num_heads,tgt_mask,src_mask
    )
    out = apply_final_output_projection(decoder_output,
    output_projection,None)
    out = apply_log_softmax_over_vocab(out)
    return out

# Step 52 - init_encoder_layer_parameters
import torch
import math



def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    D = d_model
    F = d_ff
    w_q = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_v = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_k = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_o = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w1 = torch.rand(D,F,dtype=torch.float32,requires_grad=True)
    w2 = torch.rand(F,D,dtype=torch.float32,requires_grad=True)
    b1 = torch.zeros(F,dtype=torch.float32,requires_grad=True)
    b2 = torch.zeros(D,dtype=torch.float32,requires_grad=True)
    attn_gamma = torch.ones(D,dtype=torch.float32,requires_grad=True)
    att_beta = torch.zeros(D,dtype=torch.float32,requires_grad=True)
    ffn_gamma = torch.ones(D,dtype=torch.float32,requires_grad=True)
    ffn_beta = torch.zeros(D,dtype=torch.float32,requires_grad=True)
    return {
        'w_q':w_q,'w_k':w_k,'w_v':w_v,
        'w_o':w_o,'w1':w1,'w2':w2,
        'b1':b1,'b2':b2,
        'attn_gamma':attn_gamma,
        'attn_beta':att_beta,
        'ffn_gamma':ffn_gamma,
        'ffn_beta':ffn_beta
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):

    D = d_model
    F = d_ff
    # TODO: return a dict of requires_grad tensors for one decoder layer
    w_q_self = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_v_self = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_k_self = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_o_self = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    
    w_q_cross = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_v_cross = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_k_cross = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    w_o_cross = torch.randn(D,D,dtype=torch.float32,requires_grad=True)
    


    w1 = torch.rand(D,F,dtype=torch.float32,requires_grad=True)
    w2 = torch.rand(F,D,dtype=torch.float32,requires_grad=True)
    b1 = torch.zeros(F,dtype=torch.float32,requires_grad=True)
    b2 = torch.zeros(D,dtype=torch.float32,requires_grad=True)
    
    

    self_gamma = torch.ones(D,dtype=torch.float32,requires_grad=True)
    self_beta = torch.zeros(D,dtype=torch.float32,requires_grad=True)
    
    cross_gamma = torch.ones(D,dtype=torch.float32,requires_grad=True)
    cross_beta = torch.zeros(D,dtype=torch.float32,requires_grad=True)
    
    
    ffn_gamma = torch.ones(D,dtype=torch.float32,requires_grad=True)
    ffn_beta = torch.zeros(D,dtype=torch.float32,requires_grad=True)
    return {
        'w_q_self':w_q_self,
        'w_v_self':w_v_self,
        'w_k_self':w_k_self,
        'w_o_self':w_o_self,
        'w_q_cross':w_q_cross,
        'w_v_cross':w_v_cross,
        'w_k_cross':w_k_cross,
        'w_o_cross':w_o_cross,
        'self_gamma':self_gamma,
        'self_beta':self_beta,
        'cross_gamma':cross_gamma,
        'cross_beta':cross_beta,
        'ffn_gamma':ffn_gamma,
        'ffn_beta':ffn_beta,
        'w1':w1,
        'w2':w2,
        'b1':b1,
        'b2':b2
    }

# Step 54 - init_embedding_and_projection_parameters
import torch



def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    src_embedding = torch.randn(vocab_size,d_model,dtype=torch.float32,requires_grad=True)
    tgt_embedding = torch.randn(vocab_size,d_model,dtype=torch.float32,requires_grad=True)

    if tie_weights:
        output_projection = tgt_embedding      # SAME object
    else:
        output_projection = torch.randn(
            vocab_size, d_model,
            dtype=torch.float32,
            requires_grad=True
        )
    return {
        'src_embedding':src_embedding,
        'tgt_embedding':tgt_embedding,
        'output_projection':output_projection
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    seen = set()
    info = []
    for layer in encoder_layer_params:
        for _,value in layer.items():
            if id(value) not in seen:
                seen.add(id(value))
                info.append(value)
    for layer in decoder_layer_params:
        for _,value in layer.items():
            if id(value) not in seen:
                seen.add(id(value))
                info.append(value)
    for _,value in embedding_params.items():
        if id(value) not in seen:
            seen.add(id(value))
            info.append(value)
    return info

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    new_shifted = torch.empty_like(target_ids)
    new_shifted[:,1:] = target_ids[:,:-1]
    new_shifted[:,0] = start_token_id 
    return new_shifted

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    lr = d_model**(-0.5) * min(step**(-0.5),step*(warmup_steps**(-1.5)))
    return lr

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    value = epsilon / (vocab_size - 2)
    tensor = torch.ones(shape,dtype=torch.float32)
    tensor = tensor * value
    return tensor

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    smooth_tensor = smoothed_distribution.clone()
    gold = gold_token_ids.unsqueeze(-1)
    smooth_tensor = smooth_tensor.scatter_(
        2,gold,confidence
    )
    return smooth_tensor

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    result = smoothed_distribution.clone()
    result[:,:,pad_id] = 0 

    mask = (gold_token_ids == pad_id)

    result[mask] = 0 
    return result

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    loss = log_probabilities * smoothed_distribution 
    loss = - torch.sum(loss)
    if loss == 0:
        loss = loss.abs()
    return loss

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    mask = (gold_token_ids == pad_id)
    counts = torch.sum(~mask).item()
    return total_loss / max(counts,1)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    argmaxed = torch.argmax(log_probabilities,dim=2)
    mask = (gold_token_ids!=pad_id)
    counts = mask.sum()
    correct = ((argmaxed==gold_token_ids)&mask).sum()
    return correct / max(1,counts)

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    t = 0
    m = [torch.zeros_like(p,requires_grad=False) for p in parameter_list]
    v = [torch.zeros_like(p,requires_grad=False) for p in parameter_list]
    return {
        'm':m,
        'v':v,
        't':t
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    return beta1 * m_prev + (1- beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    return beta2 * v_prev + (1 - beta2) * grad **2

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    mhat = m_t/ (1 - beta1**step)
    vhat = v_t / (1 - beta2**step)
    return (mhat,vhat)

# Step 69 - apply_adam_step_to_all_parameters
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    return beta1 * m_prev + (1- beta1) * grad

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    return beta2 * v_prev + (1 - beta2) * grad **2

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    mhat = m_t/ (1 - beta1**step)
    vhat = v_t / (1 - beta2**step)
    return (mhat,vhat)


def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
  
    optimizer_state['t'] += 1
    with torch.no_grad():
        for i , param in enumerate(parameter_list):
            if param.grad is None:
                continue
            
            grad = param.grad 
            m = update_adam_first_moment(
                optimizer_state['m'][i],grad,beta1
            )
            optimizer_state['m'][i] = m
            v = update_adam_second_moment(
                optimizer_state['v'][i],grad,beta2
            )
            optimizer_state['v'][i] = v
            m_h,v_h = apply_adam_bias_correction(m,v,beta1,beta2,optimizer_state['t'])
            param -= learning_rate * m_h / (torch.sqrt(v_h)+epsilon)
        
    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for i, param in enumerate(parameter_list):
        if param is None:
            continue
        param.grad = None

# Step 71 - compute_batch_training_loss
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    new_shifted = torch.empty_like(target_ids)
    new_shifted[:,1:] = target_ids[:,:-1]
    new_shifted[:,0] = start_token_id 
    return new_shifted

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    value = epsilon / (vocab_size - 2)
    tensor = torch.ones(shape,dtype=torch.float32)
    tensor = tensor * value
    return tensor

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    loss = log_probabilities * smoothed_distribution 
    loss = - torch.sum(loss)
    if loss == 0:
        loss = loss.abs()
    return loss 

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    mask = (gold_token_ids == pad_id)
    counts = torch.sum(~mask).item()
    return total_loss / max(counts,1)

def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    pad_id = config['pad_id']
    start_id = config['start_id']
    vocab_size = config['vocab_size']
    smoothing = config['smoothing']
    num_heads = config['num_heads']
    tgt_shifted = shift_targets_right_with_start_token(
        tgt_batch,start_id
    )
    tf_op = run_transformer_forward(src_batch, 
                tgt_shifted, model_params, num_heads, pad_id)
    sm_tgt = build_uniform_smoothing_distribution(tf_op,vocab_size,1e-6)
    total_loss = compute_label_smoothed_kl_loss(tf_op,sm_tgt)
    avg_loss = average_loss_over_non_pad_tokens(total_loss,sm_tgt,pad_id)
    return avg_loss

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

