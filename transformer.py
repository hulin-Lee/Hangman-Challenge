from torch import nn
from torch.nn.functional import cross_entropy, softmax, relu
import torch
import numpy as np

class MultiHead(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head  
        self.n_head = n_head 
        self.model_dim = model_dim  # equal to emb_dim
        # Linear transformations for queries, keys, and values
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)
        # Output transformation
        self.o_dense = nn.Linear(model_dim, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.attention = None

    def forward(self,q,k,v,mask,training):
        # residual connect
        residual = q
        dim_per_head= self.head_dim
        num_heads = self.n_head
        batch_size = q.size(0)

        # linear projection
        key = self.wk(k)    # [batch_size, seq_len, num_heads * head_dim]
        value = self.wv(v)  # [batch_size, seq_len, num_heads * head_dim]
        query = self.wq(q)  # [batch_size, seq_len, num_heads * head_dim]

        # split by head
        query = self.split_heads(query)       # [batch_size, n_head, seq_len, head_dim]
        key = self.split_heads(key)
        value = self.split_heads(value) 
        context = self.scaled_dot_product_attention(query,key, value, mask)    # [batch_size, seq_len, model_dim]
        o = self.o_dense(context)  # [batch_size, seq_len, model_dim]
        o = self.o_drop(o)

        o = self.layer_norm(residual+o)
        return o

    def split_heads(self, x):
        x = torch.reshape(x,(x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0,2,1,3)  # [batch_size, n_head, seq_len, head_dim]
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = torch.tensor(k.shape[-1]).type(torch.float)
        score = torch.matmul(q,k.permute(0,1,3,2)) / (torch.sqrt(dk) + 1e-8)    # [batch_size, n_head, seq_len, seq_len]
        if mask is not None:
            # change the value at masked position to negative infinity,
            # so the attention score at these positions after softmax will close to 0.
            # mask: batch_size, 1, seq_len, seq_len, or batch_size, 1, 1, seq_len
            score = score.masked_fill_(mask,-np.inf)
        self.attention = softmax(score,dim=-1)
        context = torch.matmul(self.attention,v)    # [batch_size, n_head, seq_len, head_dim]
        context = context.permute(0,2,1,3)          # [batch_size, seq_len, n_head, head_dim]
        context = context.reshape((context.shape[0], context.shape[1],-1))  
        return context  # [batch_size, seq_len, model_dim]

class PositionWiseFFN(nn.Module):
    def __init__(self,model_dim, dropout = 0.0):
        super().__init__()
        dff = model_dim*4
        self.l = nn.Linear(model_dim,dff)
        self.o = nn.Linear(dff,model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self,x):
        o = relu(self.l(x))
        o = self.o(o)
        o = self.dropout(o)

        o = self.layer_norm(x + o)
        return o     # [batch_size, seq_len, model_dim]



class EncoderLayer(nn.Module):

    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        self.mh = MultiHead(n_head, emb_dim, drop_rate)
        self.ffn = PositionWiseFFN(emb_dim,drop_rate)
    
    def forward(self, xz, training, mask):
        # xz: [batch_size, seq_len, model_dim]
        context = self.mh(xz, xz, xz, mask, training)   # [batch_size, seq_len, model_dim]
        o = self.ffn(context)
        return o

class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate, n_layer):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(n_head, emb_dim, drop_rate) for _ in range(n_layer)]
        )    
    def forward(self, xz, training, mask):

        for encoder in self.encoder_layers:
            xz = encoder(xz,training,mask)
        return xz       # [batch_size, seq_len, model_dim]

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab):
        super().__init__()
        pos = np.expand_dims(np.arange(max_len),1)  # [max_len, 1]
        pe = pos / np.power(1000, 2*np.expand_dims(np.arange(emb_dim)//2,0)/emb_dim)  # [max_len, emb_dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe,0) # [1, max_len, emb_dim]
        self.pe = torch.from_numpy(pe).type(torch.float32)
        self.embeddings = nn.Embedding(n_vocab,emb_dim)
        self.embeddings.weight.data.normal_(0,0.1)
        
    def forward(self, x):
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)    
        x_embed = self.embeddings(x) + self.pe  # [batch_size, max_len, emb_dim]
        return x_embed  # [batch_size, max_len, emb_dim]