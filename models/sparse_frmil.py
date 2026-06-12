import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseMAB(nn.Module):
    """
    稀疏多头注意力机制
    """
    def __init__(self, dim_Q, dim_V, num_heads, sparsity_ratio=0.1, ln=False):
        super(SparseMAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.sparsity_ratio = sparsity_ratio
        
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_Q, dim_V)
        self.fc_v = nn.Linear(dim_Q, dim_V)
        
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        
        self.fc_o = nn.Linear(dim_V, dim_V)
        
    def sparsify_attention(self, attention_scores, k_ratio=None):
        """
        attention_scores: [B*H, Lq, Lk]
        返回：按行Top-k稀疏softmax后的注意力 [B*H, Lq, Lk]
        """
        bh, Lq, Lk = attention_scores.shape
        k_ratio = self.sparsity_ratio if k_ratio is None else k_ratio
        k = max(1, int(Lk * k_ratio))

        # 取每行Top-k
        topk_vals, topk_idx = torch.topk(attention_scores, k, dim=-1)  # [bh, Lq, k]

        # 构造全为 -inf 的张量，然后把Top-k位置填回原值
        sparse_scores = attention_scores.new_full((bh, Lq, Lk), float('-inf'))

        # 批量/行索引
        bh_arange = torch.arange(bh, device=attention_scores.device).view(bh, 1, 1).expand(bh, Lq, k)
        q_arange  = torch.arange(Lq, device=attention_scores.device).view(1, Lq, 1).expand(bh, Lq, k)

        sparse_scores[bh_arange, q_arange, topk_idx] = topk_vals  # 仅Top-k处有真实分数，其余为 -inf
        # 归一化：非Top-k位置的exp(-inf)=0
        sparse_attn = F.softmax(sparse_scores, dim=-1)
        return sparse_attn

    
    def forward(self, Q, K, inst_mode=False):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        d_head = self.dim_V // self.num_heads

        def split_heads(x):  # [B, L, V] -> [B*H, L, d_head]
            return torch.cat(x.split(d_head, dim=2), dim=0)
        
        Q_ = split_heads(Q)
        K_ = split_heads(K)
        V_ = split_heads(V)
        
        # 计算注意力分数
        attention_scores = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(d_head)
        
        # 稀疏化注意力
        sparse_attention = self.sparsify_attention(attention_scores)
        O_ = Q_ + sparse_attention.bmm(V_)               # 残差

        # 拼回多头：把 [B*H, L, d_head] 还原为 [B, L, V]
        B = Q.size(0)
        H = self.num_heads
        L = Q.size(1)
        O = O_.view(B, H, L, d_head).transpose(1, 2).contiguous().view(B, L, H * d_head)

        # MAB 的 FFN + 残差 + LN
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        # 若外部保证 L==1 才 squeeze；否则直接返回 [B, L, V]
        return O if inst_mode else (O if O.size(1) != 1 else O.squeeze(1))

class SparseFRMIL(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.data_name = args.model_ext
        self.num_outputs = args.num_class
        dim_hidden = args.num_feats
        num_heads = args.n_heads
        self.k = 1
        
        # 编码器
        self.enc = nn.Sequential(
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )
        
        # 稀疏注意力机制
        self.sparse_att = SparseMAB(dim_hidden, dim_hidden, num_heads, sparsity_ratio=0.1)
        
        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, self.num_outputs)
        )
        
        self.mode = 0
        
    def recalib(self, inputs, option='max'):
        A1 = []
        Q = []
        bs = inputs.shape[0]
        
        if option == 'mean':
            Q = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):
                a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                _, m_indices = torch.sort(a1, 0, descending=True)
                
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(self.k):
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
                    else:
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[len_i - i_q, :])
                    feat_q.append(feats)
                
                feats = torch.stack(feat_q)
                A1.append(a1.squeeze(1))
                Q.append(feats.mean(0))
            
            A1 = torch.stack(A1)
            Q = torch.stack(Q)
            return A1, Q
    
    def forward(self, inputs):
        if self.mode == 1:
            return self.sparse_att(inputs, inputs, True)
        
        A1, Q = self.recalib(inputs, 'max')
        
        # 特征偏移
        if self.data_name == 'msi':
            i_shift = inputs
        else:
            inputs = F.relu(inputs - Q)
            i_shift = inputs
        
        # 稀疏注意力聚合
        bag = self.sparse_att(Q, inputs)
        out = self.fc(bag)
        
        if self.training:
            return out, i_shift, A1
        else:
            return out 