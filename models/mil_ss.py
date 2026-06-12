import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  
from timm.models.layers import DropPath

class TopKSelector(nn.Module):
    def __init__(self, input_dim, keep_num=256):
        super().__init__()
        self.keep_num = keep_num
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(), 
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        B, N, D = x.shape
        raw_scores = self.scorer(x) # [B, N, 1]
        
        gate = torch.sigmoid(raw_scores) 
        
        k = min(self.keep_num, N)
        topk_values, topk_indices = torch.topk(gate.squeeze(-1), k, dim=1)
        
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        x_selected = torch.gather(x, 1, idx_expanded)
        
        gate_weights = topk_values.unsqueeze(-1)
        x_selected = x_selected * gate_weights
        
        return x_selected, gate.squeeze(-1), topk_indices

class StablePVMLayer(nn.Module):
    def __init__(self, input_dim, d_state=16, d_conv=4, expand=2, drop_path=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        
        self.mamba_fwd = Mamba(
            d_model=input_dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.mamba_bwd = Mamba(
            d_model=input_dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        
        self.proj = nn.Linear(input_dim * 2, input_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, L, C]
        
        x_norm = self.norm(x_flat)
        
        out_fwd = self.mamba_fwd(x_norm)
        out_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.proj(out) # [B, L, C]
        
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return x + self.drop_path(out)

class StableMAB(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, Q, K, V, key_padding_mask=None, need_weights=False):
        Q_norm = self.norm1(Q)
        K_norm = self.norm1(K)
        V_norm = self.norm1(V)
        
        attn_out, attn_weights = self.mha(Q_norm, K_norm, V_norm, key_padding_mask=key_padding_mask, need_weights=need_weights)
        x = Q + attn_out
        
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights

class TopoMIL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_outputs = getattr(args, "num_class", 2)
        dim_hidden = getattr(args, "num_feats", 512)
        self.grid_size = getattr(args, "grid_size", 20) 
        self.num_keep = self.grid_size * self.grid_size 
        
        self.selector = TopKSelector(dim_hidden, keep_num=self.num_keep)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.conv_head = nn.Conv2d(dim_hidden, dim_hidden, 3, 1, 1, groups=dim_hidden)
        self.down1 = StablePVMLayer(dim_hidden, drop_path=0.1, d_state=32, expand=4, d_conv=3)
        self.down2 = StablePVMLayer(dim_hidden, drop_path=0.1, d_state=32, expand=4, d_conv=3)
        self.up1 = StablePVMLayer(dim_hidden, drop_path=0.1, d_state=32, expand=4, d_conv=3)
        self.aggregator = StableMAB(dim_hidden, num_heads=8, dropout=0.2)
        self.norm_final = nn.LayerNorm(dim_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim_hidden // 2, self.num_outputs)
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_keep, dim_hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, inputs, return_attn=False):
        B, N, D = inputs.shape

        x_sel, all_patch_scores, topk_indices = self.selector(inputs) 
        x_sel = x_sel + self.pos_embed[:, :x_sel.size(1), :]
        
        actual_k = x_sel.shape[1]
        if actual_k < self.num_keep:
            pad_len = self.num_keep - actual_k
            zeros = torch.zeros(B, pad_len, D, device=x_sel.device, dtype=x_sel.dtype)
            x_grid = torch.cat([x_sel, zeros], dim=1)
            mask = torch.cat([
                torch.zeros(B, actual_k, device=x_sel.device),
                torch.ones(B, pad_len, device=x_sel.device)
            ], dim=1).bool()
        else:
            x_grid = x_sel
            mask = torch.zeros(B, self.num_keep, device=x_sel.device).bool()
            
        x_img = x_grid.transpose(1, 2).view(B, D, self.grid_size, self.grid_size)
        x_img = self.conv_head(x_img) + x_img
        
        t0 = self.down1(x_img)
        d1 = F.max_pool2d(t0, kernel_size=2, stride=2)
        t1 = self.down2(d1)
        u1 = F.interpolate(t1, size=t0.shape[2:], mode='bilinear', align_corners=True)
        f1 = self.up1(u1 + t0)
        x_tokens = f1.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls_tokens, x_tokens], dim=1) 
        
        cls_mask = torch.zeros(B, 1, device=x_sel.device).bool()
        full_mask = torch.cat([cls_mask, mask], dim=1)
        
        out_tokens, mab_attn_weights = self.aggregator(all_tokens, all_tokens, all_tokens, key_padding_mask=full_mask, need_weights=return_attn)
        
        cls_feat = out_tokens[:, 0, :]
        out_patch_tokens = out_tokens[:, 1:, :]
        valid_mask = (~mask).float().unsqueeze(-1)
        
        sum_feat = (out_patch_tokens * valid_mask).sum(dim=1)
        valid_count = valid_mask.sum(dim=1).clamp(min=1e-6)
        gap_feat = sum_feat / valid_count
        
        bag_feature = self.norm_final(cls_feat + gap_feat)
        logits = self.classifier(bag_feature)
        if return_attn:
            attn_dict = {
                'patch_scores': all_patch_scores,
                'topk_indices': topk_indices,
                'mab_attn': mab_attn_weights
            }
            return logits, attn_dict
            
        return logits

