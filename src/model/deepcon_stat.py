
import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DeepConSTAT(nn.Module):
    def __init__(self, num_zones, input_dim=4, embed_dim=64, num_time_steps=1440):
        super().__init__()
        
        # Input Embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional Encodings
        self.time_pos_embed = nn.Parameter(torch.zeros(1, 1, num_time_steps, embed_dim))
        self.zone_pos_embed = nn.Parameter(torch.zeros(1, num_zones, 1, embed_dim))
        
        # Axial Attention Layers
        # Time Mixing
        self.time_attn = AxialAttention(embed_dim)
        self.time_norm = nn.LayerNorm(embed_dim)
        
        # Zone Mixing
        self.zone_attn = AxialAttention(embed_dim)
        self.zone_norm = nn.LayerNorm(embed_dim)
        
        # MLP / FFN
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm_mlp = nn.LayerNorm(embed_dim)
        
        # Prediction Head
        # Global Average Pooling over Time? Or use last time step?
        # Typically forecasting uses past to predict future.
        # Here we condense time axis.
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, Z, T, D)
        B, Z, T, D = x.shape
        
        # Embedding
        x = self.input_proj(x) # (B, Z, T, E)
        
        # Add Positional Embeddings
        x = x + self.time_pos_embed[:, :, :T, :] + self.zone_pos_embed[:, :Z, :, :]
        
        # 1. Time Mixing (Pattern Learning)
        # Reshape to apply attn over Time axis independently for each Zone
        # Current: (B, Z, T, E) -> Flatten B*Z -> (B*Z, T, E)
        x_flat_z = x.view(B * Z, T, -1)
        x_t = self.time_norm(x_flat_z)
        x_t = self.time_attn(x_t)
        x_flat_z = x_flat_z + x_t
        x = x_flat_z.view(B, Z, T, -1)
        
        # 2. Zone Mixing (Spatial Propagation)
        # Reshape to apply attn over Zone axis independently for each Time
        # Current: (B, Z, T, E) -> Transpose to (B, T, Z, E) -> Flatten B*T -> (B*T, Z, E)
        x_trans = x.permute(0, 2, 1, 3).reshape(B * T, Z, -1)
        x_z = self.zone_norm(x_trans)
        x_z = self.zone_attn(x_z)
        x_trans = x_trans + x_z
        x = x_trans.view(B, T, Z, -1).permute(0, 2, 1, 3) # Back to (B, Z, T, E)
        
        # 3. FFN
        x = x + self.norm_mlp(self.mlp(x))
        
        # 4. Aggregation & Prediction
        # E.g., MaxPool or AvgPool over Time to summarize history
        # Or take last state?
        # Let's use Global Avg Pool over Time
        x_pool = x.mean(dim=2) # (B, Z, E)
        
        # Final prediction per zone
        risk_scores = self.head(x_pool) # (B, Z, 1)
        
        return risk_scores

if __name__ == "__main__":
    # Test
    model = DeepConSTAT(num_zones=50)
    dummy_input = torch.randn(2, 50, 1440, 4) # Batch=2
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (2, 50, 1)
