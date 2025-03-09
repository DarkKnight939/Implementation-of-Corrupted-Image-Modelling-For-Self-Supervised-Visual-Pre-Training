import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dall_e import load_model

class DVAETokenizer:
    def __init__(self, device):
        self.encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", device)
        self.decoder = load_model("https://cdn.openai.com/dall-e/decoder.pkl", device)
        self.vocab_size = 8192
        self.device = device

    def encode(self, image):
        if image.size(1) != 3:
            image = image[:, :3, :, :]
        with torch.no_grad():
            tokens = self.encoder(image)
            z = torch.argmax(tokens, axis=1)
            return F.one_hot(z, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()

    def decode(self, tokens):
        with torch.no_grad():
            decoded = self.decoder(tokens)
            if decoded.size(1) != 3:
                decoded = decoded[:, :3, :, :]
            return decoded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_patches=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_patches, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# BEIT Embedding
class BEITEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=768, patches=196, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Linear(vocab_size, embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.position_embedding = PositionalEncoding(embed_size, patches + 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens, bool_masked_pos=None):
        batch_size = tokens.size(0)
        x = self.token_embedding(tokens)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply masking if specified
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, x.size(1) - 1, -1)
            w = bool_masked_pos.unsqueeze(-1)
            x[:, 1:] = x[:, 1:] * (1 - w) + mask_tokens * w

        # Add positional encoding
        x = x + self.position_embedding(x)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), x, x)[0]
        x = x + self.mlp(self.norm2(x))
        return x

#Attention Components
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.scale = head_size ** -0.5

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        return attention @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.head_size = d_model // n_heads
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.head_size) for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        out = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Generator(nn.Module):
    def __init__(self, device, d_model=384, n_heads=6, n_layers=6, temperature=1.0):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.tokenizer = DVAETokenizer(device)
        self.patch_embedding = nn.Linear(8192, d_model).to(device)
        self.pos_embedding = PositionalEncoding(d_model, max_patches=1024, dropout=0.1).to(device)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(d_model, n_heads) for _ in range(n_layers)
        ]).to(device)
        self.token_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 8192)
        ).to(device)

    def forward(self, image, x=None):
        image = image.float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.size(1) != 3:
            image = image[:, :3, :, :]
        with torch.no_grad():
            visual_tokens = self.tokenizer.encode(image)
            original_tokens = visual_tokens.clone()
        B, C, H, W = visual_tokens.shape
        visual_tokens = visual_tokens.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.patch_embedding(visual_tokens) + self.pos_embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        token_logits = self.token_predictor(x)
        return token_logits