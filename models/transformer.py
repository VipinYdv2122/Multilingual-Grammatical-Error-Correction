"""
Transformer Encoder-Decoder for GEC.

T4 Config: 6 encoder layers, 6 decoder layers, 8 heads, d_model=512, d_ff=2048, dropout=0.3
Uses PyTorch's built-in TransformerEncoder/Decoder.

Design choices:
  - Pre-LN (norm_first=True) : more stable than Post-LN without careful warmup;
    allows higher learning rates and converges faster on small-data GEC tasks.
  - forward() returns (B, T-1, V) to match RNN/LSTM convention — no zero-pad
    trick; train.py computes loss directly as criterion(out, tgt[:, 1:]).
  - generate() uses length-normalised beam search (see generate.py).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    """
    Transformer Encoder-Decoder for GEC.

    Args:
        src_vocab    : source vocabulary size
        tgt_vocab    : target vocabulary size
        d_model      : embedding / hidden dimension (default 256)
        nhead        : attention heads              (default 4)
        n_enc_layers : encoder depth                (default 2)
        n_dec_layers : decoder depth                (default 2)
        d_ff         : feed-forward hidden size     (default 512)
        dropout      : dropout probability          (default 0.1)
        max_len      : max sequence length          (default 512)
        pad_idx      : padding token index          (default 0)
        norm_first   : Pre-LN if True (recommended) (default True)
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 256,
        nhead: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0,
        norm_first: bool = True,
    ):
        super().__init__()
        self.pad_idx        = pad_idx
        self.tgt_vocab_size = tgt_vocab
        self.scale          = math.sqrt(d_model)
        self.d_model        = d_model

        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        enc_layer    = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(
            enc_layer, n_enc_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False)

        dec_layer    = nn.TransformerDecoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True, norm_first=norm_first)
        self.decoder = nn.TransformerDecoder(
            dec_layer, n_dec_layers, norm=nn.LayerNorm(d_model))

        self.fc_out  = nn.Linear(d_model, tgt_vocab)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, size: int, device) -> torch.Tensor:
        """Upper-triangular bool mask — prevents attending to future tokens."""
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def _embed_src(self, src: torch.Tensor) -> torch.Tensor:
        return self.pos_enc(self.src_emb(src) * self.scale)

    def _embed_tgt(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.pos_enc(self.tgt_emb(tgt) * self.scale)

    # ── Training forward ────────────────────────────────────────────────────
    def forward(
        self,
        src: torch.Tensor,                 # (B, S)
        src_lengths: torch.Tensor,         # (B,) — kept for API consistency
        tgt: torch.Tensor,                 # (B, T)
        teacher_forcing_ratio=None,        # unused; transformer always uses TF
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Decoder input  : tgt[:, :-1]   (tokens 0 … T-2)
        Decoder target : tgt[:, 1:]    (tokens 1 … T-1)

        Returns:
            logits : (B, T-1, tgt_vocab)
            Loss  : criterion(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        """
        tgt_input    = tgt[:, :-1]              # (B, T-1)
        src_pad_mask = (src      == self.pad_idx)
        tgt_pad_mask = (tgt_input == self.pad_idx)
        tgt_mask     = self._causal_mask(tgt_input.size(1), tgt_input.device)

        memory = self.encoder(
            self._embed_src(src),
            src_key_padding_mask=src_pad_mask)

        out = self.decoder(
            self._embed_tgt(tgt_input),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask)

        return self.fc_out(out)   # (B, T-1, tgt_vocab)

    # ── Greedy inference ─────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 100,
        rep_penalty: float = 1.0,   # transformer uses beam search instead
        rep_last_k:  int   = 8,
    ) -> torch.Tensor:
        """
        Auto-regressive greedy decoding.
        For beam search, call beam_search_transformer() in generate.py instead.
        Returns (B, out_len) token ids (sos stripped).
        """
        self.eval()
        B, device = src.size(0), src.device

        src_pad_mask = (src == self.pad_idx)
        memory = self.encoder(
            self._embed_src(src),
            src_key_padding_mask=src_pad_mask)

        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask  = self._causal_mask(generated.size(1), device)
            out = self.decoder(
                self._embed_tgt(generated),
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad_mask)

            next_tok = self.fc_out(out[:, -1]).argmax(dim=-1)   # (B,)
            next_tok = next_tok.masked_fill(finished, self.pad_idx)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished  = finished | (next_tok == eos_idx)
            if finished.all():
                break

        return generated[:, 1:]   # strip <sos>
