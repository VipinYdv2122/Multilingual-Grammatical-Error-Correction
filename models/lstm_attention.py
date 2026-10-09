"""
LSTM Encoder-Decoder with Bahdanau (Additive) Attention for GEC.

Architecture (from reference project, adapted for BPE pipeline):
  Encoder  : bidirectional LSTM, packed sequences for variable-length inputs
  Attention: Bahdanau (additive) — aligns decoder hidden state to encoder outputs
  Decoder  : single-layer LSTM with attention context injected at each step

Additions over baseline reference:
  - src_lengths  : packed-sequence encoding (correct padding handling)
  - EOS masking  : finished sentences frozen at PAD after first EOS
  - Rep. penalty : breaks morpheme copy loops across all languages

References:
  Bahdanau et al. (2015) ICLR — attention mechanism
  Hochreiter & Schmidhuber (1997) — LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.rnn_attention import BahdanauAttention, _apply_rep_penalty

PAD_IDX = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Encoder
# ══════════════════════════════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with packed-sequence support.

    pack_padded_sequence ensures the LSTM never processes padding tokens,
    giving the correct (h, c) states for any sentence length.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            outputs : (B, S, hidden_dim*2)
            (h, c)  : each (B, hidden_dim) — last layer fwd+bwd combined
        """
        embedded = self.dropout(self.embedding(src))
        packed   = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1))

        # Merge last-layer fwd + bwd hidden states → single decoder initial state
        # hidden/cell shape: (num_directions * num_layers, B, hidden_dim)
        # For bidirectional: [-2] = last forward, [-1] = last backward
        if hidden.size(0) < 2:
            raise RuntimeError(
                f"Expected bidirectional encoder (hidden size >= 2), got {hidden.size(0)}"
            )
        h = torch.tanh(self.fc_hidden(torch.cat([hidden[-2], hidden[-1]], dim=1)))
        c = torch.tanh(self.fc_cell(  torch.cat([cell[-2],   cell[-1]],   dim=1)))
        return outputs, (h, c)


# ══════════════════════════════════════════════════════════════════════════════
#  Decoder
# ══════════════════════════════════════════════════════════════════════════════
class LSTMDecoder(nn.Module):
    """
    Single-layer LSTM decoder with Bahdanau attention.

    At each step:
      input  → [embedding ; attention_context]  fed to LSTM
      output → fc_out([LSTM_out ; context ; embedding])
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        enc_hidden_dim: int,
        dropout: float = 0.3,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim, enc_hidden_dim)
        self.lstm = nn.LSTM(
            embed_dim + enc_hidden_dim,
            hidden_dim,
            batch_first=True,
        )
        self.fc_out  = nn.Linear(hidden_dim + enc_hidden_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token: torch.Tensor,                       # (B,)
        hidden: torch.Tensor,                           # (B, hidden_dim)
        cell: torch.Tensor,                             # (B, hidden_dim)
        encoder_outputs: torch.Tensor,                  # (B, S, enc_hidden)
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One decoder step.
        Returns: prediction (B, vocab), hidden (B, hidden), cell (B, hidden), attn (B, S)
        """
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))  # (B,1,E)
        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)

        rnn_input          = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (h, c)     = self.lstm(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        hidden, cell       = h.squeeze(0), c.squeeze(0)

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden, cell, attn_weights


# ══════════════════════════════════════════════════════════════════════════════
#  Seq2Seq Model
# ══════════════════════════════════════════════════════════════════════════════
class LSTMSeq2Seq(nn.Module):
    """
    Full LSTM encoder-decoder with Bahdanau attention for GEC.

    Args:
        src_vocab_size : source vocabulary size
        tgt_vocab_size : target vocabulary size
        embed_dim      : embedding dimension  (default 256)
        hidden_dim     : LSTM hidden size     (default 512)
        enc_layers     : encoder LSTM layers  (default 2)
        dropout        : dropout rate         (default 0.3)
        sos_idx        : <sos> token index    (default 1)
        eos_idx        : <eos> token index    (default 2)
        pad_idx        : <pad> token index    (default 0)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        enc_layers: int = 2,
        dropout: float = 0.3,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.sos_idx        = sos_idx
        self.eos_idx        = eos_idx
        self.pad_idx        = pad_idx
        self.tgt_vocab_size = tgt_vocab_size

        enc_hidden_dim = hidden_dim * 2
        self.encoder = LSTMEncoder(
            src_vocab_size, embed_dim, hidden_dim, enc_layers, dropout, pad_idx)
        self.decoder = LSTMDecoder(
            tgt_vocab_size, embed_dim, hidden_dim, enc_hidden_dim, dropout, pad_idx)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """True where src == PAD."""
        return src == self.pad_idx

    # ── Training forward ────────────────────────────────────────────────────
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Teacher-forced training pass.

        Returns:
            outputs : (B, T-1, tgt_vocab_size)
            Loss targets: tgt[:, 1:]
        """
        B, T    = tgt.size()
        device  = src.device
        src_mask = self.make_src_mask(src)

        enc_outputs, (hidden, cell) = self.encoder(src, src_lengths)

        dec_input = tgt[:, 0]
        outputs   = torch.zeros(B, T - 1, self.tgt_vocab_size, device=device)

        for t in range(1, T):
            pred, hidden, cell, _ = self.decoder(
                dec_input, hidden, cell, enc_outputs, src_mask)
            outputs[:, t - 1] = pred
            use_tf    = (self.training
                         and teacher_forcing_ratio > 0.0
                         and torch.rand(1).item() < teacher_forcing_ratio)
            dec_input = tgt[:, t] if use_tf else pred.argmax(dim=1)

        return outputs   # (B, T-1, V)

    # ── Greedy inference with repetition penalty ─────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 100,
        rep_penalty: float = 1.3,
        rep_last_k:  int   = 8,
    ) -> torch.Tensor:
        """
        Greedy decoding with per-item EOS masking and repetition penalty.
        Returns (B, out_len) token ids.
        """
        self.eval()
        B, device = src.size(0), src.device
        src_mask  = self.make_src_mask(src)

        enc_outputs, (hidden, cell) = self.encoder(src, src_lengths)

        dec_input = torch.full((B,), sos_idx, dtype=torch.long, device=device)
        result    = []
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            pred, hidden, cell, _ = self.decoder(
                dec_input, hidden, cell, enc_outputs, src_mask)
            pred      = _apply_rep_penalty(pred, result, rep_penalty, rep_last_k)
            dec_input = pred.argmax(dim=1)
            dec_input = dec_input.masked_fill(finished, self.pad_idx)
            result.append(dec_input.clone())
            finished  = finished | (dec_input == eos_idx)
            if finished.all():
                break

        if not result:
            return torch.zeros(B, 1, dtype=torch.long, device=device)
        return torch.stack(result, dim=1)


# Alias for backward compatibility
LSTMAttentionModel = LSTMSeq2Seq
