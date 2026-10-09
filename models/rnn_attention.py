"""
RNN Encoder-Decoder with Bahdanau (Additive) Attention for GEC.

Architecture (from reference project, adapted for BPE pipeline):
  Encoder  : bidirectional GRU, packed sequences for variable-length inputs
  Attention: Bahdanau (additive) — aligns each decoder step to encoder outputs
  Decoder  : single-layer GRU with attention context

Additions over baseline reference:
  - src_lengths  : packed-sequence encoding (correct padding handling)
  - EOS masking  : once a sentence emits EOS its token is frozen at PAD
  - Rep. penalty : divides logit of recently-seen tokens to break copy loops
                   (e.g. "вать вать вать" in Russian or Tamil morpheme chains)

Reference: Bahdanau et al. (2015) "Neural Machine Translation by Jointly
           Learning to Align and Translate." ICLR 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

PAD_IDX = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Bahdanau Attention
# ══════════════════════════════════════════════════════════════════════════════
class BahdanauAttention(nn.Module):
    """
    Additive attention.
    score(s_t, h_i) = v^T · tanh(W_dec·s_t + W_enc·h_i)

    Args:
        dec_hidden_dim : decoder hidden size
        enc_hidden_dim : encoder output size (hidden_dim × 2 if bidirectional)
    """

    def __init__(self, dec_hidden_dim: int, enc_hidden_dim: int):
        super().__init__()
        self.W_dec = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
        self.W_enc = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
        self.v     = nn.Linear(dec_hidden_dim, 1,              bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,           # (B, dec_hidden)
        encoder_outputs: torch.Tensor,          # (B, S, enc_hidden)
        src_mask: Optional[torch.Tensor] = None # (B, S) bool — True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns context (B, enc_hidden) and weights (B, S)."""
        S     = encoder_outputs.size(1)
        dec_h = self.W_dec(decoder_hidden).unsqueeze(1).expand(-1, S, -1)
        enc_h = self.W_enc(encoder_outputs)
        energy = self.v(torch.tanh(dec_h + enc_h)).squeeze(2)  # (B, S)

        if src_mask is not None:
            energy = energy.masked_fill(src_mask, float("-inf"))

        weights = F.softmax(energy, dim=1)                     # (B, S)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


# ══════════════════════════════════════════════════════════════════════════════
#  Encoder
# ══════════════════════════════════════════════════════════════════════════════
class RNNEncoder(nn.Module):
    """
    Bidirectional GRU encoder with packed-sequence support.

    Using pack_padded_sequence ensures the GRU never sees padding tokens,
    giving the correct hidden state regardless of sentence length.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,         # (B, S)
        src_lengths: torch.Tensor  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            outputs : (B, S, hidden_dim*2)
            hidden  : (B, hidden_dim)  — last layer fwd+bwd combined
        """
        embedded = self.dropout(self.embedding(src))
        packed   = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        # total_length keeps shape consistent across DataParallel sub-batches
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1))

        # Merge last-layer fwd + bwd hidden → single decoder initial state
        # hidden shape: (num_directions * num_layers, B, hidden_dim)
        # For bidirectional: hidden[-2] = last forward, hidden[-1] = last backward
        if hidden.size(0) < 2:
            raise RuntimeError(
                f"Expected bidirectional encoder (hidden size >= 2), got {hidden.size(0)}"
            )
        hidden = torch.tanh(
            self.fc_hidden(torch.cat([hidden[-2], hidden[-1]], dim=1))
        )  # (B, hidden_dim)
        return outputs, hidden


# ══════════════════════════════════════════════════════════════════════════════
#  Decoder
# ══════════════════════════════════════════════════════════════════════════════
class RNNDecoder(nn.Module):
    """
    Single-layer GRU decoder with Bahdanau attention.

    At each step:
      input  → [embedding ; attention_context]  fed to GRU
      output → fc_out([GRU_out ; context ; embedding])
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
        self.rnn = nn.GRU(
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
        encoder_outputs: torch.Tensor,                  # (B, S, enc_hidden)
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One decoder step.
        Returns: prediction (B, vocab), hidden (B, hidden), attn (B, S)
        """
        embedded = self.dropout(
            self.embedding(tgt_token.unsqueeze(1)))     # (B, 1, embed)
        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)

        rnn_input      = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden         = hidden.squeeze(0)              # (B, hidden_dim)

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden, attn_weights


# ══════════════════════════════════════════════════════════════════════════════
#  Repetition Penalty Helper
# ══════════════════════════════════════════════════════════════════════════════
def _apply_rep_penalty(
    logits: torch.Tensor,
    generated: list,
    penalty: float,
    last_k: int,
) -> torch.Tensor:
    """
    Divide the logit of every token seen in the last `last_k` steps by
    `penalty` (CTRL / Keskar et al. 2019 formulation).

    Positive logit → divided by penalty (less likely).
    Negative logit → multiplied by penalty (even less likely).

    penalty=1.0 is a no-op.  penalty=1.3 reliably breaks copy loops without
    noticeably degrading fluent, non-repetitive output.
    """
    if penalty == 1.0 or not generated:
        return logits
    logits = logits.clone()
    for tok_tensor in generated[-last_k:]:
        for b in range(logits.size(0)):
            t = tok_tensor[b].item()
            if logits[b, t] > 0:
                logits[b, t] /= penalty
            else:
                logits[b, t] *= penalty
    return logits


# ══════════════════════════════════════════════════════════════════════════════
#  Seq2Seq Model
# ══════════════════════════════════════════════════════════════════════════════
class RNNSeq2Seq(nn.Module):
    """
    Full GRU encoder-decoder with Bahdanau attention for GEC.

    Args:
        src_vocab_size : source vocabulary size
        tgt_vocab_size : target vocabulary size
        embed_dim      : embedding dimension  (default 256)
        hidden_dim     : GRU hidden size      (default 512)
        enc_layers     : encoder GRU layers   (default 1)
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
        enc_layers: int = 1,
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

        enc_hidden_dim = hidden_dim * 2        # bidirectional encoder output
        self.encoder = RNNEncoder(
            src_vocab_size, embed_dim, hidden_dim, enc_layers, dropout, pad_idx)
        self.decoder = RNNDecoder(
            tgt_vocab_size, embed_dim, hidden_dim, enc_hidden_dim, dropout, pad_idx)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """True where src == PAD (positions to mask in attention)."""
        return src == self.pad_idx  # (B, S)

    # ── Training forward ────────────────────────────────────────────────────
    def forward(
        self,
        src: torch.Tensor,           # (B, S)
        src_lengths: torch.Tensor,   # (B,)
        tgt: torch.Tensor,           # (B, T)
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Teacher-forced training pass.

        Returns:
            outputs : (B, T-1, tgt_vocab_size)
            Loss targets: tgt[:, 1:]  — no alignment tricks needed.
        """
        B, T    = tgt.size()
        device  = src.device
        src_mask = self.make_src_mask(src)

        enc_outputs, hidden = self.encoder(src, src_lengths)

        dec_input = tgt[:, 0]   # <sos>
        outputs   = torch.zeros(B, T - 1, self.tgt_vocab_size, device=device)

        for t in range(1, T):
            pred, hidden, _ = self.decoder(dec_input, hidden, enc_outputs, src_mask)
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
        Greedy decoding with:
          - Per-item EOS masking  : finished sentences stay at PAD
          - Repetition penalty    : breaks copy loops (e.g. morpheme repetition)

        Returns: (B, out_len) token ids
        """
        self.eval()
        B, device = src.size(0), src.device
        src_mask  = self.make_src_mask(src)

        enc_outputs, hidden = self.encoder(src, src_lengths)

        dec_input = torch.full((B,), sos_idx, dtype=torch.long, device=device)
        result    = []
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            pred, hidden, _ = self.decoder(dec_input, hidden, enc_outputs, src_mask)
            pred     = _apply_rep_penalty(pred, result, rep_penalty, rep_last_k)
            dec_input = pred.argmax(dim=1)
            dec_input = dec_input.masked_fill(finished, self.pad_idx)
            result.append(dec_input.clone())
            finished  = finished | (dec_input == eos_idx)
            if finished.all():
                break

        if not result:
            return torch.zeros(B, 1, dtype=torch.long, device=device)
        return torch.stack(result, dim=1)   # (B, out_len)


# Keep old name as alias for backward compatibility with train.py
RNNAttentionModel = RNNSeq2Seq
