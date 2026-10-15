"""Token-level contextual embeddings for a focal word, per layer.

HFEmbedder needs torch and transformers (see requirements.txt). MockEmbedder is for tests only.
Vectors follow the protocol: mean over the focal word's subword tokens -> L2-normalize -> center
(subtract the mean of L2-normalized vectors of a random token sample) -- see drift.prepare().
"""
import numpy as np


class HFEmbedder:
    def __init__(self, name, revision=None, max_length=256, batch_size=32, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        rev = None if (revision in (None, "", "FILL_COMMIT_HASH")) else revision
        self.name, self.revision = name, rev
        self.tok = AutoTokenizer.from_pretrained(name, revision=rev, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(name, revision=rev, torch_dtype=dtype).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.n_layers = self.model.config.num_hidden_layers  # hidden_states has n_layers + 1 entries

    def resolved_revision(self):
        return getattr(self.model.config, "_commit_hash", None) or self.revision

    def _forward(self, texts):
        enc = self.tok(texts, return_offsets_mapping=True, truncation=True, max_length=self.max_length,
                       padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").numpy()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hs = [h.float().cpu().numpy() for h in out.hidden_states[1:]]  # drop embedding layer
        return hs, offsets, enc["attention_mask"].cpu().numpy()

    def span_vectors(self, texts, spans):
        """Return array (n, n_layers, dim) of mean subword vectors for each char span; NaN if lost to truncation."""
        res = None
        for b in range(0, len(texts), self.batch_size):
            bt, bs = texts[b:b + self.batch_size], spans[b:b + self.batch_size]
            hs, offs, mask = self._forward(bt)
            if res is None:
                res = np.full((len(texts), len(hs), hs[0].shape[-1]), np.nan, dtype=np.float32)
            for i, (s, e) in enumerate(bs):
                o = offs[i]
                idx = np.where((o[:, 0] < e) & (o[:, 1] > s) & (o[:, 1] > o[:, 0]) & (mask[i] == 1))[0]
                if len(idx):
                    res[b + i] = np.stack([h[i, idx].mean(0) for h in hs])
        return res

    def random_token_mean(self, texts, n_tokens, rng):
        """Mean of L2-normalized token vectors over ~n_tokens random tokens, per layer -> (n_layers, dim)."""
        order = rng.permutation(len(texts))
        sums, count = None, 0
        for b in range(0, len(order), self.batch_size):
            bt = [texts[j] for j in order[b:b + self.batch_size]]
            hs, offs, mask = self._forward(bt)
            keep = (mask == 1) & (offs[:, :, 1] > offs[:, :, 0])  # real (non-special) tokens
            vecs = np.stack([h[keep] for h in hs])  # (layers, k, dim)
            vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12
            sums = vecs.sum(1) if sums is None else sums + vecs.sum(1)
            count += vecs.shape[1]
            if count >= n_tokens:
                break
        return sums / max(count, 1), count


class MockEmbedder:
    """Deterministic fake embedder for tests: vector = hash-seeded noise + signal from cue words."""

    def __init__(self, n_layers=4, dim=32, cues=None, seed=0):
        self.n_layers, self.dim, self.name, self.revision = n_layers, dim, "mock", "mock"
        self.cues = cues or {}
        self.rng = np.random.default_rng(seed)
        self.basis = {k: self.rng.normal(size=dim) for k in self.cues}

    def resolved_revision(self):
        return "mock"

    def _vec(self, text):
        r = np.random.default_rng(abs(hash(text)) % (2**32))
        v = r.normal(size=self.dim) * 0.6
        low = text.lower()
        for k, words in self.cues.items():
            if any(w in low for w in words):
                v += self.basis[k]
        return v

    def span_vectors(self, texts, spans):
        out = np.zeros((len(texts), self.n_layers, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = self._vec(t)
            for l in range(self.n_layers):
                out[i, l] = v * (0.5 + l / self.n_layers) + 0.1 * l
        return out

    def random_token_mean(self, texts, n_tokens, rng):
        return np.full((self.n_layers, self.dim), 0.01, dtype=np.float32), n_tokens
