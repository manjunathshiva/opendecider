"""A tiny model with opendecider-nano's architecture and random weights (tiny.onnx), and its probabilities from native
ONNX Runtime and from PyTorch for the prompts in parity.json (tiny.json), so the tests run the whole pipeline in CI
without the real 450 MiB build. Needs torch; the committed tiny.onnx and tiny.json are what CI uses.

    PYTHONPATH=.. python test/fixtures/make_tiny_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import ModernBertConfig, ModernBertModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "packaging" / "onnx"))
from quantize import plain  # noqa: E402  (the exporter's metadata out: no build paths in the committed file)
PAD = 50283
MASK = 50284


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = ModernBertConfig(vocab_size=50368, hidden_size=8, intermediate_size=16, num_hidden_layers=3,
                               num_attention_heads=2, global_attn_every_n_layers=3, local_attention=8,
                               max_position_embeddings=2048, pad_token_id=PAD, cls_token_id=50281, sep_token_id=50282,
                               bos_token_id=50281, eos_token_id=50282)
        self.enc = ModernBertModel(cfg)
        self.head = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.LayerNorm(8),
                                        torch.nn.Linear(8, 1))

    def forward(self, input_ids, attention_mask):
        return self.head(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state).squeeze(-1)


def probs(logits_row, ids):
    z = np.array([logits_row[j] for j, t in enumerate(ids) if t == MASK], dtype=np.float64)
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def main():
    torch.manual_seed(0)
    model = Tiny().eval()
    ids = torch.tensor([[50281, 10, MASK, 11, MASK, 12, 50282] + [13] * 20, [50281, 10, MASK, 11, 50282] + [PAD] * 22])
    att = (ids != PAD).long()
    B, L = torch.export.Dim("batch", min=1, max=1024), torch.export.Dim("seq", min=2, max=2048)
    torch.onnx.export(model, (ids, att), HERE / "tiny.onnx", input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], dynamic_shapes={"input_ids": {0: B, 1: L}, "attention_mask": {0: B, 1: L}},
                      dynamo=True, external_data=False, opset_version=18)
    import onnx
    onnx.save(plain(onnx.load(HERE / "tiny.onnx")), HERE / "tiny.onnx")
    sess = ort.InferenceSession(str(HERE / "tiny.onnx"), providers=["CPUExecutionProvider"])
    prompts = [p for p in json.loads((HERE / "parity.json").read_text())["prompts"] if p["max_len"] == 2048]
    out = []
    for p in prompts:   # one at a time: each row as the model sees it alone
        x = np.array([p["ids"]], dtype=np.int64)
        lg = sess.run(["logits"], {"input_ids": x, "attention_mask": np.ones_like(x)})[0][0]
        with torch.no_grad():
            tl = model(torch.tensor(x), torch.ones_like(torch.tensor(x)))[0].numpy()
        pt, po = probs(tl, p["ids"]), probs(lg, p["ids"])
        assert max(abs(a - b) for a, b in zip(pt, po)) < 1e-5, "ONNX and PyTorch disagree"
        out.append({"probs": po})
    (HERE / "tiny.json").write_bytes((json.dumps({"prompts": out}, indent=0) + "\n").encode("utf-8"))
    print(f"tiny.onnx {(HERE / 'tiny.onnx').stat().st_size} bytes, {len(out)} prompts")


if __name__ == "__main__":
    main()
