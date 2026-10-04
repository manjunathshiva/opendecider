"""Export opendecider-nano (encoder + head) to one ONNX graph: (input_ids, attention_mask) -> logits [batch, tokens].
The head runs on every token; @opendecider/web keeps the logits at the [MASK] markers and applies softmax.

    python packaging/onnx/export_nano.py --out build/          # writes build/model.onnx (+ .data), float32
Then quantize.py. Built with torch 2.14.1, transformers 5.18.0, onnx 1.23.1, onnxscript 0.7.2, onnxruntime 1.30.0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from opendecider.nano import NanoModel


class Graph(torch.nn.Module):
    def __init__(self, nano: NanoModel):
        super().__init__()
        self.enc, self.head = nano.enc, nano.head

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default=None, help="opendecider-nano's Hub revision (default: main)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    nano = NanoModel(snapshot_download("manjunathshiva/opendecider-nano", revision=a.revision), "cpu", 2048, "float32")
    ids = torch.tensor([nano.build({"a": 1}, "Which?", {"x": "", "y": "", "z": ""})] * 2)
    att = torch.ones_like(ids)
    att[1, -2:] = 0   # a padded row, so the mask path is traced
    B, L = torch.export.Dim("batch", min=1, max=1024), torch.export.Dim("seq", min=2, max=2048)
    torch.onnx.export(Graph(nano).eval(), (ids, att), a.out / "model.onnx", input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], dynamic_shapes={"input_ids": {0: B, 1: L}, "attention_mask": {0: B, 1: L}},
                      dynamo=True, external_data=True, opset_version=18)
    print(f"wrote {a.out / 'model.onnx'}")


if __name__ == "__main__":
    main()
