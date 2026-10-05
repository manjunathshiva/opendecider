"""The browser builds from export_nano.py's float32 graph: 8-bit weights (MatMulNBits, block 32, symmetric), the rest
float32 (q8) or float16 (q8f16), weight-only, so ONNX Runtime Web's WebAssembly gives the same logits as native ONNX
Runtime (dynamic int8 did not: its WebAssembly kernels round differently); and float16 throughout (fp16: WebAssembly
within 0.01 of native), whose plain MatMul runs many questions at once on WebGPU about 7 times faster than MatMulNBits,
which ONNX Runtime Web tunes for one row at a time. The exporter's metadata is removed, so the files carry no build
paths and rebuild byte for byte from the same versions.

    python packaging/onnx/quantize.py build/model.onnx build/onnx/
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.transformers.float16 import convert_float_to_float16


def plain(model):
    """Without the exporter's per-node metadata and doc strings (stack traces with the build machine's file paths):
    the published files carry only the graph and the weights, and rebuild byte for byte."""
    model.doc_string = ""
    del model.metadata_props[:]
    graphs = [model.graph]
    while graphs:
        g = graphs.pop()
        g.doc_string = ""
        for n in g.node:
            n.doc_string = ""
            del n.metadata_props[:]
            graphs += [a.g for a in n.attribute if a.HasField("g")] + [x for a in n.attribute for x in a.graphs]
        for x in list(g.input) + list(g.output) + list(g.value_info) + list(g.initializer):
            x.doc_string = ""
            if hasattr(x, "metadata_props"):
                del x.metadata_props[:]
    for f in model.functions:
        f.doc_string = ""
        del f.metadata_props[:]
        for n in f.node:
            n.doc_string = ""
            del n.metadata_props[:]
    return model


def q8(model):
    q = MatMulNBitsQuantizer(model, bits=8, block_size=32, is_symmetric=True)
    q.process()
    return q.model.model


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(plain(q8(onnx.load(src))), out / "model_q8.onnx")
    # logits stay float32 (keep_io_types); LayerNorm and attention run in float16
    onnx.save(plain(convert_float_to_float16(q8(onnx.load(src)), keep_io_types=True, disable_shape_infer=True)),
              out / "model_q8f16.onnx")
    onnx.save(plain(convert_float_to_float16(onnx.load(src), keep_io_types=True, disable_shape_infer=True)),
              out / "model_fp16.onnx")
    for p in sorted(out.glob("model_*.onnx")):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
