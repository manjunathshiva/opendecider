/** The opendecider-nano build this version of the package runs: one Hub revision, every file pinned by size and SHA-256. */
import type { FileSpec } from "./hub.js";

/** The Hugging Face repository with the ONNX builds. */
export const REPO = "manjunathshiva/opendecider-nano-ONNX";
/** The commit of REPO the files below come from. */
export const REVISION = "53da62af62275ff5076e29b962dd965f38107c50";
/** Tokens in one input, as opendecider-nano was trained (the state is shortened first). */
export const MAX_TOKENS = 2048;

/** The ONNX builds: q8 (8-bit weights, float32 elsewhere; WebAssembly's default), q8f16 (8-bit weights, float16
 * elsewhere; WebGPU's default) and fp16 (float16 throughout: a larger download, and on WebGPU about 7 times faster than
 * q8f16 for many questions at once). */
export type Dtype = "q8" | "q8f16" | "fp16";

export const MODELS: Readonly<Record<Dtype, FileSpec>> = {
  q8: {
    path: "onnx/model_q8.onnx",
    bytes: 594006972,
    sha256: "3920323e41bb107f28566be7b45e6cb55a2bdc1951b27a6e62625857e22542c4",
  },
  q8f16: {
    path: "onnx/model_q8f16.onnx",
    bytes: 469218222,
    sha256: "80436dbed1a284548b5e6c3975bb26df81fb493fc4e64629e5a6ef490bf0e0ae",
  },
  fp16: {
    path: "onnx/model_fp16.onnx",
    bytes: 791886899,
    sha256: "ce3b92ed327c394570123867d48a73383e748b015c08ca5b735670d6601ab44c",
  },
};

/** opendecider-nano's tokenizer for tokenizers.js: the same ids as the Python package's (see the repository's
 * packaging/onnx/make_web_tokenizer.py). */
export const TOKENIZER: FileSpec = {
  path: "tokenizer.web.json",
  bytes: 1570653,
  sha256: "fd4146f0082d98fd1fc03bf41349f4d1eed478987ce9a64addc1654dfcad14ee",
};

/** Where the files are downloaded from by default. */
export const HUB_URL = `https://huggingface.co/${REPO}/resolve/${REVISION}/`;
