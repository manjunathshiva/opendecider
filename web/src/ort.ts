/** ONNX Runtime Web, the build that fits where this runs (package.json "imports"): its WebGPU build in browsers (the
 * default build's WebGPU backend has no 8-bit weights), its default build in Node, Bun and Deno (WebAssembly there). */
import type * as Ort from "onnxruntime-web";
// @ts-ignore -- "#ort" is onnxruntime-web itself, whose types are declared under that name (imported above)
import * as impl from "#ort";

export const ort = impl as typeof Ort;
