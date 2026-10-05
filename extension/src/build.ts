/** Which opendecider-nano build this computer gets: fp16 on WebGPU (about 7 times faster than q8f16 for a feed's many
 * videos at once), q8 on WebAssembly (as fast as fp16 there, in about 3 GiB instead of 4.8). One is downloaded. */
import { MODELS, type Dtype } from "@opendecider/web";

export interface Build {
  device: "webgpu" | "wasm";
  dtype: Dtype;
  bytes: number;
}

export async function pickBuild(): Promise<Build> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  const adapter = gpu ? await gpu.requestAdapter().catch(() => null) : null;
  const dtype: Dtype = adapter ? "fp16" : "q8";
  return { device: adapter ? "webgpu" : "wasm", dtype, bytes: MODELS[dtype].bytes };
}
