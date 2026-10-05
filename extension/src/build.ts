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

/** Opens the build, and when WebGPU cannot run it (a GPU adapter, but a session that fails), q8 on WebAssembly instead,
 * so the filter still works on this computer. Not after a file error (a failed download, a file that did not match its
 * pin): the CPU build would fail the same way. */
export async function openBuild<T>(
  build: Pick<Build, "device" | "dtype">,
  open: (device: Build["device"], dtype: Dtype) => Promise<T>,
  isFileError: (e: unknown) => boolean,
): Promise<T> {
  if (build.device !== "webgpu") return open(build.device, build.dtype);
  try {
    return await open("webgpu", build.dtype);
  } catch (e) {
    if (isFileError(e)) throw e;
    return open("wasm", "q8");
  }
}
