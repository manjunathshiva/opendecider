import { describe, expect, it } from "vitest";
import { openBuild } from "../src/build.js";

class FileError extends Error {}
const isFileError = (e: unknown) => e instanceof FileError;

/** An open() that fails as told, recording what was asked. */
function opener(fail: Partial<Record<string, Error>>) {
  const asked: string[] = [];
  const open = async (device: string, dtype: string) => {
    asked.push(`${device}/${dtype}`);
    const e = fail[device];
    if (e) throw e;
    return `${device}/${dtype}`;
  };
  return { asked, open };
}

describe("openBuild", () => {
  it("opens fp16 on WebGPU when it works", async () => {
    const o = opener({});
    expect(await openBuild({ device: "webgpu", dtype: "fp16" }, o.open, isFileError)).toBe("webgpu/fp16");
    expect(o.asked).toEqual(["webgpu/fp16"]);
  });

  it("falls back to q8 on WebAssembly when WebGPU cannot run the model", async () => {
    const o = opener({ webgpu: new Error("no session") });
    expect(await openBuild({ device: "webgpu", dtype: "fp16" }, o.open, isFileError)).toBe("wasm/q8");
    expect(o.asked).toEqual(["webgpu/fp16", "wasm/q8"]);
  });

  it("does not fall back after a file error: the CPU build would fail the same way", async () => {
    const o = opener({ webgpu: new FileError("offline") });
    await expect(openBuild({ device: "webgpu", dtype: "fp16" }, o.open, isFileError)).rejects.toThrow("offline");
    expect(o.asked).toEqual(["webgpu/fp16"]);
  });

  it("opens the CPU build directly without a GPU, and reports its error", async () => {
    const o = opener({ wasm: new Error("no memory") });
    await expect(openBuild({ device: "wasm", dtype: "q8" }, o.open, isFileError)).rejects.toThrow("no memory");
    expect(o.asked).toEqual(["wasm/q8"]);
  });
});
