// Builds the extension into dist/ (load it unpacked from there), and with --zip the package for the Chrome Web Store:
// the same files in the same order with the same dates, so the zip rebuilds byte for byte from the same sources.
import { build } from "esbuild";
import { cpSync, readFileSync, readdirSync, rmSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const dist = path.join(here, "dist");
const { version } = JSON.parse(readFileSync(path.join(here, "package.json"), "utf8"));
// Tests only: the model files from this folder URL (a local server) instead of the pinned Hub revision. Their SHA-256
// is still checked against the package's pins. A Web Store zip never takes it.
const modelUrl = process.argv.find((a) => a.startsWith("--model-url="))?.slice("--model-url=".length) ?? "";
if (modelUrl && process.argv.includes("--zip"))
  throw new Error("--model-url is for tests: a zip downloads from the Hub");
rmSync(dist, { recursive: true, force: true });
cpSync(path.join(here, "static"), dist, { recursive: true });
const manifest = JSON.parse(readFileSync(path.join(dist, "manifest.json"), "utf8"));
manifest.version = version;
if (modelUrl) {
  const csp = manifest.content_security_policy;
  csp.extension_pages = csp.extension_pages.replace(
    "connect-src 'self'",
    `connect-src 'self' ${new URL(modelUrl).origin}`,
  );
}
writeFileSync(path.join(dist, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
// ONNX Runtime's own WebAssembly and loader: shipped in the extension (threads need them; nothing comes from a CDN)
// (the version @opendecider/web depends on, wherever npm put it)
const fromWeb = createRequire(createRequire(import.meta.url).resolve("@opendecider/web"));
const ort = path.dirname(fromWeb.resolve("onnxruntime-web"));
for (const f of ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"])
  cpSync(path.join(ort, f), path.join(dist, "ort", f));
cpSync(path.join(here, "../LICENSE"), path.join(dist, "LICENSE"));
cpSync(path.join(here, "../NOTICE"), path.join(dist, "NOTICE"));

const common = { bundle: true, target: "chrome124", legalComments: "linked", logLevel: "warning", minify: false };
await build({
  ...common,
  entryPoints: ["background", "offscreen", "popup", "check"].map((n) => path.join(here, `src/${n}.ts`)),
  outdir: dist,
  format: "esm",
  conditions: ["browser"],
  define: { __MODEL_URL__: JSON.stringify(modelUrl) },
});
await build({ ...common, entryPoints: [path.join(here, "src/content.ts")], outdir: dist, format: "iife" });

if (process.argv.includes("--zip")) {
  const files = [];
  const walk = (d) => {
    for (const n of readdirSync(d).sort()) {
      const p = path.join(d, n);
      if (statSync(p).isDirectory()) walk(p);
      else files.push(path.relative(dist, p));
    }
  };
  walk(dist);
  const when = new Date(Number(process.env.SOURCE_DATE_EPOCH ?? 315532800) * 1000); // 1980-01-01, zip's first date
  for (const f of files) utimesSync(path.join(dist, f), when, when);
  const zip = path.join(here, `opendecider-focus-${version}.zip`);
  rmSync(zip, { force: true });
  execFileSync("zip", ["-X", "-D", "-q", zip, ...files], { cwd: dist, env: { ...process.env, TZ: "UTC" } });
  console.log(zip);
}
