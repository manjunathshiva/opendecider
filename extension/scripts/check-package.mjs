// The built extension (dist/) asks for exactly what it needs: CI fails if a permission, a site or a network
// destination is added without this list changing with it.
import { readFileSync } from "node:fs";

const m = JSON.parse(readFileSync(new URL("../dist/manifest.json", import.meta.url), "utf8"));
const { version } = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
const want = {
  version,
  permissions: ["storage", "offscreen", "contextMenus", "alarms"],
  host_permissions: undefined,
  sites: ["https://www.youtube.com/*"],
  csp: "script-src 'self' 'wasm-unsafe-eval'; object-src 'none'; connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co; worker-src 'self'",
};
const got = {
  version: m.version,
  permissions: m.permissions,
  host_permissions: m.host_permissions,
  sites: m.content_scripts.flatMap((c) => c.matches),
  csp: m.content_security_policy.extension_pages,
};
if (JSON.stringify(got) !== JSON.stringify(want)) {
  console.error("the manifest changed:", { got, want });
  process.exit(1);
}
console.log("manifest as expected");
