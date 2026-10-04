/**
 * OpenDecider on the device: opendecider-nano in the browser (WebGPU or WebAssembly), Node, Bun or Deno. The text is
 * decided where it is, and never sent anywhere.
 *
 *     import { loadNano, choice, noul, Guard } from "@opendecider/web";
 *
 *     const model = await loadNano();             // downloads and checks the pinned build once, then the Cache API
 *     const r = await model.systemOne("I was charged twice for order 1182.", {
 *       team: choice("Which team should handle this?", { billing: "charges, refunds", tech: "bugs" }),
 *       urgent: noul("Does this need a reply today?"),
 *     });
 *     const guard = new Guard({ model });         // the prompt guard, on the same model
 *
 * Everything else (questions, answers, Router, Guard, the agent tools) is @opendecider/client's, re-exported here.
 */
export * from "@opendecider/client";
export { ModelFileError, type FileSpec, type LoadProgress } from "./hub.js";
export { HUB_URL, MAX_TOKENS, MODELS, REPO, REVISION, TOKENIZER, type Dtype } from "./manifest.js";
export { NanoBackend, loadNano, type Device, type LoadOptions } from "./nano.js";
