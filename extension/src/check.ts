/** The guard's answer for text selected on a page (the right-click menu): opened in a small window. */
import type { GuardResult } from "@opendecider/web";

const $ = (id: string) => document.getElementById(id)!;

async function main() {
  const key = `check:${location.hash.slice(1)}`;
  const text = (await chrome.storage.session.get(key))[key] as string | undefined;
  await chrome.storage.session.remove(key);
  if (!text) {
    $("verdict").textContent = "Nothing to check: select text, then right-click it again.";
    return;
  }
  $("text").textContent = text.length > 600 ? `${text.slice(0, 600)}…` : text;
  const res = await chrome.runtime.sendMessage({ type: "guard", text });
  const v = $("verdict");
  if (!res?.ok) {
    v.textContent =
      res?.error === "no-model"
        ? "Download the model first: click the OpenDecider Focus button in the toolbar."
        : `Could not check: ${res?.error ?? "no answer"}`;
    return;
  }
  const r = res.r as GuardResult;
  if (r.reason === "error") {
    // the text could not be checked: never shown as a clean result
    v.textContent = `Could not check this text: ${r.error ?? "unknown error"}`;
    v.className = "verdict flagged";
    return;
  }
  const top = Math.max(0, ...Object.values(r.probabilities));
  if (r.reason === "flagged") {
    v.textContent = "Looks like a prompt injection or jailbreak";
    v.className = "verdict flagged";
  } else {
    v.textContent = r.reason === "empty_input" ? "Nothing to check" : "No attack found";
  }
  $("detail").textContent =
    `Highest attack probability ${top.toFixed(2)} (flagged from ${Math.min(...Object.values(r.thresholds)).toFixed(2)})${r.windows > 1 ? `, over ${r.windows} windows` : ""}.`;
}

void main();
