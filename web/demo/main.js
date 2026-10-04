import { loadNano } from "@opendecider/web";

const $ = (id) => document.getElementById(id);
$("questions").value = JSON.stringify(
  {
    department: { type: "choice", instructions: "Which department should handle this?",
                  criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages", other: "everything else" } },
    urgency: { type: "score", instructions: "How urgent is this?", criteria: ["not urgent", "soon", "blocking"] },
    churn_risk: { type: "noul", instructions: "Does the user threaten to cancel or leave?" },
  },
  null,
  1,
);

let model = null;
$("run").disabled = false; // enabled once this script runs: an earlier click would do nothing
const status = (text) => ($("status").textContent = text);

async function ensureModel() {
  if (model) return model;
  status("Loading the model…");
  model = await loadNano({
    onProgress: (p) => status(`${p.cached ? "Reading" : "Downloading"} ${p.file}: ${Math.round((100 * p.loaded) / p.total)}%`),
  });
  return model;
}

function bar(label, p) {
  const row = document.createElement("div");
  row.className = "row";
  const name = document.createElement("span");
  name.textContent = label;
  const track = document.createElement("div");
  const fill = document.createElement("div");
  fill.className = "bar";
  fill.style.width = `${Math.max(1, Math.round(p * 100))}%`;
  track.append(fill);
  const num = document.createElement("span");
  num.className = "num";
  num.textContent = p.toFixed(3);
  row.append(name, track, num);
  return row;
}

$("run").addEventListener("click", async () => {
  $("run").disabled = true;
  $("answers").replaceChildren();
  try {
    const text = $("state").value.trim();
    let state = text;
    if (/^[[{]/.test(text)) { try { state = JSON.parse(text); } catch { /* plain text */ } }
    const questions = JSON.parse($("questions").value);
    const m = await ensureModel();
    status("Deciding…");
    const t = performance.now();
    const r = await m.systemOne(state, questions);
    status(`${Object.keys(r.answers).length} answers in ${Math.round(performance.now() - t)} ms on ${m.meta.device} (${m.name}).`);
    for (const [name, a] of Object.entries(r.answers)) {
      const box = document.createElement("div");
      box.className = "q";
      const h = document.createElement("h2");
      const q = questions[name];
      const levels = a.type === "score" ? q.criteria : null;
      h.textContent = `${name} (${a.type})`;
      box.append(h);
      for (const [opt, p] of Object.entries(a.probabilities)) {
        const label = a.type === "noul" ? (opt === "true" ? "yes" : "no") : levels ? `${opt}: ${levels[Number(opt)]}` : opt;
        box.append(bar(label, p));
      }
      $("answers").append(box);
    }
  } catch (e) {
    status(`Could not decide: ${e.message}`);
  } finally {
    $("run").disabled = false;
  }
});
