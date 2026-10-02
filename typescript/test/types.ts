// Compile-time checks of the public types (`npm run typecheck` compiles this file; nothing runs it).
import { generateText, stepCountIs, wrapLanguageModel, type LanguageModel } from "ai";
import { Agent } from "@mastra/core/agent";
import { choice, load, noul, score } from "../src/index.js";
import { guardMiddleware, opendeciderTools } from "../src/ai-sdk.js";
import { GuardProcessor, opendeciderTools as mastraTools } from "../src/mastra.js";

export async function answersAreTypedByTheirQuestion() {
  const m = await load("http://x");
  const r = await m.systemOne("t", {
    team: choice("Which?", ["a", "b"]),
    urgent: noul("Urgent?"),
    mood: score("Mood?", ["calm", "angry"]),
    raw: { type: "choice", instructions: "x", criteria: ["a", "b"] },
  });
  const team: string = r.answers.team.choice;
  const p: number = r.answers.urgent.noul;
  const level: number = r.answers.mood.score;
  const raw: string = r.answers.raw.choice;
  // @ts-expect-error a yes/no answer has no choice
  void r.answers.urgent.choice;
  // @ts-expect-error no such question
  void r.answers.nope;
  const b = await m.systemOneBatch(["x"], { q: noul("y?") });
  const p2: number = b[0]!.answers.q.noul;
  return [team, p, level, raw, p2];
}

// The adapters fit the frameworks' own types without casts
export async function frameworksTakeTheAdapters(model: Exclude<LanguageModel, string>) {
  const guarded = wrapLanguageModel({ model, middleware: guardMiddleware({ model: "http://x" }) });
  await generateText({
    model: guarded,
    tools: opendeciderTools({ model: "http://x" }),
    stopWhen: stepCountIs(3),
    prompt: "hi",
  });
  return new Agent({
    id: "a",
    name: "a",
    instructions: "x",
    model: "openai/gpt-5",
    tools: mastraTools({ model: "http://x" }),
    inputProcessors: [new GuardProcessor({ model: "http://x" })],
  });
}
