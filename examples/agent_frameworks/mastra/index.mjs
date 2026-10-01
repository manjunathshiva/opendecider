// OpenDecider in Mastra (TypeScript): its MCP server's tools, as a Mastra agent gets them.
//
//     pip install "opendecider[mcp]"          # the server: `opendecider mcp`
//     cd examples/agent_frameworks/mastra && npm ci && node index.mjs
//
// MCPClient starts `opendecider mcp` and turns its tools into Mastra tools, named `opendecider_<tool>`. Give them to an
// agent with `new Agent({ name, instructions, model, tools: await mcp.listTools() })`; this script calls them directly,
// as the agent would, so it runs without an API key. An invalid call fails with a message saying what to fix.
import { MCPClient } from "@mastra/mcp";

const mcp = new MCPClient({
  servers: { opendecider: { command: "opendecider", args: ["mcp", ...process.argv.slice(2)] } },  // e.g. --model ...
});
const TICKET = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.";

try {
  const tools = await mcp.listTools();
  console.log("tools:", Object.keys(tools).join(", "));

  const route = await tools.opendecider_choose.execute({
    state: TICKET, question: "Which department should handle this?",
    options: { billing: "invoices, payments, refunds", technical: "bugs, outages, system errors",
               other: "everything else" },
  }, {});
  console.log("choose:", JSON.stringify(route));
  console.log("yes_no:", JSON.stringify(await tools.opendecider_yes_no.execute(
    { state: TICKET, question: "Does the customer threaten to leave?" }, {})));

  try {
    await tools.opendecider_choose.execute({ state: TICKET, question: "Which team?", options: ["billing"] }, {});
  } catch (e) {
    console.log("invalid call ->", e.message);   // inside an agent, Mastra hands this back to the model
  }
} finally {
  await mcp.disconnect();
}
