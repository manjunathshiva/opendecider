import { describe, expect, it } from "vitest";
import { DEFAULTS, FOCUS_HIDES, KIND_NAMES, RULE_CHARS, readSettings } from "../src/settings.js";

describe("readSettings", () => {
  it("gives the defaults for nothing stored, or anything that is not an object", () => {
    for (const raw of [undefined, null, 3, "x", []]) expect(readSettings(raw)).toEqual(DEFAULTS);
    expect(DEFAULTS.hide).toEqual([...FOCUS_HIDES]);
  });

  it("keeps only known kinds, once each", () => {
    expect(readSettings({ hide: ["music", "nope", "music", 7, "gaming"] }).hide).toEqual(["music", "gaming"]);
    expect(readSettings({ hide: [] }).hide).toEqual([]);
    expect(readSettings({ hide: "music" }).hide).toEqual(DEFAULTS.hide);
  });

  it("keeps strictness between 0.2 and 0.8, and a number", () => {
    expect(readSettings({ strictness: 0 }).strictness).toBe(0.2);
    expect(readSettings({ strictness: 5 }).strictness).toBe(0.8);
    expect(readSettings({ strictness: Number.NaN }).strictness).toBe(0.5);
    expect(readSettings({ strictness: "0.7" }).strictness).toBe(0.5);
  });

  it("puts a rule on one line and shortens it", () => {
    expect(readSettings({ rule: "  crypto,\n\tcelebrity   gossip " }).rule).toBe("crypto, celebrity gossip");
    expect(readSettings({ rule: "x".repeat(500) }).rule).toHaveLength(RULE_CHARS);
    expect(readSettings({ ruleMode: "only" }).ruleMode).toBe("only");
    expect(readSettings({ ruleMode: "delete" }).ruleMode).toBe("hide");
  });

  it("names eleven kinds", () => expect(KIND_NAMES).toHaveLength(11));
});
