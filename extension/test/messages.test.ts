import { describe, expect, it } from "vitest";
import { CHANNEL_CHARS, MAX_VIDEOS, TITLE_CHARS, readVideos } from "../src/messages.js";

describe("readVideos (what a YouTube page sends)", () => {
  it("keeps well-formed videos and drops the rest", () => {
    const list = [
      { id: "kqtD5dpn9C8", title: " Python\n tutorial ", channel: "Mosh" },
      { id: "bad id!!!!!", title: "x" },
      { id: "kqtD5dpn9C8", title: "duplicate" },
      { id: "aaaaaaaaaa1", title: "   " },
      { id: "aaaaaaaaaa2", title: 5 },
      null,
      "aaaaaaaaaa3",
      { id: "aaaaaaaaaa4", title: "no channel", channel: { evil: true } },
    ];
    expect(readVideos(list)).toEqual([
      { id: "kqtD5dpn9C8", title: "Python tutorial", channel: "Mosh" },
      { id: "aaaaaaaaaa4", title: "no channel", channel: "" },
    ]);
    expect(readVideos("nope")).toEqual([]);
  });

  it("caps the count and the lengths", () => {
    const many = Array.from({ length: 40 }, (_, i) => ({
      id: `v${String(i).padStart(10, "0")}`,
      title: "t".repeat(1000),
      channel: "c".repeat(1000),
    }));
    const out = readVideos(many);
    expect(out).toHaveLength(MAX_VIDEOS);
    expect(out[0]!.title).toHaveLength(TITLE_CHARS);
    expect(out[0]!.channel).toHaveLength(CHANNEL_CHARS);
  });
});
