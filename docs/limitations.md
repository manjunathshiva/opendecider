# Limitations

What OpenDecider is not good at yet, measured on the same benchmarks as everything else.

- **Phishing detection is the weakest task:** 0.63–0.65 on Laya's battery for every OpenDecider model, against Jev's
  0.90 and Laya's 0.98 (Laya trained on that dataset).
- **TypeSafe Jev leads Laya's application battery** (0.774 vs 0.725 medium-td, 0.702 small, 0.656 nano).
- **opendecider-small is zero-shot on typed-decisions** and trails Jev there (0.672 vs 0.754). Use small-td or nano for
  business workflows like those.
- **opendecider-nano trails Laya on Laya's battery overall** (0.656 vs 0.695), because half its tasks are Laya's
  training data.
- **medium-td needs about 61 GB and large-td about 160 GB of GPU memory** across NVIDIA GPUs; neither has a Mac build.
- **large-td is not more accurate than medium-td** on unseen decisions (0.750 vs 0.765); it is better calibrated.
- **The Qwen-based models answer questions one at a time** (small: about 137 ms per question on a Mac, about 40 ms on an
  L40S). Use nano when you need many decisions per second, or `--small-batch 16` when serving.
- **More than 26 options:** the Qwen-based models switch from reading lettered options to scoring each option name,
  which is weaker (BANKING77's 78 options still score 0.70 for small, but on long rule-based states it can fall close
  to chance). A shortlist-then-letters fix is planned, measured on every benchmark before it ships. Through a model
  server (LM Studio, Ollama, vLLM) the limit is 26 options.
- **English only so far.** The training data includes some Spanish, German, French, Portuguese, Italian and Dutch, but
  no multilingual evaluation has been run.
- **Terse labels are harder.** Give options a short description when you can.
- **Benchmark labels are imperfect.** typed-decisions' gold labels come from a ~4B teacher whose own fresh samples agree
  with them 73.5% of the time, so read fine-tuned scores near 0.8 as fitting those workflows, not as general
  superiority.

Found another weakness? [Open an issue](https://github.com/manjunathshiva/opendecider/issues) with an example.
