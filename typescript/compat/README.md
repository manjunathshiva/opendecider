# Oldest tested versions

CI runs the test suite against the oldest versions `@opendecider/client` is tested on, each pinned by its own lockfile
(Dependabot leaves these folders alone, so they stay at the floor):

- `ai5/`: `ai` 5.0.207 (the AI SDK tests; the Mastra tests need `ai/test`'s v3 mock, from `ai` 6)
- `ai6-mastra/`: `ai` 6.0.214 and `@mastra/core` 1.55.0

Each is the earliest release with no known advisory in its install, so these lockfiles add no security alerts to the
repository: `ai` 5.0.207 and 6.0.214 are the first to depend on a fixed `@ai-sdk/provider-utils` (GHSA-866g-f22w-33x8),
and `@mastra/core` 1.55.0 the first without the old `@ai-sdk/ui-utils` (Mastra 1.11 to 1.54 also pass the tests, and
releases before 1.11 drop parts of the tools' input schemas before the model sees them).

```bash
cp -R src test vitest.config.ts tsconfig.json compat/ai6-mastra/ && cd compat/ai6-mastra && npm ci --ignore-scripts && npx vitest run
# ai5/ has no Mastra: run the other test files there
cp -R src test vitest.config.ts tsconfig.json compat/ai5/ && cd compat/ai5 && npm ci --ignore-scripts && \
  npx vitest run test/ai-sdk.test.ts test/client.test.ts test/decisions.test.ts test/parity.test.ts
```

Raise a floor (here, and in package.json's peerDependencies and the docs when it is a requirement) only when a
feature or a security fix needs it.
