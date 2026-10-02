// The package ships the repository's LICENSE and NOTICE (npm takes them from this folder).
import { copyFileSync } from "node:fs";

for (const name of ["LICENSE", "NOTICE"])
  copyFileSync(new URL(`../../${name}`, import.meta.url), new URL(`../${name}`, import.meta.url));
