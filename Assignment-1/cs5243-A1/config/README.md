# Public validation contracts

`A0.yml` through `A5.yml` are the active Version 1.0 contracts. They control structural, identity, version, artifact, file-safety, notebook, and packaging checks. A1-A5 also define non-executing preflight asset/signature checks and a flexible single extension artifact. `schema.yml` documents the supported keys, including optional image, numeric-array, and ASCII-PLY validation.

Validation never executes student code. Errors block packaging; warnings remain visible in `validation-report.json` for student and grader review.
