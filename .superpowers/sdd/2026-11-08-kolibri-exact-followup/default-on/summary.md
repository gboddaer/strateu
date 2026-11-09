# Default-on composition for Kolibri

## Change

The three retained Kolibri knobs are now selected by the engine when their variables are unset:

- `STRATA_KOLIBRI_MMVQ_SELECTED` (exact selected MMVQ)
- `STRATA_KOLIBRI_SKIP_ZERO_OVERFLOW` (existing zero-overflow skip)
- `STRATA_KOLIBRI_ZERO_BUFFER` (immutable zero operand)

`env_on_default_on` is used at these three newly changed call sites; other flags keep their existing use. An
explicit `=0` still turns each one off; the env parser and other flags are unchanged.

## Results (all three unset vs all three explicitly `=0`)

Five alternating 512/128 pairs, same executable and forced histories:

| pair | off decode tok/s | default decode tok/s | delta |
|---|---:|---:|---:|
| 1 | 12.33 | 16.83 | +36.5% |
| 2 | 12.33 | 16.97 | +37.7% |
| 3 | 12.34 | 14.10 | +14.2% |
| 4 | 9.64 | 12.72 | +31.9% |
| 5 | 10.87 | 14.69 | +35.2% |

Paired median **+35.2%**, 5/5 positive; ratio of medians 12.34 -> 14.69 (+19.2%). Complete logits identical
`6ffefa8f...8721` in all ten runs.

Five alternating cache8 pairs: paired median **+2.8%**, 5/5 positive; ratio of medians +2.8%; all hashes
`466f48f2...c3d6`.

Full integration with all three unset: 10/10 short/long/hot/profile dumps finite and byte-identical to the Task 2
control (`default-on/integration/compare.txt`).

Near-boundary long context, 131064 + 8, ctx 131072, all three unset: prefill 19.66 tok/s, decode 6.48 tok/s, wall
6765.6 s, SHA `5c85e7f6...a322`, identical to the previous all-off reference (`final/final-131064-off.bin`).
Previous all-off run was 19.60/5.43 tok/s, wall 6801.9 s: prefill flat, decode +19.3% in this pair, wall -0.5%.

## Scope effects recorded

- `ZERO_BUFFER` allocates K*N floats during every Kolibri scratch initialization, including configurations that do not
  consume it. Allocation precedes auto-fit, so at a tight memory boundary it can change fitted capacity.
- The skip gate no longer follows profile-derived defaults: an unset variable permits skipping even under the
  `baseline` profile, so `baseline` alone is not an all-off control. Use explicit `=0` values for the A/B arm.
- The code is in the Kolibri engine, not restricted to the W7800; only this HIP/gfx1100 host was measured.

## Decision

Keep the three selected by default for Kolibri. All are exact on the tested histories; cache8 and long-context
checks are positive/no-worse. Explicit `=0` remains the fallback. Targeted Task 10 miss/poison/mixed-format tests and
CUDA/SYCL qualification are still unrun. This is a locally qualified default, not release readiness.

No server. No release-readiness claim.
