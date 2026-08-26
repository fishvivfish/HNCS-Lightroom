# HNCS Color algorithm

This document describes the final method used by this repository. It focuses on
the scientific and numerical structure of the implementation rather than the
history of intermediate experiments.

## 1. Design goal

The objective is to reproduce the white-balance-dependent colour rendering of
the recovered Phocus 4.0.1 HNCS path inside Adobe Lightroom while preserving
the camera-specific Adobe Standard raw interpretation.

The resulting pipeline is deliberately two-stage:

```text
camera RAW
-> camera-specific Adobe Standard rendering
-> HNCS colour rendering
-> optional HNCS Film Tone
```

The camera profile is therefore not replaced by a generic Sony-to-Hasselblad
matrix. Adobe Standard remains responsible for the first-stage camera colour
reconstruction. HNCS is implemented as the second-stage rendering transform.

For a different camera, the HNCS target remains the same, but the carrier and
HueSatMap solution must be solved again from that camera's own Adobe Standard
profile.

## 2. Recovered WB-dependent HNCS target

The colour target is the Kelvin-indexed HNCS ColorCorrect path recovered from
Phocus 4.0.1 / X2D 100C Standard.

The recovered selector has two temperature-dependent components:

- ColorCorrect LUT anchors at 3100 K, 5550 K and 9100 K;
- camera-matrix anchors at 2400 K, 3100 K, 5550 K and 9100 K.

Phocus converts the requested white-balance temperature to an integer Kelvin
value before selecting parameters. The recovered implementation then applies
piecewise-linear interpolation between the surrounding anchors. Outside the
inner anchors, the same outer segment is used for extrapolation after the
selector input is clamped: 2000-10000 K for the LUT path and 2000-8000 K for
the recovered matrix path.

The HNCS ColorCorrect stage itself is Kelvin-dependent; it does not use the
Lightroom tint value as an HNCS parameter.

The public source contains the evaluator for this recovered model in
`hncs_core/phocus401_wb.py`. The raw recovered numerical records are kept as a
local research input and are not distributed in the repository.

## 3. Why Adobe Standard is preserved

A Sony RAW file first needs a camera-specific mapping from sensor RGB to a
well-defined colour space. The installed Adobe Standard profile already
contains that camera characterization, including its dual-illuminant
ColorMatrix, ForwardMatrix, HueSatMap and LookTable behaviour.

Removing this stage would change the problem from

```text
camera colour reconstruction -> HNCS rendering
```

to a new camera-characterization problem. The final HNCS solution is therefore
constructed around the existing Adobe rendering rather than replacing it.

The reference implementation reproduces the relevant DNG SDK colour-spec path,
including matrix normalization, four-decimal matrix storage, ForwardMatrix
normalization, white adaptation, camera neutral handling and the iterative
`NeutralToXY` conversion used by the SDK.

This is important because Lightroom does not simply interpolate matrices from a
nominal Kelvin number. The effective HSM interpolation weights depend on the
actual DNG white/neutral path.

## 4. Constrained three-slot carrier

Adobe Standard for the Sony ILCE-7RM5 is a dual-illuminant profile, but a
WB-dependent HNCS implementation needs three independently stored HueSatMaps.
DNG triple-illuminant profiles provide three HSM slots, so the dual Adobe
characterization is embedded into a formal three-slot carrier.

Let the original Adobe endpoints be

```text
CM_A, CM_D65
FM_A, FM_D65
```

For each formal slot `i`, the new matrices are restricted to the original Adobe
line:

```text
CM_i = c_i CM_A   + (1 - c_i) CM_D65
FM_i = c_i FM_A   + (1 - c_i) FM_D65
```

with `0 <= c_i <= 1`.

This restriction is fundamental. It gives the DCP three interpolation slots
without introducing a new camera characterization. The new matrices can move
only along the one-dimensional family already defined by Adobe.

If the triple-illuminant interpolation weights are `w_i(T)`, the effective
position on the original Adobe line is

```text
g_hat(T) = sum_i w_i(T) c_i
```

The original dual profile has its own Adobe interpolation coordinate `g(T)`.
For any fixed set of descriptor whites, the coefficients `c_i` are therefore
projected by bounded least squares so that

```text
g_hat(T) ~= g(T).
```

The same coefficients are used for both ColorMatrix and ForwardMatrix.

Because `NeutralToXY` depends on the candidate matrices, and the resulting white
position changes the triple-illuminant weights, this projection is iterated:
construct the slots, recompute the SDK-exact weights, solve the bounded least
squares problem again, and continue until the coefficients stabilize.

The three descriptor temperatures and tints are optimization variables that
control the geometry of the DNG triple-illuminant interpolation. They are not
physical Hasselblad illuminants and should not be interpreted as factory
calibration temperatures.

## 5. SDK-exact triple-illuminant weights

The carrier uses the DNG SDK triple-illuminant weighting rule rather than a
custom Kelvin interpolation.

For each query white, Lightroom's temperature/tint coordinates are converted to
the DNG interpolation coordinates. The three descriptor whites define three
centres in that space. Inverse-distance weights are computed, passed through the
SDK smooth-step and cutoff operations, and renormalized.

The HSM blend then follows the SDK float32 arithmetic contract:

```text
HSM(T) = w1(T) B1 + w2(T) B2 + w3(T) B3.
```

The complete temperature/tint conversion and triple-weight implementation is in
`hncs_core/adobe_triple_illuminant.py`.

For validation against a real RAW white balance, the candidate profile is not
queried directly with the nominal Kelvin value. The original Adobe profile is
first evaluated at that white to obtain its camera neutral. The candidate
profile then solves `NeutralToXY` for the same camera neutral and uses the
resulting candidate white and interpolation weights. This matches the host path
that matters for the final DCP.

## 6. Fixed 5550 K stage and WB residual

The downstream Creative Profile contains the fixed 5550 K HNCS colour stage.
The three camera-profile HSMs therefore do not need to encode the complete HNCS
rendering independently at every white balance. They carry the WB-dependent
preimage needed so that, after the fixed downstream 5550 K stage, the final
result matches HNCS at the requested temperature.

Conceptually, the HNCS-specific temperature term is

```text
Delta_HNCS(T) = HNCS(T) - HNCS(5550 K).
```

At 5550 K this differential term vanishes. The actual HSM payload at 5550 K is
not necessarily an identity map: it also has to preserve the camera-specific
Adobe HSM behaviour and compensate the exact coordinate mapping of the
three-slot carrier.

The solver therefore uses a hard 5550 K payload anchor rather than forcing the
5550 K HSM to identity.

## 7. The exact target seen by the HSM solver

The final solver works in the actual Lightroom-style rendering order.

For a source sample `x`, white balance `T` and exposure offset `e`, the reference
path is:

```text
x
-> original Adobe camera state at T
-> original Adobe HSM
-> Adobe Standard LookTable
-> recovered HNCS(T)
```

The candidate path is:

```text
x
-> candidate WB-dependent HSM(T)
-> Adobe Standard LookTable
-> fixed recovered HNCS(5550 K)
```

The candidate HSM is therefore solved as a preimage through the complete fixed
downstream chain, not as a simple difference between two LUT arrays.

This distinction is especially important around gamut boundaries and
highlights, where the downstream transforms are nonlinear.

## 8. HDR HueSatMap representation

The final HSM resolution is fixed at

```text
72 x 32 x 32
```

for hue, saturation and value.

The solver covers exposure offsets

```text
EV = -3, -2, -1, 0, +1, +2, +3.
```

A reversible overrange encoding is used so the HSM can represent values above
normal SDR range while still being interpolated on the DNG HueSatMap domain.
The DCP also carries the corresponding dynamic-range hint.

For each Kelvin state, the ideal HSM preimage is solved against all seven
exposure levels. During this inversion, exposure states are scaled by

```text
weight(e) = 2^(-p e),    p = 0.53.
```

`p=0.53` is a frozen numerical setting of the final solver. It stabilizes the
multi-exposure inverse problem; the final validation metric itself is still
computed per state and is not replaced by this weighted objective.

The ideal preimage is obtained by iterative local inversion of the downstream
Adobe LookTable + fixed-5550 HNCS transform. The resulting RGB mapping is then
converted to HSM hue shift, saturation scale and value scale.

Legal neutral/value cells are fixed explicitly, and unused upper value slices
are extended from the highest solved slice rather than filled by unconstrained
extrapolation.

## 9. Variable projection for the three HSM bases

For a chosen carrier, let `Y_T` be the exact ideal HSM payload required at each
active temperature, and let `w(T)` be the corresponding three DNG interpolation
weights.

The three stored basis maps `B1, B2, B3` should satisfy

```text
w1(T) B1 + w2(T) B2 + w3(T) B3 ~= Y_T.
```

The 5550 K state is imposed exactly:

```text
w(5550) B = Y_5550.
```

After this affine constraint, only two independent basis directions remain.
The solver constructs the two-dimensional nullspace of the 5550 K weight row and
solves the remaining HSM coefficients by linear least squares in that
nullspace.

This is the separable part of the optimization: for every nonlinear carrier
trial, the best HSM bases are recomputed analytically instead of being treated
as millions of independent nonlinear optimization variables.

In other words:

```text
outer variables: descriptor whites / carrier geometry
inner variables: best HSM bases for that carrier
```

The inner problem is projected out. This is the Variable Projection structure
used by `tools/optimize_hsm_varpro.py` and the final refinement code.

## 10. Carrier-dependent ideal refresh

The exact ideal HSM preimage is weakly dependent on the carrier itself. Changing
the descriptor whites or carrier matrices changes the camera-to-PCS mapping,
which changes the coordinates in which the downstream preimage must be solved.

Therefore an improvement measured against an old ideal-HSM cache is only
provisional.

The final refinement uses repeated cycles:

```text
1. build exact ideal HSMs for the current carrier;
2. optimize the carrier with those ideals;
3. rebuild the ideals at the moved carrier;
4. re-evaluate and refine again;
5. accept only after an exact refresh confirms the improvement.
```

This prevents the optimizer from exploiting a stale target representation.

## 11. Adobe-base preservation as an epsilon constraint

The HNCS fit is not allowed to improve by damaging the underlying Adobe camera
characterization.

The carrier is separately evaluated against the original Adobe Standard base.
The metric is the 95th percentile OKLab error over the camera sample set for each
temperature.

The final constraint is

```text
worst AdobeBase P95 <= 0.004.
```

This is an epsilon constraint, not a weighted penalty. A candidate that violates
`0.004` is infeasible regardless of how much it improves the HNCS error.

During exploratory optimization a smaller Adobe-base sample sketch can be used
for speed, but selected candidates are rechecked with 40,000 samples per
Kelvin.

## 12. Minimax objective and exchange active set

The HNCS objective is the worst active-state OKLab P95 over temperature and the
seven exposure levels. The optimization therefore targets the tail rather than
only the mean error.

Running the full 2400-10000 K integer grid inside every nonlinear optimization
step would be unnecessarily expensive. The final method uses an exchange-style
active set:

```text
1. optimize on a compact set of difficult Kelvin states;
2. scan the complete integer-Kelvin domain;
3. locate new worst or near-worst temperatures;
4. add them to the active set;
5. refine again;
6. repeat until the full scan no longer reveals a new limiting region.
```

The frozen ILCE-7RM5 workset is stored in `config/final_active_set.json`. It
contains endpoints, the 5550 K hard anchor, recovered HNCS anchor regions and
additional temperatures found by the exhaustive exchange scans.

The active set is only an optimization device. The final profile is always
re-audited over the full domain.

## 13. Final audit

For Sony ILCE-7RM5 the final audit evaluates every integer Kelvin from
2400 K through 10000 K and all seven exposure states.

The HNCS test grid contains both deterministic HSV lattice samples and random
RGB samples. Adobe-base preservation is checked with 40,000 samples per
Kelvin.

The final published ILCE-7RM5 result is summarized in
`docs/VALIDATION_ILCE7RM5.md`.

The final validation is followed by a real Lightroom host test. Offline agreement
is necessary but is not treated as sufficient evidence that the DCP/XMP pair is
correctly interpreted by Lightroom.

## 14. Film Tone

`HNCS Color` contains the WB-dependent HNCS colour rendering only.

`HNCS` uses the same colour solution and adds the fixed HNCS Film Tone curve as
a downstream Creative Profile tone stage. The tone curve does not participate
in the carrier/HSM optimization and does not change the solved WB-dependent
colour transform.

The fixed-Daylight profiles use the same distinction:

```text
HNCS Daylight Color  = Daylight colour only
HNCS Daylight        = Daylight colour + Film Tone
```

## 15. Porting the method to another camera

The algorithm is intended to be camera-specific but structurally portable.
Supporting another camera should keep the recovered HNCS target and the solver
architecture fixed while replacing the camera-dependent Adobe Standard input.

For a new camera:

```text
1. load that camera's original Adobe Standard DCP;
2. reconstruct its dual CM/FM/HSM/LookTable behaviour;
3. create the constrained three-slot Adobe-line carrier;
4. optimize descriptor geometry and c_i for that camera;
5. rebuild the exact ideal HSM preimages;
6. solve the three HSM bases by variable projection;
7. enforce AdobeBase P95 <= 0.004;
8. run the exchange/minimax refinement;
9. audit 2400-10000 K at 1 K spacing and EV -3..+3;
10. perform a real Lightroom host test.
```

The final ILCE-7RM5 descriptor values, coefficients and HSM payload are therefore
camera-specific results, not universal HNCS constants. They may be useful as a
warm start for a related camera, but they are not a substitute for a new
optimization.

## 16. Source map

The main implementation pieces are:

- `hncs_core/phocus401_wb.py` — recovered Kelvin-indexed Phocus model;
- `hncs_core/adobe_triple_illuminant.py` — DNG temperature/tint conversion and
  triple-illuminant HSM weights;
- `hncs_core/adobe_sdk_color.py` — camera colour-spec and `NeutralToXY` path;
- `tools/hncs_full_probe.py` — exact HNCS target, HDR HSM preimage and metrics;
- `tools/optimize_carrier.py` — carrier optimization;
- `tools/optimize_hsm_varpro.py` — variable-projection HSM solve;
- `tools/refine_final_profile.py` — exact-refresh final refinement;
- `tools/audit_profile.py` — exhaustive final audit;
- `tools/build_profile.py` — final DCP/XMP construction.

The numerical result for each supported camera lives separately from the
algorithm so that the same method can be re-run for additional camera models.
