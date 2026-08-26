# Technical overview

HNCS Color uses a two-stage rendering path:

```text
Sony RAW
-> Adobe Standard camera rendering
-> HNCS colour rendering
```

For Sony ILCE-7RM5, the original Adobe dual-illuminant camera characterization
is expanded to three formal DCP slots while keeping every ColorMatrix and
ForwardMatrix on the original Adobe A-D65 line.

The three WB interpolation weights then drive three `72 x 32 x 32` HueSatMaps
that represent the temperature-dependent HNCS residual. The fixed 5550 K HNCS
colour stage remains downstream in the Creative Profile.

The final carrier parameters and HSM payload are included in
`data/sony-ilce-7rm5/final_profile_payload.npz`.
