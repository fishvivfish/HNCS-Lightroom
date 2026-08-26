# Build

## Rebuild the published Sony ILCE-7RM5 profile

Put the original Adobe Standard DCP at:

```text
local_assets/Sony ILCE-7RM5 Adobe Standard.dcp
```

Then run:

```bash
python tools/build_profile.py
```

Output:

```text
build/sony-ilce-7rm5/HNCS Base - ILCE-7RM5.dcp
build/sony-ilce-7rm5/HNCS Color.xmp
```

The final carrier/HSM payload is already included in the repository. Rebuilding
the published profile does not require the raw Phocus extraction data.

## Research tools

`tools/` also contains the optimization and audit code used to obtain and verify
the final payload. Full re-optimization requires local recovered Phocus WB data,
which is not distributed here.
