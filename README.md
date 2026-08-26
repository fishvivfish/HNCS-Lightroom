# HNCS for Lightroom

Unofficial HNCS colour profiles for Adobe Lightroom.

## Profiles

### Sony ILCE-7RM5 / A7R V

Use both files in `profiles/sony-ilce-7rm5/`:

- `HNCS Base - ILCE-7RM5.dcp`
- `HNCS Color.xmp` — HNCS colour only
- `HNCS.xmp` — HNCS colour + Film Tone

Select **HNCS Color** or **HNCS** in Lightroom. `HNCS Base` is the technical camera profile,
not the final look by itself.

### Daylight

- `profiles/HNCS Daylight Color.xmp`
- `profiles/HNCS Daylight.xmp`

These are fixed-Daylight Creative Profiles.

## Pipeline

```text
RAW
-> Adobe Standard camera rendering
-> WB-dependent HNCS colour transform
-> Lightroom
```

The Sony/Adobe camera profile remains the first-stage colour restoration.
HNCS is applied as the second-stage colour rendering.

## Rebuild

The repository includes the final Sony ILCE-7RM5 carrier/HSM payload in
`data/sony-ilce-7rm5/final_profile_payload.npz`.

Place your locally installed `Sony ILCE-7RM5 Adobe Standard.dcp` in
`local_assets/`, then run:

```bash
python tools/build_profile.py
```

The original Adobe DCP and raw Phocus extraction data are not included.

## Validation

The Sony ILCE-7RM5 profile was validated from 2400 K to 10000 K at 1 K spacing,
over EV -3 to +3, and tested in Lightroom for Android.

See `docs/TECHNICAL.md`, `docs/BUILD.md`, and `docs/VALIDATION_ILCE7RM5.md`.

## Credits

HNCS/Phocus reverse-engineering work builds on
[V-Log-Alchemy](https://github.com/shenmintao/V-Log-Alchemy).

This project is unofficial and is not affiliated with or endorsed by
Hasselblad, Adobe, or Sony.

## License

Project source code and project-authored material are licensed under Apache-2.0.
See `NOTICE` for third-party attribution.
