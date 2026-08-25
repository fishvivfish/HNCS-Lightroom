# HNCS Daylight for Lightroom

Two unofficial Lightroom Creative Profiles that bring the publicly recovered
HNCS Daylight color transform and Film Tone to a non-destructive RAW workflow.

## Profiles

| Profile | Rendering | Recommended use |
| --- | --- | --- |
| [HNCS Daylight Color](profiles/HNCS%20Daylight%20Color.xmp) | HNCS Daylight color | Lightroom editing |
| [HNCS Daylight](profiles/HNCS%20Daylight.xmp) | HNCS Daylight color + Film Tone | More finished rendering |

## Installation

1. Download one or both XMP files from [`profiles`](profiles/).
2. Open Lightroom's profile management or profile import interface. The wording
   varies by Lightroom version and platform.
3. Import each XMP as a Creative Profile.
4. Select **HNCS Daylight Color** or **HNCS Daylight** from Profiles.

Both profiles use Adobe Standard as their base. They do not set white balance,
Exposure, Basic-panel edits, HSL adjustments, masks, or local adjustments.

## How it works

The profiles use the publicly recovered HNCS Daylight numerical transform from
[V-Log-Alchemy](https://github.com/shenmintao/V-Log-Alchemy). The color transform
is expressed as a general RGB mapping in a Lightroom Creative Profile. The full
profile also applies the reconstructed HNCS Film Tone through a relative tone
transform.

```text
Camera RAW
  -> Adobe camera-specific RAW interpretation
  -> Adobe Standard
  -> HNCS Daylight color transform
  -> optional HNCS Film Tone
  -> Lightroom output
```

See [Technical Overview](docs/TECHNICAL.md) for implementation details.

## Compatibility

The profiles are not restricted to a specific camera model. Current photographic
validation covers Sony A7R V / ILCE-7RM5 RAW files in Adobe Lightroom for
Android.

## Lighting

The current implementation covers the HNCS Daylight family: daylight, cloudy,
shade, and normal white-light environments. Very warm, narrow-spectrum, or
strongly mixed lighting is less well covered.

## Validation

The HNCS Daylight color transform was numerically checked against the upstream reference over 35,937 samples.

## Limitations

- Warm and tungsten HNCS ColorCorrect numerical assets are not included in the
  current public upstream release.
- Lightroom supplies the camera-specific RAW interpretation and Adobe Standard
  characterization.
- The color transform is carried through Adobe's Creative Profile LookTable.
- Some proprietary Phocus highlight and perceptual gamut behavior is outside the
  public reconstruction.

## Credits

[V-Log-Alchemy](https://github.com/shenmintao/V-Log-Alchemy), by Shen Min Tao and
contributors, provides the public Hasselblad/Phocus reverse-engineering work and
recovered HNCS numerical assets used by this project.

This project is unofficial and is not affiliated with or endorsed by Hasselblad
or Adobe.

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for
upstream attribution and modification information.
