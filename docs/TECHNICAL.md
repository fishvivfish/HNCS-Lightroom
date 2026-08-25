# Technical Overview

## Processing Pipeline

The profiles add the recovered HNCS Daylight rendering after Lightroom's
camera-specific RAW interpretation:

```text
Camera RAW
  -> Adobe demosaic, white balance, and camera characterization
  -> Adobe Standard
  -> HNCS Daylight color transform
  -> optional relative HNCS Film Tone
  -> Lightroom output and user edits
```

Adobe Standard is the defined base profile. Lightroom remains responsible for
the camera-specific RAW front end, highlight reconstruction, and scene-referred
working representation.

## HNCS Daylight Color

The color stage comes from the HNCS Daylight numerical transform recovered by
[V-Log-Alchemy](https://github.com/shenmintao/V-Log-Alchemy). Its public processing
structure includes the Daylight ColorCorrect data and Hasselblad RGB working
space.

This project evaluates that transform as a general RGB mapping and carries it in
Adobe's Creative Profile LookTable. Both distributed profiles use the same color
payload. **HNCS Daylight Color** omits Film Tone so Lightroom's normal tonal
workflow remains available.

## Film Tone

Lightroom already applies baseline tonal mapping. The full profile compensates
that mapping before reproducing the recovered HNCS Film Tone:

```text
T_relative = T_HNCS ∘ T_Adobe^(-1)
```

The relative transform is carried by Lightroom's profile tone-curve mechanism.
Adobe does not publish the coordinate definition of this profile property. The
relative tone transform was validated against Lightroom output.

## Lightroom Integration

The XMP files are Creative Profiles, not presets. They explicitly select Adobe
Standard and contain no white-balance values, Basic-panel edits, HSL adjustments,
masks, local adjustments, or camera calibration matrices. Lightroom's editing
adjustments remain available after profile selection.

## Camera Independence

Neither profile contains `CameraModelRestriction`. Adobe supplies the appropriate
camera characterization before the HNCS stages, and the profiles add no
brand-specific camera matrices. Current photographic validation covers Sony
ILCE-7RM5 / A7R V RAW files in Lightroom for Android.

## Validation

The HNCS Daylight color transform was checked over all 35,937 samples of the upstream Daylight/Standard/sRGB reference at the established numerical tolerance.

## Limitations

- The publicly available ColorCorrect data covers the Daylight family. Warm and
  tungsten numerical assets are not currently bundled upstream.
- Adobe's RAW engine and Adobe Standard supply camera-specific characterization
  and highlight reconstruction.
- The Creative Profile LookTable represents the recovered color transform within
  Lightroom; it is not a native Phocus processing environment.
- Some proprietary Phocus highlight rolloff and perceptual gamut-mapping behavior
  is outside the public reconstruction.
- Photographic validation on additional camera models remains future work.

## Upstream Attribution

[V-Log-Alchemy](https://github.com/shenmintao/V-Log-Alchemy), Copyright Shen Min
Tao and contributors, provides the public Hasselblad/Phocus reverse-engineering
work, recovered HNCS numerical assets, and processing structure used here. It is
licensed under Apache License 2.0.

This downstream project adapts that work to Lightroom Creative Profiles. It is
unofficial and is not affiliated with or endorsed by Hasselblad or Adobe. See
`LICENSE` and `NOTICE` in the repository root.
