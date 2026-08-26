#!/usr/bin/env python3
"""Build the camera-specific HNCS Color DCP/XMP from the original Adobe DCP.

The normal reproduction path uses the bundled project-derived final carrier/HSM
payload under data/sony-ilce-7rm5/. It therefore does not require any numbered
development DCP or recovered Phocus dump. The user supplies only the original
Sony ILCE-7RM5 Adobe Standard DCP when rebuilding the camera profile locally.

Full re-optimization from recovered Phocus WB data remains a separate research
workflow documented in docs/BUILD.md.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
import sys

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hncs_core.adobe_sdk_color import TripleColorSpec, adobe_line_slots
from hncs_core.adobe_triple_illuminant import (
    IlluminantDescriptor,
    temperature_tint_to_xy,
    xy_to_temperature_tint,
)
from tools.carrier_model import Experiment

TYPE_SIZE = {2: 1, 3: 2, 4: 4, 7: 1, 10: 8, 11: 4}
HSM_DIMS = (72, 32, 32)
PROFILE_NAME = "HNCS Base (ILCE-7RM5)"
DCP_FILENAME = "HNCS Base - ILCE-7RM5.dcp"
XMP_NAME = "HNCS Color"
XMP_FILENAME = "HNCS Color.xmp"
# DNG 1.7 ProfileDynamicRange payload used by the validated HDR/overrange path:
# version=1, min=1, HintMaxOutputValue=16.0f.
PROFILE_DYNAMIC_RANGE = bytes.fromhex("0100010000008041")


def _encode_tag(tag):
    typ = int(tag.dtype)
    value = tag.value
    if typ == 2:
        data = value.encode("utf-8") + b"\0" if isinstance(value, str) else bytes(value)
        count = len(data)
    elif typ == 3:
        vals = (value,) if np.isscalar(value) else tuple(value)
        data = struct.pack("<" + "H" * len(vals), *map(int, vals))
        count = len(vals)
    elif typ == 4:
        vals = (value,) if np.isscalar(value) else tuple(value)
        data = struct.pack("<" + "I" * len(vals), *map(int, vals))
        count = len(vals)
    elif typ == 7:
        data = bytes(value)
        count = len(data)
    elif typ == 10:
        vals = tuple(value)
        data = struct.pack("<" + "i" * len(vals), *map(int, vals))
        count = len(vals) // 2
    elif typ == 11:
        arr = np.asarray(value, dtype="<f4").reshape(-1)
        data = arr.tobytes()
        count = len(arr)
    else:
        raise ValueError((tag.code, typ))
    return tag.code, typ, count, data


def _read_tags(path: Path):
    with tifffile.TiffFile(path) as tf:
        return {tag.code: _encode_tag(tag) for tag in tf.pages[0].tags.values()}


def _set_ascii(tags, code: int, text: str):
    data = text.encode("utf-8") + b"\0"
    tags[code] = (code, 2, len(data), data)


def _set_short(tags, code: int, value: int):
    tags[code] = (code, 3, 1, struct.pack("<H", int(value)))


def _set_long(tags, code: int, value: int):
    tags[code] = (code, 4, 1, struct.pack("<I", int(value)))


def _set_long3(tags, code: int, values):
    vals = tuple(map(int, values))
    tags[code] = (code, 4, len(vals), struct.pack("<" + "I" * len(vals), *vals))


def _set_float(tags, code: int, array):
    arr = np.asarray(array, dtype="<f4").reshape(-1)
    tags[code] = (code, 11, len(arr), arr.tobytes())


def _set_undefined(tags, code: int, data: bytes):
    tags[code] = (code, 7, len(data), bytes(data))


def _set_srational_matrix(tags, code: int, matrix):
    # Match the Adobe SDK storage precision used by the validated simulator.
    nums = np.rint(np.asarray(matrix, dtype=np.float64).reshape(-1) * 10000.0).astype(np.int32)
    vals = []
    for n in nums:
        vals.extend((int(n), 10000))
    tags[code] = (code, 10, 9, struct.pack("<" + "i" * 18, *vals))


def _set_illuminant_xy(tags, code: int, xy):
    den = 100_000_000
    nx = int(round(float(xy[0]) * den))
    ny = int(round(float(xy[1]) * den))
    raw = struct.pack("<Hiiii", 0, nx, den, ny, den)
    tags[code] = (code, 7, len(raw), raw)


def _write_dcp(path: Path, tags):
    ordered = sorted(tags.values(), key=lambda item: item[0])
    n = len(ordered)
    data_start = 8 + 2 + 12 * n + 4
    data_start += (-data_start) % 4
    entries = []
    blob = bytearray()
    for code, typ, count, data in ordered:
        if len(data) != TYPE_SIZE[typ] * count:
            raise ValueError((code, typ, count, len(data)))
        if len(data) <= 4:
            field = data + b"\0" * (4 - len(data))
        else:
            while (data_start + len(blob)) % 4:
                blob.append(0)
            field = struct.pack("<I", data_start + len(blob))
            blob.extend(data)
        entries.append(struct.pack("<HHI", code, typ, count) + field)
    raw = bytearray(b"IIRC") + struct.pack("<I", 8) + struct.pack("<H", n)
    raw += b"".join(entries) + struct.pack("<I", 0)
    while len(raw) < data_start:
        raw.append(0)
    raw.extend(blob)
    path.write_bytes(raw)


def _rat_matrix(value) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    return (raw[:, 0] / raw[:, 1]).reshape(3, 3)


def _decode_illuminant(value):
    kind, nx, dx, ny, dy = struct.unpack("<Hiiii", bytes(value))
    return kind, (nx / dx, ny / dy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-dcp",
        type=Path,
        default=ROOT / "local_assets" / "Sony ILCE-7RM5 Adobe Standard.dcp",
    )
    parser.add_argument(
        "--solution",
        type=Path,
        default=ROOT / "data" / "sony-ilce-7rm5" / "final_profile_payload.npz",
    )
    parser.add_argument(
        "--xmp-template",
        type=Path,
        default=ROOT / "profiles" / "sony-ilce-7rm5" / "HNCS Color.xmp",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "sony-ilce-7rm5",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    solution = np.load(args.solution)
    values = np.asarray(solution["values"], dtype=np.float64)
    bases = np.asarray(solution["bases"], dtype=np.float32).copy()
    descriptor_xy = np.asarray(solution["descriptor_xy"], dtype=np.float64)
    if values.shape != (9,) or bases.shape != (3, 32, 72, 32, 3):
        raise RuntimeError(f"unexpected final solution shape: values={values.shape}, bases={bases.shape}")

    exp = Experiment(args.original_dcp, sample_count=40000)

    descriptor_values = values[:6]
    coefficients = values[6:]
    descriptors = tuple(
        IlluminantDescriptor(float(descriptor_values[i]), float(descriptor_values[i + 3]))
        for i in range(3)
    )
    colors, forwards = adobe_line_slots(exp.cm1, exp.cm2, exp.fm1, exp.fm2, coefficients)

    # Start from the original Adobe camera profile, not an internal intermediate DCP.
    tags = _read_tags(args.original_dcp)
    _set_ascii(tags, 50936, PROFILE_NAME)
    _set_ascii(tags, 50942, "Unofficial HNCS reimplementation; see repository NOTICE")
    _set_ascii(tags, 52552, "HNCS")
    _set_long3(tags, 50937, HSM_DIMS)
    _set_long(tags, 51107, 0)  # ProfileHueSatMapEncoding = Linear
    _set_undefined(tags, 52551, PROFILE_DYNAMIC_RANGE)

    for code, matrix in zip((50721, 50722, 52531), colors):
        _set_srational_matrix(tags, code, matrix)
    for code, matrix in zip((50964, 50965, 52532), forwards):
        _set_srational_matrix(tags, code, matrix)
    for code in (50778, 50779, 52529):
        _set_short(tags, code, 255)  # CalibrationIlluminant = Other
    for code, white_xy in zip((52533, 52534, 52535), descriptor_xy):
        _set_illuminant_xy(tags, code, white_xy)
    for code, payload in zip((50938, 50939, 52537), bases):
        _set_float(tags, code, payload)

    dcp = args.output / DCP_FILENAME
    _write_dcp(dcp, tags)

    # Use the public Daylight Color XMP as the fixed 5550 K downstream LookTable template.
    text = args.xmp_template.read_text(encoding="utf-8")
    text = re.sub(r'x:xmptk="[^"]*"', 'x:xmptk="HNCS-Lightroom"', text, count=1)
    text = re.sub(r'crs:CameraProfile="[^"]+"', f'crs:CameraProfile="{PROFILE_NAME}"', text, count=1)
    text = re.sub(r'crs:Cluster="[^"]+"', 'crs:Cluster="HNCS"', text, count=1)
    text = re.sub(
        r'crs:Copyright="[^"]*"',
        'crs:Copyright="Unofficial HNCS reimplementation; see repository NOTICE"',
        text,
        count=1,
    )
    for tag in ("Name", "ShortName", "SortName"):
        text = re.sub(
            rf'(<crs:{tag}>\s*<rdf:Alt>\s*<rdf:li xml:lang="x-default">).*?(</rdf:li>)',
            rf'\g<1>{XMP_NAME}\2',
            text,
            count=1,
            flags=re.S,
        )
    text = re.sub(
        r'(<crs:Group>\s*<rdf:Alt>\s*<rdf:li xml:lang="x-default">).*?(</rdf:li>)',
        r'\g<1>HNCS\2',
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r'(<crs:Description>\s*<rdf:Alt>\s*<rdf:li xml:lang="x-default">).*?(</rdf:li>)',
        r'\g<1>WB-dependent HNCS color rendering for Sony ILCE-7RM5 RAW files. Film Tone is not included.\2',
        text,
        count=1,
        flags=re.S,
    )
    xmp = args.output / XMP_FILENAME
    xmp.write_text(text, encoding="utf-8")

    raw = dcp.read_bytes()
    if raw[:4] != b"IIRC":
        raise RuntimeError("bad DCP magic")
    with tifffile.TiffFile(dcp) as tf:
        t = tf.pages[0].tags
        dims = tuple(map(int, t[50937].value))
        parsed_colors = np.stack([_rat_matrix(t[c].value) for c in (50721, 50722, 52531)])
        parsed_forwards = np.stack([_rat_matrix(t[c].value) for c in (50964, 50965, 52532)])
        decoded = [_decode_illuminant(t[c].value) for c in (52533, 52534, 52535)]
        parsed_desc = tuple(
            IlluminantDescriptor(*xy_to_temperature_tint(*xy)) for _, xy in decoded
        )
        parsed_bases = np.stack([
            np.asarray(t[c].value, dtype=np.float32).reshape(32, 72, 32, 3)
            for c in (50938, 50939, 52537)
        ])
        look = np.asarray(t[50982].value, dtype=np.float32).copy()
        profile_name = t[50936].value
        group_name = t[52552].value
        dynamic_range = bytes(t[52551].value)

    # Original Adobe LookTable must remain bitwise identical.
    with tifffile.TiffFile(args.original_dcp) as tf:
        original_look = np.asarray(tf.pages[0].tags[50982].value, dtype=np.float32)
    look_equal = bool(np.array_equal(look, original_look))

    report = {
        "classification": "HNCS Color build; exhaustive 1-K audit required after build",
        "source": {            "solution": str(args.solution),
            "xmp_template": str(args.xmp_template),
        },
        "carrier": {
            "descriptors_stored_roundtrip": [
                {"temperature_K": float(d.temperature), "tint": float(d.tint)}
                for d in parsed_desc
            ],
            "coefficients": [float(x) for x in coefficients],
        },
        "numerical_checkpoint": {
            "hsm_dims": list(HSM_DIMS),
            "p": 0.53,
            "anchor_K": 5550,
            "final_payload_source": "data/sony-ilce-7rm5/final_profile_payload.npz",
        },
        "sanity": {
            "magic": raw[:4].decode("latin1"),
            "profile_name": profile_name,
            "profile_group_name": group_name,
            "looktable_bitwise_preserved_from_original_adobe": look_equal,
            "profile_dynamic_range_hex": dynamic_range.hex(),
            "matrix_roundtrip_max_abs_vs_expected": float(np.max(np.abs(parsed_colors - colors))),
            "forward_roundtrip_max_abs_vs_expected": float(np.max(np.abs(parsed_forwards - forwards))),
            "hsm_roundtrip_max_abs": float(np.max(np.abs(parsed_bases - bases))),
        },
        "artifacts": {
            "dcp": {
                "path": str(dcp),
                "bytes": dcp.stat().st_size,            },
            "xmp": {
                "path": str(xmp),
                "bytes": xmp.stat().st_size,            },
        },
        "packaging": {        },
        "status": "BUILD COMPLETE; RUN tools/audit_profile.py AND REAL LIGHTROOM VALIDATION",
    }
    (args.output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
