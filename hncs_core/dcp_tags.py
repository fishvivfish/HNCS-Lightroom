"""Small TIFF/DCP tag reader for the numeric fields used by HNCS Color."""
from __future__ import annotations

from pathlib import Path
import struct

TIFF_TYPE_SIZES = {
    1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
    8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4,
}

TAG_NAMES = {
    50708: "UniqueCameraModel",
    50721: "ColorMatrix1",
    50722: "ColorMatrix2",
    50778: "CalibrationIlluminant1",
    50779: "CalibrationIlluminant2",
    50936: "ProfileName",
    50937: "ProfileHueSatMapDims",
    50938: "ProfileHueSatMapData1",
    50939: "ProfileHueSatMapData2",
    50964: "ForwardMatrix1",
    50965: "ForwardMatrix2",
    50981: "ProfileLookTableDims",
    50982: "ProfileLookTableData",
    51107: "ProfileHueSatMapEncoding",
    51108: "ProfileLookTableEncoding",
}


def _decode_tiff_value(data: bytes, endian: str, field_type: int, count: int):
    if field_type == 2:
        return data[:count].rstrip(b"\0").decode("utf-8", "replace")
    if field_type in (5, 10):
        code = "I" if field_type == 5 else "i"
        raw = struct.unpack(endian + code * (count * 2), data[: count * 8])
        values = [raw[i] / raw[i + 1] for i in range(0, len(raw), 2)]
    else:
        codes = {1: "B", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i", 11: "f", 12: "d", 13: "I"}
        code = codes.get(field_type)
        if code is None:
            return data.hex()
        values = list(struct.unpack(endian + code * count, data[: count * TIFF_TYPE_SIZES[field_type]]))
    return values[0] if count == 1 else values


def read_dcp_tags(path: Path) -> dict[str, object]:
    """Read the first DCP IFD and decode only named numeric/profile tags."""
    payload = Path(path).read_bytes()
    if payload[:2] == b"II":
        endian = "<"
    elif payload[:2] == b"MM":
        endian = ">"
    else:
        raise ValueError(f"not a TIFF/DCP file: {path}")
    ifd_offset = struct.unpack_from(endian + "I", payload, 4)[0]
    entry_count = struct.unpack_from(endian + "H", payload, ifd_offset)[0]
    tags: dict[str, object] = {}
    for index in range(entry_count):
        offset = ifd_offset + 2 + index * 12
        tag, field_type, count = struct.unpack_from(endian + "HHI", payload, offset)
        size = TIFF_TYPE_SIZES.get(field_type, 1) * count
        field = payload[offset + 8 : offset + 12]
        if size <= 4:
            raw = field[:size]
        else:
            data_offset = struct.unpack(endian + "I", field)[0]
            raw = payload[data_offset : data_offset + size]
        name = TAG_NAMES.get(tag)
        if name:
            tags[name] = _decode_tiff_value(raw, endian, field_type, count)
    return tags
