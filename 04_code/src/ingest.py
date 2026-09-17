"""Standard-library readers and validators for Project 1 raw sources."""

from __future__ import annotations

import csv
import gzip
import io
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path


SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    result = 0
    for char in letters.upper():
        result = result * 26 + ord(char) - 64
    return result - 1


def read_xlsx_records(path: Path, sheet_name: str | None = None) -> list[dict[str, str]]:
    """Return records from one XLSX worksheet without third-party packages."""
    ns = {"m": SHEET_NS, "r": REL_NS, "p": PACKAGE_REL_NS}
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{SHEET_NS}}}t"))
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_map = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships
        }
        sheets = workbook.findall("m:sheets/m:sheet", ns)
        selected = next(
            (
                item
                for item in sheets
                if sheet_name is None or item.attrib.get("name") == sheet_name
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"Worksheet not found in {path.name}: {sheet_name}")
        relationship_id = selected.attrib[f"{{{REL_NS}}}id"]
        target = relationship_map[relationship_id].replace("\\", "/")
        worksheet_path = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
        worksheet = ET.fromstring(archive.read(worksheet_path))

    rows: list[list[str]] = []
    for row in worksheet.findall(".//m:sheetData/m:row", ns):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", ns):
            index = _column_index(cell.attrib.get("r", "A1"))
            cell_type = cell.attrib.get("t")
            value_node = cell.find("m:v", ns)
            if cell_type == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.iter(f"{{{SHEET_NS}}}t")
                )
            elif value_node is None:
                value = ""
            elif cell_type == "s":
                value = shared_strings[int(value_node.text or "0")]
            else:
                value = value_node.text or ""
            values[index] = value.strip()
        if values:
            rows.append([values.get(index, "") for index in range(max(values) + 1)])

    if not rows:
        return []
    header = rows[0]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, padded, strict=True)))
    return records


def iter_zip_csv(path: Path) -> Iterator[dict[str, str]]:
    """Iterate the single data CSV in an archive, excluding citation files."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path.name}, found {len(names)}")
        with archive.open(names[0]) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                yield from csv.DictReader(stream)


def iter_gzip_csv(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_country_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_value = ascii_value.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


def parse_age_band(value: str) -> tuple[int, int] | None:
    """Parse GBD labels such as ``20-24 years``."""
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s+years?\s*", value)
    if not match:
        return None
    lower, upper = int(match.group(1)), int(match.group(2))
    return (lower, upper) if lower <= upper else None


def parse_hpv_target_age(value: str) -> tuple[int, int] | None:
    """Parse current-profile or first-round schedule age expressions.

    Supported examples include ``9``, ``9-14``, ``Y9`` and ``Y9-Y14``.
    Month-offset expressions such as ``+M6`` are dose intervals, not target ages.
    """
    cleaned = value.strip().upper()
    if not cleaned or cleaned.startswith("+"):
        return None
    cleaned = cleaned.replace("YEARS", "").replace("YEAR", "").replace("Y", "")
    match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", cleaned)
    if not match:
        return None
    lower = int(match.group(1))
    upper = int(match.group(2) or match.group(1))
    if not (0 <= lower <= upper <= 30):
        return None
    return lower, upper
