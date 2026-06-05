"""CIF/CALDB query helpers copied into the local codebase."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.time import Time

from fxtcaldb.env import CaldbPaths
from fxtcaldb.metadata import normalize_detector, normalize_filter


CBD_COMPARATORS = {
    ".LT.": "LT",
    ".LE.": "LE",
    ".EQ.": "EQ",
    ".GE.": "GE",
    ".GT.": "GT",
}


@dataclass(frozen=True)
class CBDValue:
    """One parsed CALDB CBD token value."""

    name: str
    sval: str
    value_type: str
    min_val: float | None = None
    max_val: float | None = None


@dataclass(frozen=True)
class CBDCondition(CBDValue):
    """One parsed query condition against ``CAL_CBD``."""

    op2: str = "EQ"
    op1: str = "AND"


def _to_text(value: object) -> str:
    """Convert a FITS table value to stripped text."""
    if isinstance(value, bytes):
        return value.decode(errors="ignore").strip()
    return str(value).strip()


def _normalize_name(name: str | None) -> str:
    """Normalize a metadata key or codename for comparisons."""
    return "" if name is None else str(name).strip().upper()


def _is_number(text: str) -> bool:
    """Return whether a token can be parsed as a number."""
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _classify_value(token: str) -> tuple[str, float | None, float | None, str]:
    """Classify one CBD token as text, scalar number, or numeric range."""
    cleaned = token.strip()
    if not cleaned:
        return "str", None, None, ""
    if cleaned.startswith('"') and cleaned.endswith('"'):
        return "str", None, None, cleaned[1:-1]
    if "-" in cleaned:
        left, right = cleaned.split("-", 1)
        if _is_number(left) and _is_number(right):
            return "range", float(left), float(right), cleaned
    if _is_number(cleaned):
        value = float(cleaned)
        return "val", value, value, cleaned
    return "str", None, None, cleaned


def _parse_cbd_entries(text: str) -> dict[str, list[CBDValue]]:
    """Parse one ``CAL_CBD`` text string into structured entries."""
    entries: dict[str, list[CBDValue]] = {}
    if not text or text.strip().upper() == "NONE":
        return entries
    cursor = 0
    length = len(text)
    while cursor < length:
        start = text.find("(", cursor)
        if start == -1:
            break
        name = _normalize_name(text[cursor:start])
        end = text.find(")", start)
        if end == -1:
            break
        payload = text[start + 1 : end]
        for raw_value in payload.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            vtype, vmin, vmax, sval = _classify_value(raw_value)
            entries.setdefault(name, []).append(
                CBDValue(name=name, sval=sval, value_type=vtype, min_val=vmin, max_val=vmax)
            )
        cursor = end + 1
        while cursor < length and text[cursor].isspace():
            cursor += 1
    return entries


def _parse_single_condition(text: str, logical_op: str) -> CBDCondition | None:
    """Parse one query condition token."""
    stripped = text.strip()
    if not stripped:
        return None
    for token, op_name in CBD_COMPARATORS.items():
        pattern = re.compile(rf"(?i)^([A-Z0-9_]+)\s*{re.escape(token)}\s*(.+)$")
        match = pattern.match(stripped)
        if match:
            name = _normalize_name(match.group(1))
            vtype, vmin, vmax, sval = _classify_value(match.group(2).strip())
            return CBDCondition(
                name=name,
                sval=sval,
                value_type=vtype,
                min_val=vmin,
                max_val=vmax,
                op2=op_name,
                op1=logical_op,
            )
    if "(" in stripped and ")" in stripped:
        name = _normalize_name(stripped.split("(")[0])
        payload = stripped[stripped.find("(") + 1 : stripped.find(")")]
        for raw_value in payload.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            vtype, vmin, vmax, sval = _classify_value(raw_value)
            return CBDCondition(
                name=name,
                sval=sval,
                value_type=vtype,
                min_val=vmin,
                max_val=vmax,
                op2="EQ",
                op1=logical_op,
            )
    tokens = stripped.split()
    if len(tokens) >= 2:
        name = _normalize_name(tokens[0])
        vtype, vmin, vmax, sval = _classify_value(" ".join(tokens[1:]))
        return CBDCondition(
            name=name,
            sval=sval,
            value_type=vtype,
            min_val=vmin,
            max_val=vmax,
            op2="EQ",
            op1=logical_op,
        )
    return None


def _parse_expr_conditions(expr: str) -> list[CBDCondition]:
    """Parse a ``CAL_CBD`` query expression."""
    if not expr:
        return []
    parts = re.split(r"(?i)(\.AND\.|\.OR\.)", expr)
    conditions: list[CBDCondition] = []
    pending_op = "AND"
    for part in parts:
        if not part:
            continue
        upper = part.upper()
        if upper == ".AND.":
            pending_op = "AND"
            continue
        if upper == ".OR.":
            pending_op = "OR"
            continue
        condition = _parse_single_condition(part, pending_op)
        if condition is not None:
            conditions.append(condition)
        pending_op = "AND"
    return conditions


def _compare_numeric(op: str, row_value: float | None, target: float | None) -> bool:
    """Compare one numeric CALDB value to a query target."""
    if row_value is None or target is None:
        return False
    if op == "LT":
        return row_value < target
    if op == "LE":
        return row_value <= target
    if op == "EQ":
        return row_value == target
    if op == "GE":
        return row_value >= target
    if op == "GT":
        return row_value > target
    return False


def _compare_range(op: str, row_min: float | None, row_max: float | None, target: float | None) -> bool:
    """Compare one numeric target against a row range."""
    if row_min is None or row_max is None or target is None:
        return False
    if op == "LT":
        return row_min < target
    if op == "LE":
        return row_min <= target
    if op == "EQ":
        return row_min <= target <= row_max
    if op == "GE":
        return row_max >= target
    if op == "GT":
        return row_max > target
    return False


def _compare_condition(condition: CBDCondition, values: list[CBDValue]) -> bool:
    """Evaluate one parsed condition against row CBD values."""
    if not values:
        return False
    for candidate in values:
        if condition.value_type == "str" and candidate.value_type == "str":
            if candidate.sval.upper() == condition.sval.upper():
                return True
        elif condition.value_type == "val":
            if candidate.value_type == "val" and _compare_numeric(condition.op2, candidate.min_val, condition.min_val):
                return True
            if candidate.value_type == "range" and _compare_range(condition.op2, candidate.min_val, candidate.max_val, condition.min_val):
                return True
        elif condition.value_type == "range" and candidate.value_type == "range" and condition.op2 == "EQ":
            cond_min = condition.min_val or 0.0
            cond_max = condition.max_val or cond_min
            if candidate.min_val is not None and candidate.max_val is not None:
                if candidate.min_val <= cond_min <= candidate.max_val and candidate.min_val <= cond_max <= candidate.max_val:
                    return True
    return False


def _evaluate_expr_conditions(expr: str, cal_cbd_text: str) -> bool:
    """Evaluate a query expression against one row ``CAL_CBD`` string."""
    conditions = _parse_expr_conditions(expr)
    if not conditions:
        return True
    row_entries = _parse_cbd_entries(cal_cbd_text)
    result = True
    for condition in conditions:
        match = _compare_condition(condition, row_entries.get(condition.name, []))
        result = (result and match) if condition.op1 == "AND" else (result or match)
    return result


def _normalize_optional(token: str | None) -> str:
    """Normalize optional detector/filter tokens for wildcard matching."""
    if token is None:
        return "NONE"
    stripped = str(token).strip()
    if not stripped or stripped == "-":
        return "NONE"
    return stripped.upper()


def _detector_aliases(value: str) -> set[str]:
    """Return accepted aliases for one detector token."""
    norm = _normalize_optional(value)
    if norm == "NONE":
        return {"NONE"}
    base = normalize_detector(norm)
    aliases = {base}
    if base == "A":
        aliases.add("FXTA")
    elif base == "B":
        aliases.add("FXTB")
    return aliases


def _filter_aliases(value: str) -> set[str]:
    """Return accepted aliases for one filter token."""
    norm = _normalize_optional(value)
    if norm == "NONE":
        return {"NONE"}
    base = normalize_filter(norm)
    aliases = {base}
    if base.isdigit():
        aliases.add(f"{int(base):02d}")
    return aliases


def _match_optional_with_aliases(row_value: str, requested: str, alias_builder) -> bool:
    """Match one CALDB optional field using alias-aware wildcard logic."""
    row_norm = _normalize_optional(row_value)
    req_norm = _normalize_optional(requested)
    if row_norm == "NONE" or req_norm == "NONE":
        return True
    return not alias_builder(row_norm).isdisjoint(alias_builder(req_norm))


def _filter_by_expr(rows, expr: str):
    """Filter selected CIF rows by one CBD expression."""
    if not expr:
        return rows
    mask = []
    for row in rows:
        mask.append(_evaluate_expr_conditions(expr, _to_text(row["CAL_CBD"])))
    return rows[np.asarray(mask, dtype=bool)]


def _filter_rows(calidx, telescope: str, instrument: str, detname: str, filt: str, codename: str, expr: str):
    """Filter CIF rows by mission, detector, filter, codename, quality, and CBD."""
    tele_req = _normalize_name(telescope)
    inst_req = _normalize_name(instrument)
    code_req = _normalize_name(codename)
    mask = []
    for row in calidx:
        tele_ok = _normalize_name(_to_text(row["TELESCOP"])) == tele_req
        inst_ok = _normalize_name(_to_text(row["INSTRUME"])) == inst_req
        det_ok = _match_optional_with_aliases(_to_text(row["DETNAM"]), detname, _detector_aliases)
        filt_ok = _match_optional_with_aliases(_to_text(row["FILTER"]), filt, _filter_aliases)
        code_ok = _normalize_name(_to_text(row["CAL_CNAM"])) == code_req
        qual_ok = int(row["CAL_QUAL"]) == 0
        mask.append(bool(tele_ok and inst_ok and det_ok and filt_ok and code_ok and qual_ok))
    rows = calidx[np.asarray(mask, dtype=bool)]
    return _filter_by_expr(rows, expr) if expr else rows


def _mjd_bounds(start_date: str, start_time: str, stop_date: str, stop_time: str) -> tuple[float | None, float | None]:
    """Convert date/time bounds to MJDs."""
    def _convert(date_str: str, time_str: str) -> float | None:
        if not date_str or date_str.strip() == "-":
            return None
        if date_str.lower() == "now":
            return float(Time.now().mjd)
        clean_time = "00:00:00" if not time_str or time_str.strip() == "-" else time_str
        return float(Time(f"{date_str}T{clean_time}", scale="utc", format="isot").mjd)

    return _convert(start_date, start_time), _convert(stop_date, stop_time)


def _extract_version(filename: str) -> int:
    """Extract the integer ``_vNN`` suffix from one calibration filename."""
    match = re.search(r"_v(\d+)\.", filename)
    return int(match.group(1)) if match else 0


def _choose_record(rows, start_mjd: float | None, stop_mjd: float | None):
    """Choose one best-matching row from an already filtered subset."""
    subset = rows
    if start_mjd is not None:
        subset = subset[subset["REF_TIME"] <= start_mjd]
        if len(subset) == 0:
            raise RuntimeError("No calibration rows found before the requested start time")
        subset = subset[subset["REF_TIME"] == np.max(subset["REF_TIME"])]
    if stop_mjd is not None:
        subset = subset[subset["REF_TIME"] <= stop_mjd]
        if len(subset) == 0:
            raise RuntimeError("No calibration rows satisfy the requested stop time")
    if len(subset) == 0:
        raise RuntimeError("No calibration rows satisfy the requested time range")
    versions = np.asarray([_extract_version(_to_text(name)) for name in subset["CAL_FILE"]], dtype=np.int64)
    sort_idx = np.lexsort((versions, subset["REF_TIME"]))
    return subset[sort_idx][-1]


def find_calibration_file(
    telescope: str,
    instrument: str,
    detname: str,
    filt: str,
    codename: str,
    start_date: str,
    start_time: str,
    stop_date: str,
    stop_time: str,
    expr: str = "",
    caldb_root: str | None = None,
    caldb_config: str | None = None,
) -> tuple[str, int]:
    """Resolve one calibration file path and extension from the local CIF."""
    paths = CaldbPaths.resolve(caldb_root=caldb_root, caldb_config=caldb_config)
    with fits.open(paths.index) as hdul:
        calidx = hdul[1].data
    rows = _filter_rows(calidx, telescope, instrument, detname, filt, codename, expr)
    if len(rows) == 0:
        raise RuntimeError("No calibration rows matched the selection criteria")
    start_mjd, stop_mjd = _mjd_bounds(start_date, start_time, stop_date, stop_time)
    record = _choose_record(rows, start_mjd, stop_mjd)
    cal_dir = _to_text(record["CAL_DIR"])
    cal_file = _to_text(record["CAL_FILE"])
    filepath = paths.root
    if cal_dir:
        filepath = f"{filepath}/{cal_dir}"
    return f"{filepath}/{cal_file}", int(record["CAL_XNO"])
