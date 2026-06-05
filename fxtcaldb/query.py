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


@dataclass(frozen=True)
class CaldbRequest:
    """One normalized CALDB lookup request."""

    telescope: str
    instrument: str
    detname: str
    filt: str
    codename: str
    start_date: str
    start_time: str
    stop_date: str
    stop_time: str
    expr: str


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


def _parse_prefixed_numeric(token: str) -> tuple[str, str, float | None, float | None] | None:
    """Parse ``PREFIX:value`` or ``PREFIX:min-max`` CBD payloads."""
    match = re.match(
        r"^\s*([A-Z0-9_]+)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*(?:-\s*([+-]?\d+(?:\.\d+)?))?\s*$",
        token,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    prefix = _normalize_name(match.group(1))
    first = float(match.group(2))
    second = match.group(3)
    if second is None:
        return "val", prefix, first, first
    return "range", prefix, first, float(second)


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


def _compare_prefixed_numeric_strings(condition_text: str, candidate_text: str) -> bool:
    """Compare structured CBD strings such as ``G0:12`` and ``G0:0-12``."""
    condition = _parse_prefixed_numeric(condition_text)
    candidate = _parse_prefixed_numeric(candidate_text)
    if condition is None or candidate is None:
        return False
    condition_type, condition_prefix, condition_min, condition_max = condition
    candidate_type, candidate_prefix, candidate_min, candidate_max = candidate
    if condition_prefix != candidate_prefix:
        return False
    if condition_type == "val":
        if candidate_type == "val":
            return candidate_min == condition_min
        return candidate_min is not None and candidate_max is not None and condition_min is not None and candidate_min <= condition_min <= candidate_max
    if condition_type == "range":
        if candidate_type != "range":
            return False
        return (
            candidate_min is not None
            and candidate_max is not None
            and condition_min is not None
            and condition_max is not None
            and candidate_min <= condition_min <= candidate_max
            and candidate_min <= condition_max <= candidate_max
        )
    return False


def _compare_condition(condition: CBDCondition, values: list[CBDValue]) -> bool:
    """Evaluate one parsed condition against row CBD values."""
    if not values:
        return False
    for candidate in values:
        if condition.value_type == "str" and candidate.value_type == "str":
            if candidate.sval.upper() == condition.sval.upper():
                return True
            if _compare_prefixed_numeric_strings(condition.sval, candidate.sval):
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


def _filter_rows_by_metadata(calidx, request: CaldbRequest):
    """Filter CIF rows by mission, detector, filter, codename, and quality."""
    tele_req = _normalize_name(request.telescope)
    inst_req = _normalize_name(request.instrument)
    code_req = _normalize_name(request.codename)
    mask = []
    for row in calidx:
        tele_ok = _normalize_name(_to_text(row["TELESCOP"])) == tele_req
        inst_ok = _normalize_name(_to_text(row["INSTRUME"])) == inst_req
        det_ok = _match_optional_with_aliases(_to_text(row["DETNAM"]), request.detname, _detector_aliases)
        filt_ok = _match_optional_with_aliases(_to_text(row["FILTER"]), request.filt, _filter_aliases)
        code_ok = _normalize_name(_to_text(row["CAL_CNAM"])) == code_req
        qual_ok = int(row["CAL_QUAL"]) == 0
        mask.append(bool(tele_ok and inst_ok and det_ok and filt_ok and code_ok and qual_ok))
    return calidx[np.asarray(mask, dtype=bool)]


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


def _describe_row(row) -> str:
    """Build one compact description for a candidate CALDB row."""
    return (
        f"file={_to_text(row['CAL_FILE'])}, dir={_to_text(row['CAL_DIR'])}, "
        f"ext={int(row['CAL_XNO'])}, det={_to_text(row['DETNAM'])}, "
        f"filter={_to_text(row['FILTER'])}, cbd={_to_text(row['CAL_CBD'])}, "
        f"reftime={float(row['REF_TIME']):.5f}"
    )


def _sample_rows(rows, limit: int = 5) -> str:
    """Return a compact sample of candidate rows for diagnostics."""
    if len(rows) == 0:
        return "<none>"
    return " | ".join(_describe_row(row) for row in rows[:limit])


def _format_request(request: CaldbRequest) -> str:
    """Format one normalized CALDB request for diagnostics."""
    return (
        f"telescope={request.telescope}, instrument={request.instrument}, "
        f"detname={request.detname}, filter={request.filt}, codename={request.codename}, "
        f"start={request.start_date}T{request.start_time}, stop={request.stop_date}T{request.stop_time}, "
        f"expr={request.expr or '<none>'}"
    )


def _raise_lookup_error(
    stage: str,
    request: CaldbRequest,
    paths: CaldbPaths,
    total_rows: int,
    metadata_rows,
    cbd_rows,
    extra: str = "",
) -> None:
    """Raise one detailed CALDB lookup failure."""
    message = (
        f"CALDB lookup failed at stage={stage}; "
        f"CALDB={paths.root}; CALDBCONFIG={paths.config}; CALDBINDEX={paths.index}; "
        f"request=({_format_request(request)}); "
        f"row_counts=(total={total_rows}, metadata={len(metadata_rows)}, cbd={len(cbd_rows)}); "
        f"metadata_candidates=[{_sample_rows(metadata_rows)}]; "
        f"cbd_candidates=[{_sample_rows(cbd_rows)}]"
    )
    if extra:
        message = f"{message}; {extra}"
    raise RuntimeError(message)


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
    request = CaldbRequest(
        telescope=_normalize_name(telescope),
        instrument=_normalize_name(instrument),
        detname=_normalize_optional(detname),
        filt=_normalize_optional(filt),
        codename=_normalize_name(codename),
        start_date=str(start_date).strip(),
        start_time=str(start_time).strip(),
        stop_date=str(stop_date).strip(),
        stop_time=str(stop_time).strip(),
        expr=str(expr).strip(),
    )
    paths = CaldbPaths.resolve(caldb_root=caldb_root, caldb_config=caldb_config)
    with fits.open(paths.index) as hdul:
        calidx = hdul[1].data
    metadata_rows = _filter_rows_by_metadata(calidx, request)
    if len(metadata_rows) == 0:
        _raise_lookup_error("metadata", request, paths, len(calidx), metadata_rows, metadata_rows)
    cbd_rows = _filter_by_expr(metadata_rows, request.expr)
    if len(cbd_rows) == 0:
        _raise_lookup_error("cbd", request, paths, len(calidx), metadata_rows, cbd_rows)
    start_mjd, stop_mjd = _mjd_bounds(request.start_date, request.start_time, request.stop_date, request.stop_time)
    try:
        record = _choose_record(cbd_rows, start_mjd, stop_mjd)
    except RuntimeError as exc:
        _raise_lookup_error("time", request, paths, len(calidx), metadata_rows, cbd_rows, extra=str(exc))
    cal_dir = _to_text(record["CAL_DIR"])
    cal_file = _to_text(record["CAL_FILE"])
    filepath = paths.root
    if cal_dir:
        filepath = f"{filepath}/{cal_dir}"
    return f"{filepath}/{cal_file}", int(record["CAL_XNO"])
