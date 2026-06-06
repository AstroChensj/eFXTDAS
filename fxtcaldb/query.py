"""Generic CALDB lookup core for EP-FXT products.

This module owns only the generic pieces of local CALDB access:

- reading observation metadata from FITS products
- basic detector/filter normalization
- parsing the EP/FXT ``caldb.indx`` calibration index
- filtering rows by mission/instrument/detector/filter/codename
- evaluating ``CAL_CBD`` selection expressions
- selecting the best matching row in time/version space

The EP/FXT ``caldb.indx`` file is a FITS binary table. The columns used here
are:

- ``TELESCOP``: mission name, such as ``EP``
- ``INSTRUME``: instrument name, such as ``FXT``
- ``DETNAM``: detector identifier used by that calibration family
- ``FILTER``: filter identifier used by that calibration family
- ``CAL_CNAM``: calibration codename, for example ``SPECRESP`` or ``VIGNET``
- ``CAL_CBD``: structured selection payload such as ``DATAMODE(FF)``
- ``CAL_FILE``: calibration filename relative to ``CAL_DIR``
- ``CAL_DIR``: directory relative to the CALDB root
- ``CAL_XNO``: extension number containing the payload of interest
- ``REF_TIME``: reference time used for time-based row selection
- ``CAL_QUAL``: calibration quality flag; this code uses only rows with value 0

The main ``CAL_CNAM`` values currently consumed by this repository are:

- ``SPECRESP``: base ARF calibration
- ``MATRIX``: RMF redistribution matrix
- ``VIGNET``: vignetting calibration
- ``TELDEF``: detector geometry and optical-axis calibration

EP-FXT uses inconsistent ``DETNAM`` and ``FILTER`` conventions across those
``CAL_CNAM`` values. This module intentionally stays generic and does not apply
codename-specific canonicalization. Product modules such as
``fxtcaldb.response``, ``fxtcaldb.vignetting``, and ``fxtcaldb.optics`` are
responsible for mapping :class:`ObservationMetadata` into the exact
``DETNAM``/``FILTER`` request strings expected by a specific calibration
family.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.time import Time

from fxtcaldb.env import CaldbPaths


CBD_COMPARATORS = {
    ".LT.": "LT",
    ".LE.": "LE",
    ".EQ.": "EQ",
    ".GE.": "GE",
    ".GT.": "GT",
}


def _split_isot(value: str) -> tuple[str | None, str | None]:
    """Split an ISO timestamp into date and time components.

    Parameters
    ----------
    value : str
        FITS timestamp token such as ``2026-01-01T00:00:00``.

    Returns
    -------
    tuple[str | None, str | None]
        ``(date, time)`` with ``None`` for missing pieces.
    """
    cleaned = str(value).strip()
    if not cleaned or cleaned == "-":
        return None, None
    if "T" not in cleaned:
        return cleaned, None
    return tuple(cleaned.split("T", 1))


def _normalize_optional_text(value: object | None) -> str | None:
    """Normalize one optional FITS header token.

    Parameters
    ----------
    value : object | None
        Raw FITS header token.

    Returns
    -------
    str | None
        Stripped text value, or ``None`` when empty.
    """
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _parse_max_grade(header: fits.Header) -> int | None:
    """Recover the highest selected grade from ``DSTYP*``/``DSVAL*``.

    Parameters
    ----------
    header : fits.Header
        FITS header containing optional grade selection keywords.

    Returns
    -------
    int | None
        Highest selected grade, or ``None`` when not encoded.
    """
    for key in header:
        if not key.startswith("DSTYP"):
            continue
        if str(header[key]).strip().upper() != "GRADE":
            continue
        suffix = key[5:]
        dsval = str(header.get(f"DSVAL{suffix}", "")).strip()
        if ":" in dsval:
            try:
                return int(dsval.split(":")[-1].strip())
            except ValueError:
                continue
    return None


def normalize_detector(value: str) -> str:
    """Normalize detector identifiers to the short FXT form.

    Parameters
    ----------
    value : str
        Detector token such as ``FXTA`` or ``A``.

    Returns
    -------
    str
        Canonical short detector code, usually ``A`` or ``B``.
    """
    text = str(value).strip().upper()
    if text in {"A", "FXTA"}:
        return "A"
    if text in {"B", "FXTB"}:
        return "B"
    return text


def normalize_filter(value: str | int | None) -> str | None:
    """Normalize one filter token without applying product policy.

    Parameters
    ----------
    value : str | int | None
        Raw filter token from headers or request inputs.

    Returns
    -------
    str | None
        Generic normalized filter string.
    """
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text == "-":
        return None
    if text in {"NONE", "OPEN", "THIN", "MEDIUM", "HOLE"}:
        return text
    try:
        return str(int(text))
    except ValueError:
        return text


@dataclass(frozen=True)
class ObservationMetadata:
    """Observation metadata needed for calibration selection and geometry."""

    telescope: str | None
    instrument: str | None
    detector_code: str | None
    detnam: str | None
    filt: str | None
    datamode: str | None
    start_date: str | None
    start_time: str | None
    stop_date: str | None
    stop_time: str | None
    max_grade: int | None
    ra_pnt: float | None
    dec_pnt: float | None
    pa_pnt: float | None


def _select_header(path: str, preferred_ext: int | None) -> fits.Header:
    """Select one FITS header from a product for metadata extraction.

    Parameters
    ----------
    path : str
        FITS product path.
    preferred_ext : int | None
        Preferred extension number, if known.

    Returns
    -------
    fits.Header
        Header chosen for metadata extraction.
    """
    with fits.open(path) as hdul:
        if preferred_ext is not None:
            return hdul[preferred_ext].header.copy()
        for ext in (1, 0):
            if ext < len(hdul):
                header = hdul[ext].header
                if "TELESCOP" in header and "INSTRUME" in header:
                    return header.copy()
        return hdul[0].header.copy()


def read_observation_metadata(path: str, preferred_ext: int | None = None) -> ObservationMetadata:
    """Read observation metadata from a FITS product.

    Parameters
    ----------
    path : str
        FITS product path.
    preferred_ext : int | None, optional
        Preferred extension number for products whose science metadata does not
        live in the primary header.

    Returns
    -------
    ObservationMetadata
        Generic observation metadata used by downstream calibration modules.
    """
    header = _select_header(path, preferred_ext)
    detnam = _normalize_optional_text(header.get("DETNAM"))
    datamode = _normalize_optional_text(header.get("DATAMODE"))
    start_date, start_time = _split_isot(header.get("DATE-OBS", ""))
    stop_date, stop_time = _split_isot(header.get("DATE-END", ""))
    return ObservationMetadata(
        telescope=_normalize_optional_text(header.get("TELESCOP")),
        instrument=_normalize_optional_text(header.get("INSTRUME")),
        detector_code=normalize_detector(detnam) if detnam is not None else None,
        detnam=detnam,
        filt=normalize_filter(header.get("FILTER")),
        datamode=datamode.upper() if datamode is not None else None,
        start_date=start_date,
        start_time=start_time,
        stop_date=stop_date,
        stop_time=stop_time,
        max_grade=_parse_max_grade(header),
        ra_pnt=float(header["RA_PNT"]) if "RA_PNT" in header else None,
        dec_pnt=float(header["DEC_PNT"]) if "DEC_PNT" in header else None,
        pa_pnt=float(header["PA_PNT"]) if "PA_PNT" in header else None,
    )


def require_caldb_metadata(metadata: ObservationMetadata) -> None:
    """Validate that metadata can drive CALDB response lookups.

    Parameters
    ----------
    metadata : ObservationMetadata
        Metadata to validate.

    Returns
    -------
    None
        This function raises on failure.
    """
    missing = []
    for field_name, keyword in (
        ("telescope", "TELESCOP"),
        ("instrument", "INSTRUME"),
        ("detector_code", "DETNAM"),
        ("filt", "FILTER"),
        ("datamode", "DATAMODE"),
        ("start_date", "DATE-OBS"),
        ("stop_date", "DATE-END"),
    ):
        if getattr(metadata, field_name) is None:
            missing.append(keyword)
    if metadata.max_grade is None:
        missing.append("DSTYP*/DSVAL* GRADE")
    if missing:
        raise ValueError(f"Missing required CALDB metadata keywords: {', '.join(missing)}")


def require_optaxis_metadata(metadata: ObservationMetadata) -> None:
    """Validate that metadata can project the detector optical axis.

    Parameters
    ----------
    metadata : ObservationMetadata
        Metadata to validate.

    Returns
    -------
    None
        This function raises on failure.
    """
    missing = []
    for field_name, keyword in (
        ("telescope", "TELESCOP"),
        ("instrument", "INSTRUME"),
        ("detector_code", "DETNAM"),
        ("start_date", "DATE-OBS"),
        ("stop_date", "DATE-END"),
        ("ra_pnt", "RA_PNT"),
        ("dec_pnt", "DEC_PNT"),
        ("pa_pnt", "PA_PNT"),
    ):
        if getattr(metadata, field_name) is None:
            missing.append(keyword)
    if missing:
        raise ValueError(f"Missing required optical-axis metadata keywords: {', '.join(missing)}")


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
    """Convert a FITS table value to stripped text.

    Parameters
    ----------
    value : object
        Raw FITS table value.

    Returns
    -------
    str
        Normalized text.
    """
    if isinstance(value, bytes):
        return value.decode(errors="ignore").strip()
    return str(value).strip()


def _normalize_name(name: str | None) -> str:
    """Normalize a metadata key or codename for comparisons.

    Parameters
    ----------
    name : str | None
        Input token.

    Returns
    -------
    str
        Upper-case normalized token.
    """
    return "" if name is None else str(name).strip().upper()


def _is_number(text: str) -> bool:
    """Return whether a token can be parsed as a number.

    Parameters
    ----------
    text : str
        Text token.

    Returns
    -------
    bool
        Whether ``text`` parses as a float.
    """
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _parse_prefixed_numeric(token: str) -> tuple[str, str, float | None, float | None] | None:
    """Parse ``PREFIX:value`` or ``PREFIX:min-max`` CBD payloads.

    Parameters
    ----------
    token : str
        CBD payload token.

    Returns
    -------
    tuple[str, str, float | None, float | None] | None
        Structured prefix/range tuple, or ``None`` when the token does not
        follow the expected format.
    """
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
    """Classify one CBD token as text, scalar number, or numeric range.

    Parameters
    ----------
    token : str
        Raw token from ``CAL_CBD``.

    Returns
    -------
    tuple[str, float | None, float | None, str]
        ``(value_type, min_val, max_val, text_value)``.
    """
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
    """Parse one ``CAL_CBD`` text string into structured entries.

    Parameters
    ----------
    text : str
        Raw ``CAL_CBD`` string.

    Returns
    -------
    dict[str, list[CBDValue]]
        Parsed CBD entries keyed by name.
    """
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
    """Parse one query condition token.

    Parameters
    ----------
    text : str
        Raw condition string.
    logical_op : str
        Logical operator linking this condition to the previous one.

    Returns
    -------
    CBDCondition | None
        Parsed condition, or ``None`` if the token is unsupported.
    """
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
    """Parse a ``CAL_CBD`` query expression.

    Parameters
    ----------
    expr : str
        Query expression.

    Returns
    -------
    list[CBDCondition]
        Parsed CBD conditions.
    """
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
    """Compare one numeric CALDB value to a query target.

    Parameters
    ----------
    op : str
        Comparison operator.
    row_value : float | None
        Candidate scalar value.
    target : float | None
        Query target value.

    Returns
    -------
    bool
        Whether the comparison succeeds.
    """
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
    """Compare one numeric target against a row range.

    Parameters
    ----------
    op : str
        Comparison operator.
    row_min : float | None
        Range lower bound.
    row_max : float | None
        Range upper bound.
    target : float | None
        Query scalar target.

    Returns
    -------
    bool
        Whether the comparison succeeds.
    """
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
    """Compare structured CBD strings such as ``G0:12`` and ``G0:0-12``.

    Parameters
    ----------
    condition_text : str
        Query token.
    candidate_text : str
        Candidate row token.

    Returns
    -------
    bool
        Whether the structured payloads match.
    """
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
        return (
            candidate_min is not None
            and candidate_max is not None
            and condition_min is not None
            and candidate_min <= condition_min <= candidate_max
        )
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
    """Evaluate one parsed condition against row CBD values.

    Parameters
    ----------
    condition : CBDCondition
        Query condition.
    values : list[CBDValue]
        Candidate row values for the same CBD key.

    Returns
    -------
    bool
        Whether any candidate row value satisfies the condition.
    """
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
    """Evaluate a query expression against one row ``CAL_CBD`` string.

    Parameters
    ----------
    expr : str
        Query expression.
    cal_cbd_text : str
        Candidate row ``CAL_CBD`` string.

    Returns
    -------
    bool
        Whether the row satisfies the expression.
    """
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
    """Normalize optional detector/filter tokens for wildcard matching.

    Parameters
    ----------
    token : str | None
        Optional token.

    Returns
    -------
    str
        Generic normalized token or ``NONE``.
    """
    if token is None:
        return "NONE"
    stripped = str(token).strip()
    if not stripped or stripped == "-":
        return "NONE"
    return stripped.upper()


def _detector_aliases(value: str) -> set[str]:
    """Return accepted aliases for one detector token.

    Parameters
    ----------
    value : str
        Detector token.

    Returns
    -------
    set[str]
        Accepted detector aliases.
    """
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
    """Return accepted aliases for one filter token.

    Parameters
    ----------
    value : str
        Filter token.

    Returns
    -------
    set[str]
        Accepted filter aliases.
    """
    norm = _normalize_optional(value)
    if norm == "NONE":
        return {"NONE"}
    base = normalize_filter(norm)
    aliases = {base}
    if base is not None and base.isdigit():
        aliases.add(f"{int(base):02d}")
    return aliases


def _match_optional_with_aliases(row_value: str, requested: str, alias_builder) -> bool:
    """Match one CALDB optional field using alias-aware wildcard logic.

    Parameters
    ----------
    row_value : str
        Candidate row value.
    requested : str
        Requested value.
    alias_builder : callable
        Alias-expansion callback.

    Returns
    -------
    bool
        Whether the row matches the request.
    """
    row_norm = _normalize_optional(row_value)
    req_norm = _normalize_optional(requested)
    if row_norm == "NONE" or req_norm == "NONE":
        return True
    return not alias_builder(row_norm).isdisjoint(alias_builder(req_norm))


def _filter_by_expr(rows, expr: str):
    """Filter selected CIF rows by one CBD expression.

    Parameters
    ----------
    rows : fits.FITS_rec
        Candidate rows.
    expr : str
        CBD expression.

    Returns
    -------
    fits.FITS_rec
        Expression-matched rows.
    """
    if not expr:
        return rows
    mask = []
    for row in rows:
        mask.append(_evaluate_expr_conditions(expr, _to_text(row["CAL_CBD"])))
    return rows[np.asarray(mask, dtype=bool)]


def _filter_rows_by_metadata(calidx, request: CaldbRequest):
    """Filter CIF rows by mission, detector, filter, codename, and quality.

    Parameters
    ----------
    calidx : fits.FITS_rec
        Full calibration index table.
    request : CaldbRequest
        Normalized lookup request.

    Returns
    -------
    fits.FITS_rec
        Metadata-matched rows.
    """
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
    """Convert date/time bounds to MJDs.

    Parameters
    ----------
    start_date, start_time, stop_date, stop_time : str
        Request time bounds.

    Returns
    -------
    tuple[float | None, float | None]
        Start and stop times in MJD.
    """
    def _convert(date_str: str, time_str: str) -> float | None:
        if not date_str or date_str.strip() == "-":
            return None
        if date_str.lower() == "now":
            return float(Time.now().mjd)
        clean_time = "00:00:00" if not time_str or time_str.strip() == "-" else time_str
        return float(Time(f"{date_str}T{clean_time}", scale="utc", format="isot").mjd)

    return _convert(start_date, start_time), _convert(stop_date, stop_time)


def _extract_version(filename: str) -> int:
    """Extract the integer ``_vNN`` suffix from one calibration filename.

    Parameters
    ----------
    filename : str
        Calibration filename.

    Returns
    -------
    int
        Parsed version number, or zero when absent.
    """
    match = re.search(r"_v(\d+)\.", filename)
    return int(match.group(1)) if match else 0


def _choose_record(rows, start_mjd: float | None, stop_mjd: float | None):
    """Choose one best-matching row from an already filtered subset.

    Parameters
    ----------
    rows : fits.FITS_rec
        Candidate rows.
    start_mjd : float | None
        Request start time in MJD.
    stop_mjd : float | None
        Request stop time in MJD.

    Returns
    -------
    numpy.record
        Selected row.
    """
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
    """Build one compact description for a candidate CALDB row.

    Parameters
    ----------
    row : numpy.record
        Candidate row.

    Returns
    -------
    str
        Compact diagnostic summary.
    """
    return (
        f"file={_to_text(row['CAL_FILE'])}, dir={_to_text(row['CAL_DIR'])}, "
        f"ext={int(row['CAL_XNO'])}, det={_to_text(row['DETNAM'])}, "
        f"filter={_to_text(row['FILTER'])}, cbd={_to_text(row['CAL_CBD'])}, "
        f"reftime={float(row['REF_TIME']):.5f}"
    )


def _sample_rows(rows, limit: int = 5) -> str:
    """Return a compact sample of candidate rows for diagnostics.

    Parameters
    ----------
    rows : fits.FITS_rec
        Candidate rows.
    limit : int, optional
        Maximum number of row summaries to include.

    Returns
    -------
    str
        Joined row summaries.
    """
    if len(rows) == 0:
        return "<none>"
    return " | ".join(_describe_row(row) for row in rows[:limit])


def _format_request(request: CaldbRequest) -> str:
    """Format one normalized CALDB request for diagnostics.

    Parameters
    ----------
    request : CaldbRequest
        Normalized request.

    Returns
    -------
    str
        Human-readable request summary.
    """
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
    """Raise one detailed CALDB lookup failure.

    Parameters
    ----------
    stage : str
        Failure stage name.
    request : CaldbRequest
        Lookup request.
    paths : CaldbPaths
        Active CALDB paths.
    total_rows : int
        Total number of rows in the index.
    metadata_rows : fits.FITS_rec
        Rows that survived metadata filtering.
    cbd_rows : fits.FITS_rec
        Rows that survived CBD filtering.
    extra : str, optional
        Extra diagnostic note.

    Returns
    -------
    None
        This function raises an exception.
    """
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
    """Resolve one calibration file path and extension from the local CIF.

    Parameters
    ----------
    telescope, instrument, detname, filt, codename : str
        Generic CALDB selection keys.
    start_date, start_time, stop_date, stop_time : str
        Requested time range.
    expr : str, optional
        Additional ``CAL_CBD`` expression.
    caldb_root : str | None, optional
        Explicit CALDB root override.
    caldb_config : str | None, optional
        Explicit CALDB config override.

    Returns
    -------
    tuple[str, int]
        Resolved file path and extension number.
    """
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
    filepath = f"{paths.root}/{_to_text(record['CAL_DIR'])}/{_to_text(record['CAL_FILE'])}"
    return filepath, int(record["CAL_XNO"])
