"""Helpers for parsing and naming channel-range selections."""

from __future__ import annotations


def parse_channel_ranges(range_spec: str | None, *, default: list[tuple[int, int]]) -> list[tuple[int, int]]:
	"""Parse a comma-separated list of inclusive channel ranges.

	Parameters
	----------
	range_spec : str | None
		Range specification such as ``"38:925,100:300"``.
	default : list[tuple[int, int]]
		Default ranges used when ``range_spec`` is empty.

	Returns
	-------
	list[tuple[int, int]]
		Parsed ``(lo, hi)`` channel ranges.
	"""
	if range_spec is None:
		return list(default)
	text = str(range_spec).strip()
	if not text:
		return list(default)
	channel_ranges = []
	for item in text.split(","):
		piece = item.strip()
		if not piece:
			continue
		if ":" not in piece:
			raise ValueError(f"Invalid channel range '{piece}'. Use lo:hi syntax.")
		chan_lo_text, chan_hi_text = piece.split(":", 1)
		chan_lo = int(chan_lo_text)
		chan_hi = int(chan_hi_text)
		if chan_lo < 0 or chan_hi > 1023 or chan_lo > chan_hi:
			raise ValueError(f"Invalid channel range '{piece}'. Valid channels are 0-1023 and lo <= hi.")
		channel_ranges.append((chan_lo, chan_hi))
	if not channel_ranges:
		raise ValueError("At least one valid channel range is required.")
	return channel_ranges


def channel_range_suffix(channel_range: tuple[int, int]) -> str:
	"""Return a stable filename suffix for one channel range."""
	chan_lo, chan_hi = channel_range
	return f"ch{int(chan_lo):04d}_{int(chan_hi):04d}"
