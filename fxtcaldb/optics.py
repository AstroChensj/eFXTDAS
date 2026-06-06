"""Optics calibration helpers for EP-FXT.

This module groups the TELDEF lookup, attitude rotation, and optical-axis
projection logic that used to live in separate modules. The public entry point
is :func:`compute_optical_axis_pixel`, which projects the detector optical axis
onto an image WCS using calibrated TELDEF geometry and the observation
pointing/roll metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from fxtcaldb.query import ObservationMetadata, find_calibration_file, require_optaxis_metadata


@dataclass(frozen=True)
class TeldefInfo:
    """TELDEF quantities needed by current PSF/response tools."""

    alignment_matrix: np.ndarray
    focal_length: float
    pixel_size: float
    optaxis_x: float
    optaxis_y: float


def _canonicalize_teldef_detector(metadata: ObservationMetadata) -> str:
    """Map metadata detector identity to ``TELDEF`` detector conventions.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for TELDEF lookup.

    Returns
    -------
    str
        Canonical ``TELDEF`` detector code, ``FXTA`` or ``FXTB``.
    """
    if metadata.detector_code == "A":
        return "FXTA"
    if metadata.detector_code == "B":
        return "FXTB"
    raise ValueError(f"Unsupported TELDEF detector value: {metadata.detector_code!r}")


def resolve_teldef(metadata: ObservationMetadata) -> tuple[str, int]:
    """Resolve the CALDB TELDEF file.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for TELDEF lookup.

    Returns
    -------
    tuple[str, int]
        Resolved TELDEF file path and extension number.
    """
    require_optaxis_metadata(metadata)
    return find_calibration_file(
        telescope=metadata.telescope,
        instrument=metadata.instrument,
        detname=_canonicalize_teldef_detector(metadata),
        filt="NONE",
        codename="TELDEF",
        start_date=metadata.start_date,
        start_time=metadata.start_time or "00:00:00",
        stop_date=metadata.stop_date,
        stop_time=metadata.stop_time or "00:00:00",
        expr="",
    )


def read_teldef_info(filepath: str) -> TeldefInfo:
    """Read the alignment and optical-axis quantities from one TELDEF file.

    Parameters
    ----------
    filepath : str
        TELDEF file path.

    Returns
    -------
    TeldefInfo
        Parsed TELDEF payload used by optics and response tools.
    """
    header = fits.getheader(filepath)
    return TeldefInfo(
        alignment_matrix=np.array(
            [
                [header["ALIGNM11"], header["ALIGNM12"], header["ALIGNM13"]],
                [header["ALIGNM21"], header["ALIGNM22"], header["ALIGNM23"]],
                [header["ALIGNM31"], header["ALIGNM32"], header["ALIGNM33"]],
            ],
            dtype=np.float64,
        ),
        focal_length=float(header["FOCALLEN"]),
        pixel_size=float(header["DET_XSCL"]),
        optaxis_x=float(header["OPTAXISX"]),
        optaxis_y=float(header["OPTAXISY"]),
    )


def quaternion_to_matrix(q1: float, q2: float, q3: float, q4: float) -> tuple[float, ...]:
    """Convert one quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    q1, q2, q3, q4 : float
        Quaternion components.

    Returns
    -------
    tuple[float, ...]
        Flattened 3x3 rotation matrix.
    """
    r11 = 2 * (q4**2 + q1**2) - 1
    r22 = 2 * (q4**2 + q2**2) - 1
    r33 = 2 * (q4**2 + q3**2) - 1
    r12 = 2 * (q1 * q2 - q3 * q4)
    r13 = 2 * (q1 * q3 + q2 * q4)
    r21 = 2 * (q1 * q2 + q3 * q4)
    r23 = 2 * (q2 * q3 - q4 * q1)
    r31 = 2 * (q1 * q3 - q2 * q4)
    r32 = 2 * (q2 * q3 + q1 * q4)
    return r11, r12, r13, r21, r22, r23, r31, r32, r33


def matrix_to_quaternion_vec(
    r11: float,
    r12: float,
    r13: float,
    r21: float,
    r22: float,
    r23: float,
    r31: float,
    r32: float,
    r33: float,
) -> tuple[float, float, float, float]:
    """Convert one 3x3 rotation matrix to a quaternion.

    Parameters
    ----------
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : float
        Rotation-matrix elements.

    Returns
    -------
    tuple[float, float, float, float]
        Quaternion components.
    """
    eps = 1e-12
    q1 = np.sqrt(max(1 + r11 - r22 - r33, 0.0)) / 2.0
    q2 = np.sqrt(max(1 - r11 + r22 - r33, 0.0)) / 2.0
    q3 = np.sqrt(max(1 - r11 - r22 + r33, 0.0)) / 2.0
    q4 = np.sqrt(max(1 + r11 + r22 + r33, 0.0)) / 2.0
    q1x = (r12 + r21) / (4.0 * q1 + eps)
    q1y = (r31 + r13) / (4.0 * q1 + eps)
    q1z = (r32 - r23) / (4.0 * q1 + eps)
    q2x = (r12 + r21) / (4.0 * q2 + eps)
    q2y = (r23 + r32) / (4.0 * q2 + eps)
    q2z = (r13 - r31) / (4.0 * q2 + eps)
    q3x = (r13 + r31) / (4.0 * q3 + eps)
    q3y = (r23 + r32) / (4.0 * q3 + eps)
    q3z = (r21 - r12) / (4.0 * q3 + eps)
    qx = (r32 - r23) / (4.0 * q4 + eps)
    qy = (r13 - r31) / (4.0 * q4 + eps)
    qz = (r21 - r12) / (4.0 * q4 + eps)
    if r11 >= r22 + r33:
        return float(q1), float(q1x), float(q1y), float(q1z)
    if r22 >= r11 + r33:
        return float(q2x), float(q2), float(q2y), float(q2z)
    if r33 >= r11 + r22:
        return float(q3x), float(q3y), float(q3), float(q3z)
    return float(qx), float(qy), float(qz), float(q4)


def equtor_to_matrix(ra: float, dec: float, roll: float) -> tuple[float, ...]:
    """Convert sky Euler angles to a rotation matrix.

    Parameters
    ----------
    ra, dec, roll : float
        Pointing right ascension, declination, and position angle in degrees.

    Returns
    -------
    tuple[float, ...]
        Flattened 3x3 rotation matrix.
    """
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    roll_rad = -np.deg2rad(roll)
    r11 = np.cos(dec_rad) * np.cos(ra_rad)
    r12 = np.sin(roll_rad) * np.sin(dec_rad) * np.cos(ra_rad) - np.cos(roll_rad) * np.sin(ra_rad)
    r13 = -np.cos(roll_rad) * np.sin(dec_rad) * np.cos(ra_rad) - np.sin(roll_rad) * np.sin(ra_rad)
    r21 = np.cos(dec_rad) * np.sin(ra_rad)
    r22 = np.sin(roll_rad) * np.sin(dec_rad) * np.sin(ra_rad) + np.cos(roll_rad) * np.cos(ra_rad)
    r23 = -np.cos(roll_rad) * np.sin(dec_rad) * np.sin(ra_rad) + np.sin(roll_rad) * np.cos(ra_rad)
    r31 = np.sin(dec_rad)
    r32 = -np.sin(roll_rad) * np.cos(dec_rad)
    r33 = np.cos(roll_rad) * np.cos(dec_rad)
    return r11, r12, r13, r21, r22, r23, r31, r32, r33


def eq_to_quat(ra: float, dec: float, roll: float) -> tuple[float, float, float, float]:
    """Convert sky Euler angles to a quaternion.

    Parameters
    ----------
    ra, dec, roll : float
        Pointing right ascension, declination, and position angle in degrees.

    Returns
    -------
    tuple[float, float, float, float]
        Quaternion components.
    """
    return matrix_to_quaternion_vec(*equtor_to_matrix(ra, dec, roll))


def calculate_ra_dec(direction: np.ndarray) -> tuple[float, float]:
    """Convert one direction vector to RA and Dec.

    Parameters
    ----------
    direction : np.ndarray
        Three-vector in the sky frame.

    Returns
    -------
    tuple[float, float]
        Right ascension and declination in degrees.
    """
    vector = np.asarray(direction, dtype=np.float64).reshape(3)
    dec = np.rad2deg(np.pi / 2 - np.arccos(vector[2] / np.sqrt(np.sum(vector * vector))))
    if vector[0] < 0:
        ra = np.arctan(vector[1] / vector[0]) + np.pi
    elif vector[0] > 0:
        ra = np.arctan(vector[1] / vector[0]) if vector[1] >= 0 else np.arctan(vector[1] / vector[0]) + 2 * np.pi
    elif vector[1] == 0:
        ra = 0.0
    elif vector[1] > 0:
        ra = np.pi / 2
    else:
        ra = np.pi * 3.0 / 2.0
    return float(np.rad2deg(ra)), float(dec)


def det2radecpix(
    det_pixel: list[float],
    detcenter: list[float],
    pixel_size: float,
    focal_length: float,
    quaternion: tuple[float, float, float, float],
    alignment_matrix: np.ndarray,
) -> tuple[float, float]:
    """Project one detector pixel to sky coordinates.

    Parameters
    ----------
    det_pixel : list[float]
        Detector pixel to project.
    detcenter : list[float]
        Detector reference point in detector pixels.
    pixel_size : float
        Detector pixel size from TELDEF.
    focal_length : float
        Telescope focal length from TELDEF.
    quaternion : tuple[float, float, float, float]
        Observation attitude quaternion.
    alignment_matrix : np.ndarray
        Detector-to-sky alignment matrix.

    Returns
    -------
    tuple[float, float]
        Sky right ascension and declination in degrees.
    """
    axis_column = detcenter[0]
    axis_row = detcenter[1]
    column = det_pixel[0]
    row = det_pixel[1]
    ax = float((axis_column - column) * pixel_size)
    ay = float((axis_row - row) * pixel_size)
    az = float(-focal_length)
    rotation = np.asarray(quaternion_to_matrix(*quaternion), dtype=np.float64).reshape(3, 3)
    vector = np.array([ax, ay, az], dtype=np.float64).reshape(3, 1)
    direction = rotation @ (alignment_matrix @ vector)
    return calculate_ra_dec(direction)


def compute_optical_axis_pixel(metadata: ObservationMetadata, image_wcs: WCS) -> tuple[float, float]:
    """Project the telescope optical axis onto the image pixel grid.

    Notes
    -----
    Some jargons:

    - detector plane:
        the CCD where events are actually being recorded.

    - optical-axis location (``det_optaxis_xy`` stores it in detector pixel
      coordinates):
        optical axis location in detector pixel coordinates, recorded in the
        TELDEF as ``OPTAXISX``/``OPTAXISY``. This is the point where the
        optical axis intersects the detector plane, and is the natural
        reference point for PSF and vignetting calibration.

    - pointing center:
        the sky coordinates where the telescope is pointed, recorded in
        ``RA_PNT``/``DEC_PNT``. This does not have to coincide with the optical
        axis.

    The detector geometric center, detector optical axis, and pointing center
    do not have to coincide.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata containing TELDEF-identification and pointing
        keywords.
    image_wcs : WCS
        Celestial WCS of the target image.

    Returns
    -------
    tuple[float, float]
        Optical-axis position in 1-based image pixels.
    """
    require_optaxis_metadata(metadata)
    if image_wcs is None or not getattr(image_wcs, "has_celestial", False):
        raise ValueError("A celestial image WCS is required to project the optical axis.")
    teldef_path, _ = resolve_teldef(metadata)
    teldef = read_teldef_info(teldef_path)
    det_optaxis_xy = [teldef.optaxis_x - 1.0, teldef.optaxis_y - 1.0]
    quat = eq_to_quat(float(metadata.ra_pnt), float(metadata.dec_pnt), float(metadata.pa_pnt))
    axis_ra, axis_dec = det2radecpix(
        det_optaxis_xy,
        det_optaxis_xy,
        teldef.pixel_size,
        teldef.focal_length,
        quat,
        teldef.alignment_matrix,
    )
    xpix, ypix = image_wcs.all_world2pix(float(np.asarray(axis_ra).ravel()[0]), float(np.asarray(axis_dec).ravel()[0]), 0)
    return float(xpix + 1.0), float(ypix + 1.0)
