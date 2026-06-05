"""Minimal attitude helpers vendored for standalone ``fxtrspgen``."""

from __future__ import annotations

import numpy as np


def quaternion_to_matrix(q1: float, q2: float, q3: float, q4: float) -> tuple[float, ...]:
    """Convert one quaternion to a 3x3 rotation matrix."""
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
    """Convert one 3x3 rotation matrix to a quaternion."""
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
    """Convert sky Euler angles to a rotation matrix."""
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
    """Convert sky Euler angles to a quaternion."""
    return matrix_to_quaternion_vec(*equtor_to_matrix(ra, dec, roll))


def calculate_ra_dec(direction: np.ndarray) -> tuple[float, float]:
    """Convert one direction vector to RA and Dec."""
    dec = np.rad2deg(np.pi / 2 - np.arccos(direction[2] / np.sqrt(np.sum(direction * direction))))
    if direction[0] < 0:
        ra = np.arctan(direction[1] / direction[0]) + np.pi
    elif direction[0] > 0:
        ra = np.arctan(direction[1] / direction[0]) if direction[1] >= 0 else np.arctan(direction[1] / direction[0]) + 2 * np.pi
    elif direction[1] == 0:
        ra = 0.0
    elif direction[1] > 0:
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
    """Project one detector pixel to sky coordinates."""
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
