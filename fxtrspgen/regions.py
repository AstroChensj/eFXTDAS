"""DS9 region parsing and rasterization for ``fxtrspgen``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from regions import (
    CircleAnnulusPixelRegion,
    CircleAnnulusSkyRegion,
    CirclePixelRegion,
    CircleSkyRegion,
    EllipsePixelRegion,
    EllipseSkyRegion,
    PixCoord,
    PolygonPixelRegion,
    PolygonSkyRegion,
    RectanglePixelRegion,
    RectangleSkyRegion,
    Regions,
)


SUPPORTED_REGION_TYPES = (
    CirclePixelRegion,
    CircleSkyRegion,
    CircleAnnulusPixelRegion,
    CircleAnnulusSkyRegion,
    EllipsePixelRegion,
    EllipseSkyRegion,
    RectanglePixelRegion,
    RectangleSkyRegion,
    PolygonPixelRegion,
    PolygonSkyRegion,
)


class UnsupportedRegionError(ValueError):
    """Raised when a DS9 file contains an unsupported region type."""


@dataclass(frozen=True)
class RegionSet:
    """Parsed region collection and its combined inclusion mask.
    
    - pixel_regions:
        The individual parsed region objects, already converted into image
        /pixel-space regions. This preserves the original DS9 region 
        components so later code can still inspect the underlying shapes 
        if needed. Note that subtraction flag is not labeled.

    - mask:
        The combined 2D inclusion mask on the exposure/image grid. This is 
        the actual rasterized aperture used for calculations. It includes 
        positive regions and subtracts excluded regions, and can contain 
        fractional values if oversampling is used.

    - first_positive_center_xy:
        The center of the first positive DS9 region component in pixel 
        coordinates. This is only used as the fallback source position when 
        the user did not explicitly provide srcx/srcy or ra/dec.
    """

    pixel_regions: tuple[object, ...]
    mask: np.ndarray
    first_positive_center_xy: tuple[float, float]


def _mask_to_image(mask_object, image_shape: tuple[int, int]) -> np.ndarray:
    """Convert a regions mask object to an image array."""
    if mask_object is None:
        return np.zeros(image_shape, dtype=np.float64)
    if isinstance(mask_object, list):
        image = np.zeros(image_shape, dtype=np.float64)
        for item in mask_object:
            array = item.to_image(image_shape)
            if array is not None:
                image += np.nan_to_num(array, nan=0.0)
        return np.clip(image, 0.0, 1.0)
    array = mask_object.to_image(image_shape)
    if array is None:
        return np.zeros(image_shape, dtype=np.float64)
    return np.clip(np.nan_to_num(array, nan=0.0), 0.0, 1.0)


def _region_to_mask_image(pixel_region: object, image_shape: tuple[int, int], oversample: int) -> np.ndarray:
    """Rasterize one pixel region to an image mask.
    
    WARNING: only circle region is being handled at the moment.
    """
    if isinstance(pixel_region, CircleAnnulusPixelRegion):
        outer_mask = _mask_to_image(
            CirclePixelRegion(center=pixel_region.center, radius=pixel_region.outer_radius).to_mask(
                mode="subpixels",
                subpixels=max(1, int(oversample)),
            ),
            image_shape,
        )
        inner_mask = _mask_to_image(
            CirclePixelRegion(center=pixel_region.center, radius=pixel_region.inner_radius).to_mask(
                mode="subpixels",
                subpixels=max(1, int(oversample)),
            ),
            image_shape,
        )
        return np.clip(outer_mask - inner_mask, 0.0, 1.0)
    mask_object = pixel_region.to_mask(mode="subpixels", subpixels=max(1, int(oversample)))
    return _mask_to_image(mask_object, image_shape)


def _pixel_region(region: object, wcs: WCS | None) -> object:
    """Convert a sky region to a pixel region when needed."""
    if isinstance(
        region,
        (
            CirclePixelRegion,
            CircleAnnulusPixelRegion,
            EllipsePixelRegion,
            RectanglePixelRegion,
            PolygonPixelRegion,
        ),
    ):
        return region
    if wcs is None:
        raise ValueError("Sky regions require a WCS for rasterization.")
    return region.to_pixel(wcs)


def _validate_region(region: object) -> None:
    """Reject unsupported region classes explicitly."""
    if not isinstance(region, SUPPORTED_REGION_TYPES):
        raise UnsupportedRegionError(
            f"Unsupported DS9 region type: {region.__class__.__name__}"
        )


def _polygon_centroid(vertices: PixCoord) -> tuple[float, float]:
    """Compute a polygon centroid from pixel vertices."""
    x = np.asarray(vertices.x, dtype=np.float64)
    y = np.asarray(vertices.y, dtype=np.float64)
    if x.size < 3:
        return float(np.mean(x)), float(np.mean(y))
    x2 = np.append(x, x[0])
    y2 = np.append(y, y[0])
    cross = x2[:-1] * y2[1:] - x2[1:] * y2[:-1]
    area = np.sum(cross) / 2.0
    if np.isclose(area, 0.0):
        return float(np.mean(x)), float(np.mean(y))
    cx = np.sum((x2[:-1] + x2[1:]) * cross) / (6.0 * area)
    cy = np.sum((y2[:-1] + y2[1:]) * cross) / (6.0 * area)
    return float(cx), float(cy)


def region_center_xy(pixel_region: object) -> tuple[float, float]:
    """Return a representative pixel-space center for one region."""
    if hasattr(pixel_region, "center"):
        center = pixel_region.center
        return float(center.x), float(center.y)
    if isinstance(pixel_region, PolygonPixelRegion):
        return _polygon_centroid(pixel_region.vertices)
    raise UnsupportedRegionError(
        f"Cannot determine a center for {pixel_region.__class__.__name__}"
    )


def load_region_set(
    regionfile: str,
    image_shape: tuple[int, int],
    wcs: WCS | None,
    oversample: int = 5,
) -> RegionSet:
    """Parse a DS9 region file and rasterize it onto an image grid.

    Parameters
    ----------
    regionfile : str
        External DS9 region file.
    image_shape : tuple[int, int]
        Target image shape as ``(ny, nx)``.
    wcs : WCS | None
        WCS used for sky-region conversion.
    oversample : int, optional
        Subpixel rasterization factor.

    Returns
    -------
    RegionSet : RegionSet
        Parsed region set with a combined mask and fallback center.

        - RegionSet.pixel_regions:
            The individual parsed region objects, already converted into image
            /pixel-space regions. This preserves the original DS9 region 
            components so later code can still inspect the underlying shapes 
            if needed. Note that subtraction flag is not labeled.

        - RegionSet.mask:
            The combined 2D inclusion mask on the exposure/image grid. This is 
            the actual rasterized aperture used for calculations. It includes 
            positive regions and subtracts excluded regions, and can contain 
            fractional values if oversampling is used.

        - RegionSet.first_positive_center_xy:
            The center of the first positive DS9 region component in pixel 
            coordinates. This is only used as the fallback source position when 
            the user did not explicitly provide srcx/srcy or ra/dec.
    """
    parsed = Regions.read(Path(regionfile), format="ds9")
    include_mask = np.zeros(image_shape, dtype=np.float64)
    exclude_mask = np.zeros(image_shape, dtype=np.float64)
    pixel_regions: list[object] = []
    first_positive_center_xy: tuple[float, float] | None = None

    for region in parsed:
        _validate_region(region)
        pixel_region = _pixel_region(region, wcs)
        pixel_regions.append(pixel_region)
        include = bool(region.meta.get("include", 1))
        if include and first_positive_center_xy is None:
            first_positive_center_xy = region_center_xy(pixel_region)
        mask_image = _region_to_mask_image(pixel_region, image_shape, oversample)
        if include:
            include_mask = np.clip(include_mask + mask_image, 0.0, 1.0)
        else:
            exclude_mask = np.clip(exclude_mask + mask_image, 0.0, 1.0)

    if first_positive_center_xy is None:
        raise ValueError("The region file does not contain a positive source component.")
    if np.sum(include_mask) <= 0.0 and np.sum(exclude_mask) > 0.0:
        include_mask = np.ones(image_shape, dtype=np.float64)
    mask = np.clip(include_mask * (1.0 - exclude_mask), 0.0, 1.0)
    return RegionSet(
        pixel_regions=tuple(pixel_regions),
        mask=mask,
        first_positive_center_xy=first_positive_center_xy,
    )
