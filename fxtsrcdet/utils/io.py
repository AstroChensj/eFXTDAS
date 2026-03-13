from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Column, Table
from astropy.wcs import FITSFixedWarning

from fxtsrcdet.models import CatalogRow


def read_ccd(path: Path) -> CCDData:
    """Read a FITS image into a CCDData object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        try:
            ccd = CCDData.read(path)
        except ValueError:
            ccd = CCDData.read(path, unit=u.dimensionless_unscaled)
    if ccd.data.ndim != 2:
        raise ValueError(f"{path} is not a 2D FITS image.")
    return ccd


def load_img(path: Path) -> np.ndarray:
    """Load a 2D FITS image as a floating-point array."""
    return np.asarray(read_ccd(path).data, dtype=np.float64)


def load_pipeline_inputs(
    image: np.ndarray | str | Path,
    exposure: np.ndarray | str | Path | None,
    wcs: Any | None,
) -> tuple[np.ndarray, np.ndarray | None, Any | None]:
    """Normalize file and array inputs into in-memory arrays.

    Parameters
    ----------
    image : np.ndarray | str | Path
        Input counts image array or image-file path.
    exposure : np.ndarray | str | Path | None
        Optional exposure map array or file path matched to ``image``.
    wcs : Any | None
        Optional celestial WCS for sky-coordinate outputs.

    Returns
    -------
    normalized_inputs : tuple[np.ndarray, np.ndarray | None, Any | None]
        Tuple ``(image_data, exposure_data, wcs)`` ready for pipeline use.
    """
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        image_data = load_img(image_path)
        if exposure is not None and isinstance(exposure, (str, Path)):
            exposure_data = load_img(Path(exposure))
        elif exposure is not None and not isinstance(exposure, np.ndarray):
            raise TypeError("exposure must be a numpy array, path-like, or None")
        else:
            exposure_data = None if exposure is None else np.asarray(exposure, dtype=np.float64)
        if wcs is None:
            wcs = load_wcs(image_path)
    elif isinstance(image, np.ndarray):
        image_data = np.asarray(image, dtype=np.float64)
        if exposure is not None and isinstance(exposure, (str, Path)):
            exposure_data = load_img(Path(exposure))
        elif exposure is not None and not isinstance(exposure, np.ndarray):
            raise TypeError("exposure must be a numpy array, path-like, or None")
        else:
            exposure_data = None if exposure is None else np.asarray(exposure, dtype=np.float64)
    else:
        raise TypeError("image must be a numpy array or path-like")
    return image_data, exposure_data, wcs


def load_header(path: Path) -> Any | None:
    """Load the primary FITS header from a file."""
    if path.suffix.lower() not in {".fits", ".fit", ".fts"}:
        return None
    with fits.open(path) as hdul:
        return hdul[0].header.copy()


def save_img(path: Path, array: np.ndarray, header: Any | None = None) -> None:
    """Write a 2D array to a FITS image while preserving metadata."""
    out_header = header.copy() if header is not None else fits.Header()
    ccd = CCDData(np.asarray(array, dtype=np.float32), unit=u.ct, meta=out_header)
    ccd.write(path, overwrite=True)


def _row_dict(row: CatalogRow | dict[str, Any]) -> dict[str, Any]:
    """Normalize one row-like object into a plain dictionary."""
    if isinstance(row, CatalogRow):
        return row.to_dict()
    return dict(row)


def _catalog_value(row: CatalogRow | dict[str, Any], field: str) -> Any:
    """Return one internal catalog value from either a dataclass row or mapping."""
    if isinstance(row, CatalogRow):
        return getattr(row, field)
    return row.get(field)


def write_sources_fits(path: Path | str, rows: list[CatalogRow] | list[dict[str, Any]], debug_columns: bool = False) -> None:
    """Write the final source catalog to a FITS binary table.

    Parameters
    ----------
    path : Path
        Output FITS path.
    rows : list[CatalogRow] | list[dict[str, Any]]
        Catalog rows to serialize.
    debug_columns : bool
        Whether to append internal/debug fields beyond the standard catalog.
    """
    #--- standard esass-like columns (public name, internal name, unit, description)
    standard_columns = [
        ("ID_SRC", "id_src", "", "Source ID"),
        ("ID_BAND", "id_band", "", "Band number; 0 denotes total-band results"),
        ("ID_CLUSTER", "id_cluster", "", "Local source-group identifier"),
        ("SOURCE_TYPE", "source_type", "", "Final source classification: background, point, or extended"),
        ("RA", "ra", "deg", "Right ascension"),
        ("RA_LOWERR", "ra_lowerr", "arcsec", "1 sigma lower RA error"),
        ("RA_UPERR", "ra_uperr", "arcsec", "1 sigma upper RA error"),
        ("DEC", "dec", "deg", "Declination"),
        ("DEC_LOWERR", "dec_lowerr", "arcsec", "1 sigma lower Dec error"),
        ("DEC_UPERR", "dec_uperr", "arcsec", "1 sigma upper Dec error"),
        ("RADEC_ERR", "radec_err", "arcsec", "Combined position error"),
        ("LII", "lii", "deg", "Galactic longitude"),
        ("BII", "bii", "deg", "Galactic latitude"),
        ("DIST_NN", "dist_nn", "arcsec", "Distance to nearest catalog neighbor"),
        ("X_IMA", "x_ima", "pixel", "Image x coordinate"),
        ("X_IMA_ERR", "x_ima_err", "pixel", "1 sigma x-position error"),
        ("X_IMA_LOWERR", "x_ima_lowerr", "pixel", "1 sigma lower x-position error"),
        ("X_IMA_UPERR", "x_ima_uperr", "pixel", "1 sigma upper x-position error"),
        ("Y_IMA", "y_ima", "pixel", "Image y coordinate"),
        ("Y_IMA_ERR", "y_ima_err", "pixel", "1 sigma y-position error"),
        ("Y_IMA_LOWERR", "y_ima_lowerr", "pixel", "1 sigma lower y-position error"),
        ("Y_IMA_UPERR", "y_ima_uperr", "pixel", "1 sigma upper y-position error"),
        ("EXT", "ext", "arcsec", "Best-fit source extent"),
        ("EXT_ERR", "ext_err", "arcsec", "1 sigma extent error"),
        ("EXT_LOWERR", "ext_lowerr", "arcsec", "1 sigma lower extent error"),
        ("EXT_UPERR", "ext_uperr", "arcsec", "1 sigma upper extent error"),
        ("EXT_LIKE", "ext_like", "", "Extent likelihood"),
        ("ML_RADIUS", "ml_radius", "arcsec", "Single source fit stamp radius"),
        ("MASKFRAC", "maskfrac", "", "Fraction of fit area within the valid exposure mask"),
        ("ML_CTS_0", "ml_cts", "count", "Best-fit source counts"),
        ("ML_CTS_ERR_0", "ml_cts_err", "count", "1 sigma error on ML_CTS_0"),
        ("ML_CTS_LOWERR_0", "ml_cts_lowerr", "count", "1 sigma lower error on ML_CTS_0"),
        ("ML_CTS_UPERR_0", "ml_cts_uperr", "count", "1 sigma upper error on ML_CTS_0"),
        ("ML_RATE_0", "ml_rate", "count/s", "Vignetting-corrected count rate"),
        ("ML_RATE_ERR_0", "ml_rate_err", "count/s", "1 sigma error on ML_RATE_0"),
        ("ML_RATE_LOWERR_0", "ml_rate_lowerr", "count/s", "1 sigma lower error on ML_RATE_0"),
        ("ML_RATE_UPERR_0", "ml_rate_uperr", "count/s", "1 sigma upper error on ML_RATE_0"),
        ("ML_FLUX_0", "ml_flux", "erg cm-2 s-1", "Source flux derived from ML_RATE_0 and ECF"),
        ("ML_FLUX_ERR_0", "ml_flux_err", "erg cm-2 s-1", "1 sigma error on ML_FLUX_0"),
        ("ML_FLUX_LOWERR_0", "ml_flux_lowerr", "erg cm-2 s-1", "1 sigma lower error on ML_FLUX_0"),
        ("ML_FLUX_UPERR_0", "ml_flux_uperr", "erg cm-2 s-1", "1 sigma upper error on ML_FLUX_0"),
        ("DET_LIKE_0", "det_like", "", "Detection likelihood"),
        ("ML_BKG_0", "ml_bkg", "count/arcmin2", "Local fitted background surface brightness"),
        ("ML_EXP_0", "ml_exp", "s", "Vignetting-corrected exposure at source position"),
        ("ML_EFF_0", "ml_eff", "", "Fraction of PSF in single source fitting stamp"),
    ]
    #--- debug columns (public name, internal name, unit, description)
    debug_column_meta = [
        #--- detection summary
        ("scale", "scale", "pixel", "Representative wavelet scale"),
        ("support_scales", "support_scales", "", "Wavelet scales that supported this source"),
        ("wavelet_peak_score", "wavelet_peak_score", "", "Wavelet peak ranking score"),
        ("min_significance", "min_significance", "", "Smallest per-pixel detection significance in the source support"),
        ("npix", "npix", "pixel", "Number of source-mask pixels assigned to the candidate"),
        ("counts", "counts", "count", "Raw counts inside the wavelet detection support"),
        ("net_counts", "net_counts", "count", "Approximate net counts inside the wavelet detection support"),
        ("bkg_counts", "bkg_counts", "count", "Approximate background counts inside the wavelet detection support"),
        ("major", "major", "pixel", "Wavelet-ellipse semi-major axis"),
        ("minor", "minor", "pixel", "Wavelet-ellipse semi-minor axis"),
        ("theta_deg", "theta_deg", "deg", "Wavelet-ellipse position angle"),
        #--- grouping / deblending
        ("group_id", "group_id", "", "Local grouped-fit identifier"),
        ("group_size", "group_size", "", "Number of members in the local grouped fit"),
        ("group_stamp_radius_pix", "group_stamp_radius_pix", "pixel", "Radius of the grouped-fit stamp"),
        ("theta_arcmin", "theta_arcmin", "arcmin", "Off-axis angle from the optical axis"),
        #--- psf context
        ("psf_r50_pix", "psf_r50_pix", "pixel", "Local PSF r50 radius"),
        ("psf_r75_pix", "psf_r75_pix", "pixel", "Local PSF r75 radius"),
        ("psf_r80_pix", "psf_r80_pix", "pixel", "Local PSF r80 radius"),
        ("psf_r90_pix", "psf_r90_pix", "pixel", "Local PSF r90 radius"),
        ("psf_instrument", "psf_instrument", "", "Which instrument used to construct local PSF"),
        ("psf_filter", "psf_filter", "", "Which filter used to construct local PSF"),
        ("psf_line", "psf_line", "", "Which line used to construct local PSF"),
        ("psf_energy_keV", "psf_energy_keV", "keV", "Which energy used to construct local PSF"),
        #--- morphology / fitting
        ("ml_radius_pix", "ml_radius_pix", "pixel", "Radius of the single-source fit stamp"),
        ("extent_ratio", "extent_ratio", "", "Measured-to-PSF size ratio used in extent diagnostics"),
        ("fitted_extent_sigma_pix", "fitted_extent_sigma_pix", "pixel", "Best-fit extended-model core scale"),
        ("meas_r50_pix", "meas_r50_pix", "pixel", "Measured residual-profile r50"),
        ("meas_r80_pix", "meas_r80_pix", "pixel", "Measured residual-profile r80"),
        ("meas_r90_pix", "meas_r90_pix", "pixel", "Measured residual-profile r90"),
        #--- final region
        ("catalog_shape", "catalog_shape", "", "Final catalog-region shape"),
        ("catalog_radius_pix", "catalog_radius_pix", "pixel", "Final catalog-region radius in pixels"),
        ("catalog_radius_arcsec", "catalog_radius_arcsec", "arcsec", "Final catalog-region radius in sky units"),
        ("major_arcsec", "major_arcsec", "arcsec", "Wavelet-ellipse semi-major axis in sky units"),
        ("minor_arcsec", "minor_arcsec", "arcsec", "Wavelet-ellipse semi-minor axis in sky units"),
        ("radius_arcsec", "radius_arcsec", "arcsec", "Geometric-mean wavelet radius in sky units"),
        #--- energy band
        ("emin_keV", "emin_keV", "keV", "Lower image energy bound"),
        ("emax_keV", "emax_keV", "keV", "Upper image energy bound"),
    ]
    row_dicts = [_row_dict(row) for row in rows]
    fields = [public for public, _, _, _ in standard_columns]
    if debug_columns:
        fields += [
            public
            for public, _, _, _ in debug_column_meta
            if public not in fields and any(public in row for row in row_dicts)
        ]
    table = Table()
    descriptions: dict[str, str] = {}
    standard_lookup = {public: (internal, unit, description) for public, internal, unit, description in standard_columns}
    debug_lookup = {public: (internal, unit, description) for public, internal, unit, description in debug_column_meta}
    for field in fields:
        if field in standard_lookup:
            internal, unit, description = standard_lookup[field]
            values = [_catalog_value(row, internal) for row in rows]
        elif field in debug_lookup:
            internal, unit, description = debug_lookup[field]
            values = [_catalog_value(row, internal) for row in rows]
        else:
            values = [row.get(field) for row in row_dicts]
            unit, description = "", ""
        if any(isinstance(value, (list, tuple)) for value in values):
            values = [",".join(str(item) for item in value) if isinstance(value, (list, tuple)) else str(value) for value in values]
        column = Column(name=field, data=values)
        if unit:
            column.unit = unit
        if description:
            column.description = description
            descriptions[field] = description
        table.add_column(column)
    path = Path(path)
    table_hdu = fits.BinTableHDU(table)
    for idx, field in enumerate(table.colnames, start=1):
        description = descriptions.get(field)
        if description:
            table_hdu.header[f"TDESC{idx}"] = description
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(path, overwrite=True)


def write_ds9_regions(path: Path, rows: list[CatalogRow] | list[dict[str, Any]]) -> None:
    """Write image-coordinate DS9 regions from the fitted catalog model."""
    with path.open("w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("image\n")
        for row in rows:
            row_dict = _row_dict(row)
            radius = max(float(row_dict.get("catalog_radius_pix", row_dict.get("psf_r90_pix", row_dict.get("major", 0.5)))), 0.5)
            f.write(
                "circle({x:.3f},{y:.3f},{r:.3f}) # text={{{sid}}}\n".format(
                    x=row_dict["x_ima"], y=row_dict["y_ima"], r=radius, sid=row_dict["id"]
                )
            )


def write_ds9_sky_regions(path: Path, rows: list[CatalogRow] | list[dict[str, Any]]) -> None:
    """Write sky-coordinate DS9 regions from the fitted catalog model."""
    with path.open("w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("fk5\n")
        for row in rows:
            row_dict = _row_dict(row)
            if "ra" not in row_dict or "dec" not in row_dict:
                continue
            radius = max(float(row_dict.get("catalog_radius_arcsec", row_dict.get("major_arcsec", 0.1))), 0.1)
            f.write(
                (
                    "circle({ra:.8f},{dec:.8f},{r:.3f}\") # text={{{sid}}}\n"
                ).format(
                    ra=row_dict["ra"], dec=row_dict["dec"], r=radius, sid=row_dict["id"]
                )
            )


def load_wcs(path: Path) -> Any | None:
    """Load celestial WCS metadata from a FITS image."""
    if path.suffix.lower() not in {".fits", ".fit", ".fts"}:
        return None
    wcs = read_ccd(path).wcs
    if wcs is None or not getattr(wcs, "has_celestial", False):
        return None
    return wcs.celestial
