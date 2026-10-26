"""Pixelwise GeoTIFF comparison of two QGIS outputs; requires GDAL (QGIS Python)."""
import argparse
import json


def compare_rasters(a, b, block_rows=256):
    from osgeo import gdal
    import numpy as np
    left = gdal.Open(str(a), gdal.GA_ReadOnly)
    right = gdal.Open(str(b), gdal.GA_ReadOnly)
    if left is None or right is None:
        raise FileNotFoundError('Unable to open both rasters')
    meta_match = (left.RasterXSize == right.RasterXSize and
                  left.RasterYSize == right.RasterYSize and
                  left.RasterCount == right.RasterCount and
                  left.GetGeoTransform() == right.GetGeoTransform() and
                  left.GetProjection() == right.GetProjection())
    if not meta_match:
        return {'pass': False, 'reason': 'grid/band/CRS mismatch'}
    mismatches = 0
    valid_comparisons = 0
    for band_index in range(1, left.RasterCount + 1):
        lb, rb = left.GetRasterBand(band_index), right.GetRasterBand(band_index)
        if lb.DataType != rb.DataType or lb.GetNoDataValue() != rb.GetNoDataValue():
            return {'pass': False, 'reason': 'data type or NoData mismatch', 'band': band_index}
        for y in range(0, left.RasterYSize, block_rows):
            h = min(block_rows, left.RasterYSize - y)
            x = lb.ReadAsArray(0, y, left.RasterXSize, h)
            z = rb.ReadAsArray(0, y, right.RasterXSize, h)
            same = (x == z) | (np.isnan(x) & np.isnan(z)) if np.issubdtype(x.dtype, np.floating) else x == z
            mismatches += int(np.count_nonzero(~same))
            valid_comparisons += int(x.size)
    return {'pass': mismatches == 0, 'different_pixels': mismatches,
            'compared_values': valid_comparisons, 'grid_match': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('baseline')
    p.add_argument('candidate')
    args = p.parse_args()
    result = compare_rasters(args.baseline, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['pass'] else 1)
