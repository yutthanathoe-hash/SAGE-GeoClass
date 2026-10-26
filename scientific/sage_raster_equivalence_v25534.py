"""Streaming, full-pixel raster equivalence checker. Run with QGIS Python or rasterio-enabled Python."""
import argparse, json
import numpy as np

def compare(a_path, b_path, tolerance=0.0, block_rows=128):
    from osgeo import gdal
    a, b = gdal.Open(str(a_path)), gdal.Open(str(b_path))
    if a is None or b is None:
        raise FileNotFoundError('Unable to open input GeoTIFF(s)')
    out = {'files':[str(a_path),str(b_path)], 'tolerance':float(tolerance)}
    out['shape_equal'] = (a.RasterXSize,a.RasterYSize,a.RasterCount)==(b.RasterXSize,b.RasterYSize,b.RasterCount)
    out['grid_equal'] = a.GetGeoTransform()==b.GetGeoTransform()
    from osgeo import osr
    ca,cb=osr.SpatialReference(),osr.SpatialReference()
    ca.ImportFromWkt(a.GetProjection());cb.ImportFromWkt(b.GetProjection())
    out['crs_equal'] = bool(ca.IsSame(cb))
    out['dtype_equal'] = a.GetRasterBand(1).DataType == b.GetRasterBand(1).DataType
    out['nodata_equal'] = a.GetRasterBand(1).GetNoDataValue()==b.GetRasterBand(1).GetNoDataValue()
    out.update(total_pixels=0,mask_mismatches=0,value_mismatches=0,max_abs_diff=0.0,mean_abs_diff=0.0,valid_both=0)
    if not out['shape_equal']:
        out['pass']=False;return out
    delta_sum=0.
    for band in range(1,a.RasterCount+1):
        ba,bb=a.GetRasterBand(band),b.GetRasterBand(band)
        if ba.GetNoDataValue()!=bb.GetNoDataValue():out['nodata_equal']=False
        for y in range(0,a.RasterYSize,block_rows):
            h=min(block_rows,a.RasterYSize-y)
            x=ba.ReadAsArray(0,y,a.RasterXSize,h)
            z=bb.ReadAsArray(0,y,b.RasterXSize,h)
            # GDAL mask bands cover nodata, explicit mask, and alpha masks.
            ma=ba.GetMaskBand().ReadAsArray(0,y,a.RasterXSize,h)!=0
            mb=bb.GetMaskBand().ReadAsArray(0,y,b.RasterXSize,h)!=0
            out['total_pixels']+=x.size
            out['mask_mismatches']+=int(np.count_nonzero(ma!=mb))
            common=ma&mb
            out['valid_both']+=int(np.count_nonzero(common))
            if np.any(common):
                xx=x[common];zz=z[common]
                equal=np.isclose(xx,zz,rtol=0,atol=tolerance,equal_nan=True)
                out['value_mismatches']+=int(np.count_nonzero(~equal))
                diff=np.abs(xx.astype('float64')-zz.astype('float64'))
                diff=np.nan_to_num(diff,nan=0,posinf=float('inf'))
                out['max_abs_diff']=max(out['max_abs_diff'],float(np.max(diff)))
                delta_sum+=float(np.sum(diff))
    out['mean_abs_diff']=delta_sum/out['valid_both'] if out['valid_both'] else 0.
    out['pass']=all(out[k] for k in ('shape_equal','grid_equal','crs_equal','dtype_equal','nodata_equal')) and out['mask_mismatches']==0 and out['value_mismatches']==0
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raster_1024');p.add_argument('raster_4096');p.add_argument('--tolerance',type=float,default=0.);p.add_argument('--output');args=p.parse_args()
    result=compare(args.raster_1024,args.raster_4096,args.tolerance)
    data=json.dumps(result,indent=2,ensure_ascii=False)
    print(data)
    if args.output:
        with open(args.output,'w',encoding='utf-8') as f:f.write(data+'\n')
    raise SystemExit(0 if result['pass'] else 1)
