import json, math, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from webserver.coverage import plan,tile_bounds,tile_xy
from webserver.naip import raster_tile

class CoverageTests(unittest.TestCase):
    def test_area_download_keeps_the_requested_coordinates(self):
        request={'bounds':[-74.480,40.530,-74.425,40.565],'zoom':16}
        result=plan(request)
        self.assertEqual(result['requested'],request)
        self.assertGreater(len(result['imagery_tiles']),1)
        w,s,e,n=result['bounds']; a,b,c,d=request['bounds']
        self.assertLessEqual(w,a);self.assertLessEqual(s,b)
        self.assertGreaterEqual(e,c);self.assertGreaterEqual(n,d)

    def test_route_covers_samples_without_filling_the_entire_box(self):
        route=[[-74.47,40.54],[-74.40,40.59]]
        result=plan(dict(route=route,buffer_m=100,zoom=16))
        tiles={tuple(t) for t in result['imagery_tiles']}
        for i in range(101):
            p=[route[0][j]+(route[1][j]-route[0][j])*i/100 for j in range(2)]
            x,y=tile_xy(*p,16);self.assertIn((16,math.floor(x),math.floor(y)),tiles)
        area=plan(dict(bounds=result['bounds'],zoom=16))
        self.assertLess(len(tiles),len(area['imagery_tiles']))

    def test_large_and_invalid_requests_fail_before_provider_access(self):
        for value in [dict(bounds=[-180,-80,180,80]),dict(bounds=[1,2,0,3]),dict(route=[[1,2],[math.nan,3]]),dict(route=[[1,2],[1,3]],buffer_m=-1),dict(bounds=[0,0,1,1],zoom=99)]:
            with self.assertRaises(ValueError):plan(value)

    def test_raster_reprojection_preserves_validity_and_band_order(self):
        import numpy as np,rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds
        xyz=[14,4800,6200];bounds=tile_bounds(*xyz)
        data=np.empty((4,64,64),dtype=np.uint8);data[0]=21;data[1]=64;data[2]=149;data[3]=200
        data[:,:,:8]=0
        with MemoryFile() as file:
            with file.open(driver='GTiff',width=64,height=64,count=4,dtype='uint8',crs='EPSG:4326',photometric='RGB',ALPHA='NO',transform=from_bounds(*bounds,64,64),nodata=0) as source:
                source.write(data)
                source.colorinterp=(rasterio.enums.ColorInterp.red,rasterio.enums.ColorInterp.green,rasterio.enums.ColorInterp.blue,rasterio.enums.ColorInterp.undefined)
            with file.open() as source:
                image=raster_tile([{'bbox':bounds}],[source],xyz)
                self.assertEqual(image.size,(512,512));self.assertEqual(image.getpixel((256,256)),(21,64,149,255));self.assertEqual(image.getpixel((0,256))[3],0)

if __name__=='__main__':unittest.main()
