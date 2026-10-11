"""Spherical terrain ray casting from a frozen XYZ reference catalog."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from photo_testset import digest

RADIUS = 6371008.8


def attitude(heading, down, roll=0):
    h, d, r = np.radians([heading, down, roll])
    forward = np.array([np.sin(h)*np.cos(d), np.cos(h)*np.cos(d), -np.sin(d)])
    right = np.array([np.cos(h), -np.sin(h), 0])
    base = np.column_stack([right, np.cross(right, forward), -forward])
    return base @ np.array([[np.cos(r),-np.sin(r),0], [np.sin(r),np.cos(r),0], [0,0,1]])


def basis(lat, lon):
    la, lo = np.radians([lat,lon])
    return np.column_stack([[-np.sin(lo), np.cos(lo), 0],
        [-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)],
        [np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]])


def frame(catalog):
    west,south,east,north=catalog['configuration']['bbox']
    matrix=basis((south+north)/2,(west+east)/2)
    return matrix,matrix[:,2]*RADIUS


def reframe_camera(eye,rotation,source,destination):
    old_basis,old_origin=source;new_basis,new_origin=destination
    return (old_basis@eye+old_origin-new_origin)@new_basis,new_basis.T@old_basis@rotation


class Surface:
    def __init__(self, catalog):
        self.root = Path(catalog).parent
        self.catalog = json.loads(Path(catalog).read_text())
        if not self.catalog.get("complete"):
            raise ValueError("reference catalog download is incomplete")
        west,south,east,north = self.catalog["configuration"]["bbox"]
        self.anchor = [(south+north)/2, (west+east)/2]
        self.basis,self.origin = frame(self.catalog)
        self.grids = {}
        for kind in ["imagery", "dem"]:
            tiles = [t for t in self.catalog["tiles"] if t["kind"] == kind]
            zoom = tiles[0]["xyz"][0]
            x0 = min(t["xyz"][1] for t in tiles); y0 = min(t["xyz"][2] for t in tiles)
            x1 = max(t["xyz"][1] for t in tiles); y1 = max(t["xyz"][2] for t in tiles)
            shape = ((y1-y0+1)*256,(x1-x0+1)*256)
            grid = np.zeros((*shape,3),np.uint8) if kind == "imagery" else np.full(shape,np.nan,np.float32)
            valid = np.zeros(shape,np.uint8)
            for t in tiles:
                if digest(self.root/t["path"]) != t["sha256"]:
                    raise ValueError(f"reference tile hash mismatch: {t['path']}")
                im = cv2.imread(str(self.root/t["path"]),cv2.IMREAD_COLOR)
                if im is None or im.shape != (256,256,3):
                    raise ValueError(f"invalid reference tile {t['path']}")
                if kind == "dem":
                    im = im[...,2].astype(np.float32)*256 + im[...,1] + im[...,0]/256 - 32768
                _,x,y = t["xyz"]; row=slice((y-y0)*256,(y-y0+1)*256); col=slice((x-x0)*256,(x-x0+1)*256)
                grid[row,col] = im; valid[row,col] = 255
            self.grids[kind] = (grid,valid,zoom,x0,y0)

    def local(self, lat, lon, altitude):
        unit = basis(lat,lon)[:,2]
        return (unit*(RADIUS+altitude)-self.origin) @ self.basis

    def geodetic(self, points):
        xyz = np.asarray(points) @ self.basis.T + self.origin
        lat = np.degrees(np.arctan2(xyz[...,2],np.hypot(xyz[...,0],xyz[...,1])))
        lon = np.degrees(np.arctan2(xyz[...,1],xyz[...,0]))
        altitude = np.linalg.norm(xyz,axis=-1)-RADIUS
        return lat,lon,altitude

    def camera_rotation(self, lat, lon, heading, down, roll=0):
        return self.basis.T @ basis(lat,lon) @ attitude(heading,down,roll)

    def sample(self, kind, lat, lon):
        grid,valid,zoom,x0,y0 = self.grids[kind]
        x = ((np.asarray(lon)+180)/360*2**zoom-x0)*256-.5
        y = ((1-np.arcsinh(np.tan(np.radians(lat)))/np.pi)/2*2**zoom-y0)*256-.5
        shape = np.shape(x)
        mx = np.atleast_2d(x).astype(np.float32); my=np.atleast_2d(y).astype(np.float32)
        # NaN padding poisons even an exact edge sample through a zero interpolation weight.
        sampled = cv2.remap(grid,mx,my,cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT if kind=="imagery" else cv2.BORDER_REPLICATE)
        covered = cv2.remap(valid,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)>254
        if kind == "dem":
            covered &= np.isfinite(sampled)
        channels = (3,) if kind=="imagery" else ()
        return sampled.reshape(shape+channels), covered.reshape(shape)

    def render(self, eye, rotation, intrinsics, size, far_m=40000):
        width,height = size
        x,y = np.meshgrid(np.arange(width),np.arange(height))
        rays = np.stack([(x-intrinsics[0,2])/intrinsics[0,0],
                         (intrinsics[1,2]-y)/intrinsics[1,1], -np.ones_like(x)],-1) @ rotation.T
        near = np.zeros((height,width)); far=np.full_like(near,far_m)
        hit=np.zeros_like(near,bool); previous=0.0
        previous_above=np.zeros_like(hit)
        for distance in np.geomspace(10,far_m,100):
            points=eye+distance*rays
            lat,lon,altitude=self.geodetic(points)
            terrain,covered=self.sample("dem",lat,lon)
            # A ray entering DEM coverage below ground has no observed surface
            # crossing; treating that coverage edge as terrain creates deep false hits.
            found=(altitude<=terrain)&covered&previous_above&~hit
            near[found]=previous; far[found]=distance; hit|=found; previous=distance
            previous_above=(altitude>terrain)&covered
        for _ in range(12):
            mid=(near+far)/2; points=eye+mid[...,None]*rays
            lat,lon,altitude=self.geodetic(points); terrain,covered=self.sample("dem",lat,lon)
            above=(altitude>terrain)|~covered
            near=np.where(above,mid,near); far=np.where(above,far,mid)
        depth=(near+far)/2; points=eye+depth[...,None]*rays
        lat,lon,_=self.geodetic(points); color,covered=self.sample("imagery",lat,lon)
        hit &= covered
        color[~hit]=0; depth[~hit]=0
        return color,depth,points


class ContextSurface(Surface):
    """Keep detailed local samples and fill uncovered rays from a wider catalog."""
    def __init__(self, base, context):
        self.__dict__.update(base.__dict__)
        self.base = base
        self.context = context

    def sample(self, kind, lat, lon):
        local, covered = self.base.sample(kind, lat, lon)
        distant, available = self.context.sample(kind, lat, lon)
        selected = covered[..., None] if kind == 'imagery' else covered
        return np.where(selected, local, distant), covered | available


def horizon_line(points,width,height):
    tolerance=max(3.,height*.01)
    best=None
    for i,a in enumerate(points):
        for b in points[i+1:]:
            if b[0]-a[0]<width*.2:
                continue
            slope=(b[1]-a[1])/(b[0]-a[0])
            residual=np.abs(points[:,1]-a[1]-slope*(points[:,0]-a[0]))/math.hypot(1,slope)
            selected=residual<tolerance
            merit=(int(selected.sum()),float(np.ptp(points[selected,0])),-float(np.median(residual[selected])))
            if best is None or merit>best[0]:
                best=(merit,selected)
    if best is None or best[0][0]<15 or best[0][1]<width*.3:
        return None
    return cv2.fitLine(points[best[1]].astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel(),best[0][0]


def horizon_attitude(labels, intrinsics):
    sky = labels == 2
    xs=[]; ys=[]
    for x in range(0,sky.shape[1],max(1,sky.shape[1]//100)):
        rows=np.flatnonzero(sky[:,x])
        if len(rows)>5 and rows[-1]<sky.shape[0]-10:
            xs.append(x); ys.append(rows[-1])
    if len(xs)<15:
        return None
    # Small sky-labelled reflections, strut gaps and low clouds must not tilt the
    # horizon. Fit the dominant boundary consensus before estimating attitude.
    fitted=horizon_line(np.column_stack([xs,ys]),sky.shape[1],sky.shape[0])
    if fitted is None:
        return None
    line,support=fitted
    vx,vy,x0,y0=line
    if abs(vx)<.2:
        return None
    slope=vy/vx; centre_y=y0+slope*(intrinsics[0,2]-x0)
    roll=math.atan(slope)
    down=math.atan((intrinsics[1,2]-centre_y)*math.cos(roll)/intrinsics[1,1])
    return {"down_deg": math.degrees(down), "roll_deg": math.degrees(roll),
            "support_columns": support, "source": "semantic sky boundary consensus; terrain horizon bias unresolved"}
