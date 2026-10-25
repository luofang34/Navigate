"""Geometric verification of masked image-to-terrain correspondences."""
import cv2
import numpy as np


def spatial_holdout(pixels, size):
    cells = np.floor(pixels / (np.asarray(size) / 12)).astype(int)
    return (cells[:, 0] * 17 + cells[:, 1] * 31) % 5 == 0


def prior_radius(position):
    reported=position.get('reported_horizontal_error_m') or 100
    explicit=position.get('search_radius_m') or 0
    if not np.isfinite([reported,explicit]).all() or min(reported,explicit)<0:
        raise ValueError('position uncertainty must be finite and nonnegative')
    return max(1500.,3*reported,explicit)


def candidate_priority(candidate):
    score=candidate['score']
    depth=candidate.get('anchor_depth',int(candidate.get('evidence_kind')=='photo_to_terrain_anchor'))
    verified=candidate.get('fixed_render',{}).get('passed') is True
    if verified:
        rendered=candidate['fixed_render']
        quality=rendered['image_support_fraction']*rendered['inlier_fraction']/(1+rendered['median_px'])
    else:
        quality=score['image_support_fraction']*score['holdout_inlier_fraction']/(1+score['holdout_median_px'])
    # Dense matchers emit different sample counts. Prefer direct terrain evidence
    # and short anchor chains, then fixed-render agreement before fitting support.
    return (int(depth==0),-depth,int(verified),quality,min(score['inliers'],1000))


def training_error(points,pixels,K,rvec,tvec):
    matrix=cv2.Rodrigues(rvec)[0]
    if not np.all((points@matrix.T+tvec.ravel())[:,2]>0):
        return float('inf')
    predicted=cv2.projectPoints(points,rvec,tvec,K,None)[0].reshape(-1,2)
    return float(np.mean(np.sum((predicted-pixels)**2,axis=1)))


def fit(points, pixels, intrinsics, size, focal_bounds=None):
    if len(points) < 30:
        return None
    width,height = size
    # Spatial buckets keep neighboring correspondences on the same side of the split.
    held = spatial_holdout(pixels, size)
    if held.sum() < 6 or (~held).sum() < 24:
        return None
    cv2.setRNGSeed(0)
    # Nearly planar shoreline points can make EPNP's final inlier refit diverge
    # even when its RANSAC consensus is good. Refine that consensus iteratively.
    ok,rvec,tvec,indices = cv2.solvePnPRansac(points[~held], pixels[~held], intrinsics, None,
        iterationsCount=2000, reprojectionError=4, confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or indices is None or len(indices)<20:
        return None
    training = np.flatnonzero(~held)[indices.ravel()]
    error=training_error(points[training],pixels[training],intrinsics,rvec,tvec)
    ok,refined_r,refined_t = cv2.solvePnP(points[training],pixels[training],intrinsics,None,
                               rvec.copy(),tvec.copy(),True,flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        refined_error=training_error(points[training],pixels[training],intrinsics,refined_r,refined_t)
        if np.isfinite(refined_error) and refined_error<=error:
            rvec,tvec,error=refined_r,refined_t,refined_error
    if focal_bounds is not None:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
                 cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
                 cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        try:
            _,candidate,_,rs,ts = cv2.calibrateCamera([points[training].astype(np.float32)],
                [pixels[training].astype(np.float32)],size,intrinsics.copy(),None,flags=flags)
            if focal_bounds[0] <= candidate[0,0] <= focal_bounds[1]:
                refined_error=training_error(points[training],pixels[training],candidate,rs[0],ts[0])
                if np.isfinite(refined_error) and refined_error<=error:
                    intrinsics=candidate; rvec=rs[0]; tvec=ts[0]
        except cv2.error:
            pass
    predicted = cv2.projectPoints(points,rvec,tvec,intrinsics,None)[0].reshape(-1,2)
    errors = np.linalg.norm(predicted-pixels,axis=1)
    inliers = errors < 4
    tested = errors[held]
    support = cv2.convexHull(pixels[inliers].astype(np.float32)) if inliers.sum()>3 else None
    coverage = float(cv2.contourArea(support)/(width*height)) if support is not None else 0
    matrix = cv2.Rodrigues(rvec)[0]
    eye = (-matrix.T@tvec).ravel(); rotation = matrix.T@np.diag([1,-1,-1])
    positive = ((points[inliers]@matrix.T+tvec.ravel())[:,2]>0).mean() if inliers.any() else 0.
    score = {"pairs":len(points), "inliers":int(inliers.sum()), "train_inliers":len(training),
             "focal_px":float(intrinsics[0,0]),
             "holdout_count":int(held.sum()), "holdout_inlier_fraction":float(np.mean(tested<4)),
             "holdout_median_px":float(np.median(tested)),
             "train_rms_px":float(np.sqrt(np.mean(errors[training]**2))),
             "image_support_fraction":coverage, "positive_depth_fraction":float(positive)}
    accepted = (inliers.sum()>=30 and coverage>=.05 and positive==1
                and score["holdout_inlier_fraction"]>=.6 and score["holdout_median_px"]<=4)
    return eye,rotation,{**score,"geometry_accepted":bool(accepted)},inliers


def ground_pairs(pairs, depth, world, valid_mask):
    pairs=np.asarray(pairs,dtype=np.float64).reshape(-1,2,2)
    h,w=depth.shape
    if not len(pairs):
        return np.empty((0,3)),np.empty((0,2))
    xy=np.rint(pairs).astype(int)
    query_h,query_w=valid_mask.shape
    inside=(np.all(xy>=0,axis=(1,2))&(xy[:,0,0]<w)&(xy[:,0,1]<h)
            &(xy[:,1,0]<query_w)&(xy[:,1,1]<query_h))
    pairs=pairs[inside]; xy=xy[inside]
    ref,query=xy[:,0],xy[:,1]
    good=(depth[ref[:,1],ref[:,0]]>0)&(valid_mask[query[:,1],query[:,0]]>0)
    return world[ref[good,1],ref[good,0]],pairs[good,1]
