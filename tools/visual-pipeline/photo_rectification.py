"""Planar view hypotheses remove foreshortening before regional image retrieval."""
import math

import cv2
import numpy as np

from photo_surface import attitude, horizon_attitude


def planar_views(image,mask,labels):
    height,width=image.shape[:2]
    y,x=np.nonzero(mask[::8,::8]>0)
    pixels=np.column_stack([x*8,y*8,np.ones(len(x))])
    if len(pixels)<20:
        return
    for equivalent in [20,35,70]:
        focal=equivalent*math.hypot(width,height)/math.hypot(36,24)
        K=np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
        horizon=horizon_attitude(labels,K)
        angles=[horizon['down_deg']] if horizon else [35,60,85]
        roll=horizon['roll_deg'] if horizon else 0
        for down in angles:
            # The ground plane lies one unit below the optical centre. Unknown
            # altitude changes scale, which the multi-scale reference bank searches.
            rays=attitude(0,down,roll)@np.diag([1,-1,-1])@np.linalg.inv(K)
            homography=np.diag([1,1,-1])@rays
            ground=pixels@homography.T
            ground=ground[ground[:,2]>.05]
            if len(ground)<20:
                continue
            ground=ground[:,:2]/ground[:,2,None]
            low,high=np.percentile(ground,[5,95],axis=0)
            if np.any(high-low<.1):
                continue
            scale=min(760/(high[0]-low[0]),560/(high[1]-low[1]))
            resize=np.array([[scale,0,20-scale*low[0]],[0,-scale,580+scale*low[1]],[0,0,1.]])
            transform=resize@homography
            warped=cv2.warpPerspective(image,transform,(800,600))
            valid=cv2.warpPerspective(mask,transform,(800,600),flags=cv2.INTER_NEAREST)
            warped[valid==0]=0
            yield warped,np.linalg.inv(transform),{'rectified':True,'focal_equivalent_mm':equivalent,
                                                     'down_deg':down,'roll_deg':roll}
