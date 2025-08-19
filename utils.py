\
import cv2
import numpy as np
from shapely.geometry import Point, Polygon

def denorm_pts(poly_norm, w, h):
    return [(int(x*w), int(y*h)) for x,y in poly_norm]

def point_in_polygon(pt, polygon_pts):
    # polygon_pts: list of (x,y) ints
    poly = Polygon(polygon_pts)
    return poly.contains(Point(pt[0], pt[1]))

def draw_translucent_poly(img, pts, color=(0,255,0), alpha=0.2, thickness=2):
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

def put_text(img, text, org, scale=0.7, color=(255,255,255), thickness=2, bg=True):
    if bg:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x-4, y-h-6), (x+w+4, y+4), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
