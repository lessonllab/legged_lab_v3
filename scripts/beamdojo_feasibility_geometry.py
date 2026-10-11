"""Continuous planar support for a conservative G1 collision-envelope rectangle.

Only same-height horizontal top surfaces are modeled. This is NOT dynamics,
contact pressure, leg reach, friction, collision-free swing, or actual foot-mesh
feasibility. A margin expands the local sole envelope. Coverage is of a UNION,
not a sampled sole, terrain convex hull, or a single supporting rectangle.

Algorithm: subtract each AABB rectangle from the convex uncovered pieces. A
convex polygon minus a rectangle is partitioned into disjoint convex polygons
by the rectangle's four halfplanes. Any final positive-area piece is a hole.
Floating-point geometric clipping is used; tiny areas below a scale-dependent
roundoff tolerance cannot be distinguished from numerical zero.
"""
import math
import numpy as np


def _array(value, name):
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + ' must be numeric') from error
    if not np.isfinite(a).all():
        raise ValueError(name + ' must be finite')
    return a


def _scalar(value, name):
    a = _array(value, name)
    if a.ndim:
        raise ValueError(name + ' must be scalar')
    return float(a)


def foot_polygon(ankle_xy, yaw, *, x_bounds=(-.055, .145),
                 y_bounds=(-.035, .035), margin=0.):
    """CCW expanded sole corners [4,2], anchored at ankle-roll link XY, meters."""
    ankle = _array(ankle_xy, 'ankle_xy')
    x = _array(x_bounds, 'x_bounds'); y = _array(y_bounds, 'y_bounds')
    angle = _scalar(yaw, 'yaw'); m = _scalar(margin, 'margin')
    if ankle.shape != (2,) or x.shape != (2,) or y.shape != (2,):
        raise ValueError('ankle and each bound must have shape [2]')
    if m < 0 or x[0] >= x[1] or y[0] >= y[1]:
        raise ValueError('margin must be nonnegative and bounds strictly increasing')
    local = np.array([[x[0]-m,y[0]-m],[x[1]+m,y[0]-m],
                      [x[1]+m,y[1]+m],[x[0]-m,y[1]+m]])
    c,s = math.cos(angle),math.sin(angle)
    polygon = local @ np.array([[c,s],[-s,c]]) + ankle
    if not np.isfinite(polygon).all():
        raise ValueError('foot coordinates overflow')
    return polygon


def _area(poly):
    if len(poly) < 3:
        return 0.
    p = poly-poly[0]  # Avoid cancellation for translated maps.
    return abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))))*.5


def _clip(poly, axis, bound, keep_greater):
    """Sutherland-Hodgman against an axis-aligned closed halfplane."""
    if len(poly) == 0:
        return poly
    out=[]; prev=poly[-1]; pd=(prev[axis]-bound)*(1 if keep_greater else -1)
    for curr in poly:
        cd=(curr[axis]-bound)*(1 if keep_greater else -1)
        if (cd>=0) != (pd>=0):
            t=pd/(pd-cd)
            p=prev+t*(curr-prev);p[axis]=bound
            out.append(p)
        if cd>=0:
            out.append(curr)
        prev,pd=curr,cd
    return np.asarray(out,dtype=np.float64).reshape(-1,2)


def _subtract_rectangle(poly, rect, area_epsilon):
    # At every stage, outside pieces are disjoint; only inside continues.
    remaining=poly; pieces=[]
    for axis,bound,greater in ((0,rect[0],True),(0,rect[2],False),
                               (1,rect[1],True),(1,rect[3],False)):
        outside=_clip(remaining,axis,bound,not greater)
        if _area(outside)>area_epsilon:
            pieces.append(outside)
        remaining=_clip(remaining,axis,bound,greater)
        if _area(remaining)<=area_epsilon:
            break
    return pieces


def foot_supported(rectangles_m, ankle_xy, yaw, *, x_bounds=(-.055,.145),
                   y_bounds=(-.035,.035), margin=0.):
    """Whether the complete continuous envelope is covered by rectangle union.

    rectangles_m must be [N,4] xmin,ymin,xmax,ymax, with positive dimensions.
    Empty [0,4] is valid and returns False. Shared edges count as coverage;
    holes and gaps of positive area reject support, subject to FP64 roundoff.
    """
    rects=_array(rectangles_m,'rectangles_m')
    if rects.ndim!=2 or rects.shape[1]!=4:
        raise ValueError('rectangles_m must have shape [N,4]')
    if np.any(rects[:,0]>=rects[:,2]) or np.any(rects[:,1]>=rects[:,3]):
        raise ValueError('terrain rectangles must have positive dimensions')
    foot=foot_polygon(ankle_xy,yaw,x_bounds=x_bounds,y_bounds=y_bounds,margin=margin)
    # Translate together before clipping, improving large-coordinate accuracy.
    origin=_array(ankle_xy,'ankle_xy');foot=foot-origin
    rects=rects-np.tile(origin,2)
    lo=foot.min(0);hi=foot.max(0)
    rects=rects[(rects[:,2]>lo[0])&(rects[:,0]<hi[0])&
                (rects[:,3]>lo[1])&(rects[:,1]<hi[1])]
    if not len(rects):
        return False
    if np.any((rects[:,0]<=lo[0])&(rects[:,1]<=lo[1])&
              (rects[:,2]>=hi[0])&(rects[:,3]>=hi[1])):
        return True
    eps=32*np.finfo(np.float64).eps*max(float(np.max(hi-lo)**2),1e-12)
    uncovered=[foot]
    for rect in rects:
        new=[]
        for poly in uncovered:
            pl=poly.min(0);ph=poly.max(0)
            if rect[2]<=pl[0] or rect[0]>=ph[0] or rect[3]<=pl[1] or rect[1]>=ph[1]:
                new.append(poly)
            else:
                new.extend(_subtract_rectangle(poly,rect,eps))
        uncovered=new
        if not uncovered:
            return True
    return sum(_area(p) for p in uncovered)<=eps


def _convex_polygon(value,name):
    p=_array(value,name)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3:
        raise ValueError(name+' must be [V,2], V>=3')
    if np.array_equal(p[0],p[-1]):
        p=p[:-1]
    if len(p)<3 or _area(p)==0:
        raise ValueError(name+' must have positive area')
    edges=np.roll(p,-1,axis=0)-p
    if np.any(np.linalg.norm(edges,axis=1)==0):
        raise ValueError(name+' has duplicate adjacent vertices')
    # Every vertex must lie in the same closed halfplane of every edge;
    # this rejects concave and self-crossing orderings, including star polygons.
    delta=p[None,:,:]-p[:,None,:]
    crosses=edges[:,None,0]*delta[:,:,1]-edges[:,None,1]*delta[:,:,0]
    eps=32*np.finfo(float).eps*max(float(np.ptp(p,axis=0).max()**2),1e-12)
    if not (np.all(crosses>=-eps) or np.all(crosses<=eps)):
        raise ValueError(name+' must be a convex boundary-ordered polygon')
    return p


def polygons_overlap(a,b,tolerance=1e-9):
    """Strict interior overlap of convex polygons (either orientation).

    Shared edges/vertices are NOT overlap. tolerance is a nonnegative separating
    axis penetration tolerance in meters, not area. Nonconvex polygons reject.
    """
    a=_convex_polygon(a,'a');b=_convex_polygon(b,'b')
    tol=_scalar(tolerance,'tolerance')
    if tol<0:
        raise ValueError('tolerance must be nonnegative')
    origin=a[0].copy();a=a-origin;b=b-origin
    for poly in (a,b):
        e=np.roll(poly,-1,axis=0)-poly
        normals=np.stack((-e[:,1],e[:,0]),axis=-1)
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        pa=a@normals.T;pb=b@normals.T
        penetration=np.minimum(pa.max(0),pb.max(0))-np.maximum(pa.min(0),pb.min(0))
        if np.any(penetration<=tol):
            return False
    return True
