"""Robust BHA for broken / sliver-heavy meshes (intrinsic-Delaunay / robust Laplacian).

Stock BHA's biharmonic solve and heat-method geodesics produce garbage on meshes with disconnected
components or many sliver/obtuse triangles: negative cotangent weights make the 4th-order biharmonic
solve ill-conditioned (distances blow up / go negative). This variant builds the operator from
`robust_laplacian.mesh_laplacian` (Sharp & Crane's intrinsic, mollified, guaranteed-PSD Laplacian +
lumped mass) and gets landmark geodesics from `potpourri3d.MeshHeatMethodDistanceSolver` (intrinsic
Delaunay heat method). Mesh-agnostic -- operates on arbitrary (points, faces); no vertex is moved.

Requires `robust_laplacian` and `potpourri3d`.
"""
import numpy as np
from scipy import sparse
import cupy
import robust_laplacian
import potpourri3d as pp3d
from .bha_cupy import BHA


class _ShapeOnly:
    """Minimal stand-in passed to BHA.fit for its `n, d = X.shape` (distances come from the
    potpourri3d closure, not from X)."""
    def __init__(self, n):
        self.shape = (n, n)


class RobustBHA(BHA):
    """BHA whose biharmonic operator uses a robust (intrinsic, mollified) Laplacian and whose
    landmark geodesics use potpourri3d's intrinsic-Delaunay heat method."""

    mollify_factor = 1e-5

    def preprocessing(self, pts, polys):
        self._pts = pts
        self._polys = polys
        L, Mmass = robust_laplacian.mesh_laplacian(
            np.asarray(pts, dtype=np.float64), np.asarray(polys, dtype=np.int64),
            mollify_factor=self.mollify_factor)        # L: PSD stiffness, Mmass: lumped mass
        d = np.asarray(Mmass.diagonal(), dtype=np.float64)
        Minv = sparse.dia_matrix((1.0 / d, [0]), Mmass.shape).tocsr()
        self._M = (L.dot(Minv.dot(L))).tocsr()         # biharmonic operator, well-conditioned
        self._lapW = None
        self._lapV = None
        self._Dinv = Minv

    @classmethod
    def from_surface(cls, pts, polys, l, nnz_row=100, m=1.0, **kwargs):
        V = np.asarray(pts, dtype=np.float64)
        F = np.asarray(polys, dtype=np.int64)
        solver = pp3d.MeshHeatMethodDistanceSolver(V, F)

        def get_rows(rows):
            return np.vstack([solver.compute_distance(int(i)) for i in rows])

        def geodesic(_, source=None, dest=None):
            if source is None and dest is None:
                return np.vstack([solver.compute_distance(i) for i in range(len(pts))])
            elif dest is None:
                return get_rows(source).T
            else:
                return get_rows(source)[:, dest].T

        bha = cls(l, geodesic, nnz_row=nnz_row, **kwargs)
        bha.preprocessing(pts, polys)
        bha.fit(_ShapeOnly(len(pts)))
        return bha


def make_bha_get_row(pts, polys, l=960, nnz_row=100, sanity=True):
    """Return a `get_row(v) -> numpy (nv,)` closure backed by a robust BHA at landmark count `l`.

    Materializes the (sources x nv) distance matrix one source-row at a time.
    """
    if sanity:
        s = pp3d.MeshHeatMethodDistanceSolver(np.asarray(pts, float), np.asarray(polys, np.int64))
        g = np.asarray(s.compute_distance(0)).ravel()
        print(f"[robust_bha] sanity: finite={np.isfinite(g).all()} "
              f"min={np.nanmin(g):.2f} max={np.nanmax(g):.2f} negatives={(g < 0).sum()}", flush=True)
    bha = RobustBHA.from_surface(pts, polys, l, nnz_row=nnz_row)
    return lambda v: cupy.asnumpy(bha.get_row(int(v))).ravel()
