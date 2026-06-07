This example spot-checks imperfect mode sorting for the `distance_3_point_sources_hg_xy`
geometry.

It uses a fixed skew-symmetric generator so the orthogonal mixing family is fully
reproducible and `epsilon` scales the rotation angle directly.

Run it with:

```bash
env UV_CACHE_DIR=/private/tmp/rayfish-uv-cache MPLCONFIGDIR=/private/tmp/rayfish-mpl \
  uv run python examples/distance_3_point_exp_imperfection/simulation.py
```
