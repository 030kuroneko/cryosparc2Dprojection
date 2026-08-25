# Use the native Rendering Map grid by default

When `--render-grid-size` is omitted, extract the Camera View Render surface
from the selected Rendering Map at its complete native grid. Comparison DPI
continues to control only the raster presentation size; it does not select the
surface sampling resolution. This avoids enlarging a low-resolution surface
mesh merely because a high-DPI figure makes its facets easier to see.

An explicit `--render-grid-size` remains a presentation override for reducing
rendering cost. It has no fixed software maximum, but its effective value never
upsamples past the selected Rendering Map. Non-cubic maps preserve their aspect
ratio when downsampled. These choices affect only surface presentation and must
not change Class Camera Orientation, Matched Projection, or matching scores.

Full native grids may require substantial memory and rendering time. Before
surface extraction, report the original map shape, automatic or manual mode,
requested grid, effective sampled shape, whether downsampling occurs, and a
lower-bound working-memory estimate in the terminal, CryoSPARC Job Log, and
result JSON. Warn, but continue, when the estimate exceeds approximately 1 GiB,
regardless of whether the grid was automatic or manual.

Never silently lower the grid after a rendering failure. Report the effective
grid that failed and recommend the next smaller standard tier from
`512, 384, 256, 192, 128` that is below it, phrased as that value or smaller.
The scientific result remains reproducible because any retry is an explicit
user choice.
