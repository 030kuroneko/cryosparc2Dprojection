# Separate Search Projections from native-grid Matched Projections

Use a bounded Search Projection of at most 128 pixels per side to select and
raw-score each Class Camera Orientation. After selection, keep the camera fixed
and regenerate a Matched Projection from the unsharpened Matching Map at the
native Class Average box and pixel size. Re-optimize only XY translation on
that native grid. Comparison DPI changes raster layout, never projection data
resolution.

This preserves existing search cost and raw-score behavior while preventing a
high-DPI Class Result from merely enlarging a 128-pixel projection. It also
keeps presentation resolution independent of the Surface Sampling Grid Size,
which belongs only to 3D isosurface rendering.

Publish native-grid images as `matched_projections` and publish the actual
bounded scoring images separately as `search_projections`. Label the raw score
as a search score. Recalculate the Diagnostic Band-Limited Score on the native
Class Average and native-grid Matched Projection, because that diagnostic is
post-selection and should describe the displayed pair. Record both search and
native projection shifts with their respective pixel sizes.

All selected classes must share one native box size to form the CryoSPARC MRCS
outputs. Fail clearly on inconsistent template sizes or native reprojection
failure; never silently resample classes to a common smaller box or substitute
the bounded Search Projection.
