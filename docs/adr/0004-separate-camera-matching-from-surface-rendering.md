# Separate camera matching from surface rendering

_Oblique-render portions superseded by ADR 0005._

Determine every Class Camera Orientation only from the unsharpened Matching Map, then treat the Rendering Map, Surface Level, and surface styling as presentation choices. This boundary preserves the scientific camera result when users switch to a sharpened map or adjust the surface for clearer inspection, while retaining an exact orthographic Camera View Render.
