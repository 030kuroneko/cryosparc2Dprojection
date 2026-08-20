# Use hybrid Class Camera matching

Seed each Class Camera Orientation from overlapping particles' refinement poses combined with their 2D alignments, then refine it by matching a map projection to the actual selected class average. This avoids the cost of unconditional global search while recovering full in-plane rotation and provides a projection-based validation of the pose-derived result; low-confidence local results trigger a symmetry-aware global recovery search.
