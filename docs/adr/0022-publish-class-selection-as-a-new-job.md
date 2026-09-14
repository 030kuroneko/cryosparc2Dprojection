# Publish Class Selection as a new job

After Class Orientation completes, users select original classes in the web results page and export the four Class Selection Outputs to a new CryoSPARC External Job in the source workflow's project and workspace. Each export creates a new result job, preserving prior selections and downstream provenance rather than rewriting completed outputs. This feature is scoped to Class Orientation; Axis Search is excluded.

Integrate with the image-only fallback implemented on `codex/image-camera-fallback` (implementation commit `ae02b1b`, inspected branch tip `7165823`). Classes with no usable refinement particle poses remain eligible for selection alongside pose-derived results. Export partitions the original 2D input datasets by source class identity, not the refinement-overlap subset or pose-validity-filtered arrays used for orientation estimation; synthetic projections never replace the original class averages.

The fallback branch deliberately does not trigger global recovery for low-scoring pose-seeded results. This selection feature preserves that boundary and does not require a new low-confidence recovery algorithm.
