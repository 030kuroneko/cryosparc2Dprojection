# Publish Class Selection as a new job

After Class Orientation completes, users select original classes in the web results page and export the four Class Selection Outputs to a new CryoSPARC External Job in the source workflow's project and workspace. Each export creates a new result job, preserving prior selections and downstream provenance rather than rewriting completed outputs. This feature is scoped to Class Orientation; Axis Search is excluded.

Integrate with the image-only fallback implemented on `codex/image-camera-fallback` (implementation commit `ae02b1b`, inspected branch tip `7165823`). Classes with no usable refinement particle poses remain eligible for selection alongside pose-derived results. Export partitions the original 2D input datasets by source class identity, not the refinement-overlap subset or pose-validity-filtered arrays used for orientation estimation; synthetic projections never replace the original class averages.

The fallback branch deliberately does not trigger global recovery for low-scoring pose-seeded results. This selection feature preserves that boundary and does not require a new low-confidence recovery algorithm.

Post-computation Class Selection export is a narrow exception to [ADR 0015](0015-add-private-multi-user-web-launcher.md)'s shared dispatch queue and worker path. It partitions and publishes existing datasets through the SDK using bounded web background threads (at most four active exports), without GPU computation or Slurm execution. This avoids rescheduling or rerunning completed Class Orientation work for each selection.

SQLite records the immutable selection and manifest in `preparing` before dispatch, commits `creating` before remote job creation, and stores the returned job UID with `publishing` before writing outputs. After restart, `preparing` is retryable; an uncertain creation without a recorded UID blocks retries and replacement exports for that source job until administrator reconciliation. Retries with a known UID reuse and reconcile that job instead of creating another one. These durable records, rather than thread lifetime, govern recovery.
