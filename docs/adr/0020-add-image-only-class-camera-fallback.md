# Add image-only Class Camera fallback

Retain the Select 2D Classes + Refinement workflow and implement image-only Class Camera Orientation search inside this project for each class lacking usable refinement particle poses, using the supplied reference map. Do not require a separate Reference Based Auto Select 2D job: its reusable pose output contract is unverified, and the user wants the existing workflow to handle fallback directly.

Use global coarse search followed by local refinement of multiple candidates, with standard and fine settings calibrated by measurements. Group symmetry-equivalent cameras into Orientation Groups before comparing competing solutions; show the best group's representative, flag ambiguity between distinct groups, and preserve alternative candidates. Missing-pose fallback does not fabricate particle-derived spread or confidence.

Support NVIDIA GPU execution with CPU execution when no usable GPU is available. Both backends use the same search rules and require numerical consistency checks; acceleration is not claimed until measured. On GPU out-of-memory errors, reduce the batch and retry before restarting on CPU; report the reason. Other runtime errors fail visibly.

Acceptance covers known synthetic cameras, partial and complete absence of usable particle poses, symmetry-equivalent groups, CPU/GPU agreement and a real CryoSPARC projection comparison with timings. Missing GPU hardware or real data must be reported as unverified rather than counted as a successful validation. These behaviors are tested at the public image-camera search and External Job result boundaries, with launcher argument validation covering the user-facing settings.

This narrows the first implementation relative to ADR 0002: low-scoring pose-seeded results do not trigger global recovery in this change. See the [research and design record](../research/reference-based-class-orientation.md) for evidence and unresolved implementation prerequisites.
