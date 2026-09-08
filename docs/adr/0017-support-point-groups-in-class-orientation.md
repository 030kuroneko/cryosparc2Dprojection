# Support point groups in Class Orientation

Supersedes the accepted-name restriction in ADR 0006, following approval to add
Cn, Dn, T, and O alongside C1 and I.

Use a validated symmetry value instead of an enum with a fixed number of names.
Share a single coordinate-convention adapter between direction folding and camera
folding. Apply the explicit Dn and T coordinate conversions documented in
[the convention research](../research/cryosparc-point-groups.md).

CLI, desktop, and web input accept any positive integer order through the same
validator. Keep I1/I2 and unsupported symmetry families rejected. Preserve the
symmetry name in existing result metadata. Axis Search support is extended by [ADR 0018](0018-support-point-groups-in-axis-search.md).

Local convention and workflow tests accompany this change. Unlike the original
ADR's intended prerequisite, live CryoSPARC operator/projection validation has
not been completed; document this limitation, particularly the inferred T and O
embeddings, rather than presenting synthetic checks as installed-version proof.
