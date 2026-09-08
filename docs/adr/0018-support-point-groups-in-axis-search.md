# Support point groups in Axis Search

Extend the shared Cn/Dn/T/O/I conventions to Axis Search. Keep default I and its
existing cameras. Reject C1 before connecting or creating an External Job.

Build new registries from the shared operator sets. Each record is one directed
orbit at its maximal rotation order. Preserve inequivalent orbits, including
opposite cyclic poles and the two tetrahedral threefold orbits. Use Nfold for
the first orbit and Nfold-2, Nfold-3, etc. for additional orbits. New cameras
use a projected Cartesian reference for display roll; I retains its old rule.

Resolve family and display-roll options against the selected symmetry after
parsing all arguments. Carry family records through ranking and refinement;
write their symmetry into result metadata. Expose symmetry in desktop/web
basic settings, using the same validator as the CLI.

See [sources and validation limits](../research/cryosparc-point-groups.md#axis-search).
Local tests are not live CryoSPARC convention validation; T/O embedding
limitations remain as recorded in the preceding decision.
