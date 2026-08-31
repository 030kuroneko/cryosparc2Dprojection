# Use one External Job adapter for the supported workflows

## Status

Accepted

## Decision

The Class Orientation and Symmetry-Axis Search workflows use one small
adapter for the CryoSPARC 5.0.6 External Job API. The adapter owns input and
output registration, Dataset slot details, source and job paths, allocation,
publication, and best-effort preview attachment. Workflows receive typed
template and volume values and retain ownership of scientific calculations,
stage messages, and log wording.

Each workflow stages its complete scientific output set locally before the
adapter publishes outputs in order. A publication failure identifies the
output that failed and does not claim that earlier publications were rolled
back. Preview attachment failures are warnings and do not invalidate
scientific outputs. `ExternalJobSource` is canonical; `SourceOutput` and
`AxisSourceOutput` remain aliases for compatibility.

## Consequences

Both workflows share one in-memory adapter test seam and can evolve without
duplicating CryoSPARC Dataset mechanics. The adapter deliberately supports
the current two workflows and CryoSPARC 5.0.6 rather than speculating about a
general job framework or other server versions.
