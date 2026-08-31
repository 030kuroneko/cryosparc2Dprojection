# 2D Class to 3D Camera Orientation

This context describes how a CryoSPARC 2D class is related to a reproducible view of a refined 3D density map.

## Language

**Class Camera Orientation**:
The complete 3D camera rotation whose viewing direction and in-plane rotation make the map projection correspond to a 2D class average.
_Avoid_: Viewing angle, view vector, camera direction

**View Direction**:
The axis from which the map is observed, without an in-plane rotation. A view direction alone is not a Class Camera Orientation.
_Avoid_: Camera orientation

**Matched Projection**:
A simulated 2D image generated from the Matching Map at a Class Camera Orientation, resampled to the native Class Average box and pixel size, and aligned on that native grid for presentation and export. Its displayed resolution is independent of comparison DPI.
_Avoid_: Camera view, 3D view

**Search Projection**:
A bounded-resolution projection generated from the Matching Map while selecting and scoring a Class Camera Orientation. It may be downsampled for search performance and is distinct from the native-grid Matched Projection shown in a Class Result.
_Avoid_: Matched Projection, presentation projection, Camera View Render

**CryoSPARC Display Orientation**:
The vertical image orientation shown by the CryoSPARC UI for a 2D class average. Static Class Result images use this orientation as their display reference without redefining the underlying Class Camera Orientation.
_Avoid_: Raw array orientation, arbitrary display flip

**CryoSPARC Output Thumbnail**:
A presentation-only image attached to a registered CryoSPARC output card. The Matched Projection output uses the lowest Class Number at its native pixel dimensions, in CryoSPARC Display Orientation, without labels or axes. Thumbnail generation and upload do not alter scientific output data, and an upload failure is reported without failing the job.
_Avoid_: Matched Projection data, Event Log preview

**Axis Search Dashboard Preview**:
A presentation-only summary attached to both the dedicated axis-search preview output card and the job tile, whose underlying dataset contains the aligned classes in displayed order. It shows the first preview page with each result row labelled outside its scientific image panels by Axis Family, family-local rank, Class Number, four-decimal Axis Class Score, and optional four-decimal Near-Axis score plus three-decimal Near-Axis angular distance in degrees. Results are grouped 2fold, 3fold, then 5fold and ranked within each family; the same class remains visible in every matching family. Upload failure is warned without failing completed scientific outputs.
_Avoid_: CryoSPARC Output Thumbnail, scientific output data, complete Event Log preview

**Camera View Render**:
An orthographic 3D isosurface rendering of the Rendering Map observed at the exact Class Camera Orientation.
_Avoid_: Projection, voxel scatter, 3D projection

**Representative Symmetry View**:
A deterministic member of the symmetry-equivalent Class Camera Orientations. It represents the class but is not a unique physical orientation.
_Avoid_: Unique orientation, true orientation

**Supported Symmetry Convention**:
The symmetry coordinate convention whose pose folding and Class Camera Orientation behavior have been explicitly accepted for this tool. Version 0.1 supports no symmetry (`C1`) and CryoSPARC's documented icosahedral convention (`I`).
_Avoid_: Treating `I1`, `I2`, or another unverified convention as an alias

**Class Result**:
The 2D class average, Matched Projection, Camera View Render, orientation metadata, match score, and uncertainty for one selected class.
_Avoid_: Projection only

**Class Result Presentation Resolution**:
The raster dimensions and display scale of a static three-column Class Result. Changing it does not alter the Class Average, Matched Projection, Class Camera Orientation, matching scores, or Camera View Render geometry.
_Avoid_: Matching resolution, map resolution, projection calculation grid

**Match Confidence**:
Evidence that a Class Camera Orientation is reliable, based on projection similarity, separation from competing orientations, and orientation spread.
_Avoid_: Correlation alone, certainty

**Diagnostic Band-Limited Score**:
A softly masked, physically band-limited similarity score calculated only for the Class Camera Orientation selected by the existing search. It diagnoses whether that selected match remains similar within the declared frequency band and spatial region; it does not rank orientations, establish a global margin, or represent a probability.
_Avoid_: Match Confidence, ranking score, probability

**Low-Confidence Class Result**:
A visible Class Result whose Match Confidence does not meet the reporting threshold. It is flagged rather than hidden or treated as a failed job.
_Avoid_: Rejected class, failed class

**Class Number**:
The one-based source class identifier shown in the CryoSPARC UI and preserved when a selected subset is reordered. It is not the selected input row position.
_Avoid_: Zero-based class ID

**Handedness Warning**:
A warning that a mirrored projection fits substantially better than every non-mirrored candidate. Mirroring is diagnostic evidence and is not silently applied to a Class Result.
_Avoid_: Automatic mirror correction

**Matching Map**:
The unsharpened refined density map used to determine and score a Class Camera Orientation.
_Avoid_: Rendering map, sharpened map

**Rendering Map**:
The density map used only for the Camera View Render. It may be sharpened, but changing it must not change the Class Camera Orientation.
_Avoid_: Matching map

**Surface Level**:
The raw density contour value whose isosurface defines the visible boundary of a Camera View Render. An automatically selected value may be explicitly overridden without changing the Class Camera Orientation.
_Avoid_: Match threshold, confidence threshold

**Surface Sampling Grid Size**:
The maximum side length used when sampling the Rendering Map before extracting its isosurface. It controls presentation geometry detail and rendering cost without changing the Class Camera Orientation, Matched Projection, or matching scores. It never upsamples beyond the original Rendering Map grid.
_Avoid_: Render image size, comparison DPI, map resolution

**Camera Metadata**:
The rotation matrix, quaternion, View Direction, in-plane rotation, symmetry information, and coordinate-convention declaration that reproduce a Class Camera Orientation.
_Avoid_: Euler angles alone

**Orientation Group**:
A Class Camera Orientation together with every camera related to it by the map's declared symmetry. Members of one group are not competing orientation answers.
_Avoid_: Individual symmetry mate

**Projection Shift**:
The two-dimensional translation that aligns a Matched Projection to its class average. It is reported separately and does not change the Class Camera Orientation.
_Avoid_: Camera translation, camera rotation

**Interactive Class Volume**:
A rotated copy of the Rendering Map whose voxel grid makes the CryoSPARC volume viewer's default view correspond to one requested Class Camera Orientation.
_Avoid_: Camera metadata, static camera render

**Axis Search Run**:
The complete CryoSPARC workflow that loads selected class averages and a map, performs Exact-Axis Ranking, optionally performs Near-Axis Refinement, and produces rendered and machine-readable results.
_Avoid_: Exact-Axis Ranking, axis-search algorithm

**Axis Family**:
A family of candidate directions that shares one symmetry-axis order and canonical camera convention, such as 2fold, 3fold, or 5fold.
_Avoid_: Class, search result, individual candidate

**Axis Class Score**:
The maximum band-limited, soft-masked normalized cross-correlation across the permitted in-plane rotations and XY shifts on bounded Search Projections. It ranks classes within one Axis Family and is neither a probability nor a native-grid diagnostic score.
_Avoid_: Match Confidence, probability, Diagnostic Band-Limited Score

**Exact-Axis Ranking**:
The stage that scores selected class averages against the canonical orientation of each requested Axis Family and ranks the matching classes.
_Avoid_: Axis Search Run, Near-Axis Refinement

**Near-Axis Refinement**:
The stage that searches a bounded orientation cone around an Exact-Axis Ranking candidate without changing its Axis Family.
_Avoid_: Exact-Axis Ranking, global orientation search

**Result Rendering**:
The stage that turns all completed axis-search candidates from one Axis Search Run into one result set containing native-grid Matched Projections, Camera View Renders, template stacks, previews, and machine-readable results. The result set is complete only when every candidate appears consistently in all required outputs; Result Rendering never completes a partial result set. It completes before those results are published to CryoSPARC; failure to attach a presentation preview does not change completed scientific results.
_Avoid_: Axis Search Run, orientation search
