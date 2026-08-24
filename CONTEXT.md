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
A simulated 2D image generated from the refined map at a Class Camera Orientation and aligned to the corresponding 2D class average.
_Avoid_: Camera view, 3D view

**CryoSPARC Display Orientation**:
The vertical image orientation shown by the CryoSPARC UI for a 2D class average. Static Class Result images use this orientation as their display reference without redefining the underlying Class Camera Orientation.
_Avoid_: Raw array orientation, arbitrary display flip

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
The one-based class identifier shown in the CryoSPARC UI and exposed to users for filtering and filenames.
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
