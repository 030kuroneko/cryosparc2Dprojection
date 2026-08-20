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

**Camera View Render**:
A 3D surface rendering of the refined map observed at a Class Camera Orientation.
_Avoid_: Projection

**Representative Symmetry View**:
A deterministic member of the symmetry-equivalent Class Camera Orientations. It represents the class but is not a unique physical orientation.
_Avoid_: Unique orientation, true orientation

**Class Result**:
The 2D class average, Matched Projection, Camera View Render, orientation metadata, match score, and uncertainty for one selected class.
_Avoid_: Projection only

**Match Confidence**:
Evidence that a Class Camera Orientation is reliable, based on projection similarity, separation from competing orientations, and orientation spread.
_Avoid_: Correlation alone, certainty

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

**Camera Metadata**:
The rotation matrix, quaternion, View Direction, in-plane rotation, symmetry information, and coordinate-convention declaration that reproduce a Class Camera Orientation.
_Avoid_: Euler angles alone

**Orientation Group**:
A Class Camera Orientation together with every camera related to it by the map's declared symmetry. Members of one group are not competing orientation answers.
_Avoid_: Individual symmetry mate

**Symmetry Axis Assignment**:
The nearest 2-fold, 3-fold, or 5-fold axis and its angular distance from a View Direction. A class is named as an axis view only when that distance is within the configured threshold.
_Avoid_: Forced axis class

**General View**:
A View Direction that is outside the configured angular threshold for every named symmetry axis.
_Avoid_: Other axis

**Projection Shift**:
The two-dimensional translation that aligns a Matched Projection to its class average. It is reported separately and does not change the Class Camera Orientation.
_Avoid_: Camera translation, camera rotation

**Interactive Class Volume**:
A rotated copy of the Rendering Map whose voxel grid makes the CryoSPARC volume viewer's default view correspond to one requested Class Camera Orientation.
_Avoid_: Camera metadata, static camera render
