# Limit Interactive Class Volumes to requested classes

_Oblique-render portions superseded by ADR 0005._

Register one original Rendering Map for CryoSPARC's integrated volume viewer and generate static Camera View Renders and ChimeraX cameras for every processed class. Create rotated Interactive Class Volumes only for classes explicitly requested by Class Number, because CryoSPARC's public External Job API does not document a way to preset the integrated viewer camera and duplicating a full map for every class would multiply storage use.
