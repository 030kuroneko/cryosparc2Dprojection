# cryosparc2Dprojection

CryoSPARC tool to map 2D class averages to 3D viewing directions using NU refinement particle poses.

## Goal

Input:
- CryoSPARC Select 2D output
- CryoSPARC NU refinement particles + volume

Output:
- class -> viewing direction
- 3D projection from the corresponding camera
- camera position / direction visualization

## Development

This project follows TDD.

Initial milestone:
- Parse CryoSPARC particle metadata
- Match 2D class assignment with NU refinement poses
- Calculate class viewing vectors
