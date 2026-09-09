"""Workflow field labels and grouping for the web launcher."""

TITLES = {'orientation': 'Class Orientation', 'axis': 'Axis Search'}
DESCRIPTIONS = {
    'orientation': 'Match selected 2D classes to a refined map using particle poses.\n'
                   'Results: Class Average · Matched Projection · Camera View Render',
    'axis': 'Rank class averages against axes of the selected point-group symmetry.\n'
            'No particle-pose overlap required. Near-Axis Refinement is optional.',
}
BASIC = {
    'orientation': {'select_job', 'refinement_job', 'symmetry', 'classes'},
    'axis': {'select_job', 'volume_job', 'symmetry', 'axis_family', 'top_n', 'refine_near_axis'},
}
LABELS = {
    'url': 'CryoSPARC URL', 'project': 'Project UID', 'workspace': 'Workspace UID',
    'select_job': 'Select 2D job UID', 'refinement_job': 'Refinement job UID',
    'volume_job': 'Volume job UID', 'classes': 'Interactive class numbers',
    'axis_family': 'Axis families', 'top_n': 'Top classes per axis',
    'refine_near_axis': 'Enable Near-Axis Refinement', 'axis_roll': 'Display rolls (family=degrees; …)',
}

# Each field has a brief inline hint and optional expanded help.
FIELD_HELP = {
    'project': ('Enter the CryoSPARC project ID, e.g. P12.', 'Use the project containing your source jobs. All source job IDs are resolved within this project.'),
    'workspace': ('Enter the destination workspace ID, e.g. W3.', 'The new External Job and its results will be created in this workspace within the selected project.'),
    'select_job': ('Enter your Select 2D Classes job ID, e.g. J123.', 'Use the job containing the selected 2D classes to analyze. Copy its J-number from CryoSPARC; enter only the ID, not the job title or URL.'),
    'refinement_job': ('Enter your NU or Local Refinement job ID, e.g. J124.', 'Use the refinement job providing particle poses and the reference volume. Its particles must overlap with the selected 2D particles so their orientations can be matched. Enter only the J-number.'),
    'volume_job': ('Enter the job providing your reference volume, e.g. J125.', 'Use a job with the volume to compare against your selected classes. Axis Search compares images and does not require overlapping particle IDs.'),
    'classes': ('Optional: enter class numbers such as 3,8,12.', 'Use one-based class numbers shown in the Select 2D Classes job. All selected classes are matched; this list controls which also receive rotated volumes for interactive inspection. Leave blank to skip these additional volumes.'),
    'axis_family': ('Leave blank to search all axes of the selected symmetry.', 'Enter comma-separated families, such as 2fold,3fold for I or 4fold for O. Some symmetries have distinct directions with suffixes, such as 3fold and 3fold-2 for C3.'),
    'top_n': ('Keep 5 to report the top five matches for each axis.', 'Enter a positive integer. Increasing this value includes more candidates and may increase rendering and Near-Axis Refinement work.'),
    'refine_near_axis': ('Optionally refine matches around each exact symmetry axis.', 'After Exact-Axis Search, refine selected candidates within the configured angular cone to explore views slightly tilted away from the exact axis.'),
    'render_map': ('Use map, or select sharpened for surface rendering.', 'Selects the density map used to draw the 3D surface. It does not change matching or scoring. When entering a Surface level, use a threshold from this same map.'),
    'render_grid_size': ('Leave blank for full detail. Try 256 or 128 if memory is limited.', 'Sets the maximum grid side length in voxels for extracting the surface. For a 512 × 512 × 512 map, entering 256 uses a 256 × 256 × 256 grid. Smaller values reduce memory use but can remove surface detail. The grid never exceeds the original map size. This does not set the output image size. Minimum: 2.'),
    'render_size': ('Leave blank for automatic sizing, or enter 1024 for a 1024 × 1024 image.', 'Sets the square Camera View Render PNG dimensions in pixels. Automatic sizing uses the larger of 1024 pixels or three times the Comparison DPI. Larger images may increase rendering time and file size, but do not improve map resolution. Minimum: 64 pixels.'),
    'surface_level': ('Leave blank for automatic selection, or copy the map threshold from CryoSPARC.', 'In CryoSPARC, open the source volume job → Volumes → select the same map used under Rendering map. Adjust the viewer’s Threshold until the desired surface is visible, then copy that numeric value here. Enter a raw density value, not a sigma multiplier. Higher values retain stronger density; lower values include more weak density. Thresholds are map-specific: there is no universal recommended number. This affects the Camera View Render only.'),
    'render_background': ('Choose a dark or light background for the 3D render.', ''),
    'comparison_dpi': ('Keep 100 for previews; use 300 for larger exported images.', 'Controls the pixel dimensions of the three-column comparison images. Higher DPI produces larger images and files. It also increases the automatic Camera View Render size when needed.'),
    'preview_page_size': ('Show 10 classes per CryoSPARC preview page by default.', 'Enter a positive integer. Use fewer classes for shorter pages, or more to reduce the number of pages. This does not change how many classes are analyzed.'),
    'auto_crop_2d': ('Crop the 2D panels to match the 3D camera framing.', 'Adjusts the displayed framing of Class Average and Matched Projection panels in comparison previews. It affects presentation only. If a suitable crop cannot be determined, the full frame is retained.'),
    'axis_roll': ('Optional: enter 2fold=90;3fold=30 to rotate displayed results.', 'Set a display rotation for each named axis family in degrees. Separate entries with semicolons. This changes presentation only, not search scores or rankings. Leave blank for the default orientation.'),
    'low_resolution_A': ('Keep 80 Å as the coarse end of the search scoring band.', 'Sets the low-resolution edge of the frequency band used to score and rank matches. Must be greater than the high-resolution limit.'),
    'high_resolution_A': ('Keep 15 Å as the fine end of the search scoring band.', 'Must be positive and smaller than the low-resolution limit. A smaller value includes finer image detail and may change rankings.'),
    'mask_radius_fraction': ('Keep 0.45 for a mask radius of 45% of the image width.', 'Sets the central image region receiving full scoring weight. The radius plus its soft edge must not exceed 0.50 of the image width.'),
    'mask_edge_fraction': ('Keep 0.05 for a soft edge spanning 5% of the image width.', 'Controls the fade from full scoring weight to zero outside the central mask. Use 0 for a hard edge. The radius plus edge must not exceed 0.50.'),
    'roll_coarse_step': ('Keep 5° for the initial rotation search.', 'Sets the angular spacing for testing rotations within the image plane. Smaller steps sample more rotations and increase search time. Must be positive.'),
    'roll_refine_step': ('Keep 0.5° to refine the best in-plane rotation.', 'Sets the angular spacing for local rotation refinement. Smaller steps provide finer sampling but increase computation. Must be positive.'),
    'shift_bound_fraction': ('Keep 0.10 to allow shifts up to 10% of the image width.', 'Limits horizontal and vertical translation as a fraction of matching image width. Use 0 to disable translation search. Allowed range: 0 to 0.50.'),
    'mirror_warning_margin': ('Keep 0.05 to flag substantially better mirrored matches.', 'Warns when the mirrored diagnostic score exceeds the normal score by at least this amount. This is a score difference, not a percentage, and does not automatically mirror the result.'),
    'axis_cone_degrees': ('Keep 15° as the maximum tilt away from the exact axis.', 'Used only when Near-Axis Refinement is enabled. A larger cone explores more viewing directions and may increase computation. Must be greater than 0° and no more than 90°.'),
    'tilt_coarse_step': ('Keep 3° for the initial Near-Axis tilt search.', 'Sets the angular sampling interval inside the cone. Smaller positive steps test more directions and increase computation. Used only when Near-Axis Refinement is enabled.'),
    'tilt_refine_step': ('Keep 0.5° for local Near-Axis tilt refinement.', 'Sets the angular sampling interval around promising tilt directions. Smaller positive steps provide finer sampling but increase computation. Used only when Near-Axis Refinement is enabled.'),
    'diagnostic_low_resolution_A': ('Keep 80 Å as the coarse end of the diagnostic scoring band.', 'Together with the high-resolution limit, sets the frequency band for comparing a class average with its matched projection. Must be greater than the high-resolution limit. This diagnostic score does not select the particle-pose-derived orientation.'),
    'diagnostic_high_resolution_A': ('Keep 15 Å as the fine end of the diagnostic scoring band.', 'Must be positive and smaller than the low-resolution limit. A smaller value includes finer detail in diagnostic scoring.'),
    'diagnostic_mask_radius_fraction': ('Keep 0.45 for a mask radius of 45% of the image width.', 'Sets the central region receiving full weight during diagnostic scoring. The radius plus its soft edge must not exceed 0.50 of the image width.'),
    'diagnostic_mask_edge_fraction': ('Keep 0.05 for a soft edge spanning 5% of the image width.', 'Controls how gradually the diagnostic scoring mask fades from full weight to zero. Use 0 for a hard edge. The radius plus edge must not exceed 0.50.'),
}
SYMMETRY_HELP = {
    'orientation': ('Match the refinement symmetry: Cn, Dn, T, O, or I (e.g. C1 or D7).', 'Supported values are Cn and Dn with a positive integer n, plus T, O, and I. Use the symmetry of the source refinement.'),
    'axis': ('Match the reference map symmetry, e.g. I, O, or D7.', 'Supported values are Cn with n ≥ 2, Dn, T, O, and I. C1 has no symmetry axis to search.'),
}
PLACEHOLDERS = {'project': 'e.g. P12', 'workspace': 'e.g. W3',
                'select_job': 'e.g. J123', 'refinement_job': 'e.g. J124',
                'volume_job': 'e.g. J125', 'classes': 'e.g. 3,8,12',
                'render_size': 'Automatic', 'render_grid_size': 'Native grid',
                'surface_level': 'Automatic'}
HELP_SOURCES = {'surface_level': 'https://guide.cryosparc.com/application-guide-v4.0%2B/inspecting-job-data'}

LABELS.update({
    'render_map': 'Rendering map', 'render_background': 'Render background',
    'comparison_dpi': 'Comparison DPI', 'auto_crop_2d': 'Auto-crop 2D panels',
    'diagnostic_low_resolution_A': 'Diagnostic low-resolution limit (Å)',
    'diagnostic_high_resolution_A': 'Diagnostic high-resolution limit (Å)',
})
