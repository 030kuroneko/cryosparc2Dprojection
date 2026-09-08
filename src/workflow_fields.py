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
HINTS = {
    'classes': 'Optional: 3,8,12. All selected classes are matched; only these get rotated volumes.',
    'axis_family': 'Blank = all axes of the selected symmetry. Examples: O: 4fold; D7: 7fold; C3: 3fold,3fold-2. Suffixes distinguish inequivalent directions.',
    'axis_roll': 'Optional: 2fold=90;3fold=30. Presentation only.',
    'render_grid_size': 'Blank = complete native Rendering Map grid. Lower explicitly to reduce memory.',
    'surface_level': 'Blank = automatic. Affects the Camera View Render only.',
    'url': 'Use the same URL and saved CryoSPARC Tools login as your command-line workflow.',
}
