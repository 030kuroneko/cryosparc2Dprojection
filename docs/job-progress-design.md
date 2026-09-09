# Human-readable job progress

Status: confirmed by the user and implemented; live CryoSPARC validation remains pending.

## Confirmed requirements

- Cover Class Orientation and Symmetry-Axis Search.
- Show consistent human-readable progress in the web task page and CryoSPARC Job Log.
- The default summary contains the current stage, completed/total work when known, elapsed time, and estimated remaining time.
- Keep technical diagnostics available behind an expandable details view. Keep warnings and errors visible.
- Show “Estimating” until sufficient timing evidence exists; then show an approximate time range and explicitly identify whether it describes the current stage or the whole task.
- Use English for user-facing progress messages.
- Prefer remaining time through completion of all result uploads. Exclude queue waiting time. When whole-task timing cannot be estimated, show an explicitly labelled current-stage estimate instead.
- During operations without measurable progress, show the operation, its elapsed time, and the last progress update time. Advancing elapsed time does not imply advancing work; do not invent percentages or infer a stalled job from silence alone.
- Keep warnings, errors, and low-confidence scientific results visible; hiding technical progress details does not hide scientific findings.

## Behavior before this change

- Class Orientation lacks progress during input loading, pose analysis, and camera solving.
- Axis Search emits internal stage/family/class/pass counters. Its existing ETA covers one class's current search pass, not the entire task.
- The web Activity Log displays terminal output; it does not currently provide structured progress or historical runtime estimates.

## Presentation example

Illustrative values only:

```text
Finding class orientations · 12 / 40 classes completed
Elapsed: 8 min · Estimated stage time remaining: 15–20 min
```

For an unmeasured operation:

```text
Uploading results · Elapsed in this stage: 2 min
Estimated time remaining: Estimating…
Last progress update: 1 min ago
```

## Implementation follow-through

Use test-driven development. Verify both workflows, estimate scope and unknown estimates, progress during long operations, default diagnostic hiding with accessible details, visible warnings/errors, and completion only after required publication finishes. Validate the available CryoSPARC presentation mechanism for technical details before implementing it; do not assume its Job Log supports the web page's native expand/collapse controls.

## Implemented presentation and estimation

- The web page shows a persistent stage summary and a stage progress bar; technical details are collapsed. CryoSPARC updates one named progress event and keeps diagnostics in `job-details.log`, identified in its Event Log. The SDK has no native disclosure-control API.
- Thirty-second worker heartbeats update elapsed time during long operations. The browser updates displayed elapsed time each second, without increasing completed work. New search events refresh the last-work timestamp even before a whole class finishes.
- ETA uses completed units from the current stage. At least two timing samples are required; the last eight samples produce a heuristic range from 0.75 times the fastest sample to 1.5 times the slowest. This is not a statistical confidence interval. Estimates are withdrawn when an in-flight unit exceeds twice the slowest observed sample.
- There is no historical whole-task timing model, so this implementation explicitly uses stage estimates. Unmeasured reading, surface construction, and upload stages show `Estimating…`; their unknown cost is never silently excluded from a claimed whole-task ETA.
- Publication failure reports failure, and failed/interrupted/unavailable worker states suppress stale estimates. Reporting errors do not replace scientific results or trigger lower-quality retries.
- Progress callbacks run on one background sender outside the state lock. Its queue keeps at most 64 recent summaries. Shutdown waits at most one second for delivery, then retains only the newest pending terminal summary; a stalled remote logger cannot stop local progress snapshots or scientific computation. Remote delivery remains best effort if the process exits before the server recovers.
- Web log truncation preserves complete lines: an incomplete first line is discarded before diagnostic classification, while a complete line at the 64 KiB boundary is retained.
