/* Text-only progress presentation. Time passing never increments completed work. */
(() => {
  function duration(value) {
    const seconds = Math.max(0, Math.ceil(Number(value) || 0));
    if (seconds < 60) return `${seconds} sec`;
    if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
    return `${Math.floor(seconds / 3600)} hr ${Math.ceil((seconds % 3600) / 60)} min`;
  }

  function describe(progress, jobState, now = Date.now() / 1000) {
    const terminal = {completed: "Completed", failed: "Failed", interrupted: "Interrupted", unknown: "Status unavailable"};
    const status = terminal[jobState] ? jobState : progress?.state || jobState;
    if (terminal[status]) {
      const context = status === "completed" ? "All required results uploaded." :
        `Last operation: ${progress?.stage || "Not reported"}. Check the activity log.`;
      return {text: `${terminal[status]} · ${context}`, value: null, max: null};
    }
    if (!progress) {
      return {text: jobState === "queued" || jobState === "pending" ?
        "Waiting to start · Queue wait is not included in runtime estimates." :
        "Waiting for worker progress…", value: null, max: null};
    }
    const age = Math.max(0, now - (progress.updated_at || now));
    const count = progress.total != null ? ` · ${progress.completed}/${progress.total} ${progress.unit} completed` : "";
    let text = `${progress.stage}${count}\nElapsed: ${duration(progress.elapsed_seconds + age)} · Time in stage: ${duration(progress.stage_elapsed_seconds + age)}`;
    // Preserve the measured range between reports; do not fabricate a countdown.
    if (progress.remaining_seconds && age < 90) {
      const label = progress.remaining_scope === "task" ? "task" : "stage";
      text += `\nEstimated ${label} time remaining: ${duration(progress.remaining_seconds[0])}–${duration(progress.remaining_seconds[1])}`;
    } else {
      text += "\nEstimated time remaining: Estimating…";
    }
    text += ` · Last progress update: ${duration(progress.last_progress_seconds + age)} ago`;
    if (age >= 90) text += "\nWaiting for a fresh worker update.";
    if (progress.detail) text += `\n${progress.detail}`;
    return {text, value: progress.total > 0 ? progress.completed : null, max: progress.total};
  }
  window.JobProgressView = {describe};
})();
