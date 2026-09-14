/* User actions for the selected Web Job. */
(() => {
  function create({api, confirm, render, refresh}) {
    let job = null, busy = false, error = '', generation = 0;
    const terminal = state => ['completed', 'failed', 'interrupted'].includes(state);
    function draw() {
      render({visible: !!job, busy, error, canStop: !!job && !terminal(job.state),
        stopLabel: job?.state === 'stopping' ? 'Retry stop' : 'Stop job'});
    }
    async function act(remove) {
      if (!job || busy || (!remove && terminal(job.state))) return;
      const selected = job, version = generation;
      if (remove && !confirm(`${terminal(job.state) ? '' : 'This will immediately kill the computation.\n\n'}Remove this job from the interface? Local result files and CryoSPARC jobs/results are preserved. Result viewing and class selection for this job will no longer be available here.`)) return;
      busy = true; error = ''; draw();
      try {
        await api('/api/jobs/' + selected.id + (remove ? '' : '/stop'), remove ? 'DELETE' : 'POST');
        if (version === generation) await refresh();
      } catch (failure) {
        if (version === generation) error = failure.message;
      } finally {
        if (version === generation) {busy = false; draw();}
      }
    }
    return {
      setJob(value) {
        if (value?.id !== job?.id) {generation++; busy = false; error = '';}
        job = value; draw();
      },
      stop: () => act(false), remove: () => act(true),
    };
  }
  window.JobControls = {create};
})();
