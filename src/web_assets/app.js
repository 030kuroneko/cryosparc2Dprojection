"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  csrf: "",
  schema: null,
  workflow: "orientation",
  pages: {},
  selected: null,
  jobs: [],
  requestId: null,
  submitting: false,
  generation: 0,
};
const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const selectionView = window.ClassSelectionView.create({api, render:renderSelection});
const jobControls = window.JobControls.create({api, confirm: message => window.confirm(message), refresh,
  render(view) {
    $("job-controls").hidden = !view.visible;
    $("stop-job").disabled = view.busy || !view.canStop;
    $("stop-job").textContent = view.stopLabel;
    $("delete-job").disabled = view.busy;
    $("job-control-status").textContent = view.error || (view.busy ? "Sending request…" : "");
  }
});
$("stop-job").addEventListener("click", () => jobControls.stop());
$("delete-job").addEventListener("click", () => jobControls.remove());
function renderSelection(view) {
  const panel = $("class-selection");
  panel.hidden = view.hidden || !view.job || view.job.workflow !== "orientation" || view.job.state !== "completed";
  const enabled = !!view.available;
  for (const id of ["selection-all","selection-clear","selection-invert","selection-sort"]) $(id).disabled = !enabled;
  $("selection-export").disabled = !view.canExport;
  $("selection-export").textContent = view.exporting ? "Submitting…" : view.retryRequest ? "Retry export request" : "Export selection";
  $("selection-count").textContent = enabled ? `${view.selected_class_numbers.length} / ${view.classes.length} selected` : "";
  $("selection-status").textContent = view.error || (view.saving ? "Saving selections…" : enabled ? "Selections saved automatically. Each export creates a new CryoSPARC job." : view.reason || "Loading results…");
  $("selection-sort").value = view.order || "class_number";
  const selected = new Set(view.selected_class_numbers || []);
  const markup = (view.classes || []).map(c => `<label class="selection-card ${selected.has(c.class_number) ? "is-selected" : ""}">
    <span class="selection-card-title"><input type="checkbox" data-class="${c.class_number}" ${selected.has(c.class_number) ? "checked" : ""}> Class ${c.class_number}
    <span class="hint">${Number(c.particle_count).toLocaleString()} particles · Score ${c.score == null ? "unavailable" : Number(c.score).toFixed(4)}</span></span>
    <img loading="lazy" src="/api/jobs/${encodeURIComponent(view.job.id)}/selection/images/${encodeURIComponent(c.image)}" alt="Class ${c.class_number}: original class average, matched projection and camera view">
    <span class="hint">${c.orientation_method === "image_global_search" ? "Image-only fallback" : "Particle poses"}${c.confidence === "low" ? " · Low confidence" : ""}</span></label>`).join("");
  // Keep focus on a checkbox while an asynchronous save changes only its status.
  if ($("selection-cards").dataset.markup !== markup) {
    const focused = document.activeElement?.dataset.class;
    $("selection-cards").innerHTML = markup;
    $("selection-cards").dataset.markup = markup;
    if (focused) $("selection-cards").querySelector(`[data-class="${Number(focused)}"]`)?.focus();
  }
  $("selection-exports").innerHTML = (view.exports || []).map(e=>`<p class="selection-export-row"><strong>${escape(e.job_uid || "Selection export")}</strong> · ${escape(e.state)}
    ${e.detail ? `<span class="hint">${escape(e.detail)}</span>` : ""}
    ${["failed","unknown"].includes(e.state) ? `<button type="button" data-retry-export="${escape(e.id)}">Check / retry</button>` : ""}</p>`).join("");
}
$("selection-cards").addEventListener("change", e=>{if(e.target.dataset.class) selectionView.toggle(Number(e.target.dataset.class));});
$("selection-all").addEventListener("click",()=>selectionView.selectAll());
$("selection-clear").addEventListener("click",()=>selectionView.clear());
$("selection-invert").addEventListener("click",()=>selectionView.invert());
$("selection-sort").addEventListener("change",e=>selectionView.sort(e.target.value));
$("selection-reload").addEventListener("click",()=>selectionView.reload());
$("selection-export").addEventListener("click",()=>selectionView.exportSelection());
$("selection-exports").addEventListener("click", e=>{if(e.target.dataset.retryExport) selectionView.retry(e.target.dataset.retryExport);});
async function api(path, method = "GET", body) {
  const response = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": state.csrf },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const data = await response
    .json()
    .catch(() => ({ error: "The server returned an unexpected response." }));
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/login") showLogin();
    const error = new Error(data.error || "Request failed.");
    error.status = response.status;
    throw error;
  }
  return data;
}
function showLogin() {
  selectionView.setJob(null);
  state.generation++;
  document.dispatchEvent(new Event("launcher-session-reset"));
  $("workspace").hidden = true;
  $("login-screen").hidden = false;
  state.schema = null;
  state.pages = {};
  state.selected = null;
  jobControls.setJob(null);
  state.jobs = [];
  state.requestId = null;
  $("password").value = "";
  $("admin-key").value = "";
  $("slurm-fields").disabled = true;
  $("slurm-fields").hidden = true;
  $("slurm-admin").open = false;
  $("slurm-status").textContent = "";
  document.querySelectorAll("#slurm-fields input").forEach((input) => {
    input.value = "";
    input.checked = false;
  });
  $("job-list").replaceChildren();
  $("job-log").textContent = "";
  state.progress = null;
  $("progress-panel").hidden = true;
  $("job-progress").textContent = "";
  $("job-details").textContent = "";
  $("technical-details").open = false;
  $("job-cleanup-warning").textContent = "";
  $("job-cleanup-warning").hidden = true;
}
async function bootstrap() {
  try {
    const session = await api("/api/session");
    state.csrf = session.csrf;
    state.url = session.cryosparc_url;
    $("login-server").textContent = state.url;
    if (session.email) await enter(session.email);
  } catch (error) {
    $("login-error").textContent = error.message;
  }
}
async function enter(email) {
  state.schema = await api("/api/schema");
  for (const [name, workflow] of Object.entries(state.schema.workflows)) {
    state.pages[name] = Object.fromEntries(
      workflow.fields.map((f) => [f.key, f.default]),
    );
  }
  $("signed-in-email").textContent = email;
  $("open-cryosparc").href = state.url;
  $("server-url").textContent = state.url;
  $("profile").innerHTML = state.schema.profiles
    .map((p) => `<option value="${escape(p.id)}">${escape(p.label)}</option>`)
    .join("");
  $("login-screen").hidden = true;
  $("workspace").hidden = false;
  renderForm();
  profileNote();
  await refresh();
}
$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  $("login-error").textContent = "";
  try {
    const result = await api("/api/login", "POST", {
      email: $("email").value,
      password: $("password").value,
    });
    state.csrf = result.csrf;
    $("password").value = "";
    await enter(result.email);
  } catch (error) {
    $("login-error").textContent = error.message;
    $("password").value = "";
  } finally {
    button.disabled = false;
  }
});
$("logout").addEventListener("click", async () => {
  try {
    const result = await api("/api/logout", "POST");
    state.csrf = result.csrf;
    showLogin();
  } catch (error) {
    $("form-error").textContent = error.message;
  }
});
function fieldHint(f, id) {
  const help = f.help ? `<span class="field-help"><button type="button" class="help-button" aria-label="Help for ${escape(f.label)}" aria-expanded="false" aria-controls="${id}-help">?</button><span class="help-panel" id="${id}-help" hidden>${escape(f.help)}${f.help_url ? ` <a href="${escape(f.help_url)}" target="_blank" rel="noopener noreferrer">CryoSPARC guide ↗</a>` : ""}</span></span>` : "";
  return `<div class="hint-row"><small class="hint" id="${id}-hint">${escape(f.hint)}</small>${help}</div>`;
}
function field(f) {
  const value = state.pages[state.workflow][f.key], id = "field-" + f.key;
  const description = `aria-describedby="${id}-hint"`;
  if (f.type === "boolean")
    return `<div class="field"><div class="check-row"><input id="${id}" data-key="${f.key}" type="checkbox" ${description} ${value ? "checked" : ""}><label for="${id}">${escape(f.label)}</label></div>${fieldHint(f, id)}</div>`;
  let input;
  if (f.choices.length)
    input = `<select id="${id}" data-key="${f.key}" ${description}>${f.choices.map((c) => `<option ${String(c) === value ? "selected" : ""}>${escape(c)}</option>`).join("")}</select>`;
  else
    input = `<input id="${id}" data-key="${f.key}" value="${escape(value)}" placeholder="${escape(f.placeholder || "")}" ${description} ${f.required ? "required" : ""} maxlength="2048" autocomplete="off" ${["project", "workspace", "select_job", "refinement_job", "volume_job"].includes(f.key) ? `pattern="${f.key === "project" ? "P" : f.key === "workspace" ? "W" : "J"}[1-9][0-9]*"` : ""}>`;
  return `<div class="field"><label for="${id}">${escape(f.label)}${f.required ? " *" : ""}</label>${input}${fieldHint(f, id)}<span class="field-error" id="error-${f.key}"></span></div>`;
}
function renderForm() {
  const workflow = state.schema.workflows[state.workflow];
  $("workflow-title").textContent = workflow.title;
  $("workflow-description").textContent = workflow.description;
  document.querySelectorAll("[data-workflow]").forEach((b) => {
    b.classList.toggle("active", b.dataset.workflow === state.workflow);
    b.setAttribute(
      "aria-current",
      b.dataset.workflow === state.workflow ? "page" : "false",
    );
  });
  $("connection-fields").innerHTML = workflow.fields
    .filter((f) => f.group === "connection")
    .map(field)
    .join("");
  $("basic-fields").innerHTML = workflow.fields
    .filter((f) => f.group === "basic")
    .map(field)
    .join("");
  $("advanced-fields").innerHTML = ["search", "rendering"]
    .map(
      (group) =>
        `<div class="t-acc" data-open="false"><button type="button" class="t-acc-head" aria-expanded="false" aria-controls="settings-${group}">${{ search: "Search settings", rendering: "Rendering & presentation" }[group]}<span class="t-acc-chevron" aria-hidden="true"><svg viewBox="0 0 16 16" width="16" height="16"><path d="M4 6.5L8 10.5L12 6.5" fill="none" stroke="currentColor"/></svg></span></button><div class="t-acc-panel" id="settings-${group}" inert><div class="t-acc-panel-inner"><div class="fields">${workflow.fields
          .filter((f) => f.group === group)
          .map(field)
          .join("")}</div></div></div></div>`,
    )
    .join("");
  $("form-error").textContent = "";
  conditional();
}
$("advanced-fields").addEventListener("click", (event) => {
  const head = event.target.closest(".t-acc-head");
  if (!head) return;
  const acc = head.closest(".t-acc");
  const open = acc.dataset.open !== "true";
  acc.dataset.open = String(open);
  head.setAttribute("aria-expanded", String(open));
  acc.querySelector(".t-acc-panel").inert = !open;
});
document.querySelectorAll("[data-workflow]").forEach((button) =>
  button.addEventListener("click", () => {
    if (state.workflow === button.dataset.workflow) return;
    state.workflow = button.dataset.workflow;
    state.requestId = null;
    renderForm();
    // Replay only on an explicit workflow change, never on job polling.
    document.querySelectorAll(".connection, .configuration").forEach((panel) => {
      panel.classList.add("t-panel-slide");
      panel.style.transition = "none";
      panel.dataset.open = "false";
      void panel.offsetWidth;
      panel.style.removeProperty("transition");
      panel.dataset.open = "true";
    });
  }),
);
$("workflow-form").addEventListener("input", (event) => {
  const key = event.target.dataset.key;
  if (!key) return;
  state.pages[state.workflow][key] =
    event.target.type === "checkbox"
      ? event.target.checked
      : event.target.value;
  state.requestId = null;
  if (key === "project" || key === "workspace")
    for (const page of Object.values(state.pages))
      page[key] = event.target.value;
  const error = $("error-" + key);
  if (error)
    error.textContent = event.target.validity.valid
      ? ""
      : event.target.validationMessage;
  conditional();
});
function conditional() {
  if (state.workflow !== "axis") return;
  for (const key of [
    "axis_cone_degrees",
    "tilt_coarse_step",
    "tilt_refine_step",
  ]) {
    const input = $("field-" + key);
    if (input) input.disabled = !state.pages.axis.refine_near_axis;
  }
}
function profileNote() {
  const profile = state.schema.profiles.find(
    (p) => p.id === $("profile").value,
  );
  const isSlurm = !profile || profile.backend === "slurm";
  $("slurm-panel").hidden = false;
  $("profile-note").textContent = !profile
    ? "Ask the administrator to enable an execution lane below."
    : isSlurm
      ? "Submitted when this lane has capacity. Resources and concurrency are set by your administrator."
      : "Runs on the server within the Local concurrency limit. You can close this browser after submitting.";
}
$("profile").addEventListener("change", () => {
  state.requestId = null;
  profileNote();
});
$("workflow-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.submitting) return;
  state.submitting = true;
  $("submit-job").disabled = true;
  $("form-error").textContent = "";
  const generation = state.generation;
  const payload = {
    workflow: state.workflow,
    values: { ...state.pages[state.workflow] },
    profile: $("profile").value,
  };
  try {
    if ($("profile").value === "slurm-setup")
      throw new Error(
        "Slurm must be configured by the administrator before submitting.",
      );
    const requestId = state.requestId || (await api("/api/request-id")).request_id;
    if (generation !== state.generation) return;
    state.requestId = requestId;
    const job = await api("/api/jobs", "POST", {
      ...payload,
      request_id: requestId,
    });
    if (generation !== state.generation) return;
    state.requestId = null;
    state.selected = job.id;
    state.progress = null;
    $("technical-details").open = false;
    $("form-status").textContent = "Queued · " + job.id.slice(0, 8);
    await refresh();
  } catch (error) {
    if (generation !== state.generation) return;
    $("form-error").textContent = error.message;
    if (error.status === 400) state.requestId = null;
  } finally {
    state.submitting = false;
    $("submit-job").disabled = false;
  }
});
function renderJobs() {
  if (!state.jobs.length) {
    $("job-list").innerHTML =
      '<p class="empty">Your first run starts here. Submitted jobs continue after you close this page.</p>';
    return;
  }
  $("job-list").innerHTML = state.jobs
    .map(
      (job) =>
        `<button type="button" class="job-row ${job.id === state.selected ? "selected" : ""}" data-job="${job.id}"><span>${escape(state.schema.workflows[job.workflow].title)} <small> / ${escape(job.values.project)} · ${escape(job.values.workspace)}</small></span><span class="job-state ${job.state}">${escape(job.state === "interrupted" ? "stopped" : job.state)}</span><span class="meta">${escape(new Date(job.created).toLocaleString())} · ${escape(job.profile)}${job.scheduler_id ? " · Slurm " + escape(job.scheduler_id) : ""} · ${job.id.slice(0, 8)}${job.detail ? " — " + escape(job.detail) : ""}</span></button>`,
    )
    .join("");
}
$("job-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-job]");
  if (!button) return;
  state.selected = button.dataset.job;
  state.progress = null;
  $("technical-details").open = false;
  $("job-details").textContent = "";
  renderProgress();
  $("log-panel").open = true;
  renderJobs();
  await loadLog();
});
async function loadLog() {
  if (!state.selected) {
    jobControls.setJob(null);
    await selectionView.setJob(null);
    $("job-log").textContent = "Select a run to inspect its activity.";
    $("job-details").textContent = "";
    $("log-job").textContent = "";
    $("progress-panel").hidden = true;
    $("job-cleanup-warning").hidden = true;
    return;
  }
  const id = state.selected,
    generation = state.generation;
  const selectedJob = state.jobs.find((job) => job.id === id);
  jobControls.setJob(selectedJob);
  $("job-control-target").textContent = "Selected job · " + id.slice(0, 8);
  await selectionView.setJob(selectedJob);
  const warning = $("job-cleanup-warning");
  const outcome = { completed: "Computation succeeded", failed: "Computation failed", interrupted: "Computation interrupted" };
  warning.textContent = selectedJob?.cleanup_pending
    ? `${outcome[selectedJob.state] || "Execution finished"}; credential cleanup pending retry. Retrying automatically, including after server restarts.`
    : "";
  warning.hidden = !selectedJob?.cleanup_pending;
  try {
    const data = await api("/api/jobs/" + id + "/log");
    if (state.selected !== id || generation !== state.generation) return;
    $("log-job").textContent = id.slice(0, 8);
    $("job-log").textContent = data.log || "No activity messages yet.";
    $("job-details").textContent = data.details || "No technical details yet.";
    state.progress = data.progress;
    renderProgress();
    const job = state.jobs.find((j) => j.id === id);
    if (job && ["failed", "unknown"].includes(job.state))
      $("log-panel").open = true;
  } catch (error) {
    if (generation === state.generation)
      $("refresh-status").textContent = error.message;
  }
}
let refreshing = false;
function renderProgress() {
  if (!state.selected) return;
  const job = state.jobs.find((j) => j.id === state.selected);
  const view = window.JobProgressView.describe(state.progress, job?.state);
  $("progress-panel").hidden = false;
  $("job-progress").textContent = view.text;
  const bar = $("job-progress-bar");
  bar.hidden = view.value === null;
  if (!bar.hidden) {
    bar.max = view.max;
    bar.value = view.value;
  }
}
setInterval(renderProgress, 1000);
async function refresh() {
  if (!state.schema || refreshing) return;
  const generation = state.generation;
  refreshing = true;
  try {
    const data = await api("/api/jobs");
    if (!state.schema || generation !== state.generation) return;
    state.jobs = data.jobs;
    if (!state.jobs.some(job => job.id === state.selected)) state.selected = state.jobs[0]?.id || null;
    const warnings = data.deletion_warnings || [];
    $("job-deletion-warnings").textContent = warnings.map(w => `${w.id.slice(0, 8)} · ${w.detail}`).join("\n");
    $("job-deletion-warnings").hidden = !warnings.length;
    renderJobs();
    await loadLog();
    if (generation === state.generation)
      $("refresh-status").textContent =
        "Updated " + new Date().toLocaleTimeString();
  } catch (error) {
    if (generation === state.generation)
      $("refresh-status").textContent = error.message;
  } finally {
    refreshing = false;
  }
}
$("save-settings").addEventListener("click", () => {
  const pages = Object.fromEntries(
    Object.entries(state.pages).map(([name, values]) => [
      name,
      { ...values, url: state.url },
    ]),
  );
  const url = URL.createObjectURL(
    new Blob([JSON.stringify({ version: 1, pages }, null, 2)], {
      type: "application/json",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "projection-settings.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("load-settings").addEventListener("click", () => $("settings-file").click());
$("settings-file").addEventListener("change", async (event) => {
  try {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 65536) throw Error("Settings file is too large.");
    const data = JSON.parse(await file.text());
    if (data.version !== 1 || !data.pages)
      throw Error("Unsupported settings format.");
    const next = {};
    for (const [name, workflow] of Object.entries(state.schema.workflows)) {
      const page = data.pages[name];
      if (!page || page.url !== state.url)
        throw Error(
          "Settings must contain both pages and match this CryoSPARC server.",
        );
      next[name] = {};
      for (const f of workflow.fields) {
        const value = page[f.key] ?? f.default;
        if (typeof value !== typeof f.default)
          throw Error("Invalid setting: " + f.label);
        next[name][f.key] = value;
      }
    }
    if (
      next.orientation.project !== next.axis.project ||
      next.orientation.workspace !== next.axis.workspace
    )
      throw Error("Connection settings must match across both pages.");
    state.pages = next;
    state.requestId = null;
    renderForm();
    $("form-status").textContent = "Settings loaded";
  } catch (error) {
    $("form-error").textContent = error.message;
  } finally {
    event.target.value = "";
  }
});
bootstrap();
setInterval(refresh, 5000);
