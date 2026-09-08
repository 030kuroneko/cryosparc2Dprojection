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
  state.generation++;
  $("workspace").hidden = true;
  $("login-screen").hidden = false;
  state.schema = null;
  state.pages = {};
  state.selected = null;
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
  if (!state.schema.profiles.some((p) => p.id === "slurm")) {
    $("profile").insertAdjacentHTML(
      "beforeend",
      '<option value="slurm-setup">Slurm · configure in browser</option>',
    );
  }
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
function field(f) {
  const value = state.pages[state.workflow][f.key],
    id = "field-" + f.key;
  let input;
  if (f.type === "boolean")
    return `<div class="field"><div class="check-row"><input id="${id}" data-key="${f.key}" type="checkbox" ${value ? "checked" : ""}><label for="${id}">${escape(f.label)}</label></div><small class="hint">${escape(f.hint)}</small></div>`;
  if (f.choices.length)
    input = `<select id="${id}" data-key="${f.key}">${f.choices.map((c) => `<option ${String(c) === value ? "selected" : ""}>${escape(c)}</option>`).join("")}</select>`;
  else
    input = `<input id="${id}" data-key="${f.key}" value="${escape(value)}" ${f.required ? "required" : ""} maxlength="2048" autocomplete="off" ${["project", "workspace", "select_job", "refinement_job", "volume_job"].includes(f.key) ? `pattern="${f.key === "project" ? "P" : f.key === "workspace" ? "W" : "J"}[1-9][0-9]*"` : ""}>`;
  return `<label class="field" for="${id}">${escape(f.label)}${f.required ? " *" : ""}${input}<small class="hint">${escape(f.hint)}</small><span class="field-error" id="error-${f.key}"></span></label>`;
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
        `<details><summary>${{ search: "Search settings", rendering: "Rendering & presentation" }[group]}</summary><div class="fields">${workflow.fields
          .filter((f) => f.group === group)
          .map(field)
          .join("")}</div></details>`,
    )
    .join("");
  $("form-error").textContent = "";
  conditional();
}
document.querySelectorAll("[data-workflow]").forEach((button) =>
  button.addEventListener("click", () => {
    state.workflow = button.dataset.workflow;
    state.requestId = null;
    renderForm();
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
  $("slurm-panel").hidden = !isSlurm;
  $("slurm-fields").disabled = !isSlurm || $("slurm-fields").hidden;
  $("copy-command").hidden = isSlurm;
  $("profile-note").textContent = !profile
    ? "Ask the service administrator to configure Slurm below before submitting."
    : isSlurm
      ? "Submitted to Slurm after the current run. Resources are set by your administrator."
      : "Runs on the server, one job at a time. You can close this browser after submitting.";
}
$("profile").addEventListener("change", () => {
  state.requestId = null;
  profileNote();
});
$("unlock-admin").addEventListener("click", async () => {
  const generation = state.generation;
  try {
    await api("/api/admin/unlock", "POST", { token: $("admin-key").value });
    const result = await api("/api/admin/slurm");
    if (generation !== state.generation) return;
    for (const [key, value] of Object.entries(result.settings)) {
      const input = $("slurm-" + key);
      if (input.type === "checkbox") input.checked = value;
      else input.value = value;
    }
    $("slurm-fields").hidden = false;
    profileNote();
    $("slurm-status").textContent =
      "Administrator settings unlocked for this session.";
  } catch (error) {
    if (generation === state.generation)
      $("slurm-status").textContent = error.message;
  } finally {
    $("admin-key").value = "";
  }
});
$("save-slurm").addEventListener("click", async () => {
  const generation = state.generation;
  const settings = {};
  for (const input of document.querySelectorAll("#slurm-fields input")) {
    settings[input.id.slice(6)] =
      input.type === "checkbox"
        ? input.checked
        : input.type === "number"
          ? Number(input.value)
          : input.value.trim();
  }
  try {
    await api("/api/admin/slurm", "POST", settings);
    const schema = await api("/api/schema");
    if (generation !== state.generation) return;
    state.schema = schema;
    $("profile").innerHTML = schema.profiles
      .map((p) => `<option value="${escape(p.id)}">${escape(p.label)}</option>`)
      .join("");
    $("profile").value = "slurm";
    state.requestId = null;
    profileNote();
    $("slurm-status").textContent =
      "Saved. New Slurm jobs will use these settings.";
  } catch (error) {
    if (generation === state.generation)
      $("slurm-status").textContent = error.message;
  }
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
        `<button type="button" class="job-row ${job.id === state.selected ? "selected" : ""}" data-job="${job.id}"><span>${escape(state.schema.workflows[job.workflow].title)} <small> / ${escape(job.values.project)} · ${escape(job.values.workspace)}</small></span><span class="job-state ${job.state}">${escape(job.state)}</span><span class="meta">${escape(new Date(job.created).toLocaleString())} · ${escape(job.profile)}${job.scheduler_id ? " · Slurm " + escape(job.scheduler_id) : ""} · ${job.id.slice(0, 8)}${job.detail ? " — " + escape(job.detail) : ""}</span></button>`,
    )
    .join("");
}
$("job-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-job]");
  if (!button) return;
  state.selected = button.dataset.job;
  $("log-panel").open = true;
  renderJobs();
  await loadLog();
});
async function loadLog() {
  if (!state.selected) return;
  const id = state.selected,
    generation = state.generation;
  try {
    const data = await api("/api/jobs/" + id + "/log");
    if (state.selected !== id || generation !== state.generation) return;
    $("log-job").textContent = id.slice(0, 8);
    $("job-log").textContent = data.log || "Waiting for the worker to start…";
    const job = state.jobs.find((j) => j.id === id);
    if (job && ["failed", "unknown"].includes(job.state))
      $("log-panel").open = true;
  } catch (error) {
    if (generation === state.generation)
      $("refresh-status").textContent = error.message;
  }
}
let refreshing = false;
async function refresh() {
  if (!state.schema || refreshing) return;
  const generation = state.generation;
  refreshing = true;
  try {
    const data = await api("/api/jobs");
    if (!state.schema || generation !== state.generation) return;
    state.jobs = data.jobs;
    renderJobs();
    if (!state.selected && state.jobs.length) state.selected = state.jobs[0].id;
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
function quote(text) {
  return "'" + String(text).replaceAll("'", "'\\''") + "'";
}
$("copy-command").addEventListener("click", async () => {
  const parts = [
    state.workflow === "axis"
      ? "cryosparc2d-axis-search"
      : "cryosparc2d-projection",
    "--url",
    quote(state.url),
  ];
  for (const [key, value] of Object.entries(state.pages[state.workflow])) {
    if (value === false || value === "") continue;
    const flag = "--" + key.replaceAll("_", "-");
    if (value === true) {
      parts.push(flag);
      continue;
    }
    for (const part of key === "axis_roll" ? value.split(";") : [value])
      parts.push(flag, quote(part));
  }
  try {
    await navigator.clipboard.writeText(parts.join(" "));
    $("form-status").textContent =
      "Command copied · uses your local CryoSPARC login";
  } catch (error) {
    $("form-error").textContent =
      "Clipboard unavailable. Use Save settings to export your configuration.";
  }
});
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
