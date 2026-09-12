"use strict";
(() => {
  let catalog = null;
  let selected = null;
  let copyFrom = null;
  let previewToken = null;
  let draftVersion = 0;
  let busy = false;
  const fields = ["label", "work_dir", "python", "partition", "account", "qos", "cpus", "gpus", "memory_mb", "time_minutes", "max_concurrent", "enabled", "template_path", "variables", "shared_confirmed"];

  function invalidate() {
    draftVersion++;
    previewToken = null;
    $("save-slurm").disabled = true;
    $("lane-preview-panel").hidden = true;
  }
  function reset() {
    invalidate();
    catalog = selected = copyFrom = null;
    busy = false;
    $("lane-manager").hidden = true;
    $("lane-select").replaceChildren();
    $("lane-preview").textContent = "";
    $("slurm-variables").value = "{}";
    $("template-version").textContent = "";
    $("slurm-fields").hidden = true;
    $("slurm-fields").disabled = true;
  }
  document.addEventListener("launcher-session-reset", reset);
  $("slurm-fields").addEventListener("input", invalidate);
  $("slurm-fields").addEventListener("change", invalidate);

  function renderCatalog() {
    $("lane-select").innerHTML = catalog.lanes.map((lane) =>
      `<option value="${escape(lane.id)}">${escape(lane.label || lane.id)}${lane.enabled === false ? " · disabled" : ""}</option>`).join("");
    $("template-example").textContent = catalog.example_template;
    $("template-variables").textContent = "Built-in variables: " + catalog.builtin_variables.join(", ") + ". Use {{ run_cmd }} to start the workflow. Use | quote for paths and shell arguments.";
  }
  function choose(lane, duplicate = false) {
    invalidate();
    selected = duplicate ? null : lane;
    copyFrom = duplicate ? lane.id : null;
    const values = {...catalog.defaults, ...lane};
    const local = values.backend === "local";
    $("lane-id").value = duplicate ? "" : lane.id || "";
    $("lane-id").disabled = Boolean(selected?.id);
    for (const key of fields) {
      const input = $("slurm-" + key);
      let value = values[key];
      if (key === "shared_confirmed") value = false;
      if (key === "label" && duplicate) value = (lane.label || lane.id) + " copy";
      if (input.type === "checkbox") input.checked = Boolean(value);
      else input.value = key === "variables" ? JSON.stringify(value || {}, null, 2) : value ?? "";
    }
    $("slurm-label").disabled = local;
    $("slurm-enabled").disabled = local;
    $("slurm-resource-fields").hidden = local;
    $("reload-template").hidden = local;
    $("duplicate-lane").disabled = local || !selected?.id;
    $("lane-select").value = duplicate ? "" : lane.id || "";
    $("slurm-status").textContent = local ? "Local capacity is independent of Slurm. Changes apply to the next dispatch." : "Edit settings, preview, then apply. Disabling a lane still allows its accepted jobs to finish.";
    $("template-version").textContent = values.template_sha256 ? "Active template: " + values.template_sha256.slice(0, 12) : "Built-in submission template";
  }
  function settings() {
    const local = selected?.backend === "local";
    if (local) return {...selected, max_concurrent: Number($("slurm-max_concurrent").value)};
    const value = {backend: "slurm"};
    for (const key of fields) {
      const input = $("slurm-" + key);
      value[key] = key === "variables" ? JSON.parse(input.value || "{}") : input.type === "checkbox" ? input.checked : input.type === "number" ? Number(input.value) : input.value.trim();
    }
    if (selected?.id) value.revision = selected.revision;
    if (copyFrom) value.copy_from = copyFrom;
    return value;
  }
  $("unlock-admin").addEventListener("click", async () => {
    const generation = state.generation;
    try {
      await api("/api/admin/unlock", "POST", {token: $("admin-key").value});
      const result = await api("/api/admin/lanes");
      if (generation !== state.generation) return;
      catalog = result;
      renderCatalog();
      $("lane-manager").hidden = false;
      $("slurm-fields").hidden = false;
      $("slurm-fields").disabled = false;
      choose(catalog.lanes.find(lane => lane.id === $("profile").value) || catalog.lanes[0] || catalog.defaults);
    } catch (error) {
      if (generation === state.generation) $("slurm-status").textContent = error.message;
    } finally { $("admin-key").value = ""; }
  });
  $("lane-select").addEventListener("change", () => {
    const lane = catalog?.lanes.find(item => item.id === $("lane-select").value);
    if (lane) choose(lane);
  });
  $("new-lane").addEventListener("click", () => choose(catalog.defaults));
  $("duplicate-lane").addEventListener("click", () => { if (selected?.id) choose(selected, true); });

  async function preview(reload) {
    if (busy) return;
    invalidate();
    const version = draftVersion, generation = state.generation;
    busy = true;
    $("slurm-status").textContent = "Preparing preview…";
    try {
      const result = await api("/api/admin/lanes/preview", "POST", {id: $("lane-id").value.trim(), settings: settings(), reload_template: reload});
      if (generation !== state.generation || version !== draftVersion) return;
      previewToken = result.preview_token;
      $("lane-preview").textContent = result.preview || "Local concurrent jobs: " + result.settings.max_concurrent;
      $("lane-preview-panel").hidden = false;
      $("save-slurm").disabled = false;
      $("slurm-status").textContent = "Preview ready. Apply to publish these settings.";
    } catch (error) {
      if (generation === state.generation && version === draftVersion) $("slurm-status").textContent = error.message;
    } finally { busy = false; }
  }
  $("preview-lane").addEventListener("click", () => preview(false));
  $("reload-template").addEventListener("click", () => preview(true));
  $("save-slurm").addEventListener("click", async () => {
    if (busy || !previewToken) return;
    busy = true;
    $("save-slurm").disabled = true;
    const generation = state.generation, version = draftVersion;
    try {
      const result = await api("/api/admin/lanes", "POST", {preview_token: previewToken});
      const [updated, schema] = await Promise.all([api("/api/admin/lanes"), api("/api/schema")]);
      if (generation !== state.generation) return;
      catalog = updated;
      renderCatalog();
      state.schema = schema;
      const previous = $("profile").value;
      $("profile").innerHTML = schema.profiles.map(p => `<option value="${escape(p.id)}">${escape(p.label)}</option>`).join("");
      $("profile").value = schema.profiles.some(p => p.id === previous) ? previous : schema.profiles[0]?.id || "";
      state.requestId = null;
      profileNote();
      if (version === draftVersion) choose(result.lane);
      $("slurm-status").textContent = "Applied. Existing jobs retain their original settings; concurrency limits apply to subsequent dispatch.";
    } catch (error) {
      if (generation === state.generation) $("slurm-status").textContent = error.message;
    } finally { busy = false; }
  });
})();
