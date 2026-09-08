"use strict";
(() => {
  const root = document.documentElement;
  let theme = "dark";
  try {
    if (localStorage.getItem("cryosparc2d-theme") === "light") theme = "light";
  } catch (_) { /* Storage may be disabled; switching still works. */ }
  root.dataset.theme = theme;
  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("theme-toggle");
    function render() {
      const light = root.dataset.theme === "light";
      button.textContent = light ? "☾ Dark" : "☀ Light";
      button.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
      button.setAttribute("aria-pressed", String(light));
    }
    button.addEventListener("click", () => {
      root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
      try { localStorage.setItem("cryosparc2d-theme", root.dataset.theme); } catch (_) {}
      render();
    });
    render();
  });
})();
