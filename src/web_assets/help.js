"use strict";
// Delegation also covers workflow fields rebuilt when switching workflows.
(() => {
  const pendingCloses = new WeakMap();
  function cancelClose(wrapper) {
    clearTimeout(pendingCloses.get(wrapper));
    pendingCloses.delete(wrapper);
  }
  function show(wrapper, open) {
    wrapper.querySelector('.help-panel').hidden = !open;
    wrapper.querySelector('button').setAttribute('aria-expanded', String(open));
  }
  function close(wrapper) {
    cancelClose(wrapper);
    delete wrapper.dataset.pinned;
    delete wrapper.dataset.hovered;
    delete wrapper.dataset.focused;
    show(wrapper, false);
  }
  for (const [enter, leave, flag] of [
    ['pointerover', 'pointerout', 'hovered'],
    ['focusin', 'focusout', 'focused'],
  ]) {
    document.addEventListener(enter, event => {
      const wrapper = event.target.closest('.field-help');
      if (!wrapper || wrapper.contains(event.relatedTarget)) return;
      cancelClose(wrapper);
      wrapper.dataset[flag] = 'true';
      show(wrapper, true);
    });
    document.addEventListener(leave, event => {
      const wrapper = event.target.closest('.field-help');
      if (!wrapper || wrapper.contains(event.relatedTarget)) return;
      delete wrapper.dataset[flag];
      cancelClose(wrapper);
      if (!wrapper.dataset.pinned && !wrapper.dataset.hovered && !wrapper.dataset.focused) {
        // Allow the pointer to cross the gap between the icon and help content.
        pendingCloses.set(wrapper, setTimeout(() => {
          pendingCloses.delete(wrapper);
          if (!wrapper.dataset.pinned && !wrapper.dataset.hovered && !wrapper.dataset.focused)
            show(wrapper, false);
        }, 300));
      }
    });
  }
  document.addEventListener('click', event => {
    const wrapper = event.target.closest('.field-help');
    for (const other of document.querySelectorAll('.field-help'))
      if (other !== wrapper) close(other);
    if (!wrapper || event.target !== wrapper.querySelector('button')) return;
    if (wrapper.dataset.pinned) close(wrapper);
    else {
      cancelClose(wrapper);
      wrapper.dataset.pinned = 'true';
      show(wrapper, true);
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    const wrapper = event.target.closest('.field-help');
    for (const other of document.querySelectorAll('.field-help')) close(other);
    if (wrapper) {
      wrapper.querySelector('button').focus();
      close(wrapper);
    }
  });
})();
