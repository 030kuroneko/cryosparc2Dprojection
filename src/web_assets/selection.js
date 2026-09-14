/* Original class identity remains stable across sorting and asynchronous saves. */
"use strict";
window.ClassSelectionView = {
  create({api, render}) {
    let current = null, order = "class_number";
    const jobs = new Map();
    const path = item => `/api/jobs/${encodeURIComponent(item.job.id)}/selection`;
    function paint(item) {
      if (item !== current) return;
      const classes = [...(item.data.classes || [])].sort((a,b) =>
        (order === "class_number" ? a.class_number-b.class_number :
          (b[order] ?? -Infinity)-(a[order] ?? -Infinity)) || a.class_number-b.class_number);
      render({...item.data, job:item.job, classes, selected_class_numbers:[...item.selected].sort((a,b)=>a-b),
        saving:item.pending > 0, error:item.error, order,
        exporting:!!item.exporting, retryRequest:!!item.exportRequest,
        canExport:!item.exporting && (item.exportRequest || (item.data.available && item.selected.size > 0 && !item.error))});
    }
    async function refresh(item) {
      const serial = ++item.serial;
      try {
        const data = await api(path(item));
        if (serial !== item.serial || item.pending) return;
        item.data = data;
        item.selected = new Set(data.selected_class_numbers || []);
        item.error = "";
      } catch (error) { item.error = error.message; }
      paint(item);
    }
    function choose(numbers) {
      const item = current;
      if (!item?.data.available) return Promise.resolve();
      const selected = [...numbers].sort((a,b)=>a-b);
      item.selected = new Set(selected);
      item.pending++;
      item.serial++;
      item.error = "";
      paint(item);
      item.queue = item.queue.then(async () => {
        try {
          const saved = await api(path(item), "PUT", {
            selected_class_numbers:selected, revision:item.data.revision,
          });
          item.data.revision = saved.revision;
        } catch (error) { item.error = error.message + " Reload selections before continuing."; }
        finally { item.pending--; paint(item); }
      });
      return item.queue;
    }
    async function exportSelection() {
      const item = current;
      if (!item?.data.available || item.exporting || (!item.exportRequest && !item.selected.size)) return;
      const snapshot = [...item.selected].sort((a,b)=>a-b);
      item.exporting = true;
      item.error = "";
      paint(item);
      try {
        if (!item.exportRequest) item.exportRequest = {
          request_id:(await api('/api/request-id')).request_id,
          selected_class_numbers:snapshot,
        };
        const result = await api(path(item) + '/exports', 'POST', item.exportRequest);
        item.data.exports = [result, ...(item.data.exports || []).filter(e=>e.id !== result.id)];
        item.exportRequest = null;
      } catch (error) {
        item.error = error.message;
        if (error.status === 400 || error.status === 404) item.exportRequest = null;
      } finally {item.exporting = false; paint(item);}
    }
    return {
      async setJob(job) {
        if (!job) {current = null; jobs.clear(); render({available:false,hidden:true}); return;}
        let item = jobs.get(job.id);
        if (!item) {
          item = {job,data:{available:false,classes:[]},selected:new Set(),pending:0,
            queue:Promise.resolve(),serial:0,error:""};
          jobs.set(job.id, item);
        }
        current = item;
        item.job = job;
        paint(item);
        if (job.workflow === "orientation" && job.state === "completed" && !item.pending && !item.error)
          await refresh(item);
      },
      toggle(number) {
        const selected = new Set(current.selected);
        selected.has(number) ? selected.delete(number) : selected.add(number);
        return choose(selected);
      },
      selectAll:() => choose(current.data.classes.map(c=>c.class_number)),
      clear:() => choose([]),
      invert:() => choose(current.data.classes.filter(c=>!current.selected.has(c.class_number)).map(c=>c.class_number)),
      sort(value) {order = value; if (current) paint(current);},
      reload:() => current ? refresh(current) : Promise.resolve(),
      exportSelection,
      async retry(id) {
        const item = current;
        try {
          const result = await api(path(item) + '/exports/' + encodeURIComponent(id) + '/retry', 'POST', {});
          item.data.exports = (item.data.exports || []).map(e=>e.id === id ? result : e);
          item.error = "";
        } catch(error) {item.error = error.message;}
        paint(item);
      },
    };
  },
};
