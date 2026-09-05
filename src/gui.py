"""Two-page native desktop launcher for the supported CryoSPARC workflows."""
import argparse
import shlex
import sys
import webbrowser

from cryosparc_2d_projection.gui_model import (
    SHARED, WORKFLOWS, JobRunner, actions, build_arguments,
    default_values, load_settings, save_settings, validate_url,
)

TITLES = {'orientation': 'Class Orientation', 'axis': 'Axis Search'}
DESCRIPTIONS = {
    'orientation': 'Match selected 2D classes to a refined map using particle poses.\n'
                   'Results: Class Average · Matched Projection · Camera View Render',
    'axis': 'Rank class averages against icosahedral 2fold / 3fold / 5fold axes.\n'
            'No particle-pose overlap required. Near-Axis Refinement is optional.',
}
BASIC = {
    'orientation': {'select_job', 'refinement_job', 'symmetry', 'classes'},
    'axis': {'select_job', 'volume_job', 'axis_family', 'top_n', 'refine_near_axis'},
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
    'axis_family': 'Blank = all. Otherwise: 2fold,3fold,5fold. CryoSPARC I convention only.',
    'axis_roll': 'Optional: 2fold=90;3fold=30. Presentation only.',
    'render_grid_size': 'Blank = complete native Rendering Map grid. Lower explicitly to reduce memory.',
    'surface_level': 'Blank = automatic. Affects the Camera View Render only.',
    'url': 'Use the same URL and saved CryoSPARC Tools login as your command-line workflow.',
}


class Launcher:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.runner = JobRunner()
        self.variables = {}
        self.run_url = None
        self.busy = False
        root.title('CryoSPARC · 2D Projection')
        root.geometry('1080x860')
        root.minsize(780, 640)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background='#f5f6fa')
        style.configure('TLabel', background='#f5f6fa', foreground='#26334b')
        style.configure('Title.TLabel', font=('TkDefaultFont', 20, 'bold'))
        style.configure('Subtitle.TLabel', foreground='#667085')
        style.configure('TButton', padding=(12, 7))
        style.configure('TNotebook.Tab', padding=(22, 10))
        style.configure('Accent.TButton', background='#6756b3', foreground='white')
        outer = ttk.Frame(root, padding=20)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='2D Projection', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text='CryoSPARC workspace launcher', style='Subtitle.TLabel').pack(anchor='w', pady=(2, 12))
        connection = ttk.LabelFrame(outer, text='Connection · shared across both pages', padding=12)
        connection.pack(fill='x')
        shared = {name: tk.StringVar() for name in SHARED}
        for i, name in enumerate(SHARED):
            ttk.Label(connection, text=LABELS[name]).grid(row=0, column=i, sticky='w')
            ttk.Entry(connection, textvariable=shared[name], width=45 if i == 0 else 14).grid(
                row=1, column=i, sticky='ew', padx=(0, 12), pady=4)
            connection.columnconfigure(i, weight=3 if i == 0 else 1)
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill='x', pady=8)
        for label, callback in [('Load settings', self.load), ('Save settings', self.save),
                                ('Login instructions', self.login_help)]:
            ttk.Button(toolbar, text=label, command=callback).pack(side='left', padx=(0, 6))
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill='both', expand=True)
        for name in WORKFLOWS:
            defaults = default_values(name)
            self.variables[name] = {key: (tk.BooleanVar(value=value) if type(value) is bool else
                                          tk.StringVar(value=value)) for key, value in defaults.items()}
            self.variables[name].update(shared)
            self._page(name)
        footer = ttk.Frame(outer)
        footer.pack(fill='x', pady=(12, 8))
        self.run_button = ttk.Button(footer, text='Run selected workflow', style='Accent.TButton', command=self.run)
        self.run_button.pack(side='left')
        ttk.Button(footer, text='Copy command', command=self.copy_command).pack(side='left', padx=6)
        ttk.Button(footer, text='Open CryoSPARC', command=self.open_results).pack(side='right')
        self.status = tk.StringVar(value='Ready · results are published to your CryoSPARC workspace')
        ttk.Label(outer, textvariable=self.status).pack(anchor='w')
        self.progress = ttk.Progressbar(outer, mode='indeterminate')
        self.progress.pack(fill='x', pady=6)
        log_frame = ttk.Frame(outer)
        log_frame.pack(fill='x')
        self.log = tk.Text(log_frame, height=8, wrap='word', background='#192236', foreground='#e4e9f3',
                           font=('TkFixedFont', 10), state='disabled', relief='flat', padx=10, pady=8)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.log.pack(fill='x', expand=True)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.poll)

    def _page(self, name):
        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=TITLES[name])
        canvas = tk.Canvas(frame, highlightthickness=0, background='#f5f6fa')
        scroll = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        content = ttk.Frame(canvas, padding=16)
        window = canvas.create_window((0, 0), window=content, anchor='nw')
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
        content.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        # Bind to descendants too, so scrolling works while the pointer is over a field.
        def wheel(event):
            units = (-1 if event.num == 4 else 1) if event.num in (4, 5) else (-1 if event.delta > 0 else 1)
            canvas.yview_scroll(units * 3, 'units')
        ttk.Label(content, text=DESCRIPTIONS[name], justify='left', wraplength=700).pack(anchor='w', pady=(0, 12))
        basic = ttk.Frame(content)
        basic.pack(fill='x')
        advanced = ttk.LabelFrame(content, text='Advanced parameters', padding=12)
        toggle = tk.BooleanVar(value=False)
        ttk.Checkbutton(content, text='Show advanced settings', variable=toggle,
                        command=lambda: advanced.pack(fill='x', pady=10) if toggle.get() else advanced.pack_forget()).pack(anchor='w', pady=8)
        rows = {basic: 0, advanced: 0}
        for action in actions(name):
            if action.dest in SHARED:
                continue
            parent = basic if action.dest in BASIC[name] else advanced
            row = rows[parent]
            label = LABELS.get(action.dest, action.dest.replace('_', ' ').replace('resolution A', 'resolution (Å)').capitalize())
            ttk.Label(parent, text=label + (' *' if action.required else '')).grid(row=row, column=0, sticky='nw', padx=(0, 18), pady=5)
            variable = self.variables[name][action.dest]
            if isinstance(action, argparse._StoreTrueAction):
                widget = ttk.Checkbutton(parent, variable=variable)
            elif action.choices or action.dest == 'symmetry':
                widget = ttk.Combobox(parent, textvariable=variable, values=action.choices or ('C1', 'I'), state='readonly')
            else:
                widget = ttk.Entry(parent, textvariable=variable)
            widget.grid(row=row, column=1, sticky='ew', pady=5)
            hint = HINTS.get(action.dest, action.help)
            if hint:
                ttk.Label(parent, text=hint, style='Subtitle.TLabel', wraplength=570).grid(row=row+1, column=1, sticky='w', pady=(0, 6))
            rows[parent] += 2
            parent.columnconfigure(1, weight=1)
        def bind_tree(widget):
            for event in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
                widget.bind(event, wheel, add='+')
            for child in widget.winfo_children():
                bind_tree(child)
        bind_tree(content)

    def selected(self):
        return tuple(WORKFLOWS)[self.notebook.index(self.notebook.select())]

    def values(self):
        return {name: {key: var.get() for key, var in page.items()} for name, page in self.variables.items()}

    def command(self):
        name = self.selected()
        return [sys.executable, '-u', '-m', 'cryosparc_2d_projection.gui', '--worker', name,
                *build_arguments(name, self.values()[name])]

    def show_error(self, error):
        from tkinter import messagebox
        messagebox.showerror('Please check the settings', str(error), parent=self.root)

    def run(self):
        if self.busy:
            return
        try:
            command = self.command()
            self.run_url = self.values()[self.selected()]['url'].strip()
            self.runner.start(command)
        except (ValueError, RuntimeError) as error:
            self.show_error(error)
            return
        self.busy = True
        self.run_button.configure(state='disabled')
        self.progress.start(12)
        self.status.set(f'Running {TITLES[self.selected()]} · keep this window open')
        self.append_log('\n' + shlex.join(command) + '\n')

    def append_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        # Keep the desktop responsive during long searches. CryoSPARC retains job logs.
        count = int(self.log.index('end-1c').split('.')[0])
        if count > 5000:
            self.log.delete('1.0', f'{count - 5000}.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def poll(self):
        for kind, value in self.runner.drain():
            if kind == 'log':
                self.append_log(value)
            elif kind == 'finished':
                self.busy = False
                self.progress.stop()
                self.run_button.configure(state='normal')
                self.status.set('Completed · open CryoSPARC to view results' if value == 0 else
                                f'Failed (exit {value}) · inspect the log before retrying')
        self.root.after(100, self.poll)

    def copy_command(self):
        try:
            name = self.selected()
            executable = 'cryosparc-2d-projection' if name == 'orientation' else 'cryosparc-axis-search'
            command = shlex.join([executable, *build_arguments(name, self.values()[name])])
            self.root.clipboard_clear()
            self.root.clipboard_append(command)
        except ValueError as error:
            self.show_error(error)

    def save(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.json', filetypes=[('GUI settings', '*.json')])
        if path:
            try:
                save_settings(path, self.values())
            except (OSError, ValueError) as error:
                self.show_error(error)

    def load(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.root, filetypes=[('GUI settings', '*.json')])
        if path:
            try:
                pages = load_settings(path)
                for name, values in pages.items():
                    for key, value in values.items():
                        self.variables[name][key].set(value)
            except (OSError, ValueError) as error:
                self.show_error(error)

    def login_help(self):
        from tkinter import messagebox
        url = self.values()[self.selected()]['url'].strip() or 'https://cryosparc.example.org'
        command = shlex.join([sys.executable, '-m', 'cryosparc.tools', 'login', '--url', url])
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        messagebox.showinfo('CryoSPARC Tools login',
                            'Run this command in a terminal using the same Python environment:\n\n' + command +
                            '\n\nCommand copied. Complete the token login there, then return here.\n'
                            'The GUI does not ask for or save passwords.', parent=self.root)

    def open_results(self):
        try:
            url = validate_url(self.run_url or self.values()[self.selected()]['url'].strip())
            webbrowser.open(url)
        except ValueError as error:
            self.show_error(error)

    def close(self):
        from tkinter import messagebox
        if self.busy or self.runner.running:
            messagebox.showinfo('Job is running', 'Keep this launcher open until the job completes.\n'
                                'Closing the application cannot safely cancel a CryoSPARC External Job.', parent=self.root)
            return
        self.root.destroy()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == '--worker' and argv[1] in WORKFLOWS:
        return WORKFLOWS[argv[1]].main(argv[2:])
    parser = argparse.ArgumentParser(description='Launch the two-page CryoSPARC desktop GUI.')
    parser.parse_args(argv)
    try:
        import tkinter as tk
        root = tk.Tk()
    except (ImportError, RuntimeError) as error:
        print(f'GUI unavailable: {error}. Install Tk for this Python and use a desktop display.', file=sys.stderr)
        return 1
    except Exception as error:
        print(f'Cannot open desktop display: {error}. Run on a desktop or use X forwarding.', file=sys.stderr)
        return 1
    Launcher(root)
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
