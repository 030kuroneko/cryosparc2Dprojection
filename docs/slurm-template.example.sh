#!/bin/bash
#SBATCH --partition={{ partition }}
#SBATCH --cpus-per-task={{ cpus }}
#SBATCH --mem={{ memory_mb }}M
#SBATCH --time={{ time_minutes }}

# The launcher fills authoritative GPU and resource directives before execution.
# Place any additional #SBATCH directives above all shell commands.
# Add site-specific module loads or environment initialization here.
exec {{ run_cmd }}
