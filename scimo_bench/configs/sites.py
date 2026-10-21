"""Site profiles (group `site`): where a call runs and where its files live. `local` runs
in-process; a cluster site submits the steps as small CPU / GPU jobs (`scimo_bench/jobs.py`).

`paths` are defaults for the top-level `paths` (override e.g. `paths.data=/elsewhere`).
`resources[kind]` are scheduler options written verbatim (`#SBATCH --<key>=<value>`).
"""

from hydra_zen import make_config, store

site_store = store(group="site")

# Login nodes of every site, checked whatever site is selected: forgetting `site=raven` on a
# Raven login node is refused instead of running there.
LOGIN_HOSTS = {"raven": [r"raven\d+"]}


def site_conf(name, scheduler, paths, resources=None, node_local=(), setup=(), wait=False):
    """A site profile: scheduler, default paths, job resources, setup lines."""
    return make_config(
        name=name,
        scheduler=scheduler,
        paths=paths,
        login_hosts=LOGIN_HOSTS.get(name, []),
        inline_steps=["plot"],  # redrawing from the CSVs is allowed on a login node
        setup=list(setup),
        resources=resources or {},
        node_local=list(node_local),  # folders compute nodes cannot see (refused for jobs)
        wait=wait,
    )


# no data default: give `paths.data=...` (or the task's truth path) on the command line
site_store(
    site_conf(
        "local",
        "local",
        paths=dict(data=None, outputs="outputs", snapshots=None),
    ),
    name="local",
)

PTMP = "/ptmp/${oc.env:USER}"
site_store(
    site_conf(
        "raven",
        "slurm",
        paths=dict(
            data=PTMP,
            outputs=f"{PTMP}/bench/outputs",
            snapshots=f"{PTMP}/bench/snapshots",
        ),
        node_local=["/tmp"],
        resources=dict(
            # shared CPU node; the quality path is single-threaded
            cpu={"ntasks": 1, "cpus-per-task": 4, "mem": 16000, "time": "00:30:00"},
            # one of the four A100s of a shared GPU node (a quarter of its CPUs and memory)
            gpu={
                "constraint": "gpu",
                "gres": "gpu:a100:1",
                "ntasks": 1,
                "cpus-per-task": 18,
                "mem": 125000,
                "time": "00:30:00",
            },
        ),
    ),
    name="raven",
)
