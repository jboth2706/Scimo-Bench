"""Site profiles (group `site`): where a call runs and where its files live. `local` runs
in-process; a cluster site runs the steps as a Snakemake workflow of small CPU / GPU jobs
(`scimo_bench/workflow.py`), with the job resources of its Snakemake profile
`workflow/profiles/<profile>/config.yaml` (`profile`, default: the site's name).

`paths` are defaults for the top-level `paths` (override e.g. `paths.data=/elsewhere`).
For a site whose compute nodes cannot see this repo (Viper: no `/raven`), `envs` names the
pixi manifest of each environment, `bundle` the producer packages copied into the snapshot.
"""

from hydra_zen import make_config, store

site_store = store(group="site")

# Login nodes of every site, checked whatever site is selected: forgetting `site=raven` on a
# Raven login node is refused instead of running there.
LOGIN_HOSTS = {"raven": [r"raven\d+"], "viper": [r"viper\d+"]}
NVIDIA_SMI = "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true"


def site_conf(
    name,
    scheduler,
    paths,
    profile=None,
    set_resources=None,
    snakemake_args=(),
    node_local=(),
    setup=(),
    envs=None,
    bundle=(),
    pixi=None,
    gpu_info=NVIDIA_SMI,
):
    """A site profile: scheduler, default paths, Snakemake profile, setup lines, environments."""
    return make_config(
        name=name,
        scheduler=scheduler,
        paths=paths,
        login_hosts=LOGIN_HOSTS.get(name, []),
        inline_steps=["plot"],  # redrawing from the CSVs is allowed on a login node
        # Snakemake profile (job resources): a folder name in workflow/profiles, or a path
        profile=profile or name,
        # rule → resources merged into the profile's `set-resources` for one call, e.g. the
        # dev partition for the predict jobs: +site.set_resources.predict.slurm_partition=gpudev
        set_resources=set_resources or {},
        # more Snakemake options for one call, e.g. [--keep-going]; a `--set-resources` here
        # replaces the profile's GPU resources (use `set_resources`)
        snakemake_args=list(snakemake_args),
        setup=list(setup),  # shell lines a job runs after activating its env
        node_local=list(node_local),  # folders compute nodes cannot see (refused for jobs)
        # pixi env → {manifest, env, pythonpath?} a job activates (default: this repo's
        # pixi.toml, same env)
        envs=envs or {},
        # package folders (relative to the repo) copied to `<snapshot>/vendor`, first on PYTHONPATH
        bundle=list(bundle),
        pixi=pixi,  # pixi executable of the jobs (default: the one on PATH at submit time)
        gpu_info=gpu_info,  # shell line that prints the GPU into a GPU job's log
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
    ),
    name="raven",
)

# Viper (MI300A APUs, ROCm). Compute nodes see neither /raven nor the Viper login node's
# /ptmp/$USER: paths are on /viper/ptmp2 and /viper/u2, the envs are installed there
# (the model env is hep4m_scimo's ROCm env), the producer code goes into the snapshot.
# The apu partition takes jobs only from the APU login nodes viper11-13 (AllocNodes).
VPTMP = "/viper/ptmp2/${oc.env:USER}"
VHOME = "/viper/u2/${oc.env:USER}"
site_store(
    site_conf(
        "viper",
        "slurm",
        paths=dict(
            data=VPTMP,
            outputs=f"{VPTMP}/bench/outputs",
            snapshots=f"{VPTMP}/bench/snapshots",
        ),
        node_local=["/tmp", "/raven"],
        envs=dict(
            default=dict(manifest=f"{VHOME}/scimo-bench-env/pixi.toml", env="default"),
            # hep4m_scimo's training env; pyarrow (not in it) from a folder of its own
            perf=dict(
                manifest=f"{VHOME}/scimo-rocm/pixi.toml",
                env="default",
                pythonpath=[f"{VHOME}/scimo-bench-env/rocm-extras"],
            ),
        ),
        bundle=["../scimo/src/hep4m_scimo", "../nanoHEP/hep4m"],
        pixi=f"{VHOME}/.pixi/bin/pixi",
        gpu_info="rocm-smi --showproductname --showdriverversion || true",
    ),
    name="viper",
)
