import os
from absl import flags
import random
from ml_collections import ConfigDict
from pathlib import Path
import yaml
from cellot_model.utils.helpers import parse_cli_opts
from cellot_model.utils import load_config

import string

ALPHABET = string.ascii_lowercase + string.digits

flags.DEFINE_string("outroot", "./results", "Root directory to write model output")

flags.DEFINE_string("model_name", "", "Name of model class")

flags.DEFINE_string("data_name", "", "Name of dataset")

flags.DEFINE_string("preproc_name", "", "Name of dataset")

flags.DEFINE_string("experiment_name", "", "Name for experiment")

flags.DEFINE_string("submission_id", "", "UUID generated by bash script submitting job")

flags.DEFINE_string(
    "drug", "", "Compute OT map on drug, change outdir to outdir/drugs/drug"
)

flags.DEFINE_string("celldata", "", "Short cut to specify config.data.path & outdir")

flags.DEFINE_string("outdir", "", "Path to outdir")

FLAGS = flags.FLAGS


def name_expdir():
    experiment_name = FLAGS.experiment_name
    if len(FLAGS.drug) > 0:
        if len(experiment_name) > 0:
            experiment_name = f"{experiment_name}/drug-{FLAGS.drug}"
        else:
            experiment_name = f"drug-{FLAGS.drug}"

    if len(FLAGS.outdir) > 0:
        expdir = FLAGS.outdir

    else:
        expdir = os.path.join(
            FLAGS.outroot,
            FLAGS.data_name,
            FLAGS.preproc_name,
            experiment_name,
            f"model-{FLAGS.model_name}",
        )

    return Path(expdir)


def generate_random_string(n=8):
    return "".join(random.choice(ALPHABET) for _ in range(n))


def write_config(path, config):
    if isinstance(config, ConfigDict):
        full = path.resolve().with_name("." + path.name)
        config.to_yaml(stream=open(full, "w"))
        config = config.to_dict()

    yaml.dump(config, open(path, "w"))
    return


def parse_config_cli(path, args):
    if isinstance(path, list):
        config = ConfigDict()
        for path in FLAGS.config:
            config.update(yaml.load(open(path), Loader=yaml.UnsafeLoader))
    else:
        config = load_config(path)

    opts = parse_cli_opts(args)
    config.update(opts)

    if len(FLAGS.celldata) > 0:
        config.data.path = str(FLAGS.celldata)
        config.data.type = "cell"
        config.data.source = "control"

    drug = FLAGS.drug
    if len(drug) > 0:
        config.data.target = drug

    return config


def prepare(argv):
    _, *unparsed = flags.FLAGS(argv, known_only=True)

    if len(FLAGS.celldata) > 0:
        celldata = Path(FLAGS.celldata)

        if len(FLAGS.data_name) == 0:
            FLAGS.data_name = str(celldata.parent.relative_to("datasets"))

        if len(FLAGS.preproc_name) == 0:
            FLAGS.preproc_name = celldata.stem

    if FLAGS.submission_id == "":
        FLAGS.submission_id = generate_random_string()

    if FLAGS.config is not None or len(FLAGS.config) > 0:
        config = parse_config_cli(FLAGS.config, unparsed)
        if len(FLAGS.model_name) == 0:
            FLAGS.model_name = config.model.name

    outdir = name_expdir()

    if FLAGS.config is None or FLAGS.config == "":
        FLAGS.config = str(outdir / "config.yaml")
        config = parse_config_cli(FLAGS.config, unparsed)

    return config, outdir
