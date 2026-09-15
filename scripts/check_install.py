"""Run a CPU-only synthetic pretrain -> fine-tune integration check.

Synthetic labels have no scientific meaning. All generated data, logs and
checkpoints live in a temporary directory and are removed on exit.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd
from rdkit import Chem
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(arguments, env):
    completed = subprocess.run([sys.executable, *arguments], env=env,
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"Command failed: {arguments}\n{completed.stdout[-12000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-model", action="store_true",
                        help="Check the supplied 256-dimensional, six-layer encoders instead of the tiny model.")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="hypermol-check-") as temporary:
        work = Path(temporary)
        env = dict(os.environ, HYPERMOL_PROJECT_ROOT=str(ROOT),
            HYPERMOL_DATA_ROOT=str(work / "data"),
            HYPERMOL_RESULTS_ROOT=str(work / "results"),
            CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
            PYTHONDONTWRITEBYTECODE="1")
        env.pop("HYPERMOL_SERVER", None)
        env.pop("PYTHONPATH", None)
        reactions = []
        for n in range(2, 18):
            reactant = "[CH3:1]" + "".join(f"[CH2:{j}]" for j in range(2, n + 1)) + f"[OH:{n+1}]"
            product = "[CH3:1]" + "".join(f"[CH2:{j}]" for j in range(2, n)) + f"[CH:{n}]=[O:{n+1}]"
            reactions.append(dict(reactant_smiles=reactant, prod_smiles=product,
                                  unmapped_components="CCO", reaction_type=n % 2))
        raw = work / "raw_pretrain"
        raw.mkdir()
        pd.DataFrame(reactions[:12]).to_csv(raw / "train.csv", index=False)
        pd.DataFrame(reactions[12:]).to_csv(raw / "test.csv", index=False)
        run(["scripts/prepare_pretrain.py", "--raw-root", str(raw),
             "--output-root", str(work / "data/pretrain"), "--workers", "1",
             "--map-size-gb", "1"], env)
        print("PASS: mapped reaction and context preprocessing", flush=True)

        raw_reaction = work / "raw_reaction"
        raw_reaction.mkdir()
        for split, rows in (("train", reactions[:12]), ("valid", reactions[12:14]), ("test", reactions[14:])):
            pd.DataFrame(rows).to_csv(raw_reaction / f"{split}.csv", index=False)
        run(["-m", "hypermol.data.process_reaction_class", "--raw_root", str(raw_reaction),
             "--output_root", str(work / "data/schneider"), "--map_size_gb", "1"], env)

        # Distinct ring scaffolds ensure nonempty, disjoint property splits.
        smiles = ["C1" + "C" * n + "1" for n in range(2, 12)]
        smiles += ["N1" + "C" * n + "1" for n in range(2, 12)]
        assert all(Chem.MolFromSmiles(s) is not None for s in smiles)
        pd.DataFrame({"smiles": smiles,
            "measured log solubility in mols per litre": [i / 10 for i in range(20)]}
        ).to_csv(work / "synthetic_properties.csv", index=False)
        run(["-m", "hypermol.data.process_moleculenet", "--dataset", "esol",
             "--raw_csv", str(work / "synthetic_properties.csv"), "--output_root",
             str(work / "data/esol"), "--map_size_gb", "1"], env)
        print("PASS: reaction classification and property preprocessing", flush=True)

        for filename, module in (("pretrain", "pretrain"),
                ("finetune_esol", "finetune_property"),
                ("finetune_schneider", "finetune_reaction_class")):
            config = yaml.safe_load((ROOT / "configs" / f"{filename}.yaml").read_text())
            config.update(gpu=-1, epochs=1, batch_size=2, num_workers=0)
            for key in config["model"]:
                if args.full_model:
                    continue
                if key.endswith("num_layers") or key == "hg_layers":
                    config["model"][key] = 1
                elif key.endswith("num_heads"):
                    config["model"][key] = 4
                elif key.endswith("dim") or key.endswith("hidden_size") or key.endswith("num_kernel"):
                    config["model"][key] = 32
            if filename == "pretrain":
                config["validation"]["fraction"] = 0.2
                config["split_integrity"]["cache_dir"] = str(work / "split_cache")
            path = work / f"{filename}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            run(["-m", f"hypermol.training.{module}", "--config", str(path), "--max_steps", "1"], env)
            print(f"PASS: {module} forward/backward, validation and checkpoint I/O", flush=True)
        for task in ("esol", "schneider"):
            assert (work / "results/finetune" / task / "best.pt").is_file()
            metrics = json.loads((work / "results/finetune" / task / "final_metrics.json").read_text())
            assert metrics
        print("PASS: complete synthetic integration check; temporary artifacts removed on exit")


if __name__ == "__main__":
    main()
