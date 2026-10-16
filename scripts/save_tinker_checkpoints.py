"""Copy every teacher's tinker:// sampler paths out of the gitignored outputs/ tree.

outputs/ is gitignored, so outputs/rmlp/tinker/<arm>/meta.json — the only record of where a
trained teacher's weights live — exists on one laptop and nowhere else. The weights themselves
sit on Tinker's servers and are billed as checkpoint storage, so they outlive this machine; the
URI that reaches them does not. Losing the file means paying to train again.

This writes results/rmlp/checkpoints.json, which IS tracked, so the paths survive a lost
working tree. Run it after any teacher finishes.

  .venv-tinker/bin/python scripts/save_tinker_checkpoints.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TINKER = ROOT / "outputs" / "rmlp" / "tinker"
OUT = ROOT / "results" / "rmlp" / "checkpoints.json"


def main() -> None:
    arms = {}
    for meta_path in sorted(TINKER.glob("*/meta.json")):
        m = json.loads(meta_path.read_text())
        arms[m["name"]] = {
            "base_model": m["base_model"],
            "lora_rank": m["lora_rank"],
            "lora_modules": m["lora_modules"],
            "learning_rate": m["learning_rate"],
            "logical_batch": m["logical_batch"],
            "seed": m["seed"],
            "train_rows": m["train_rows"],
            "epochs_done": m["epochs_done"],
            "complete": m["epochs_done"] >= 5,
            "samplers": {str(d["epoch"]): d["sampler_path"] for d in m["history"]},
            "last_loss": {str(d["epoch"]): d["last_loss"] for d in m["history"]},
            "sampler_path": m.get("sampler_path"),
        }
    doc = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "why": "outputs/ is gitignored; these tinker:// URIs are the only way back to weights "
               "we paid to train. SAMPLER_EPOCH picks among samplers in scripts/run_rmlp.sh.",
        "arms": arms,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    for name, a in arms.items():
        flag = "complete" if a["complete"] else f"PARTIAL ({a['epochs_done']}/5)"
        print(f"  {name:8s} {flag:16s} {len(a['samplers'])} sampler(s)")
    print(f"[checkpoints] wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
