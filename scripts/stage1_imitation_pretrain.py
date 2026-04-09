"""Minimal imitation pretraining for Stage 1 controller.

This uses a synthetic teacher target around stable AdamW-like multipliers (near 1.0)
with weak dependence on feature statistics. It is intentionally lightweight and
serves as initialization for short-horizon meta-training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from bench.optimizers.learned_hybrid import TinyLayerwiseController


def main() -> None:
    cfg = yaml.safe_load(Path("configs/stage1_imitation.yaml").read_text(encoding="utf-8"))
    torch.manual_seed(int(cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Matches learned optimizer feature layout for common layer types.
    input_dim = 9 + 8
    model = TinyLayerwiseController(input_dim=input_dim, hidden_dim=int(cfg["hidden_dim"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))

    n = int(cfg["num_samples"])
    x = torch.randn(n, input_dim)

    # Teacher: conservative, bounded around no modulation (1.0).
    t_lr = 0.10 * torch.tanh(0.20 * x[:, 0] - 0.10 * x[:, 4])
    t_mom = 0.08 * torch.tanh(0.15 * x[:, 2] + 0.10 * x[:, 6])
    t_trust = 0.10 * torch.tanh(0.20 * x[:, 1] - 0.10 * x[:, 3])
    targets = torch.stack([t_lr, t_mom, t_trust], dim=1)

    bs = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    losses = []

    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            pred = torch.tanh(model(x[idx]))
            loss = torch.nn.functional.mse_loss(pred, targets[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

    out_path = output_dir / "controller_imitation.pt"
    torch.save(model.state_dict(), out_path)

    (output_dir / "imitation_summary.json").write_text(
        json.dumps(
            {
                "output_path": str(out_path),
                "num_updates": len(losses),
                "final_loss": losses[-1] if losses else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
