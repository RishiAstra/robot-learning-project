from __future__ import annotations

from rl.config import parse_args
from rl.trainer import PPOTrainer, SACTrainer


def main():
    args = parse_args()
    if args.method.startswith("sac_"):
        trainer = SACTrainer(args)
    else:
        trainer = PPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
