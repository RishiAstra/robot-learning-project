from __future__ import annotations

from rl.config import parse_args
from rl.trainer import SACTrainer


def main():
    args = parse_args()
    trainer = SACTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

