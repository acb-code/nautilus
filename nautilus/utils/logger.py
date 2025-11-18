from typing import Any

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str, config: Any, use_wandb: bool = False, run_name: str = "exp"):
        """
        Unified logger for TensorBoard and WandB.
        """
        self.use_wandb = use_wandb

        # 1. Setup TensorBoard
        self.writer = SummaryWriter(log_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n{}".format(
                "\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])
            ),
        )

        # 2. Setup WandB (Optional)
        if self.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    sync_tensorboard=True,  # Auto-upload TensorBoard logs
                    config=vars(config),
                    name=run_name,
                    monitor_gym=True,  # Auto-upload videos
                    save_code=True,
                )
            except ImportError:
                print("⚠️ WandB not installed. install with `pip install wandb`")
                self.use_wandb = False

    def log(self, metrics: dict[str, float], step: int):
        """
        Log a dictionary of metrics.
        """
        for key, value in metrics.items():
            # Handle cleanup if value is a Tensor or Array
            if hasattr(value, "item"):
                value = value.item()

            # Log to TensorBoard
            self.writer.add_scalar(key, value, step)

        # WandB handles TensorBoard sync automatically if configured,
        # but we can also log explicitly if needed.
        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=step)

    def close(self):
        self.writer.close()
        if self.use_wandb:
            import wandb

            wandb.finish()
