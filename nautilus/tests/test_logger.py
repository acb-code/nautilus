from types import SimpleNamespace

from nautilus.utils.logger import Logger


def test_logger_writes_events(tmp_path):
    config = SimpleNamespace(a=1, b="x")
    logger = Logger(log_dir=str(tmp_path), config=config, use_wandb=False, run_name="test")

    # Should accept scalar logging without raising
    logger.log({"charts/metric": 1.0}, step=0)
    logger.close()

    # TensorBoard writer should have created an event file in the log dir
    assert any(tmp_path.iterdir())
