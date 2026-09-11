from pathlib import Path

from catt_rl.config import load_config
from catt_rl.data.synthetic import make_synthetic_panel
from catt_rl.experiment import train_experiment


def test_end_to_end_training_writes_artifacts(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    panel_path = tmp_path / "smoke.npz"
    make_synthetic_panel(panel_path, assets=3, days=120, seed=19)
    config = load_config(
        root / "configs" / "smoke.yaml",
        [
            f"data.panel_path={panel_path}",
            f"experiment.output_dir={tmp_path / 'outputs'}",
            "experiment.max_updates=1",
            "model.d_model=16",
            "model.heads=4",
            "model.pw_hidden_1=12",
            "model.pw_hidden_2=8",
            "model.policy_hidden=16",
            "data.lookback=8",
            "data.episode_steps=12",
            "data.train_start=2018-01-01",
            "data.train_end=2018-03-31",
            "data.validation_start=2018-04-02",
            "data.validation_end=2018-04-30",
            "data.test_start=2018-05-01",
            "data.test_end=2018-06-15",
            "ppo.rollout_steps=4",
            "ppo.num_envs=2",
            "ppo.minibatch_size=4",
            "evaluation.bootstrap_samples=10",
        ],
    )
    run_dir = train_experiment(config)
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "test" / "daily_returns.csv").exists()
