from __future__ import annotations

import pandas as pd

from scripts.run_sequence_benchmark import (
    _atomic_write_csv,
    _model_is_complete,
    _read_csv_or_empty,
)
from scripts.run_zero_shot_benchmark import _zero_shot_model_is_complete
from scripts.prepare_model_release import _copy_result_package


def test_grouped_model_checkpoint_requires_every_seed_and_fold(tmp_path):
    metrics = pd.DataFrame({
        "model": ["gru"] * 4,
        "seed": [17, 17, 42, 42],
        "fold": [1, 2, 1, 2],
    })
    predictions = pd.DataFrame({"model": ["gru"], "y_pred": [90.0]})
    assert _model_is_complete(metrics, predictions, "gru", (17, 42), 2)
    assert not _model_is_complete(metrics.iloc[:-1], predictions, "gru", (17, 42), 2)

    path = tmp_path / "checkpoint.csv"
    _atomic_write_csv(metrics, path)
    pd.testing.assert_frame_equal(_read_csv_or_empty(path), metrics)


def test_zero_shot_checkpoint_requires_both_targets(tmp_path):
    family = "recurrent"
    model = "gru"
    for target in ("calce", "oxford"):
        pd.DataFrame({"model": [model, model], "seed": [17, 42]}).to_csv(
            tmp_path / f"nasa_to_{target}_{family}_metrics.csv", index=False
        )
        pd.DataFrame({"model": [model], "y_pred": [90.0]}).to_csv(
            tmp_path / f"nasa_to_{target}_{family}_predictions.csv", index=False
        )
    assert _zero_shot_model_is_complete(tmp_path, family, model, (17, 42))
    (tmp_path / f"nasa_to_oxford_{family}_predictions.csv").unlink()
    assert not _zero_shot_model_is_complete(tmp_path, family, model, (17, 42))


def test_release_result_package_compresses_row_predictions(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "metrics.csv").write_text("mae\n1.2\n", encoding="utf-8")
    (source / "nasa_predictions.csv").write_text("y_true,y_pred\n90,89\n", encoding="utf-8")
    _copy_result_package(source, destination)
    assert (destination / "metrics.csv").read_text(encoding="utf-8") == "mae\n1.2\n"
    import gzip
    with gzip.open(destination / "nasa_predictions.csv.gz", "rt", encoding="utf-8") as handle:
        assert handle.read() == "y_true,y_pred\n90,89\n"
