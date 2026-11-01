import json

import numpy as np
import pandas as pd
import pytest

from src.experiments import deep


def test_training_audit_preserves_roles_histories_and_selected_epoch(tmp_path):
    index = pd.DataFrame({'battery_id': ['A', 'A', 'B', 'C'], 'cycle_number': [1, 2, 1, 1]})
    diagnostics = {
        'train_losses': [3.0, 2.0, 1.0], 'val_losses': [2.0, 1.0, 1.5],
        'best_epoch': 2, 'epochs_trained': 3, 'monitor': 'normalized_mean_batch_mae',
        'validation_predictions': np.array([82.0]),
        'target_mean': 95.0, 'target_scale': 5.0,
    }
    deep.write_fit_audit(
        tmp_path, dataset='NASA', model='gru', seed=17, fold=1,
        index=index, y=np.array([100., 90., 80., 70.]),
        partitions=(np.array([0, 1]), np.array([2]), np.array([3])),
        train_predictions=np.array([99., 91.]), test_predictions=np.array([73.]),
        diagnostics=diagnostics, preprocessing={'channel_mean': [1.]},
        max_epochs=200, patience=20,
    )
    record = json.loads((tmp_path / 'nasa_gru_s17_f1.json').read_text())
    assert record['best_epoch'] == 2
    assert record['epochs_trained'] == 3
    assert record['val_losses'] == [2., 1., 1.5]
    assert record['batteries'] == {'train': ['A'], 'validation': ['B'], 'test': ['C']}
    assert record['validation_mae'] == 2.
    assert record['test_minus_validation_mae'] == 1.
    assert record['preprocessing']['channel_mean'] == [1.]
    rows = pd.read_csv(tmp_path / 'nasa_gru_s17_f1.csv')
    assert rows.role.tolist() == ['train', 'train', 'validation', 'test']
    assert rows.y_pred.tolist() == [99., 91., 82., 73.]
    assert not list(tmp_path.glob('*.tmp'))


def test_training_audit_rejects_overlapping_battery_partitions(tmp_path):
    with pytest.raises(ValueError, match='disjoint'):
        deep.write_fit_audit(
            tmp_path, dataset='NASA', model='gru', seed=17, fold=1,
            index=pd.DataFrame({'battery_id': ['A'] * 3, 'cycle_number': [1, 2, 3]}),
            y=np.array([90., 80., 70.]),
            partitions=tuple(np.array([i]) for i in range(3)),
            train_predictions=np.array([90.]), test_predictions=np.array([70.]),
            diagnostics={'validation_predictions': np.array([80.])},
            preprocessing={}, max_epochs=200, patience=20,
        )
    assert not list(tmp_path.iterdir())


def test_torch_training_exports_actual_validation_history_without_changing_return_contract():
    import torch
    deep._set_seed(17)
    X = np.ones((6, 2, 1), dtype=np.float32)
    y = np.array([100., 95., 90., 85., 80., 75.])
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 1), torch.nn.Flatten(0))
    diagnostics = {}
    train, test, epochs = deep._train_torch_model(
        'gru', model, X[:3], y[:3], X[3:5], y[3:5], X[5:],
        max_epochs=2, patience=1, batch_size=2, device='cpu', diagnostics=diagnostics,
    )
    assert train.shape == (3,) and test.shape == (1,)
    assert epochs == len(diagnostics['val_losses']) == len(diagnostics['train_losses'])
    assert diagnostics['best_epoch'] == int(np.argmin(diagnostics['val_losses'])) + 1
    assert diagnostics['monitor'] == 'normalized_mean_batch_mae'
    assert len(diagnostics['validation_predictions']) == 2


def test_tensorflow_training_exports_weighted_monitor_history():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(17)
    X = np.ones((6, 8, 5), dtype=np.float32)
    y = np.array([100., 95., 90., 85., 80., 75.])
    diagnostics = {}
    train, test, epochs = deep._train_tensorflow_model(
        'physics_itransformer', X[:3], y[:3], X[3:5], y[3:5], X[5:],
        max_epochs=1, patience=1, batch_size=2, diagnostics=diagnostics,
    )
    assert train.shape == (3,) and test.shape == (1,)
    assert epochs == diagnostics['best_epoch'] == diagnostics['epochs_trained'] == 1
    assert len(diagnostics['val_losses']) == 1
    assert diagnostics['monitor'] == 'weighted_two_head_normalized_mae'
    assert len(diagnostics['validation_predictions']) == 2
