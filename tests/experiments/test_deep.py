import numpy as np

from src.experiments.deep import SequenceStandardizer, TF_MODEL_IDS, TORCH_MODEL_IDS


def test_sequence_standardizer_is_train_fitted_and_finite():
    X = np.array([[[1.0, np.nan], [3.0, 5.0]], [[5.0, 7.0], [7.0, 9.0]]])
    scaler = SequenceStandardizer().fit(X[:1])
    transformed = scaler.transform(X)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed[:1].mean(axis=(0, 1)), [0.0, 0.0])


def test_sequence_registry_covers_ten_models():
    assert len(TORCH_MODEL_IDS) + len(TF_MODEL_IDS) == 10


def test_physics_branch_uses_current_c_rate_channel():
    import tensorflow as tf

    from src.models.deep.itransformer import AbsCumCurrentLayer, CURRENT_CHANNEL_INDEX
    from src.utils.config import SEQUENCE_FEATURE_COLS_V3

    assert CURRENT_CHANNEL_INDEX == SEQUENCE_FEATURE_COLS_V3.index("current_c_rate")
    values = np.zeros((1, 3, len(SEQUENCE_FEATURE_COLS_V3)), dtype=np.float32)
    values[0, :, CURRENT_CHANNEL_INDEX] = [-1.0, -2.0, -3.0]
    values[0, :, SEQUENCE_FEATURE_COLS_V3.index("voltage_v")] = [4.0, 3.8, 3.6]
    observed = AbsCumCurrentLayer()(tf.constant(values)).numpy().reshape(-1)
    np.testing.assert_allclose(observed, [1.0, 3.0, 6.0])


def test_physics_itransformer_carries_configured_loss_weight():
    from src.models.deep.itransformer import build_physics_itransformer

    model = build_physics_itransformer(8, 5, d_model=8, n_heads=2, n_blocks=1, lambda_phy=0.25)
    assert model.physics_loss_weight == 0.25
