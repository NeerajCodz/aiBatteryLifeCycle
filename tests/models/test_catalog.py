from src.models.catalog import MODEL_CATALOG, get_model_spec, model_catalog_records


def test_catalog_has_twenty_unique_models():
    assert len(MODEL_CATALOG) == 20
    assert len({spec.model_id for spec in MODEL_CATALOG}) == 20
    assert {spec.input_kind for spec in MODEL_CATALOG} == {"tabular", "sequence"}


def test_catalog_lookup_and_serialization():
    assert get_model_spec("extra_trees").display_name == "Extra Trees"
    assert len(model_catalog_records()) == 20
