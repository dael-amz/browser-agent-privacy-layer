from plva_proxy.providers import PROVIDERS


def test_provider_presets_use_official_endpoints_and_models() -> None:
    assert PROVIDERS["overshoot"].base_url == "https://api.overshoot.ai/v1"
    assert PROVIDERS["overshoot"].model == "Hcompany/Holo3-35B-A3B"
    assert PROVIDERS["hcompany"].base_url == "https://api.hcompany.ai/v1"
    assert PROVIDERS["hcompany"].model == "holo3-1-35b-a3b"
    assert PROVIDERS["hcompany"].key_names[0] == "HAI_API_KEY"


def test_provider_model_catalogs_include_the_default() -> None:
    for spec in PROVIDERS.values():
        assert spec.models
        assert spec.model in spec.allowed_models()
        assert all(model in spec.allowed_models() for model in spec.models)
