import pytest

torch = pytest.importorskip("torch")

from src.models import get_model  # noqa: E402
from src.models.checkpoint import load_checkpoint  # noqa: E402


def build(config, name, **model_kwargs):
    config["model"].update(name=name, pretrained=False, **model_kwargs)  # no weight download in tests
    return get_model(config).eval()


@pytest.mark.parametrize("name", ["custom_cnn", "mobilenet_v3_small", "efficientnet_b0"])
def test_forward_shape(config, name):
    model = build(config, name)
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 96)).shape == (2, 3)


def test_unknown_model_name(config):
    with pytest.raises(ValueError):
        build(config, "vgg16")


def test_head_only_trains_just_the_head(config):
    m = build(config, "efficientnet_b0", unfreeze_blocks=0)
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in m.head.parameters()}


def test_partial_fine_tuning_trains_more_than_head_less_than_all(config):
    head = build(config, "efficientnet_b0", unfreeze_blocks=0).num_trainable_params
    partial = build(config, "efficientnet_b0", unfreeze_blocks=3).num_trainable_params
    full = build(config, "efficientnet_b0", unfreeze_blocks=None).num_trainable_params
    assert head < partial < full


def test_frozen_layers_stay_in_eval_mode(config):
    m = build(config, "efficientnet_b0", unfreeze_blocks=1)
    m.train()
    assert not m.net.features[0].training   # frozen stem: BatchNorm statistics must not move
    assert m.net.features[-1].training      # trainable last block
    assert m.head.training


def test_checkpoint_roundtrip(config, tmp_path):
    model = build(config, "custom_cnn")
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": config,
                "classes": config["data"]["classes"]}, path)
    loaded, cfg = load_checkpoint(path)
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)
    assert cfg["data"]["classes"] == config["data"]["classes"]
