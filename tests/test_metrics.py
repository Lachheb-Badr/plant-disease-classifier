import pytest

pytest.importorskip("sklearn")

from src.evaluation.metrics import (  # noqa: E402
    compute_metrics,
    plot_confusion_matrix,
    wilson_interval,
)

CLASSES = ["Healthy", "Powdery", "Rust"]


def test_wilson_interval_perfect_score():
    low, high = wilson_interval(60, 60)
    assert high == pytest.approx(1.0) and 0.93 < low < 0.95


def test_wilson_interval_contains_the_estimate():
    low, high = wilson_interval(51, 60)
    assert low < 51 / 60 < high


def test_perfect_predictions():
    y = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y, y, CLASSES)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["n_errors"] == 0


def test_error_count_and_accuracy():
    m = compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], CLASSES)
    assert m["n_errors"] == 2 and m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["Rust"]["recall"] == pytest.approx(0.5)


def test_confusion_matrix_plot_is_written(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion_matrix([0, 1, 2], [0, 2, 2], CLASSES, out)
    assert out.exists() and out.stat().st_size > 0
