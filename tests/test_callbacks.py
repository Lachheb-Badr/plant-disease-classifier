from src.training.callbacks import EarlyStopping


def test_improvement_resets_the_counter():
    es = EarlyStopping(patience=3, min_delta=0.001)
    assert es.step(1.0) and es.step(0.8)
    assert es.counter == 0 and not es.should_stop


def test_stops_after_patience_epochs_without_improvement():
    es = EarlyStopping(patience=3, min_delta=0.001)
    es.step(0.8)
    for loss in (0.81, 0.82):
        assert not es.step(loss) and not es.should_stop
    es.step(0.83)
    assert es.should_stop


def test_disabled_never_stops_but_still_tracks_the_best():
    es = EarlyStopping(patience=1, enabled=False)
    es.step(0.5)
    es.step(0.9)
    es.step(0.9)
    assert not es.should_stop and es.best == 0.5
