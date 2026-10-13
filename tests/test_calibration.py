import numpy as np
import pytest

from sentiment_intelligence.calibration import calibration_report


def test_validation_calibration_is_finite():
    result = calibration_report(
        np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]),
        [0, 1, 2],
    )
    assert result["split"] == "validation"
    assert result["calibration_fitted"] is False
    assert result["samples"] == 3
    assert 0 <= result["ece"] <= 1
    assert result["brier_score"] >= 0


def test_calibration_rejects_invalid_logits():
    with pytest.raises(ValueError):
        calibration_report([[float("nan"), 1.0]], [0])


def test_calibration_rejects_bad_labels():
    with pytest.raises(ValueError):
        calibration_report([[0.0, 1.0]], [2])
