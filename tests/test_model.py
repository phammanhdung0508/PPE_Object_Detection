import numpy as np

from app.model import YoloOnnxModel


def test_normalize_prediction_keeps_nxc_layout() -> None:
    predictions = np.zeros((100, 22), dtype=np.float32)

    normalized = YoloOnnxModel._normalize_prediction(predictions)

    assert normalized.shape == (100, 22)


def test_normalize_prediction_transposes_cxn_layout() -> None:
    predictions = np.zeros((84, 100), dtype=np.float32)

    normalized = YoloOnnxModel._normalize_prediction(predictions)

    assert normalized.shape == (100, 84)


def test_predict_batch_casts_input_dtype() -> None:
    class FakeSession:
        def run(self, output_names, feed):
            tensor = next(iter(feed.values()))
            assert tensor.dtype == np.float16
            return [np.zeros((1, 3, 6), dtype=np.float32)]

    model = YoloOnnxModel()
    model.session = FakeSession()
    model.input_name = "images"
    model.input_dtype = np.float16
    model.output_names = ["output0"]

    predictions = model.predict_batch(np.zeros((1, 3, 640, 640), dtype=np.float32))

    assert len(predictions) == 1
    assert predictions[0].shape == (3, 6)
