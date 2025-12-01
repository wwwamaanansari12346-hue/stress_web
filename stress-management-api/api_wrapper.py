"""
Simple in-process wrapper used by backend/server.py to call prediction code.

Expose: predict_from_dict(data_dict) -> dict

This attempts to load the project's PredictionService from API.services.prediction_service
and call its predict_stress_level method. If the model files or dependencies aren't available
in the current environment it returns a clear stub response so the server remains functional.
"""
from typing import Any, Dict

_service = None


def _load_service():
    global _service
    if _service is not None:
        return _service
    try:
        # API package is expected to be importable when the parent folder is on sys.path
        from API.services.prediction_service import PredictionService

        _service = PredictionService()
        return _service
    except Exception as e:
        # If import or model loading fails, leave _service as None
        print('api_wrapper: failed to load PredictionService:', e)
        _service = None
        return None


def predict_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run prediction for a single survey dict.

    Returns a dict containing at least the keys:
      - prediction: model output (may be str or number)
      - meta: diagnostic metadata

    If the model can't be loaded, returns a fallback dict with prediction=None and a note.
    """
    svc = _load_service()
    if svc is None:
        return {"prediction": None, "note": "ML model not available — place model files in stress-management-api/models and ensure dependencies"}

    # PredictionService.predict_stress_level may return a label string; wrap in dict
    try:
        pred = svc.predict_stress_level(data)
        return {"prediction": pred, "meta": {"source": "in-process-prediction"}}
    except Exception as e:
        return {"prediction": None, "error": str(e)}
