"""Leaf Lens web application.

A thin Flask layer over :class:`src.model.predict.Predictor`. All classification
logic lives in the model; this file only handles HTTP, upload validation and
error presentation.

Endpoints
---------
``GET  /``             the single-page interface
``GET  /api/status``   whether a trained model is loaded, and its class list
``POST /api/predict``  multipart upload -> prediction JSON

Run::

    python app/app.py
    python app/app.py --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Allow `python app/app.py` as well as `python -m app.app`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge  # noqa: E402

from src.model.predict import (DISCLAIMER, ModelNotAvailableError,  # noqa: E402
                               Predictor)
from src.utils.config import load_config  # noqa: E402
from src.utils.helpers import get_logger  # noqa: E402
from src.utils.image_io import InvalidImageError, open_image  # noqa: E402

LOGGER = get_logger("leaf_lens.app")

# Extensions accepted by the upload form, mapped from the configured list.
CONTENT_TYPE_HINTS = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp",
}


def create_app(config_path: str | None = None) -> Flask:
    """Application factory - also used by the tests."""
    config = load_config(config_path)

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    max_mb = float(config.get("data", "max_upload_mb", default=10))
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024)
    app.config["LEAF_LENS_CONFIG"] = config
    app.config["JSON_SORT_KEYS"] = False

    # One predictor for the process; the Keras model is loaded on first use.
    predictor = Predictor(config)
    app.config["LEAF_LENS_PREDICTOR"] = predictor

    allowed_extensions = set(config.supported_extensions)
    accept_attribute = ",".join(
        sorted({CONTENT_TYPE_HINTS.get(e, e) for e in allowed_extensions} | allowed_extensions)
    )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.route("/")
    def index() -> str:
        return render_template(
            "index.html",
            project_name=config.get("project", "name", default="Leaf Lens"),
            class_names=predictor.class_names,
            confidence_threshold=predictor.confidence_threshold,
            max_upload_mb=max_mb,
            accept_attribute=accept_attribute,
            allowed_extensions=", ".join(sorted(e.lstrip(".").upper() for e in allowed_extensions)),
            model_available=predictor.is_available(),
            disclaimer=DISCLAIMER,
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    @app.route("/api/status")
    def status() -> Any:
        info = predictor.info().to_dict()
        available = predictor.is_available()
        return jsonify({
            "model_available": available,
            "model_version": info["model_version"],
            "architecture": info["architecture"],
            "class_names": info["class_names"],
            "confidence_threshold": info["confidence_threshold"],
            "image_size": info["image_size"],
            "trained_utc": info["trained_utc"],
            "max_upload_mb": max_mb,
            "allowed_extensions": sorted(allowed_extensions),
            "message": (
                "Model loaded and ready."
                if available else
                "No trained model is available yet. Train one with "
                "'python -m src.model.train' (which requires a dataset in data/raw)."
            ),
        })

    @app.route("/api/predict", methods=["POST"])
    def predict() -> Tuple[Any, int]:
        if not predictor.is_available():
            return _error(
                "No trained model is available yet, so predictions cannot be made. "
                "Train a model first (see the project README).", 503)

        if "image" not in request.files:
            return _error("No image was included in the request. Choose a file and try again.", 400)

        upload = request.files["image"]
        if not upload.filename:
            return _error("No file was selected. Choose an image and try again.", 400)

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in allowed_extensions:
            return _error(
                f"'{suffix or 'that file type'}' is not supported. "
                f"Please upload one of: "
                f"{', '.join(sorted(e.lstrip('.').upper() for e in allowed_extensions))}.", 415)

        data = upload.read()
        if not data:
            return _error("The uploaded file is empty.", 400)
        if len(data) > app.config["MAX_CONTENT_LENGTH"]:
            return _error(f"The image is larger than the {max_mb:g} MB limit.", 413)

        # Decode before touching the model so a corrupt file produces a clear
        # message instead of a TensorFlow error.
        try:
            image = open_image(io.BytesIO(data))
        except InvalidImageError as exc:
            LOGGER.info("rejected upload %s: %s", upload.filename, exc)
            return _error(
                "That file could not be read as an image. It may be corrupted, or it may "
                "not be a real image file.", 400)

        min_pixels = int(config.get("data", "min_image_pixels", default=32))
        if image.width < min_pixels or image.height < min_pixels:
            return _error(
                f"The image is too small ({image.width}x{image.height} pixels). "
                f"Please upload an image at least {min_pixels}x{min_pixels} pixels.", 400)

        try:
            result = predictor.predict(image)
        except ModelNotAvailableError as exc:
            LOGGER.error("model unavailable: %s", exc)
            return _error("The trained model could not be loaded. Check the server logs.", 503)
        except InvalidImageError as exc:
            return _error(f"The image could not be processed: {exc}", 400)
        except Exception:  # noqa: BLE001 - never leak a traceback to the browser
            LOGGER.exception("prediction failed for %s", upload.filename)
            return _error(
                "Something went wrong while analysing the image. Please try another photograph.",
                500)

        result["filename"] = upload.filename
        result["image_dimensions"] = {"width": image.width, "height": image.height}
        result["probabilities_sorted"] = [
            {"class": name, "probability": value}
            for name, value in sorted(result["probabilities"].items(),
                                      key=lambda kv: kv[1], reverse=True)
        ]
        # Class order for the chart follows the configuration, not the ranking.
        result["probabilities_ordered"] = [
            {"class": name, "probability": result["probabilities"][name]}
            for name in predictor.class_names
        ]
        LOGGER.info("predicted %s (%.3f) for %s",
                    result["predicted_class"], result["confidence"], upload.filename)
        return jsonify(result), 200

    # ------------------------------------------------------------------
    # Error handling - JSON for the API, never a stack trace
    # ------------------------------------------------------------------
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: Exception) -> Tuple[Any, int]:
        return _error(f"The image is larger than the {max_mb:g} MB limit.", 413)

    @app.errorhandler(404)
    def not_found(_: Exception) -> Tuple[Any, int]:
        if request.path.startswith("/api/"):
            return _error("Unknown endpoint.", 404)
        return _error("Page not found.", 404)

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Tuple[Any, int]:
        return _error(exc.description or "Request failed.", exc.code or 500)

    @app.errorhandler(Exception)
    def unhandled(exc: Exception) -> Tuple[Any, int]:
        LOGGER.exception("unhandled error: %s", exc)
        return _error("An unexpected server error occurred. Please try again.", 500)

    return app


def _error(message: str, status_code: int) -> Tuple[Any, int]:
    """Uniform JSON error body. No internal details are exposed."""
    return jsonify({"error": True, "message": message, "status": status_code}), status_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Leaf Lens web application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true",
                        help="enable Flask debug mode (development only)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    app = create_app(args.config)
    predictor: Predictor = app.config["LEAF_LENS_PREDICTOR"]

    LOGGER.info("Leaf Lens starting on http://%s:%d", args.host, args.port)
    if predictor.is_available():
        LOGGER.info("model: %s (version %s)", predictor.model_path, predictor.model_version)
    else:
        LOGGER.warning("no trained model at %s", predictor.model_path)
        LOGGER.warning("The interface will load but predictions are disabled until you train "
                       "a model: python -m src.model.train")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
