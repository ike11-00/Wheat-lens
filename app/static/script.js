/* ==========================================================================
   Leaf Lens - front-end behaviour
   Handles file selection, client-side validation, preview, the prediction
   request and rendering of the result. All classification happens server-side
   in the trained model; nothing here interprets the image.
   ========================================================================== */

(function () {
  "use strict";

  var settings = window.LEAF_LENS || {};
  var ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"];
  var maxBytes = (settings.maxUploadMb || 10) * 1024 * 1024;

  var form = document.getElementById("upload-form");
  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");
  var previewArea = document.getElementById("preview-area");
  var previewImage = document.getElementById("preview-image");
  var previewName = document.getElementById("preview-name");
  var previewSize = document.getElementById("preview-size");
  var predictButton = document.getElementById("predict-button");
  var resetButton = document.getElementById("reset-button");
  var loading = document.getElementById("loading");
  var errorBox = document.getElementById("error-box");
  var resultCard = document.getElementById("result-card");
  var resultClass = document.getElementById("result-class");
  var resultConfidence = document.getElementById("result-confidence");
  var lowConfidence = document.getElementById("low-confidence");
  var uncertainBox = document.getElementById("uncertain-box");
  var uncertainReasons = document.getElementById("uncertain-reasons");
  var probabilityList = document.getElementById("probability-list");
  var confidenceNote = document.getElementById("confidence-note");
  var modelStatus = document.getElementById("model-status");

  var selectedFile = null;
  var previewUrl = null;

  /* ---------------------------------------------------------------- */
  /* helpers                                                           */
  /* ---------------------------------------------------------------- */

  function show(element) { if (element) { element.hidden = false; } }
  function hide(element) { if (element) { element.hidden = true; } }

  function showError(message) {
    errorBox.textContent = message;
    show(errorBox);
  }

  function clearError() {
    errorBox.textContent = "";
    hide(errorBox);
  }

  function formatBytes(bytes) {
    if (bytes < 1024) { return bytes + " B"; }
    if (bytes < 1024 * 1024) { return (bytes / 1024).toFixed(0) + " KB"; }
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function formatPercent(value) {
    return (value * 100).toFixed(1) + "%";
  }

  function extensionOf(name) {
    var index = name.lastIndexOf(".");
    return index === -1 ? "" : name.slice(index + 1).toLowerCase();
  }

  /* ---------------------------------------------------------------- */
  /* file selection                                                    */
  /* ---------------------------------------------------------------- */

  function validateFile(file) {
    var extension = extensionOf(file.name);
    if (ALLOWED_EXTENSIONS.indexOf(extension) === -1) {
      return "“" + file.name + "” is not a supported image type. " +
             "Please choose a " + ALLOWED_EXTENSIONS.join(", ").toUpperCase() + " file.";
    }
    if (file.size === 0) {
      return "That file is empty. Please choose another image.";
    }
    if (file.size > maxBytes) {
      return "That image is " + formatBytes(file.size) + ", which is over the " +
             settings.maxUploadMb + " MB limit. Please choose a smaller image.";
    }
    return null;
  }

  function handleFile(file) {
    clearError();
    hide(resultCard);

    var problem = validateFile(file);
    if (problem) {
      showError(problem);
      clearSelection();
      return;
    }

    selectedFile = file;
    if (previewUrl) { URL.revokeObjectURL(previewUrl); }
    previewUrl = URL.createObjectURL(file);

    previewImage.src = previewUrl;
    previewImage.onerror = function () {
      showError("That file could not be displayed as an image. It may be corrupted.");
      clearSelection();
    };
    previewName.textContent = file.name;
    previewSize.textContent = formatBytes(file.size) +
      (extensionOf(file.name) ? " · " + extensionOf(file.name).toUpperCase() : "");
    show(previewArea);
    show(resetButton);
  }

  function clearSelection() {
    selectedFile = null;
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    previewImage.removeAttribute("src");
    fileInput.value = "";
    hide(previewArea);
  }

  function resetAll() {
    clearSelection();
    clearError();
    hide(resultCard);
    hide(resetButton);
    probabilityList.innerHTML = "";
    dropzone.focus();
  }

  /* ---------------------------------------------------------------- */
  /* rendering the result                                              */
  /* ---------------------------------------------------------------- */

  function renderProbabilities(rows, topClass) {
    probabilityList.innerHTML = "";
    rows.forEach(function (row) {
      var item = document.createElement("li");
      item.className = "probability-row" + (row["class"] === topClass ? " is-top" : "");

      var header = document.createElement("div");
      header.className = "probability-header";

      var name = document.createElement("span");
      name.className = "probability-name";
      name.textContent = row["class"];

      var value = document.createElement("span");
      value.className = "probability-value";
      value.textContent = formatPercent(row.probability);

      header.appendChild(name);
      header.appendChild(value);

      var track = document.createElement("div");
      track.className = "probability-track";
      track.setAttribute("role", "img");
      track.setAttribute("aria-label", row["class"] + ": " + formatPercent(row.probability));

      var fill = document.createElement("div");
      fill.className = "probability-fill";
      track.appendChild(fill);

      item.appendChild(header);
      item.appendChild(track);
      probabilityList.appendChild(item);

      // Set the width after insertion so the transition runs.
      window.requestAnimationFrame(function () {
        fill.style.width = Math.max(row.probability * 100, 0.5) + "%";
      });
    });
  }

  function renderResult(data) {
    resultClass.textContent = data.predicted_class;
    resultConfidence.textContent = formatPercent(data.confidence);

    var rows = data.probabilities_ordered || Object.keys(data.probabilities || {}).map(function (key) {
      return { "class": key, probability: data.probabilities[key] };
    });
    renderProbabilities(rows, data.predicted_class);

    if (data.low_confidence) { show(lowConfidence); } else { hide(lowConfidence); }

    var reasons = (data.uncertainty && data.uncertainty.reasons) || [];
    if (data.uncertain && reasons.length) {
      uncertainReasons.innerHTML = "";
      reasons.forEach(function (reason) {
        var item = document.createElement("li");
        item.textContent = reason;
        uncertainReasons.appendChild(item);
      });
      show(uncertainBox);
    } else {
      hide(uncertainBox);
    }

    confidenceNote.textContent = data.disclaimer +
      " The confidence threshold for a warning is " +
      formatPercent(data.confidence_threshold) + ". Model version: " +
      data.model_version + ".";

    show(resultCard);
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ---------------------------------------------------------------- */
  /* submission                                                        */
  /* ---------------------------------------------------------------- */

  function submit(event) {
    event.preventDefault();
    clearError();

    if (!selectedFile) {
      showError("Please choose an image first.");
      return;
    }

    var payload = new FormData();
    payload.append("image", selectedFile);

    predictButton.disabled = true;
    show(loading);
    hide(resultCard);

    fetch("/api/predict", { method: "POST", body: payload })
      .then(function (response) {
        return response.json()
          .catch(function () {
            throw new Error("The server returned an unreadable response (HTTP " +
                            response.status + ").");
          })
          .then(function (data) {
            if (!response.ok || data.error) {
              throw new Error(data.message || "The prediction failed (HTTP " +
                              response.status + ").");
            }
            return data;
          });
      })
      .then(renderResult)
      .catch(function (error) {
        showError(error.message || "Could not reach the server. Is it still running?");
      })
      .finally(function () {
        hide(loading);
        predictButton.disabled = !settings.modelAvailable ? true : false;
      });
  }

  /* ---------------------------------------------------------------- */
  /* wiring                                                            */
  /* ---------------------------------------------------------------- */

  dropzone.addEventListener("click", function () { fileInput.click(); });
  dropzone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });
  dropzone.addEventListener("drop", function (event) {
    var files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length) { handleFile(files[0]); }
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files.length) { handleFile(fileInput.files[0]); }
  });

  form.addEventListener("submit", submit);
  resetButton.addEventListener("click", resetAll);

  /* Model status in the footer. */
  fetch("/api/status")
    .then(function (response) { return response.json(); })
    .then(function (data) {
      settings.modelAvailable = data.model_available;
      if (data.model_available) {
        modelStatus.textContent = "model " + data.model_version +
          " (" + data.architecture + ") · " + data.class_names.length + " classes";
        predictButton.disabled = false;
        predictButton.removeAttribute("title");
      } else {
        modelStatus.textContent = "no trained model loaded";
        predictButton.disabled = true;
      }
    })
    .catch(function () {
      modelStatus.textContent = "model status unavailable";
    });
})();
