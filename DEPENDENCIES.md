# Dependencies

## Python packages

Installed automatically by `pip install .`:

| Package | Purpose |
|---------|---------|
| click | CLI framework |
| numpy | Array operations |
| opencv-python | Image processing |
| Pillow | Image I/O, JPEG/PNG encoding |
| tqdm | Progress bars |
| img2pdf | PDF assembly |
| ocrmypdf | OCR text layer in PDF |
| pytesseract | Tesseract Python bindings |

## Optional Python extras

```bash
pip install ".[ai]"             # OpenVINO for --ai-dewarp (Intel GPU)
pip install ".[omr]"            # OpenVINO for ghh omr (chant-omr model)
pip install ".[historical-ocr]" # Kraken OCR (requires Python <3.14)
pip install ".[dev]"            # pytest, ruff, coverage
```

## System packages by platform

### Fedora / RHEL

```bash
# OCR
sudo dnf install tesseract tesseract-langpack-lat

# Ghostscript (used by ocrmypdf)
sudo dnf install ghostscript

# Intel GPU for OpenVINO (ai/omr extras)
sudo dnf install intel-compute-runtime oneapi-level-zero
sudo usermod -aG render "$USER"   # re-login for /dev/dri/renderD* access
```

### Ubuntu / Debian

```bash
# OCR
sudo apt install tesseract-ocr tesseract-ocr-lat

# Ghostscript
sudo apt install ghostscript

# Intel GPU for OpenVINO (ai/omr extras)
# See https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_apt.html
```

### macOS (Homebrew)

```bash
# Python (if not already installed)
brew install python@3.13

# OCR
brew install tesseract tesseract-lang

# Ghostscript (used by ocrmypdf)
brew install ghostscript
```

OpenVINO's `--ai-dewarp` uses CPU inference on macOS (no Intel GPU
runtime needed). Install the `ai` or `omr` Python extra as usual:

```bash
pip install ".[ai]"
```

### Dependency matrix

| Dependency | Fedora | Ubuntu/Debian | macOS (Homebrew) | Required? |
|------------|--------|---------------|------------------|-----------|
| Python 3.11+ | `python3` | `python3` | `python@3.13` | Yes |
| Tesseract | `tesseract` | `tesseract-ocr` | `tesseract` | Optional (OCR stage) |
| Latin lang pack | `tesseract-langpack-lat` | `tesseract-ocr-lat` | `tesseract-lang` | Optional (OCR stage) |
| Ghostscript | `ghostscript` | `ghostscript` | `ghostscript` | Yes (PDF assembly) |
| Intel GPU runtime | `intel-compute-runtime` | see OpenVINO docs | N/A | Optional (AI dewarp) |
