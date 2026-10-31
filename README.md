# Document Ink Segmentation

Independent portfolio demo of a **classical computer-vision pipeline**, not a trained neural network or a client deployment.

```sh
python segment.py --output examples
python -m unittest discover -s tests -v
```

Python 3.10+; standard library only. Run commands from this directory. No downloads, API keys or GPU are needed.

The pipeline calculates a global Otsu threshold, extracts 4-connected ink components and filters components smaller than five pixels. Global threshold selection maximizes between-class variance. Four-connectivity keeps diagonal contacts separate. Area filtering removes isolated sensor dust without claiming to identify characters.

The generator creates glyph-like strokes and exact ink masks with reproducible random noise. Clean and shadowed cases are evaluated separately, against a fixed threshold of 127. `examples/report.json` contains actual executed pixel IoU, precision and recall; background-dominated accuracy is intentionally not used. PGM files preserve raw inputs/masks, and SVG overlays display the resulting boxes. The difficult shadowed case exposes the limitations of global thresholding rather than concealing them.

No hyperparameter was selected from the evaluation masks. There is no model training or train/test claim here: the two synthetic cases are deterministic demonstrations, not a benchmark of real scanned-document performance. Real use requires representative scans, independently labelled masks, adaptive-threshold comparisons, document-level splits and target-device evaluation.

See [MODEL_CARD.md](MODEL_CARD.md) for operating limits. The CLI generates data; external image ingestion, OCR, semantic layout segmentation and deep-learning inference are deliberately outside scope.
