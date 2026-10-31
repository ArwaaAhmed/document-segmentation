# Algorithm card

- **System:** Otsu global threshold and connected-component area filtering.
- **Intended use:** transparent educational baseline for grayscale document ink.
- **Data provenance:** repository-generated glyph strokes; no external assets, personal information or client data.
- **Evaluation:** two fixed-seed synthetic conditions, pixel IoU/precision/recall and fixed-threshold comparison. Results are in `examples/report.json` and can be reproduced with the README command.
- **Known failure modes:** uneven illumination, low contrast, halftones, inverted text, tiny punctuation lost by filtering, touching strokes, skew and textured backgrounds. Pixel segmentation is not word detection or reading.
- **Responsible claims:** no real-world accuracy, learned-model capability, security or accessibility certification is asserted.
- **Next validation:** independently collected licensed scans, manual annotation checks, adaptive baseline and tests stratified by contrast, font and scanning device.
