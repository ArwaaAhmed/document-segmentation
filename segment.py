"""Classical document-ink segmentation, deliberately independent of ML packages."""
import argparse
import base64
import json
import random
import struct
import zlib
from collections import deque
from pathlib import Path


def image_shape(image):
    if not image or not image[0] or any(len(row) != len(image[0]) for row in image):
        raise ValueError('Expected a nonempty rectangular image')
    return len(image), len(image[0])


def otsu(image):
    """Maximize between-class variance; ties choose the first separating level."""
    image_shape(image)
    values = [pixel for row in image for pixel in row]
    histogram = [0] * 256
    for pixel in values:
        if not 0 <= pixel <= 255:
            raise ValueError("Expected 8-bit grayscale")
        histogram[pixel] += 1
    total_sum = sum(i * count for i, count in enumerate(histogram))
    left_count = left_sum = 0
    best_variance, threshold = -1, 0
    for level, count in enumerate(histogram):
        left_count += count
        left_sum += level * count
        right_count = len(values) - left_count
        if not left_count or not right_count:
            continue
        variance = left_count * right_count * (
            left_sum / left_count - (total_sum - left_sum) / right_count
        ) ** 2
        if variance > best_variance:
            best_variance, threshold = variance, level
    return threshold


def components(mask, minimum_area=1):
    """Return retained 4-connected components and the cleaned mask.

    Four-connectivity avoids joining glyphs that touch only at a diagonal corner.
    The component filter is selected before evaluation, not tuned on test masks.
    """
    height, width = image_shape(mask)
    if minimum_area < 1:
        raise ValueError('Minimum component area must be positive')
    seen, regions = set(), []
    cleaned = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or (x, y) in seen:
                continue
            queue, pixels = deque([(x, y)]), []
            seen.add((x, y))
            while queue:
                px, py = queue.popleft()
                pixels.append((px, py))
                for nx, ny in ((px-1, py), (px+1, py), (px, py-1), (px, py+1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        queue.append((nx, ny))
            if len(pixels) >= minimum_area:
                for px, py in pixels:
                    cleaned[py][px] = True
                xs, ys = zip(*pixels)
                regions.append({"area": len(pixels), "bbox": [min(xs), min(ys), max(xs)+1, max(ys)+1]})
    return regions, cleaned


def segment(image, minimum_area=5):
    threshold = otsu(image)
    # A uniform page has no evidence of a foreground/background split.
    if len({pixel for row in image for pixel in row}) == 1:
        return threshold, [], [[False] * len(image[0]) for _ in image]
    mask = [[pixel <= threshold for pixel in row] for row in image]
    regions, mask = components(mask, minimum_area)
    return threshold, regions, mask


def evaluate(predicted, truth):
    if image_shape(predicted) != image_shape(truth):
        raise ValueError('Prediction and truth dimensions must match')
    pairs = list(zip((p for row in predicted for p in row), (p for row in truth for p in row)))
    tp = sum(a and b for a, b in pairs)
    fp = sum(a and not b for a, b in pairs)
    fn = sum(not a and b for a, b in pairs)
    return {"intersection_over_union": tp / (tp+fp+fn) if tp+fp+fn else 1.0,
            "precision": tp / (tp+fp) if tp+fp else 1.0,
            "recall": tp / (tp+fn) if tp+fn else 1.0,
            "true_positive_pixels": tp, "false_positive_pixels": fp, "false_negative_pixels": fn}


def synthetic_page(seed, challenging=False):
    rng, width, height = random.Random(seed), 240, 140
    truth = [[False] * width for _ in range(height)]
    # Structured strokes mimic fragmented text, without copying a real document.
    for line in range(5):
        for glyph in range(14):
            x, y = 15 + glyph*15, 13 + line*24
            for py in range(y, y+12):
                for px in range(x, x+8):
                    if px in (x, x+1) or py in (y, y+1, y+6, y+7):
                        truth[py][px] = True
    image = []
    for y in range(height):
        row = []
        for x in range(width):
            base = (105 if challenging else 45) if truth[y][x] else (130 + 100*x/width if challenging else 230)
            value = max(0, min(255, int(base + rng.gauss(0, 12 if challenging else 6))))
            # Tiny dark dust is deliberately outside ground-truth ink.
            if not truth[y][x] and rng.random() < .003:
                value = 20
            row.append(value)
        image.append(row)
    return image, truth


def write_pgm(path, image):
    height, width = len(image), len(image[0])
    Path(path).write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixel for row in image for pixel in row))


def png_bytes(image):
    """Encode lossless 8-bit grayscale PNG using only the standard library."""
    height, width = image_shape(image)
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data)&0xffffffff)
    raw = b''.join(b'\x00'+bytes(row) for row in image)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',width,height,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, seed, challenging in [("clean", 41, False), ("shadowed", 82, True)]:
        image, truth = synthetic_page(seed, challenging)
        threshold, regions, mask = segment(image)
        baseline = [[p <= 127 for p in row] for row in image]
        reports.append({"case": name, "seed": seed, "threshold": threshold,
                        "components": len(regions), "method": evaluate(mask, truth),
                        "fixed_threshold_baseline": evaluate(baseline, truth)})
        write_pgm(output / f"{name}-input.pgm", image)
        (output / f"{name}-input.png").write_bytes(png_bytes(image))
        write_pgm(output / f"{name}-truth.pgm", [[0 if p else 255 for p in row] for row in truth])
        write_pgm(output / f"{name}-mask.pgm", [[0 if p else 255 for p in row] for row in mask])
        # SVG makes the evidence inspectable in an ordinary browser.
        # Embedded PNG preserves grayscale noise without tens of thousands of SVG nodes.
        marks = [f'<image width="240" height="140" href="data:image/png;base64,{base64.b64encode(png_bytes(image)).decode()}"/>']
        boxes = [f'<rect x="{r["bbox"][0]}" y="{r["bbox"][1]}" width="{r["bbox"][2]-r["bbox"][0]}" height="{r["bbox"][3]-r["bbox"][1]}" fill="none" stroke="#be123c" stroke-width=".5"/>' for r in regions]
        (output / f"{name}-overlay.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 140"><title>Synthetic document with detected ink components</title>'+''.join(marks+boxes)+'</svg>', encoding='utf-8')
    report = {"dataset": "Generated synthetic glyph strokes; no real client data", "method": "Global Otsu plus area-filtered 4-connected components", "minimum_area": 5, "cases": reports}
    (output / "report.json").write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('examples'))
    run(parser.parse_args().output)
