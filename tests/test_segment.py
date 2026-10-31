import unittest
from segment import components, evaluate, otsu, segment, synthetic_page

class SegmentationTests(unittest.TestCase):
    def test_bimodal_threshold(self):
        self.assertEqual(otsu([[10, 10, 240, 240]]), 10)

    def test_diagonal_is_not_connected(self):
        regions, _ = components([[True, False], [False, True]])
        self.assertEqual(len(regions), 2)

    def test_remove_single_pixel_dust(self):
        regions, cleaned = components([[True, True, False], [False, False, True]], 2)
        self.assertEqual(regions[0]['area'], 2)
        self.assertFalse(cleaned[1][2])

    def test_uniform_page_is_blank(self):
        self.assertFalse(any(any(row) for row in segment([[200]*4 for _ in range(4)])[2]))

    def test_metric_is_not_accuracy(self):
        score = evaluate([[True, True], [False, False]], [[True, False], [True, False]])
        self.assertAlmostEqual(score['intersection_over_union'], 1/3)

    def test_generated_clean_page(self):
        image, truth = synthetic_page(41)
        score = evaluate(segment(image)[2], truth)
        self.assertGreater(score['intersection_over_union'], .95)

    def test_invalid_dimensions_are_not_silently_truncated(self):
        with self.assertRaises(ValueError): evaluate([[True]], [[True, False]])
        with self.assertRaises(ValueError): otsu([[1], [1, 2]])
        with self.assertRaises(ValueError): components([[True]], 0)

if __name__ == '__main__':
    unittest.main()
