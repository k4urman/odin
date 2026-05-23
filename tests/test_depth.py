"""Unit tests for monocular distance heuristics."""

import unittest

from guide_dog.depth import distance_bucket, estimate_distance_feet, feet_to_phrase


class TestDepth(unittest.TestCase):
    def test_person_closer_has_larger_bbox_and_smaller_distance(self):
        fh, fw = 480, 640
        close = (100, 50, 200, 350)
        far = (200, 120, 80, 140)
        d_close = estimate_distance_feet("person", close, fh, fw)
        d_far = estimate_distance_feet("person", far, fh, fw)
        self.assertLess(d_close, d_far)

    def test_distance_bucket_stable(self):
        self.assertEqual(distance_bucket(3.0), distance_bucket(3.2))
        self.assertNotEqual(distance_bucket(3.0), distance_bucket(5.0))

    def test_feet_to_phrase_readable(self):
        s = feet_to_phrase(3.4)
        self.assertTrue("foot" in s.lower() or "feet" in s.lower())


if __name__ == "__main__":
    unittest.main()
