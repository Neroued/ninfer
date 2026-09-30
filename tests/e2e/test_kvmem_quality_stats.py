import math
import unittest

from kvmem_quality_stats import paired_accuracy, paired_cluster_accuracy


class PairedStatistics(unittest.TestCase):
    def test_all_ties_do_not_prove_zero_uncertainty(self):
        result = paired_accuracy([True] * 500, [True] * 500)
        expected = 1 - math.pow(0.05 / 4, 1 / 500)
        self.assertAlmostEqual(result["difference_interval"][0], -expected, places=10)
        self.assertAlmostEqual(result["difference_interval"][1], expected, places=10)
        self.assertFalse(result["equivalence_established"])
        small = paired_accuracy([True] * 15, [True] * 15)
        self.assertGreater(small["difference_interval"][1], 0.2)

    def test_direction_and_swap_symmetry(self):
        a, b = [True, True, False, True], [False, False, True, True]
        left, right = paired_accuracy(a, b), paired_accuracy(b, a)
        self.assertEqual(left["difference"], -0.25)
        self.assertEqual(left["regressions"], 2)
        self.assertEqual(left["improvements"], 1)
        self.assertAlmostEqual(left["difference_interval"][0], -right["difference_interval"][1])

    def test_reject_missing_or_nonboolean_labels(self):
        for a, b in (([], []), ([True], []), ([True], [None]), ([1], [True])):
            with self.assertRaises(ValueError):
                paired_accuracy(a, b)

    def test_singleton_clusters_reduce_to_question_interval(self):
        a, b = [True, False, True], [False, True, True]
        clustered = paired_cluster_accuracy(a, b, ['a', 'b', 'c'])
        self.assertEqual(clustered['clusters'], 3)
        self.assertEqual(clustered['difference_interval'], paired_accuracy(a, b)['difference_interval'])

    def test_duplicate_variants_do_not_increase_independent_sample_count(self):
        # Fifty perfectly dependent pairs carry 50, not 100, independent outcomes.
        a, b = [True] * 100, [True] * 100
        pairs = [str(i // 2) for i in range(100)]
        clustered = paired_cluster_accuracy(a, b, pairs)
        self.assertEqual(clustered['clusters'], 50)
        self.assertGreater(clustered['difference_interval'][1], paired_accuracy(a, b)['difference_interval'][1])
        self.assertGreater(clustered['difference_interval'][1], 0)

    def test_cluster_difference_is_question_weighted_and_symmetric(self):
        a, b = [True, True, False, False, True], [False, False, True, True, True]
        groups = ['pair', 'pair', 'single1', 'single2', 'single3']
        left = paired_cluster_accuracy(a, b, groups)
        right = paired_cluster_accuracy(b, a, groups)
        self.assertEqual(left['difference'], 0)
        self.assertEqual(left['n'], 5)
        self.assertEqual(left['clusters'], 4)
        self.assertAlmostEqual(left['difference_interval'][0], -right['difference_interval'][1])
        with self.assertRaises(ValueError):
            paired_cluster_accuracy(a, b, groups[:-1])


if __name__ == "__main__":
    unittest.main()
