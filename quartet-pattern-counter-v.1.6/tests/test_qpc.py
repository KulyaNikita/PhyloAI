"""Regression checks against an independent site-pattern counter."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import QuartetPatternCounterModule as qpc


class CounterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def alignment(self, sequences, name='alignment'):
        path = Path(self.temp.name) / f'{name}.fas'
        path.write_text(''.join(f'>s{i}\n{s}\n' for i, s in enumerate(sequences)))
        return str(path)

    @staticmethod
    def expected(sequences, alphabet, quartets):
        frequencies, lengths = [], []
        for quartet in quartets:
            counts = np.zeros(len(alphabet) ** 4, np.float32)
            length = 0
            for site in zip(*(sequences[i] for i in quartet)):
                if all(c in alphabet for c in site):
                    index = 0
                    for c in site:
                        index = index * len(alphabet) + alphabet.index(c)
                    counts[index] += 1
                    length += 1
            if length:
                counts /= length
            frequencies.append(counts)
            lengths.append(length)
        return np.asarray(frequencies), np.asarray(lengths)

    def test_order_lengths_batching_threads_and_index_bounds(self):
        datasets = [
            ('D', 'ACGT', ['ACGTACGTA', 'ACG-ACGTA', 'ATGTACGTA',
                           'ACGTNCGTA', 'TCGTACGTA']),
            ('P', 'ACDEFGHIKLMNPQRSTVWY', ['ACDEFGHIK', 'ARNDCQEGL',
                                         'ACD-FGHIK', 'ACDEFGHIK', 'VCDEFGHIK']),
        ]
        for kind, alphabet, seqs in datasets:
            expected_indices = list(itertools.combinations(range(5), 4))
            expected, lengths = self.expected(seqs, alphabet, expected_indices)
            for threads in (1, 2):
                for variant in ('plain', 'lengths', 'patterns', 'both'):
                    with self.subTest(kind=kind, threads=threads, variant=variant):
                        counter = qpc.QuartetPatternCounter(self.alignment(seqs), kind, threads, 0)
                        self.assertEqual(counter.get_dataset_characteristics(), (5, 9))
                        freqs = np.empty((2, len(alphabet)**4), np.float32)
                        lens = np.empty(2, np.uint32)
                        guard = np.full((6, 4), 65535, np.uint16)
                        indices = guard[:2]
                        actual, actual_lens, actual_indices, batches = [], [], [], []
                        while True:
                            guard.fill(65535)
                            if variant == 'plain':
                                n = counter.next_quartet_pattern_frequencies(freqs)
                            elif variant == 'lengths':
                                n = counter.next_quartet_pattern_frequencies_with_lengths(freqs, lens)
                            elif variant == 'patterns':
                                n = counter.next_quartet_pattern_frequencies_with_patterns(freqs, indices)
                            else:
                                n = counter.next_quartet_pattern_frequencies_with_lengths_with_patterns(freqs, lens, indices)
                            np.testing.assert_array_equal(guard[2:], 65535)
                            if not n:
                                break
                            batches.append(n)
                            actual.extend(freqs[:n].copy())
                            actual_lens.extend(lens[:n].copy())
                            actual_indices.extend(indices[:n].copy())
                        self.assertEqual(batches, [2, 2, 1])
                        np.testing.assert_allclose(actual, expected, atol=1e-7)
                        if variant in ('lengths', 'both'):
                            np.testing.assert_array_equal(actual_lens, lengths)
                        if variant in ('patterns', 'both'):
                            np.testing.assert_array_equal(actual_indices, expected_indices)

    def test_four_clusters_preserve_supplied_taxon_order(self):
        seqs = ['ACGTA', 'ACGTT', 'T-GTA', 'ACGNA', 'TCGTA']
        groups = [np.array(x, dtype=np.int64) for x in ([4, 0], [1], [3], [2])]
        counter = qpc.QuartetPatternCounter(self.alignment(seqs), *groups, 'D', 1, 0)
        f = np.empty((3, 256), np.float32)
        l = np.empty(3, np.uint32)
        indices = np.full((3, 4), 65535, np.uint16)
        n = counter.next_quartet_pattern_frequencies_with_lengths_with_patterns(f, l, indices)
        expected_indices = list(itertools.product(*groups))
        expected, lengths = self.expected(seqs, 'ACGT', expected_indices)
        self.assertEqual(n, 2)
        np.testing.assert_allclose(f[:n], expected)
        np.testing.assert_array_equal(l[:n], lengths)
        np.testing.assert_array_equal(indices[:n], expected_indices)
        self.assertEqual(counter.next_quartet_pattern_frequencies(f), 0)

    def test_zero_valid_sites_return_zero_frequencies_and_zero_length(self):
        for kind, width in [('D', 256), ('P', 160000)]:
            counter = qpc.QuartetPatternCounter(self.alignment(['---'] * 4), kind, 1, 0)
            f = np.full((1, width), 123, np.float32)
            l = np.empty(1, np.uint32)
            self.assertEqual(counter.next_quartet_pattern_frequencies_with_lengths(f, l), 1)
            np.testing.assert_array_equal(f, 0)
            self.assertEqual(l[0], 0)

    def test_invalid_buffers_are_rejected_before_advancing(self):
        counter = qpc.QuartetPatternCounter(self.alignment(['ACGT'] * 4), 'D', 1, 0)
        readonly = np.zeros((1, 256), np.float32)
        readonly.flags.writeable = False
        for f in [np.zeros((1, 160000), np.float32), np.zeros((1, 256), np.float64),
                  np.zeros((1, 512), np.float32)[:, ::2], readonly,
                  np.zeros((0, 256), np.float32)]:
            with self.assertRaises((ValueError, TypeError)):
                counter.next_quartet_pattern_frequencies(f)
        f = np.empty((1, 256), np.float32)
        for l in [np.zeros(1, np.uint16), np.zeros(2, np.uint32)]:
            with self.assertRaises((ValueError, TypeError)):
                counter.next_quartet_pattern_frequencies_with_lengths(f, l)
        for p in [np.zeros((1, 3), np.uint16), np.zeros((1, 4), np.int64)]:
            with self.assertRaises((ValueError, TypeError)):
                counter.next_quartet_pattern_frequencies_with_patterns(f, p)
        self.assertEqual(counter.next_quartet_pattern_frequencies(f), 1)

    def test_bad_files_and_configuration_raise_instead_of_exiting(self):
        path = self.alignment(['ACGT'] * 4)
        for kind, threads in [('X', 1), ('D', 0), ('P', -1)]:
            with self.assertRaises(ValueError):
                qpc.QuartetPatternCounter(path, kind, threads, 0)
        for sequences in [[], ['ACGT'] * 3, ['ACGT', 'ACG', 'ACGT', 'ACGT']]:
            with self.assertRaises((ValueError, RuntimeError)):
                qpc.QuartetPatternCounter(self.alignment(sequences), 'D', 1, 0)
        with self.assertRaises(RuntimeError):
            qpc.QuartetPatternCounter(str(Path(self.temp.name) / 'missing'), 'D', 1, 0)
        with self.assertRaises(ValueError):
            qpc.QuartetPatternCounter(path, np.array([9]), np.array([1]),
                                    np.array([2]), np.array([3]), 'D', 1, 0)


if __name__ == '__main__':
    unittest.main()
