"""Random seeds of tf-keras for `phyloai train` (tf-keras <2.19 fails on Python >=3.12 without the fix)."""
import pytest

from phyloai import train

tf_keras = pytest.importorskip("tf_keras")


def test_fixed_seed_functions():
    assert train.fix_tf_keras_seeds(force=True)
    from tf_keras.src import backend
    from tf_keras.src.utils import tf_utils

    tf_keras.utils.set_random_seed(13)
    first = backend.RandomGenerator(None)._create_seed(None)
    assert isinstance(first, int) and isinstance(tf_utils.get_random_seed(), int)
    assert backend.RandomGenerator(None)._create_seed(7) == 7
    tf_keras.utils.set_random_seed(13)
    assert backend.RandomGenerator(None)._create_seed(None) == first  # Still reproducible.

    # The step that failed: building a layer whose initializer has no explicit seed.
    layer = tf_keras.layers.Dense(2)
    layer.build((None, 3))
    assert layer.kernel.shape == (3, 2)
