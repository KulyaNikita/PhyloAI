"""Loading SavedModel networks through their serving signature (Keras 2 and Keras 3)."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from phyloai.benchmark import ModelSpec, SavedModelNetwork, load_network, predict, validate_network_inputs

WEIGHTS = np.arange(15, dtype=np.float32).reshape(5, 3) / 10


class TinyNetwork(tf.Module):
    """Two named inputs and a `topology` output, like the released Fisher networks."""

    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32, name="selected_patterns"),
                                  tf.TensorSpec([None, 1], tf.float32, name="alignment_length")])
    def serve(self, selected_patterns, alignment_length):
        logits = tf.matmul(tf.concat([selected_patterns, alignment_length], axis=1), WEIGHTS)
        return {"topology": tf.nn.softmax(logits)}


def expected(patterns, lengths):
    logits = np.concatenate([patterns, lengths], axis=1) @ WEIGHTS
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def saved_network(tmp_path_factory):
    path = tmp_path_factory.mktemp("network") / "PhyloAI-Fisher-Tiny"
    module = TinyNetwork()
    tf.saved_model.save(module, str(path), signatures={"serving_default": module.serve})
    return path


def test_load_and_predict(saved_network):
    network = load_network(saved_network)
    assert isinstance(network, SavedModelNetwork)
    rng = np.random.default_rng(1)
    patterns = rng.random((7, 4), dtype=np.float32)
    lengths = np.log10(rng.integers(50, 5000, size=(7, 1))).astype(np.float32)
    spec = ModelSpec(name="tiny", model=saved_network.name, kind="topology_aware", length_input="separate")
    validate_network_inputs(network, [patterns, lengths], spec)
    probabilities = predict(network, [patterns, lengths], batch_size=3)  # Several batches.
    assert probabilities.shape == (7, 3)
    assert np.allclose(probabilities, expected(patterns, lengths), atol=1e-6)


def test_wrong_input_shape_is_reported(saved_network):
    network = load_network(saved_network)
    spec = ModelSpec(name="tiny", model=saved_network.name, kind="topology_aware", length_input="separate")
    with pytest.raises(ValueError, match="expects"):
        validate_network_inputs(network, [np.zeros((2, 5), np.float32), np.zeros((2, 1), np.float32)], spec)
