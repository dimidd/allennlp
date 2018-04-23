# pylint: disable=no-self-use,invalid-name
import pytest
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers.tei_text import TEITextDatasetReader


class TestTEITextReader:

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        tei_reader = TEITextDatasetReader(lazy=lazy)
        instances = tei_reader.read("tests/fixtures/data/tei.txt")
        instances = ensure_list(instances)

        expected_labels = ["I-ORG", "O", "I-PER", "O", "O", "I-LOC", "O"]

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["U.N.", "official", "Ekeus", "heads", "for", "Baghdad", "."]
        assert fields["tags"].labels == expected_labels

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["AI2", "engineer", "Joel", "lives", "in", "Seattle", "."]
        assert fields["tags"].labels == expected_labels
