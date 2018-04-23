import logging
from typing import Dict, Iterable, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils.tei import tei_to_iob
from allennlp.data.fields import Field, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("tei_txt")
class TEITextDatasetReader(DatasetReader):
    """
    Reads an unstructured text where named entities are marked is in the following format:

    <personName>Harry</personName> goes to <orgName>Hogwarts</orgName>

    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values will get loaded into the ``"tags"`` ``SequenceLabelField``.

    This dataset reader tokenizes the text and splits it to sentences, such that
    each sentence as an independent ``Instance``.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    """

    def __init__(
        self, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from text in file at: %s", file_path)
            txt = data_file.read().replace("\n", " ")
            doc, _, tags = tei_to_iob(txt)
            n_tokens = 0
            for sent in doc.sents:
                sent_tokens = list(sent)
                sent_tags = []
                for _ in sent_tokens:
                    sent_tags.append(tags[n_tokens])
                    n_tokens += 1

                tokens = [Token(token) for token in sent_tokens]
                sequence = TextField(tokens, self._token_indexers)

                instance_fields: Dict[str, Field] = {"tokens": sequence}

                # Add "tag label" to instance
                instance_fields["tags"] = SequenceLabelField(sent_tags, sequence)

                yield Instance(instance_fields)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance(
            {"tokens": TextField(tokens, token_indexers=self._token_indexers)}
        )
