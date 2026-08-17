import numpy

from ubdataloader.text_dataset import TextDataset
from ubdataloader.tokenizer_worker import create_tokens_buffer, ensure_pad_token


class FakeStream:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def state_dict(self, sample_in_epoch, from_beginning=True):
        return {
            "sample_in_epoch": sample_in_epoch,
            "from_beginning": from_beginning,
        }


class FakeTokenizer:
    eos_token = "</s>"

    def __init__(self):
        self.pad_token_id = None
        self._pad_token = None

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self.pad_token_id = 2


def make_text_dataset(rows, *, chunk_size=1, use_token_column=None):
    dataset = object.__new__(TextDataset)
    dataset.epoch = 0
    dataset.sample_in_epoch = 0
    dataset.chunk_size = chunk_size
    dataset.use_token_column = use_token_column
    dataset.stream_dataset = FakeStream(rows)
    return dataset


def test_text_dataset_defaults_missing_chunk_metadata_for_text_rows():
    dataset = make_text_dataset([{"text": "hello"}])

    chunk = next(iter(dataset))
    item = chunk.text_token_payloads[0]

    assert item.text == "hello"
    assert item.token is None
    assert item.ck_tot == 1
    assert item.ck_idx == 0


def test_text_dataset_defaults_missing_chunk_metadata_for_token_rows():
    tokens = numpy.array([11, 12, 13], dtype=numpy.int32)
    dataset = make_text_dataset(
        [{"tokens": tokens.tobytes()}],
        use_token_column="tokens",
    )

    chunk = next(iter(dataset))
    item = chunk.text_token_payloads[0]

    assert item.text is None
    assert item.token == [11, 12, 13]
    assert item.ck_tot == 1
    assert item.ck_idx == 0


def test_create_tokens_buffer_truncates_long_samples_before_native_pack():
    tokenizer = FakeTokenizer()
    ensure_pad_token(tokenizer)

    tokens, labels = create_tokens_buffer(
        [[1, 2, 3, 4, 5], [6, 7]],
        each_sample_seq_len=4,
        tokenizer=tokenizer,
        pack_method="native:truncate",
    )

    assert tokens == [1, 2, 3, 4, 6, 7, 2, 2]
    assert labels == [2, 3, 4, -100, 7, -100, -100, -100]


def test_ensure_pad_token_uses_eos_when_missing():
    tokenizer = FakeTokenizer()

    ensure_pad_token(tokenizer)

    assert tokenizer.pad_token == "</s>"
    assert tokenizer.pad_token_id == 2
