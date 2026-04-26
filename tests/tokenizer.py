from collections.abc import Iterable, Iterator

import regex as re


gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        # normalize vocab to bytes and build bi-directional maps
        self.id_to_bytes: dict[int, bytes] = {}
        self.bytes_to_id: dict[bytes, int] = {}
        for token_id, token in vocab.items():
            token_bytes = token if isinstance(token, bytes) else token.encode("utf-8")
            self.id_to_bytes[token_id] = token_bytes
            self.bytes_to_id[token_bytes] = token_id

        self.merges = [
            (
                m1 if isinstance(m1, bytes) else m1.encode("utf-8"),
                m2 if isinstance(m2, bytes) else m2.encode("utf-8"),
            )
            for m1, m2 in merges
        ]
        self.merge_ranks = {merge: rank for rank, merge in enumerate(self.merges)}
        self.special_token_ids = {
            token: self.bytes_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token and token.encode("utf-8") in self.bytes_to_id
        }
        if self.special_token_ids:
            special_pattern = "|".join(re.escape(token) for token in self.special_token_ids)
            self.special_token_pattern = re.compile(f"({special_pattern})")
        else:
            self.special_token_pattern = None

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.id_to_bytes[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        ids = []
        for chunk in self._split_on_special_tokens(text):
            if not chunk:
                continue
            if chunk in self.special_token_ids:
                ids.append(self.special_token_ids[chunk])
                continue
            for match in re.finditer(gpt2_pattern, chunk):
                ids.extend(self._encode_pretoken(match.group(0)))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def _split_on_special_tokens(self, text: str) -> list[str]:
        if self.special_token_pattern is None:
            return [text]
        return self.special_token_pattern.split(text)

    def _encode_pretoken(self, text: str) -> list[int]:
        word = tuple(bytes([byte]) for byte in text.encode("utf-8"))

        while len(word) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(word, word[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, pair_to_merge = min(ranked_pairs)
            word = self._merge_pair(word, pair_to_merge)

        return [self.bytes_to_id[token] for token in word]

    @staticmethod
    def _merge_pair(
        word: tuple[bytes, ...],
        pair_to_merge: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        merged_word = []
        i = 0
        while i < len(word):
            if (
                i < len(word) - 1
                and word[i] == pair_to_merge[0]
                and word[i + 1] == pair_to_merge[1]
            ):
                merged_word.append(word[i] + word[i + 1])
                i += 2
            else:
                merged_word.append(word[i])
                i += 1
        return tuple(merged_word)




# small tests
if __name__ == "__main__":
    vocab = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m", 13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u", 21: "v", 22: "w", 23: "x", 24: "y", 25: "z", 26: "he"}
    # merges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h"), ("h", "i"), ("i", "j"), ("j", "k"), ("k", "l"), ("l", "m"), ("m", "n"), ("n", "o"), ("o", "p"), ("p", "q"), ("q", "r"), ("r", "s"), ("s", "t"), ("t", "u"), ("u", "v"), ("v", "w"), ("w", "x"), ("x", "y"), ("y", "z")]
    
    merges = [("h","e")]
    special_tokens = [""]
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    print(tokenizer.id_to_bytes)
    print(tokenizer.encode("hello"))
    print(tokenizer.decode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]))
