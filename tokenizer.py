from collections import defaultdict
import os
import regex as re
import heapq
from typing import Iterable, List, Dict, Tuple, Iterator


GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)


def iter_pretokenize(text: str) -> Iterable[bytes]:
    for b in GPT2_RE.finditer(text):
        yield b.group(0).encode("utf-8")


def _gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


_BYTES_TO_UNICODE = _gpt2_bytes_to_unicode()
_UNICODE_TO_BYTES = {v: k for k, v in _BYTES_TO_UNICODE.items()}
        

class BPETokenizer:
    def __init__(
        self,
        special_tokens: List[str] | None = None,
        vocab: Dict[int, bytes] | None = None,
        merges: List[Tuple[bytes, bytes]] | None = None,
    ):
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [token.encode("utf-8") for token in self.special_tokens]

        self.stoi: Dict[bytes, int] = {}
        self.itos: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {}

        if vocab is not None:
            # Use provided vocab mapping as-is
            self.itos = dict(vocab)
            self.stoi = {b: i for i, b in self.itos.items()}
        else:
            # Initialize default vocab: specials first, then 256 single bytes
            for i, special_tokens_byte in enumerate(self.special_tokens_bytes):
                self.stoi[special_tokens_byte] = i
                self.itos[i] = special_tokens_byte
            offset = len(self.special_tokens_bytes)
            for i in range(256):
                b = bytes([i])
                self.stoi[b] = i + offset
                self.itos[i + offset] = b

        if merges is not None:
            self.merges = merges
            self.merges_rank = {pair: idx for idx, pair in enumerate(self.merges)}
        
        # Mirror itos for compatibility with adapters
        self.vocab = self.itos.copy()
    
    # ---------- heap utils ----------
    
    def _desc_bytes_key_from_id(self, tid: int):
        b = self.itos[tid]
        # Tie-break uses the first two bytes (padded with 0), compared in descending order.
        a0 = b[0] if len(b) > 0 else 0
        a1 = b[1] if len(b) > 1 else 0
        return (-a0, -a1)

    def _desc_pair_key_from_ids(self, pair: Tuple[int, int]):
        i1, i2 = pair
        return (self._desc_bytes_key_from_id(i1), self._desc_bytes_key_from_id(i2))

    def _heap_push(self, heap, pair: Tuple[int, int], cnt: int):
        # Max-heap by count; tie-break by descending bytes on first two bytes of each token
        heapq.heappush(heap, (-cnt, self._desc_pair_key_from_ids(pair), pair))

    def _heap_pop(self, heap, pair_cnts):
        while heap:
            negf, _, pair = heapq.heappop(heap)
            f = -negf
            if pair_cnts.get(pair, 0) == f and f > 0:
                return pair
        return None
    
    # ---------- optimized train ----------
    
    def _merge_pair_in_word(self, 
                            word: List[int], 
                            pair: Tuple[int, int], 
                            new_index: int) -> List[int]:
        a, b = pair
        out, i, L = [], 0, len(word)
        while i < L:
            if i + 1 < L and word[i] == a and word[i + 1] == b:
                out.append(new_index)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return out
    
    def _count_words_and_pairs(self, token_groups: List[List[int]]):
        word_cnts: Dict[Tuple[int, ...], int] = defaultdict(int)
        for w in token_groups:
            if len(w) >= 2:
                word_cnts[tuple(w)] += 1

        words: List[Tuple[int, ...]] = list(word_cnts.keys())
        wcnt:  List[int]             = list(word_cnts.values())

        pair_cnts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_occs: Dict[Tuple[int, int], set] = defaultdict(set)
        for wid, (wb, c) in enumerate(zip(words, wcnt)):
            if len(wb) < 2:
                continue
            for a, b in zip(wb[:-1], wb[1:]):
                pair_cnts[(a, b)] += c     
            for p in set(zip(wb[:-1], wb[1:])): 
                pair_occs[p].add(wid)
        return words, wcnt, pair_cnts, pair_occs
            
    def train(self, path: os.PathLike, vocab_size: int):
        assert vocab_size >= len(self.stoi), (
            f"vocab_size ({vocab_size}) must be >= {len(self.stoi)} "
            f"(special tokens + 256 byte tokens)"
        )   
        
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.special_tokens: 
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            text_parts = re.split(special_pattern, text)
        else:
            text_parts = [text]
            
        token_groups = []
        for part in text_parts:
            if not part:
                continue
            if part in self.special_tokens:
                # Special tokens should not be pre-tokenized.
                token_groups.append([self.stoi[part.encode("utf-8")]]) 
            else:
                for word_bytes in iter_pretokenize(part):
                    # Make any word in text a list of int.
                    token_groups.append([self.stoi[bytes([b])] for b in word_bytes])
                    
        words, wcnt, pair_cnts, pair_occs = self._count_words_and_pairs(token_groups)
                    
        heap = []
        for p, f in pair_cnts.items():
            self._heap_push(heap, p, f)

        while len(self.stoi) < vocab_size and heap:
            pair = self._heap_pop(heap, pair_cnts)
            if pair is None:
                break

            i1, i2 = pair
            b1, b2 = self.itos[i1], self.itos[i2]
            self.merges.append((b1, b2))
            new_token_byte = b1 + b2
            if new_token_byte not in self.stoi:
                new_token_id = len(self.itos)
                self.stoi[new_token_byte] = new_token_id
                self.itos[new_token_id] = new_token_byte
            else:
                new_token_id = self.stoi[new_token_byte]
                
            for wid in list(pair_occs.get(pair, ())):
                old, c = words[wid], wcnt[wid]
                # remove old pairs
                for p in zip(old[:-1], old[1:]):
                    if p in pair_cnts:
                        pair_cnts[p] -= c
                        if pair_cnts[p] <= 0: pair_cnts.pop(p, None)
                        else: self._heap_push(heap, p, pair_cnts[p])
                    if p in pair_occs:
                        pair_occs[p].discard(wid)
                        if not pair_occs[p]: pair_occs.pop(p, None)
                        
                # apply merge on this word
                new = self._merge_pair_in_word(old, pair, new_token_id)
                words[wid] = tuple(new)
                
                # add new pairs to dicts
                if len(new) >= 2:
                    for new_p in zip(new[:-1], new[1:]):
                        pair_cnts[new_p] += c
                        self._heap_push(heap, new_p, pair_cnts[new_p])
                        pair_occs[new_p].add(wid)
            pair_occs.pop(pair, None)
            
        # Update exposed vocab mapping for external callers
        self.vocab = self.itos.copy()
        return self.itos, self.merges    # self.itos is the vocabularies.

    # ---------- tokenizer construction from GPT-2 files ----------

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: List[str] | None = None,
    ) -> "BPETokenizer":
        import json

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab_bytes: Dict[int, bytes] = {
            int(idx): bytes([_UNICODE_TO_BYTES[ch] for ch in token])
            for token, idx in gpt2_vocab.items()
        }

        # Add specials if missing
        if special_tokens:
            existing = set(vocab_bytes.values())
            for tok in special_tokens:
                b = tok.encode("utf-8")
                if b not in existing:
                    vocab_bytes[len(vocab_bytes)] = b
                    existing.add(b)

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip()
                if not cleaned:
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                a, b = parts
                ab = bytes([_UNICODE_TO_BYTES[ch] for ch in a])
                bb = bytes([_UNICODE_TO_BYTES[ch] for ch in b])
                merges.append((ab, bb))

        return cls(special_tokens=special_tokens, vocab=vocab_bytes, merges=merges)

    # ---------- encoding / decoding ----------

    def _ensure_merges_rank(self):
        if not hasattr(self, "merges_rank"):
            self.merges_rank = {}
        if not self.merges_rank and self.merges:
            self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}

    def _split_by_special(self, text: str) -> List[str]:
        if not self.special_tokens:
            return [text]
        specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
        pattern = f"({'|'.join(re.escape(s) for s in specials_sorted)})"
        parts = re.split(pattern, text)
        return [p for p in parts if p != ""]

    def _bpe_merge_word(self, word_bytes: bytes) -> List[bytes]:
        seq: List[bytes] = [bytes([b]) for b in word_bytes]
        if len(seq) <= 1:
            return seq
        self._ensure_merges_rank()
        if not self.merges_rank:
            return seq
        while True:
            best_rank = None
            best_pair = None
            for x, y in zip(seq[:-1], seq[1:]):
                r = self.merges_rank.get((x, y))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, (x, y)
            if best_pair is None:
                break
            merged: List[bytes] = []
            i = 0
            L = len(seq)
            while i < L:
                if i + 1 < L and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                    merged.append(seq[i] + seq[i + 1])
                    i += 2
                else:
                    merged.append(seq[i])
                    i += 1
            seq = merged
            if len(seq) <= 1:
                break
        return seq

    def encode(self, text: str) -> List[int]:
        out_ids: List[int] = []
        specials = set(self.special_tokens)
        for part in self._split_by_special(text):
            if part in specials:
                out_ids.append(self.stoi[part.encode("utf-8")])
            else:
                for m in GPT2_RE.finditer(part):
                    w = m.group(0).encode("utf-8")
                    for tok in self._bpe_merge_word(w):
                        out_ids.append(self.stoi[tok])
        return out_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        specials = set(self.special_tokens)
        for chunk in iterable:
            for part in self._split_by_special(chunk):
                if part in specials:
                    yield self.stoi[part.encode("utf-8")]
                else:
                    for m in GPT2_RE.finditer(part):
                        w = m.group(0).encode("utf-8")
                        for tok in self._bpe_merge_word(w):
                            yield self.stoi[tok]

    def decode(self, ids: List[int]) -> str:
        b = b"".join(self.itos[i] for i in ids)
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback: map each byte through GPT-2 unicode mapping to ensure no error
            return "".join(_BYTES_TO_UNICODE[byte] for byte in b)
