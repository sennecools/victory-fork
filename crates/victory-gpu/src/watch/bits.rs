//! Sets of 64 KiB blocks for the guards, as bits.
//!
//! - [`Bits`]: a bit per block of the address space, read with no lock, for
//!   what the guards publish to the askings and to the fault handler
//!   (`super::PUBLISHED`, `super::PROTECTED`).
//! - [`Blocks`]: the guards' own sets (`super::Guards`), held under their
//!   lock: words of bits, with a summary per 4 GiB of which words have a
//!   block and which are full, so a fault or a look at a range of thousands
//!   of blocks is a few words.
//!
//! Ported from cool_fox 2aa8134, 74045e4 (`Bits`) and 7bc7c07 (`Blocks`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// A bit per block of the address space, in pieces made when first needed
/// and never let go.
pub(super) struct Bits([AtomicPtr<BitsPiece>; Bits::PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [AtomicU64; Bits::WORDS];

impl Bits {
    const WORDS: usize = 1 << 10;
    /// 128 TiB, what a process has.
    const PIECES: usize = 1 << 15;

    pub(super) const fn new() -> Bits {
        Bits([const { AtomicPtr::new(std::ptr::null_mut()) }; Bits::PIECES])
    }

    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / Bits::WORDS)?.load(Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    /// One caller at a time (under the guards' lock).
    pub(super) fn set(&self, block: usize) {
        let word = block / 64;
        let Some(place) = self.0.get(word / Bits::WORDS) else {
            return;
        };
        if place.load(Ordering::Relaxed).is_null() {
            let piece: Box<BitsPiece> = Box::new([const { AtomicU64::new(0) }; Bits::WORDS]);
            place.store(Box::into_raw(piece), Ordering::Release);
        }
        if let Some(piece) = self.piece(word) {
            piece[word % Bits::WORDS].fetch_or(1 << (block % 64), Ordering::SeqCst);
        }
    }

    pub(super) fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            // Ahead of the protection's change, and seen by whoever the
            // writer tells of its write.
            piece[word % Bits::WORDS].fetch_and(!(1 << (block % 64)), Ordering::SeqCst);
        }
    }

    /// Whether a bit of `[first, last)` is set.
    pub(super) fn any(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            match self.piece(word) {
                Some(piece) => {
                    if piece[word % Bits::WORDS].load(Ordering::SeqCst) & mask != 0 {
                        return true;
                    }
                    block = until;
                }
                // None of the piece's.
                None => block = (word / Bits::WORDS + 1) * Bits::WORDS * 64,
            }
        }
        false
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    pub(super) fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % Bits::WORDS].load(Ordering::SeqCst) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }
}

/// A set of blocks: words of bits in pieces of 4 GiB, made when a block
/// there is first put in and kept, each with a summary of its words.
pub(super) struct Blocks {
    pieces: BTreeMap<usize, Box<Piece>>,
    len: usize,
}

/// The words of 4 GiB of blocks, and a bit per word: whether it has a block
/// (`any`), whether it has all 64 (`full`).
struct Piece {
    words: [u64; Blocks::WORDS],
    any: [u64; Blocks::WORDS / 64],
    full: [u64; Blocks::WORDS / 64],
}

impl Piece {
    /// Word `k` set to `bits`, its summary bits with it.
    fn store(&mut self, k: usize, bits: u64) {
        self.words[k] = bits;
        let (s, bit) = (k / 64, 1u64 << (k % 64));
        if bits != 0 {
            self.any[s] |= bit;
        } else {
            self.any[s] &= !bit;
        }
        if bits == u64::MAX {
            self.full[s] |= bit;
        } else {
            self.full[s] &= !bit;
        }
    }
}

impl Blocks {
    /// The words of 4 GiB.
    const WORDS: usize = 1 << 10;
    const IN_PIECE: usize = Blocks::WORDS * 64;

    pub(super) const fn new() -> Blocks {
        Blocks {
            pieces: BTreeMap::new(),
            len: 0,
        }
    }

    /// The 64 bits of blocks `[word * 64, word * 64 + 64)`.
    pub(super) fn word(&self, word: usize) -> u64 {
        self.pieces
            .get(&(word / Blocks::WORDS))
            .map_or(0, |piece| piece.words[word % Blocks::WORDS])
    }

    pub(super) fn contains(&self, block: usize) -> bool {
        self.word(block / 64) >> (block % 64) & 1 != 0
    }

    /// Whether it was not there.
    pub(super) fn insert(&mut self, block: usize) -> bool {
        let piece = self
            .pieces
            .entry(block / Blocks::IN_PIECE)
            .or_insert_with(|| {
                Box::new(Piece {
                    words: [0; Blocks::WORDS],
                    any: [0; Blocks::WORDS / 64],
                    full: [0; Blocks::WORDS / 64],
                })
            });
        let k = block % Blocks::IN_PIECE / 64;
        let was = piece.words[k];
        piece.store(k, was | 1 << (block % 64));
        let new = was != piece.words[k];
        self.len += usize::from(new);
        new
    }

    /// Whether it was there.
    pub(super) fn remove(&mut self, block: usize) -> bool {
        let Some(piece) = self.pieces.get_mut(&(block / Blocks::IN_PIECE)) else {
            return false;
        };
        let k = block % Blocks::IN_PIECE / 64;
        let was = piece.words[k];
        piece.store(k, was & !(1 << (block % 64)));
        let gone = was != piece.words[k];
        self.len -= usize::from(gone);
        gone
    }

    pub(super) fn insert_all(&mut self, blocks: std::ops::Range<usize>) {
        for block in blocks {
            self.insert(block);
        }
    }

    /// The words of `[first, last)` that have a block, in order: the word's
    /// number and its bits in the range. Words with none are skipped by the
    /// summary.
    fn words_in(&self, first: usize, last: usize) -> impl Iterator<Item = (usize, u64)> + '_ {
        let pieces = if first < last {
            self.pieces
                .range(first / Blocks::IN_PIECE..=(last - 1) / Blocks::IN_PIECE)
        } else {
            self.pieces.range(1..1)
        };
        pieces.flat_map(move |(&piece, words)| {
            let base = piece * Blocks::IN_PIECE;
            let from = first.max(base) - base;
            let to = last.min(base + Blocks::IN_PIECE) - base;
            let (lo, hi) = (from / 64, to.div_ceil(64));
            (lo / 64..hi.div_ceil(64)).flat_map(move |s| {
                let mask = in_word(s * 64, lo, hi);
                BitsOf(words.any[s] & mask).map(move |bit| {
                    let k = s * 64 + bit;
                    let bits = words.words[k] & in_word(k * 64, from, to);
                    (piece * Blocks::WORDS + k, bits)
                })
            })
        })
    }

    /// Those of `[first, last)`, in order.
    pub(super) fn iter(&self, first: usize, last: usize) -> impl Iterator<Item = usize> + '_ {
        self.words_in(first, last)
            .flat_map(|(word, bits)| BitsOf(bits).map(move |bit| word * 64 + bit))
    }

    /// Takes out those of `[first, last)`, telling `gone` of each, in order.
    pub(super) fn remove_range(&mut self, first: usize, last: usize, mut gone: impl FnMut(usize)) {
        if first >= last {
            return;
        }
        let mut taken = 0;
        for (&piece, words) in self
            .pieces
            .range_mut(first / Blocks::IN_PIECE..=(last - 1) / Blocks::IN_PIECE)
        {
            let base = piece * Blocks::IN_PIECE;
            let from = first.max(base) - base;
            let to = last.min(base + Blocks::IN_PIECE) - base;
            let (lo, hi) = (from / 64, to.div_ceil(64));
            for s in lo / 64..hi.div_ceil(64) {
                for bit in BitsOf(words.any[s] & in_word(s * 64, lo, hi)) {
                    let k = s * 64 + bit;
                    let out = words.words[k] & in_word(k * 64, from, to);
                    words.store(k, words.words[k] & !out);
                    taken += out.count_ones() as usize;
                    for b in BitsOf(out) {
                        gone(base + k * 64 + b);
                    }
                }
            }
        }
        self.len -= taken;
    }

    /// The first `n` from `first` on, in order.
    pub(super) fn first_from(&self, first: usize, n: usize) -> Vec<usize> {
        self.iter(first, usize::MAX).take(n).collect()
    }

    /// How many of `[first, last)` there are.
    pub(super) fn count(&self, first: usize, last: usize) -> usize {
        self.words_in(first, last)
            .map(|(_, bits)| bits.count_ones() as usize)
            .sum()
    }

    /// Whether all of `[first, last)` are there (true of an empty range).
    pub(super) fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let piece = block / Blocks::IN_PIECE;
            let Some(words) = self.pieces.get(&piece) else {
                return false;
            };
            let base = piece * Blocks::IN_PIECE;
            let to = last.min(base + Blocks::IN_PIECE) - base;
            let mut at = block - base;
            while at < to {
                let k = at / 64;
                // Whole full words at once, by the summary.
                if at.is_multiple_of(64)
                    && k.is_multiple_of(64)
                    && to - at >= 64 * 64
                    && words.full[k / 64] == u64::MAX
                {
                    at += 64 * 64;
                    continue;
                }
                let mask = in_word(k * 64, at, to);
                if words.words[k] & mask != mask {
                    return false;
                }
                at = (k + 1) * 64;
            }
            block = base + to;
        }
        true
    }

    /// Of `[first, last)`, the blocks in neither `self` nor `other`, in
    /// order, into `out`. Words full in either are skipped by the summaries.
    pub(super) fn in_neither(
        &self,
        other: &Blocks,
        first: usize,
        last: usize,
        out: &mut Vec<usize>,
    ) {
        static EMPTY: Piece = Piece {
            words: [0; Blocks::WORDS],
            any: [0; Blocks::WORDS / 64],
            full: [0; Blocks::WORDS / 64],
        };
        let mut block = first;
        while block < last {
            let piece = block / Blocks::IN_PIECE;
            let base = piece * Blocks::IN_PIECE;
            let to = last.min(base + Blocks::IN_PIECE) - base;
            let a = self.pieces.get(&piece).map_or(&EMPTY, |p| &**p);
            let b = other.pieces.get(&piece).map_or(&EMPTY, |p| &**p);
            let from = block - base;
            let (lo, hi) = (from / 64, to.div_ceil(64));
            for s in lo / 64..hi.div_ceil(64) {
                for bit in BitsOf(!(a.full[s] | b.full[s]) & in_word(s * 64, lo, hi)) {
                    let k = s * 64 + bit;
                    let unknown = !(a.words[k] | b.words[k]) & in_word(k * 64, from, to);
                    out.extend(BitsOf(unknown).map(|n| base + k * 64 + n));
                }
            }
            block = base + to;
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }
}

/// Of the 64 blocks from `at` on, those in `[from, to)`, a bit each.
pub(super) fn in_word(at: usize, from: usize, to: usize) -> u64 {
    let low = from.saturating_sub(at).min(64);
    let high = to.saturating_sub(at).min(64);
    if low >= high {
        0
    } else {
        (u64::MAX >> (64 - (high - low))) << low
    }
}

/// The bits of a word that are set, by number.
pub(super) struct BitsOf(pub(super) u64);

impl Iterator for BitsOf {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        let bit = (self.0 != 0).then(|| self.0.trailing_zeros() as usize)?;
        self.0 &= self.0 - 1;
        Some(bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A range's bits are all of its blocks', whatever lies beside them.
    #[test]
    fn bits_of_ranges() {
        static BITS: Bits = Bits::new();
        let blocks = [
            0usize,
            1,
            63,
            64,
            65,
            127,
            128,
            200,
            65535,
            65536,
            65537,
            1 << 20,
            (1 << 31) - 1,
        ];
        for &b in &blocks {
            assert!(!BITS.all(b, b + 1) && !BITS.any(b, b + 1));
            BITS.set(b);
            assert!(BITS.all(b, b + 1) && BITS.any(b, b + 1));
        }
        assert!(BITS.all(0, 2) && BITS.all(63, 66) && BITS.all(127, 129) && BITS.all(65535, 65538));
        assert!(!BITS.all(0, 3) && !BITS.all(62, 64) && !BITS.all(65534, 65536));
        assert!(!BITS.all(5, 5) && !BITS.any(5, 5) && !BITS.any(2, 63) && BITS.any(2, 64));
        for b in 300..700 {
            BITS.set(b);
        }
        assert!(BITS.all(300, 700) && !BITS.all(299, 700) && !BITS.all(300, 701));
        BITS.clear(511);
        assert!(!BITS.all(300, 700) && BITS.all(300, 511) && BITS.all(512, 700));
        assert!(!BITS.any(511, 512));
        assert!(
            !BITS.all(1 << 40, (1 << 40) + 1),
            "outside the address space"
        );
        assert!(!BITS.any(1 << 40, (1 << 40) + 1));
    }

    /// The set of blocks against another of the same.
    #[test]
    fn blocks_are_a_set() {
        let (mut set, mut other) = (Blocks::new(), BTreeSet::new());
        let mut seed = 1u64;
        let mut next = |n: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        let near = [0, 63, Blocks::IN_PIECE, 3 * Blocks::IN_PIECE - 70];
        for round in 0..40_000 {
            // About the ends of pieces and words.
            let block = near[next(4)] + next(140);
            if next(3) == 0 {
                assert_eq!(set.remove(block), other.remove(&block));
            } else {
                assert_eq!(set.insert(block), other.insert(block));
            }
            assert_eq!(set.contains(block), other.contains(&block));
            if round % 64 == 0 {
                let a = near[next(4)] + next(140);
                let b = a + [next(10), next(300), 2 * Blocks::IN_PIECE + next(100)][next(3)];
                let want: Vec<usize> = other.range(a..b).copied().collect();
                assert_eq!(set.iter(a, b).collect::<Vec<_>>(), want, "{a}..{b}");
                assert_eq!(set.count(a, b), want.len(), "{a}..{b}");
                assert_eq!(
                    set.iter(0, usize::MAX).collect::<Vec<_>>(),
                    other.iter().copied().collect::<Vec<_>>()
                );
                assert_eq!(set.len(), other.len());
                let word = a / 64;
                let bits = other
                    .range(word * 64..word * 64 + 64)
                    .fold(0u64, |w, &b| w | 1 << (b - word * 64));
                assert_eq!(set.word(word), bits);
            }
        }
        assert_eq!(
            set.iter(5, usize::MAX).next().as_ref(),
            other.range(5..).next()
        );
        for from in [0, 5, 64, Blocks::IN_PIECE - 3, 3 * Blocks::IN_PIECE] {
            for n in [0, 1, 7, 1000] {
                let want: Vec<usize> = other.range(from..).take(n).copied().collect();
                assert_eq!(set.first_from(from, n), want, "{from} {n}");
            }
        }
    }

    /// The summaries of the words follow them: whole ranges filled and
    /// emptied across words and pieces answer `all`, `count`, `iter`,
    /// `in_neither` and `remove_range` as a plain set does, including after
    /// a word goes empty or stops being full.
    #[test]
    fn blocks_summary_matches_words() {
        let (mut set, mut other) = (Blocks::new(), BTreeSet::new());
        let (mut beside, mut beside_other) = (Blocks::new(), BTreeSet::new());
        let mut seed = 7u64;
        let mut next = |n: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        // About the end of a summary word (4096 blocks) and of a piece.
        let near = [0, 4096 - 70, Blocks::IN_PIECE - 5000, 2 * Blocks::IN_PIECE];
        for round in 0..3000 {
            let a = near[next(4)] + next(9000);
            let b = a + [next(3), next(130), next(9000)][next(3)];
            match next(5) {
                0 => {
                    let mut gone = Vec::new();
                    set.remove_range(a, b, |block| gone.push(block));
                    let want: Vec<usize> = other.range(a..b).copied().collect();
                    for block in &want {
                        other.remove(block);
                    }
                    assert_eq!(gone, want, "round {round}: removed {a}..{b}");
                }
                1 => {
                    let block = a + next(b - a + 1);
                    assert_eq!(beside.insert(block), beside_other.insert(block));
                }
                2 => {
                    for block in a..b {
                        if next(16) != 0 {
                            assert_eq!(set.remove(block), other.remove(&block));
                        }
                    }
                }
                _ => {
                    set.insert_all(a..b);
                    other.extend(a..b);
                }
            }
            let (a, b) = {
                let a = near[next(4)] + next(9000);
                (a, a + [next(3), next(130), next(9000)][next(3)])
            };
            let want: Vec<usize> = other.range(a..b).copied().collect();
            assert_eq!(
                set.iter(a, b).collect::<Vec<_>>(),
                want,
                "round {round}: {a}..{b}"
            );
            assert_eq!(set.count(a, b), want.len(), "round {round}: {a}..{b}");
            assert_eq!(
                set.all(a, b),
                want.len() == b - a,
                "round {round}: all {a}..{b}"
            );
            assert_eq!(set.len(), other.len(), "round {round}");
            let mut neither = Vec::new();
            set.in_neither(&beside, a, b, &mut neither);
            let want: Vec<usize> = (a..b)
                .filter(|b| !other.contains(b) && !beside_other.contains(b))
                .collect();
            assert_eq!(neither, want, "round {round}: in neither {a}..{b}");
        }
        assert!(Blocks::new().all(5, 5) && !Blocks::new().all(5, 6));
    }
}
