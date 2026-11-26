//! Sets of 64 KiB blocks for the guards, as bits.
//!
//! - [`Bits`]: a bit per block of the address space, read with no lock, for
//!   what the guards publish to the askings and to the fault handler
//!   (`VICTORY_LIGHT_REFRESH=1`, `super::PUBLISHED`, `super::PROTECTED`).
//! - [`Blocks`]: the guards' own sets (`super::Guards`), held under their
//!   lock. With `VICTORY_GUARD_BITS=1` a set is words of bits, so a fault or
//!   a look at a range of thousands of blocks is a few words; without it the
//!   ordered tree it was before.
//!
//! Ported from cool_fox 2aa8134, 74045e4 (`Bits`) and 7bc7c07 (`Blocks`).

use std::collections::{BTreeMap, BTreeSet};
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

/// Whether the guards' sets are words of bits (`VICTORY_GUARD_BITS=1`; off
/// by default: ordered trees, as before).
///
/// cool_fox measured, before its change, the title's write faults waiting
/// for the guards' lock most of their time, and a look at a range walking a
/// tree of every guarded block: 586 faults a frame took 69 ms of the title's
/// threads. The same switch has a refresh let go of the lock between the
/// runs of blocks it guards (`super::guard`).
pub(super) fn guard_bits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_BITS").is_some_and(|v| v == "1"))
}

/// A set of blocks: words of bits with [`guard_bits`], an ordered tree
/// without. One of the two is used for the whole run.
pub(super) struct Blocks {
    tree: BTreeSet<usize>,
    words: BTreeMap<usize, Box<[u64; Blocks::WORDS]>>,
    len: usize,
}

impl Blocks {
    /// The words of 4 GiB.
    const WORDS: usize = 1 << 10;
    const IN_PIECE: usize = Blocks::WORDS * 64;

    pub(super) const fn new() -> Blocks {
        Blocks {
            tree: BTreeSet::new(),
            words: BTreeMap::new(),
            len: 0,
        }
    }

    /// The 64 bits of blocks `[word * 64, word * 64 + 64)`.
    pub(super) fn word(&self, word: usize) -> u64 {
        if guard_bits() {
            return self
                .words
                .get(&(word / Blocks::WORDS))
                .map_or(0, |piece| piece[word % Blocks::WORDS]);
        }
        let base = word * 64;
        self.tree
            .range(base..base.saturating_add(64))
            .fold(0, |bits, &block| bits | 1 << (block - base))
    }

    pub(super) fn contains(&self, block: usize) -> bool {
        if guard_bits() {
            self.word(block / 64) >> (block % 64) & 1 != 0
        } else {
            self.tree.contains(&block)
        }
    }

    /// Whether it was not there.
    pub(super) fn insert(&mut self, block: usize) -> bool {
        let new = if guard_bits() {
            let piece = self
                .words
                .entry(block / Blocks::IN_PIECE)
                .or_insert_with(|| Box::new([0; Blocks::WORDS]));
            let word = &mut piece[block % Blocks::IN_PIECE / 64];
            let was = *word;
            *word |= 1 << (block % 64);
            was != *word
        } else {
            self.tree.insert(block)
        };
        self.len += usize::from(new);
        new
    }

    /// Whether it was there.
    pub(super) fn remove(&mut self, block: usize) -> bool {
        let was = if guard_bits() {
            let Some(piece) = self.words.get_mut(&(block / Blocks::IN_PIECE)) else {
                return false;
            };
            let word = &mut piece[block % Blocks::IN_PIECE / 64];
            let was = *word;
            *word &= !(1 << (block % 64));
            was != *word
        } else {
            self.tree.remove(&block)
        };
        self.len -= usize::from(was);
        was
    }

    pub(super) fn insert_all(&mut self, blocks: std::ops::Range<usize>) {
        for block in blocks {
            self.insert(block);
        }
    }

    /// Those of `[first, last)`, in order.
    pub(super) fn range(&self, first: usize, last: usize) -> Vec<usize> {
        let mut all = Vec::new();
        if first >= last {
            return all;
        }
        if !guard_bits() {
            all.extend(self.tree.range(first..last).copied());
            return all;
        }
        for (&piece, words) in self
            .words
            .range(first / Blocks::IN_PIECE..=(last - 1) / Blocks::IN_PIECE)
        {
            let base = piece * Blocks::IN_PIECE;
            let from = first.max(base) - base;
            let to = last.min(base + Blocks::IN_PIECE) - base;
            for (k, &bits) in words[from / 64..to.div_ceil(64)].iter().enumerate() {
                let at = (from / 64 + k) * 64;
                all.extend(BitsOf(bits & in_word(at, from, to)).map(|bit| base + at + bit));
            }
        }
        all
    }

    /// The first `n` from `first` on, in order.
    pub(super) fn first_from(&self, first: usize, n: usize) -> Vec<usize> {
        if !guard_bits() {
            return self.tree.range(first..).take(n).copied().collect();
        }
        let mut all = Vec::new();
        for (&piece, words) in self.words.range(first / Blocks::IN_PIECE..) {
            let base = piece * Blocks::IN_PIECE;
            let from = first.max(base) - base;
            for (k, &bits) in words[from / 64..].iter().enumerate() {
                let at = (from / 64 + k) * 64;
                for bit in BitsOf(bits & in_word(at, from, Blocks::IN_PIECE)) {
                    if all.len() == n {
                        return all;
                    }
                    all.push(base + at + bit);
                }
            }
        }
        all
    }

    /// How many of `[first, last)` there are.
    pub(super) fn count(&self, first: usize, last: usize) -> usize {
        if first >= last {
            return 0;
        }
        if !guard_bits() {
            return self.tree.range(first..last).count();
        }
        (first / 64..last.div_ceil(64))
            .map(|word| (self.word(word) & in_word(word * 64, first, last)).count_ones() as usize)
            .sum()
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
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

    /// The set of blocks against another of the same, whichever way
    /// `VICTORY_GUARD_BITS` keeps it.
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
                assert_eq!(set.range(a, b), want, "{a}..{b}");
                assert_eq!(set.count(a, b), want.len(), "{a}..{b}");
                assert_eq!(
                    set.range(0, usize::MAX),
                    other.iter().copied().collect::<Vec<_>>()
                );
                assert_eq!(set.len(), other.len());
                assert_eq!(set.is_empty(), other.is_empty());
                let word = a / 64;
                let bits = other
                    .range(word * 64..word * 64 + 64)
                    .fold(0u64, |w, &b| w | 1 << (b - word * 64));
                assert_eq!(set.word(word), bits);
            }
        }
        assert_eq!(set.range(5, usize::MAX).first(), other.range(5..).next());
        for from in [0, 5, 64, Blocks::IN_PIECE - 3, 3 * Blocks::IN_PIECE] {
            for n in [0, 1, 7, 1000] {
                let want: Vec<usize> = other.range(from..).take(n).copied().collect();
                assert_eq!(set.first_from(from, n), want, "{from} {n}");
            }
        }
    }
}
