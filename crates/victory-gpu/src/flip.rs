//! Two arms compared in one launch: every second of frames is one window, logged as
//! `flip <arm> <skip> ...`, and the arm changes every `VICTORY_FLIP=<windows>` of them in the
//! order ABBA, so a steady drift of the scene cancels. The first window after a change is marked
//! (`skip` 1). A switch takes part when its value is a pair, `VICTORY_X="250|20"`: the first
//! value under arm 0, the second under arm 1 ([`Pair`]). Off without `VICTORY_FLIP`.
// shortcut: an arm changes between two presented frames, with work of the old arm still in
// flight; what a switch leaves behind (a queue, a cache) reaches the other arm's first windows.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Windows seen, and the arm in force.
static WINDOWS: AtomicU32 = AtomicU32::new(0);
static ARM: AtomicU32 = AtomicU32::new(0);

/// Windows per block; 0: no comparison.
fn every() -> u32 {
    static EVERY: OnceLock<u32> = OnceLock::new();
    *EVERY.get_or_init(|| {
        let value = std::env::var("VICTORY_FLIP").ok();
        value.and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    })
}

/// The arm of window `window` in blocks of `every`: ABBA.
fn arm(window: u32, every: u32) -> u32 {
    let block = window / every;
    u32::from(matches!(block % 4, 1 | 2))
}

/// The end of a window; `line` gives its numbers.
pub(crate) fn window(line: impl FnOnce() -> String) {
    let every = every();
    if every == 0 {
        return;
    }
    let n = WINDOWS.fetch_add(1, Relaxed);
    let skip = n == 0 || arm(n, every) != arm(n - 1, every);
    tracing::info!("flip {} {} {}", arm(n, every), u32::from(skip), line());
    ARM.store(arm(n.wrapping_add(1), every), Relaxed);
}

/// An environment variable that may hold a pair of values.
pub(crate) struct Pair {
    name: &'static str,
    values: OnceLock<Option<[String; 2]>>,
}

impl Pair {
    pub(crate) const fn new(name: &'static str) -> Self {
        Pair {
            name,
            values: OnceLock::new(),
        }
    }

    /// The value now in force.
    pub(crate) fn get(&self) -> Option<&str> {
        let values = self
            .values
            .get_or_init(|| std::env::var(self.name).ok().map(|v| split(&v)));
        let arm = ARM.load(Relaxed) as usize & 1;
        values.as_ref().map(|v| v[arm].as_str())
    }

    /// The value now in force as a number, if it is one.
    pub(crate) fn parsed<T: std::str::FromStr>(&self) -> Option<T> {
        self.get().and_then(|v| v.trim().parse().ok())
    }
}

/// The two arms' values of `value`: `a|b`, or the one value for both.
fn split(value: &str) -> [String; 2] {
    let (a, b) = value.split_once('|').unwrap_or((value, value));
    [a.to_owned(), b.to_owned()]
}

#[cfg(test)]
mod tests {
    #[test]
    fn arms_run_abba() {
        let arms: Vec<u32> = (0..10).map(|w| super::arm(w, 2)).collect();
        assert_eq!(arms, [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn a_pair_gives_each_arm_its_value() {
        assert_eq!(super::split("250|20"), ["250", "20"]);
        assert_eq!(super::split("1"), ["1", "1"]);
    }
}
