//! Pipelines whose creation ends the process (`VICTORY_PIPELINE_JOURNAL`).
//!
//! Some host drivers crash while compiling a translated shader (under Wine,
//! a fault inside `vkCreate*Pipelines` ends the process with exit code 3,
//! before any handler of ours runs). Such a crash cannot be caught, so it
//! is remembered instead: each pipeline is noted in `pending/` while it is
//! created and the note is removed when creation returns.
//!
//! Pipelines are created in parallel. Notes left over at the next start name
//! the pipelines that were being created when the process ended:
//!
//! - one note names the pipeline that crashed. It goes into `crashed.txt` and
//!   is refused from then on, its draws and dispatches dropped like those of
//!   any refused pipeline;
//! - several notes name suspects, of which at most one crashed. They go into
//!   `suspects.txt` and are not refused. Each is created alone (the others
//!   wait), so if it crashes again it leaves one note and is named. A suspect
//!   whose creation returns leaves the list.
//!
//! A line deleted from `crashed.txt` lets that pipeline be tried again.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

static DIR: OnceLock<PathBuf> = OnceLock::new();

struct Journal {
    pending: PathBuf,
    crashed_file: PathBuf,
    suspects_file: PathBuf,
    /// Pipeline keys (`hash` or `hash+hash`) refused, and those logged.
    crashed: HashSet<String>,
    logged: Mutex<HashSet<String>>,
    /// Pipelines that were being created alongside others when a run ended.
    suspects: Mutex<HashSet<String>>,
    /// Shared while an ordinary pipeline is created, exclusive for a suspect.
    gate: RwLock<()>,
    next: AtomicU64,
}

/// Keeps the journal in `dir` (a folder of the cache).
pub(crate) fn use_dir(dir: PathBuf) {
    let _ = DIR.set(dir);
}

fn journal() -> Option<&'static Journal> {
    static JOURNAL: OnceLock<Option<Journal>> = OnceLock::new();
    JOURNAL
        .get_or_init(|| {
            std::env::var_os("VICTORY_PIPELINE_JOURNAL").filter(|v| !v.is_empty())?;
            let Some(dir) = DIR.get() else {
                tracing::warn!("VICTORY_PIPELINE_JOURNAL: no cache folder; journal off");
                return None;
            };
            Journal::open(dir)
        })
        .as_ref()
}

fn read_list(path: &Path) -> HashSet<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

fn write_list<'a>(path: &Path, keys: impl IntoIterator<Item = &'a String>) {
    let mut keys: Vec<&String> = keys.into_iter().collect();
    keys.sort();
    let text: String = keys.into_iter().map(|k| format!("{k}\n")).collect();
    if let Err(e) = std::fs::write(path, text) {
        tracing::warn!(error = %e, path = %path.display(), "pipeline journal list not saved");
    }
}

impl Journal {
    /// The journal in `dir`, with the notes a run that ended while creating
    /// pipelines left sorted into `crashed.txt` or `suspects.txt`.
    fn open(dir: &Path) -> Option<Journal> {
        let pending = dir.join("pending");
        let crashed_file = dir.join("crashed.txt");
        let suspects_file = dir.join("suspects.txt");
        if let Err(e) = std::fs::create_dir_all(&pending) {
            tracing::warn!(error = %e, path = %pending.display(), "VICTORY_PIPELINE_JOURNAL: journal off");
            return None;
        }
        let mut crashed = read_list(&crashed_file);
        let mut suspects = read_list(&suspects_file);
        let mut left = Vec::new();
        for entry in std::fs::read_dir(&pending).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let key = name.split('@').next().unwrap_or(&name).to_owned();
            if !crashed.contains(&key) && !left.contains(&key) {
                left.push(key);
            }
            let _ = std::fs::remove_file(entry.path());
        }
        match left.len() {
            0 => {}
            1 => {
                let key = left.pop().unwrap();
                tracing::warn!(
                    pipeline = %key,
                    "the last run ended while creating this host pipeline; it is refused from now on"
                );
                suspects.remove(&key);
                write_list(&suspects_file, &suspects);
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&crashed_file);
                match file {
                    Ok(mut f) => {
                        let _ = writeln!(f, "{key}");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, path = %crashed_file.display(), "crashed pipeline not saved")
                    }
                }
                crashed.insert(key);
            }
            _ => {
                tracing::warn!(
                    pipelines = ?left,
                    "the last run ended while creating these host pipelines together; each is created alone from now on until it succeeds"
                );
                suspects.extend(left);
                write_list(&suspects_file, &suspects);
            }
        }
        tracing::info!(
            refused = crashed.len(),
            suspects = suspects.len(),
            path = %crashed_file.display(),
            "pipeline journal on"
        );
        Some(Journal {
            pending,
            crashed_file,
            suspects_file,
            crashed,
            logged: Mutex::new(HashSet::new()),
            suspects: Mutex::new(suspects),
            gate: RwLock::new(()),
            next: AtomicU64::new(0),
        })
    }

    fn enter(&self, key: &str) -> Option<Pending<'_>> {
        if self.crashed.contains(key) {
            let mut logged = self.logged.lock().unwrap_or_else(|e| e.into_inner());
            if logged.insert(key.to_owned()) {
                tracing::warn!(
                    pipeline = key,
                    list = %self.crashed_file.display(),
                    "host pipeline skipped: creating it ended an earlier run"
                );
            }
            return None;
        }
        let suspect = self
            .suspects
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(key);
        let gate = if suspect {
            Gate::Alone(self.gate.write().unwrap_or_else(|e| e.into_inner()))
        } else {
            Gate::Shared(self.gate.read().unwrap_or_else(|e| e.into_inner()))
        };
        let path = self.pending.join(format!(
            "{key}@{}-{}",
            std::process::id(),
            self.next.fetch_add(1, Ordering::Relaxed)
        ));
        let note = match std::fs::File::create(&path) {
            Ok(_) => Some(path),
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "pipeline journal note not written");
                None
            }
        };
        Some(Pending {
            note,
            cleared: suspect.then(|| (self, key.to_owned())),
            _gate: Some(gate),
        })
    }

    /// `key` was created without ending the process: no longer a suspect.
    fn clear(&self, key: &str) {
        let mut suspects = self.suspects.lock().unwrap_or_else(|e| e.into_inner());
        if suspects.remove(key) {
            write_list(&self.suspects_file, suspects.iter());
        }
    }
}

#[allow(dead_code)]
enum Gate<'a> {
    Shared(RwLockReadGuard<'a, ()>),
    Alone(RwLockWriteGuard<'a, ()>),
}

/// A pipeline being created; its note is removed when this is dropped,
/// then a suspect waiting to be created alone may start.
pub(crate) struct Pending<'a> {
    note: Option<PathBuf>,
    cleared: Option<(&'a Journal, String)>,
    _gate: Option<Gate<'a>>,
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if let Some(path) = &self.note {
            let _ = std::fs::remove_file(path);
        }
        if let Some((j, key)) = &self.cleared {
            j.clear(key);
        }
    }
}

/// Notes that the pipeline `key` is being created, or `None` if it ended a
/// run before (it must not be created). Always `Some` with the journal off.
pub(crate) fn enter(key: &str) -> Option<Pending<'static>> {
    match journal() {
        Some(j) => j.enter(key),
        None => Some(Pending {
            note: None,
            cleared: None,
            _gate: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_note_left_by_a_crash_refuses_that_pipeline_from_then_on() {
        let dir = tempfile::tempdir().unwrap();
        {
            let j = Journal::open(dir.path()).unwrap();
            let done = j.enter("aaaaaaaaaaaaaaaa").unwrap();
            drop(done);
            let crashing = j.enter("bbbbbbbbbbbbbbbb+cccccccccccccccc").unwrap();
            // The process ends here: the note stays.
            std::mem::forget(crashing);
        }
        let j = Journal::open(dir.path()).unwrap();
        assert!(j.enter("bbbbbbbbbbbbbbbb+cccccccccccccccc").is_none());
        assert!(j.enter("aaaaaaaaaaaaaaaa").is_some());
        let listed = std::fs::read_to_string(dir.path().join("crashed.txt")).unwrap();
        assert_eq!(listed, "bbbbbbbbbbbbbbbb+cccccccccccccccc\n");
        assert_eq!(
            std::fs::read_dir(dir.path().join("pending"))
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn notes_left_by_pipelines_created_together_make_suspects_that_run_alone() {
        let dir = tempfile::tempdir().unwrap();
        {
            let j = Journal::open(dir.path()).unwrap();
            // Two pipelines in creation at once when the process ends.
            std::mem::forget(j.enter("eeeeeeeeeeeeeeee").unwrap());
            std::mem::forget(j.enter("ffffffffffffffff").unwrap());
        }
        let j = Journal::open(dir.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("crashed.txt")).unwrap_or_default(),
            ""
        );
        let suspects = std::fs::read_to_string(dir.path().join("suspects.txt")).unwrap();
        assert_eq!(suspects, "eeeeeeeeeeeeeeee\nffffffffffffffff\n");
        // A suspect is created alone: no other pipeline may hold the gate.
        let alone = j.enter("eeeeeeeeeeeeeeee").unwrap();
        assert!(j.gate.try_read().is_err());
        drop(alone);
        // Its creation returned, so it is no longer a suspect.
        let suspects = std::fs::read_to_string(dir.path().join("suspects.txt")).unwrap();
        assert_eq!(suspects, "ffffffffffffffff\n");
        // Created alone again, it crashes: now it is named.
        std::mem::forget(j.enter("ffffffffffffffff").unwrap());
        drop(j);
        let j = Journal::open(dir.path()).unwrap();
        assert!(j.enter("ffffffffffffffff").is_none());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("suspects.txt")).unwrap(),
            ""
        );
    }

    #[test]
    fn ordinary_pipelines_are_created_together() {
        let dir = tempfile::tempdir().unwrap();
        let j = Journal::open(dir.path()).unwrap();
        let a = j.enter("1111111111111111").unwrap();
        let b = j.enter("2222222222222222").unwrap();
        drop((a, b));
    }

    #[test]
    fn a_line_removed_from_the_list_is_tried_again() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("crashed.txt"), "# tried again\n").unwrap();
        let j = Journal::open(dir.path()).unwrap();
        assert!(j.enter("dddddddddddddddd").is_some());
    }
}
