//! Where the user's files live.
//!
//! Resolution order for each value: environment variable, then the config
//! file (`VICTORY_CONFIG`, else `./victory.toml`), then the default.
//! See `victory.example.toml` at the repository root.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::policy::ensure_outside_repo;

#[derive(Debug, Default, Clone)]
pub struct Config {
    /// The user's dump of the title.
    pub game_dir: Option<PathBuf>,
    /// Root for all derived data. Always outside the repository.
    pub cache_dir: PathBuf,
    /// What titles write and must keep (persistent local storage). Never
    /// inside the cache, which may be deleted at any time.
    pub data_dir: PathBuf,
    /// Other titles used to exercise platform-level code.
    pub test_titles: Vec<PathBuf>,
    /// How big the window the title is shown in is on the host screen.
    pub window: WindowSize,
    /// The config file that was read, if any.
    pub source: Option<PathBuf>,
}

/// How big the window the title is shown in is on the primary screen, in
/// screen pixels (`[window].size`, `VICTORY_WINDOW_SIZE`): the window
/// `victory` presents in. A window the title shows itself is only shrunk
/// to fit the screen. Only the window: the title renders at the size it
/// chooses either way, and its frame is stretched to the window.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WindowSize {
    /// `"title"`: the title's size (its frame), shrunk to the screen if
    /// larger, and centred.
    #[default]
    Title,
    /// `"<width>x<height>"`: a client area of that many pixels, centred, and
    /// shrunk to the screen (keeping proportions) if it does not fit.
    Pixels { width: u32, height: u32 },
    /// `"fullscreen"`: borderless over the whole screen.
    Fullscreen,
}

/// The pixel sizes `"<width>x<height>"` accepts for each side.
const WINDOW_PIXELS: std::ops::RangeInclusive<u32> = 64..=16384;

/// As the config file spells it, so it parses back.
impl std::fmt::Display for WindowSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Title => f.write_str("title"),
            Self::Pixels { width, height } => write!(f, "{width}x{height}"),
            Self::Fullscreen => f.write_str("fullscreen"),
        }
    }
}

impl std::str::FromStr for WindowSize {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("title") {
            return Ok(Self::Title);
        }
        if text.eq_ignore_ascii_case("fullscreen") {
            return Ok(Self::Fullscreen);
        }
        // Digits only: `u32::from_str` would also take a sign.
        let number = |s: &str| {
            let s = s.trim();
            (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                .then(|| s.parse::<u32>().ok())
                .flatten()
        };
        let parsed = text
            .split_once(['x', 'X'])
            .and_then(|(w, h)| Some((number(w)?, number(h)?)));
        match parsed {
            Some((width, height))
                if WINDOW_PIXELS.contains(&width) && WINDOW_PIXELS.contains(&height) =>
            {
                Ok(Self::Pixels { width, height })
            }
            _ => anyhow::bail!(
                "window size {text:?}: use \"title\", \"fullscreen\" or \"<width>x<height>\" \
                 ({} to {} pixels each, e.g. \"1920x1080\")",
                WINDOW_PIXELS.start(),
                WINDOW_PIXELS.end()
            ),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    paths: FilePaths,
    #[serde(default)]
    window: FileWindow,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileWindow {
    size: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePaths {
    game_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    #[serde(default)]
    test_titles: Vec<PathBuf>,
}

impl Config {
    /// Loads configuration from the environment and the config file.
    pub fn load() -> Result<Self> {
        let explicit = std::env::var_os("VICTORY_CONFIG").map(PathBuf::from);
        let path = explicit
            .clone()
            .unwrap_or_else(|| PathBuf::from("victory.toml"));
        let file = if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some((path, text))
        } else if let Some(p) = explicit {
            anyhow::bail!(
                "VICTORY_CONFIG points at {}, which does not exist",
                p.display()
            );
        } else {
            None
        };
        let (source, parsed) = match file {
            Some((path, text)) => {
                let parsed: FileConfig =
                    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
                (Some(path), parsed)
            }
            None => (None, FileConfig::default()),
        };
        Self::resolve(parsed.paths, parsed.window, source, |k| std::env::var_os(k))
    }

    fn resolve(
        file: FilePaths,
        window: FileWindow,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Self> {
        let env_path = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let test_titles = match env("VICTORY_TEST_TITLES") {
            Some(v) if !v.is_empty() => std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect(),
            _ => file.test_titles,
        };
        let cache_dir = env_path("VICTORY_CACHE_DIR")
            .or(file.cache_dir)
            .map_or_else(|| default_dir("cache"), Ok)?;
        let data_dir = env_path("VICTORY_DATA_DIR")
            .or(file.data_dir)
            .map_or_else(|| default_dir("data"), Ok)?;
        let window = match env("VICTORY_WINDOW_SIZE").filter(|v| !v.is_empty()) {
            Some(v) => v.to_string_lossy().parse().context("VICTORY_WINDOW_SIZE")?,
            None => match window.size {
                Some(size) => size.parse().with_context(|| match &source {
                    Some(path) => format!("[window].size in {}", path.display()),
                    None => "[window].size".to_owned(),
                })?,
                None => WindowSize::default(),
            },
        };
        Ok(Config {
            game_dir: env_path("VICTORY_GAME_DIR").or(file.game_dir),
            cache_dir,
            data_dir,
            test_titles,
            window,
            source,
        })
    }

    /// The game directory, or a helpful error if it is not configured.
    pub fn require_game_dir(&self) -> Result<&Path> {
        self.game_dir.as_deref().context(
            "no game directory configured: set VICTORY_GAME_DIR or [paths].game_dir in \
             victory.toml (see victory.example.toml)",
        )
    }

    /// Returns `<cache>/<sub>`, creating it, after checking the asset policy.
    pub fn cache_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.cache_dir, sub)
    }

    /// Returns `<data>/<sub>`, creating it, after checking the asset policy.
    pub fn data_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.data_dir, sub)
    }
}

fn subdir(root: &Path, sub: &str) -> Result<PathBuf> {
    let dir = root.join(sub);
    ensure_outside_repo(&dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// `%LOCALAPPDATA%/victory/<name>` (elsewhere the XDG or home cache).
fn default_dir(name: &str) -> Result<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory").join(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn environment_overrides_file() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            cache_dir: Some("cache-from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(|| OsString::from("from-env"));
        let cfg = Config::resolve(file, FileWindow::default(), None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-env"));
        assert_eq!(cfg.cache_dir, PathBuf::from("cache-from-file"));
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(OsString::new);
        let cfg = Config::resolve(file, FileWindow::default(), None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-file"));
    }

    #[test]
    fn empty_test_titles_env_is_ignored() {
        let file = FilePaths {
            test_titles: vec!["from-file".into()],
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(OsString::new);
        let cfg = Config::resolve(file, FileWindow::default(), None, env).unwrap();
        assert_eq!(cfg.test_titles, vec![PathBuf::from("from-file")]);
    }

    #[test]
    fn test_titles_env_splits_paths() {
        let value = std::env::join_paths(["first", "second"]).unwrap();
        let file = FilePaths::default();
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(|| value.clone());
        let cfg = Config::resolve(file, FileWindow::default(), None, env).unwrap();
        assert_eq!(
            cfg.test_titles,
            vec![PathBuf::from("first"), PathBuf::from("second")]
        );
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../victory.example.toml");
        let parsed: FileConfig = toml::from_str(text).unwrap();
        assert!(parsed.paths.game_dir.is_some());
        // The example leaves the size commented out; the value it shows parses.
        assert_eq!(parsed.window.size, None);
        let shown = text
            .lines()
            .find_map(|line| line.trim().strip_prefix("# size = "))
            .expect("the example shows a window size");
        let shown: WindowSize = shown.trim_matches('"').parse().unwrap();
        assert_eq!(
            shown,
            WindowSize::Pixels {
                width: 1920,
                height: 1080
            }
        );
    }

    #[test]
    fn a_bad_window_size_in_the_environment_names_the_variable() {
        let env = |k: &str| (k == "VICTORY_WINDOW_SIZE").then(|| OsString::from("1080p"));
        let err =
            Config::resolve(FilePaths::default(), FileWindow::default(), None, env).unwrap_err();
        assert!(
            format!("{err:#}").contains("VICTORY_WINDOW_SIZE"),
            "{err:#}"
        );
    }

    #[test]
    fn window_sizes_parse() {
        assert_eq!("title".parse::<WindowSize>().unwrap(), WindowSize::Title);
        assert_eq!(
            " Fullscreen ".parse::<WindowSize>().unwrap(),
            WindowSize::Fullscreen
        );
        assert_eq!(
            "1920x1080".parse::<WindowSize>().unwrap(),
            WindowSize::Pixels {
                width: 1920,
                height: 1080
            }
        );
        assert_eq!(
            "1280 X 720".parse::<WindowSize>().unwrap(),
            WindowSize::Pixels {
                width: 1280,
                height: 720
            }
        );
        for bad in [
            "",
            "1080p",
            "1920",
            "1920x",
            "x1080",
            "0x0",
            "63x1080",
            "1920x16385",
            "-1x5",
            "+1920x+1080",
            "big",
        ] {
            assert!(
                bad.parse::<WindowSize>().is_err(),
                "{bad:?} should be refused"
            );
        }
        for size in [
            WindowSize::Title,
            WindowSize::Fullscreen,
            WindowSize::Pixels {
                width: 1920,
                height: 1080,
            },
        ] {
            assert_eq!(size.to_string().parse::<WindowSize>().unwrap(), size);
        }
    }

    #[test]
    fn window_size_environment_overrides_file() {
        let window = FileWindow {
            size: Some("fullscreen".into()),
        };
        let env = |k: &str| (k == "VICTORY_WINDOW_SIZE").then(|| OsString::from("1600x900"));
        let cfg = Config::resolve(FilePaths::default(), window, None, env).unwrap();
        assert_eq!(
            cfg.window,
            WindowSize::Pixels {
                width: 1600,
                height: 900
            }
        );

        let window = FileWindow {
            size: Some("fullscreen".into()),
        };
        let empty = |k: &str| (k == "VICTORY_WINDOW_SIZE").then(OsString::new);
        let cfg = Config::resolve(FilePaths::default(), window, None, empty).unwrap();
        assert_eq!(cfg.window, WindowSize::Fullscreen);

        let cfg =
            Config::resolve(FilePaths::default(), FileWindow::default(), None, |_| None).unwrap();
        assert_eq!(cfg.window, WindowSize::Title);
    }

    #[test]
    fn a_bad_window_size_names_where_it_came_from() {
        let window = FileWindow {
            size: Some("huge".into()),
        };
        let source = Some(PathBuf::from("my-victory.toml"));
        let err = Config::resolve(FilePaths::default(), window, source, |_| None).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("[window].size in my-victory.toml"), "{text}");
    }

    #[test]
    fn cache_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        let cfg = Config {
            cache_dir: inside,
            ..Default::default()
        };
        assert!(cfg.cache_subdir("x").is_err());
    }

    #[test]
    fn data_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let cfg = Config {
            data_dir: inside,
            ..Default::default()
        };
        assert!(cfg.data_subdir("x").is_err());
    }
}
