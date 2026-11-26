//! Scripted input for unattended runs, in every build: `VICTORY_KEYS` (key
//! presses) and `VICTORY_ROUTE` (a file of timed keys and stick and trigger
//! positions, so a run can walk and turn). Times count from the first poll
//! of input.

use std::time::Duration;

use super::PadState;

/// How long a `VICTORY_KEYS` entry holds its key when it gives no hold.
pub(super) const TAP: Duration = Duration::from_millis(200);

/// A pad axis a route can set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Axis {
    LeftX,
    LeftY,
    RightX,
    RightY,
    LeftTrigger,
    RightTrigger,
}

/// One scripted input: a key (virtual key) held down, or an axis held at a
/// value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Input {
    Key(u8),
    Axis(Axis, f32),
}

/// An input held from `at` for `hold`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Step {
    pub at: Duration,
    pub hold: Duration,
    pub input: Input,
}

/// Axis values the script sets now; `None` leaves the host's own.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Axes([Option<f32>; 6]);

impl Axes {
    pub fn get(&self, axis: Axis) -> Option<f32> {
        self.0[axis as usize]
    }

    /// Sets the scripted axes on `pad`, over what the host reported.
    pub fn apply(&self, pad: &mut PadState) {
        let set = |value: &mut f32, axis| {
            if let Some(v) = self.get(axis) {
                *value = v;
            }
        };
        set(&mut pad.left_x, Axis::LeftX);
        set(&mut pad.left_y, Axis::LeftY);
        set(&mut pad.right_x, Axis::RightX);
        set(&mut pad.right_y, Axis::RightY);
        set(&mut pad.left_trigger, Axis::LeftTrigger);
        set(&mut pad.right_trigger, Axis::RightTrigger);
    }
}

/// What the script holds at one moment.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Held {
    pub keys: Vec<u8>,
    pub axes: Axes,
}

/// A key by name: a letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`,
/// `Down`, `Left`, `Right` (the stand-in pad's table maps them to buttons).
pub(super) fn key(name: &str) -> Option<u8> {
    Some(match name {
        "Enter" => 0x0D,
        "Esc" => 0x1B,
        "Tab" => 0x09,
        "Back" => 0x08,
        "Up" => 0x26,
        "Down" => 0x28,
        "Left" => 0x25,
        "Right" => 0x27,
        k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
            k.as_bytes()[0].to_ascii_uppercase()
        }
        _ => return None,
    })
}

/// Sticks `lx`, `ly`, `rx`, `ry` (-1 to 1, right and up positive), triggers
/// `lt`, `rt` (0 to 1).
fn axis(name: &str) -> Option<(Axis, f32)> {
    Some(match name {
        "lx" => (Axis::LeftX, -1.0),
        "ly" => (Axis::LeftY, -1.0),
        "rx" => (Axis::RightX, -1.0),
        "ry" => (Axis::RightY, -1.0),
        "lt" => (Axis::LeftTrigger, 0.0),
        "rt" => (Axis::RightTrigger, 0.0),
        _ => return None,
    })
}

fn seconds(text: &str) -> Option<Duration> {
    Duration::try_from_secs_f64(text.trim().parse().ok()?).ok()
}

/// Whether a step's end, `at + hold`, fits a `Duration`, so `held` never
/// overflows on the input poll thread.
fn ends(at: Duration, hold: Duration) -> bool {
    at.checked_add(hold).is_some()
}

/// `VICTORY_KEYS=<seconds>:<key>[:<hold seconds>],...`: each key held for
/// `hold` (0.2 s when left out) from that time on. Entries that do not parse
/// are skipped.
pub(super) fn parse_keys(list: &str) -> Vec<Step> {
    list.split(',')
        .filter_map(|entry| {
            let mut fields = entry.split(':');
            let at = seconds(fields.next()?)?;
            let vk = key(fields.next()?.trim())?;
            let hold = match fields.next() {
                Some(hold) => seconds(hold)?,
                None => TAP,
            };
            if fields.next().is_some() || !ends(at, hold) {
                return None;
            }
            Some(Step {
                at,
                hold,
                input: Input::Key(vk),
            })
        })
        .collect()
}

/// A `VICTORY_ROUTE` file: one line per step, `<seconds> <hold seconds>
/// <input>...`, where an input is a key name or `<axis>=<value>`; the inputs
/// of one line start together and hold equally long (more than zero, and
/// `at + hold` must fit a `Duration`). `#` starts a comment.
/// Unlike `VICTORY_KEYS`, a line that does not parse is an error (line
/// number and why): a benchmark that silently stands still measures the
/// wrong thing.
pub(super) fn parse_route(text: &str) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or_default();
        let mut words = line.split_whitespace();
        let Some(at) = words.next() else {
            continue;
        };
        let bad = |why: String| format!("line {}: {why}", number + 1);
        let at = seconds(at).ok_or_else(|| bad(format!("bad time {at:?}")))?;
        let hold = words.next().ok_or_else(|| bad("no hold".into()))?;
        let hold = seconds(hold)
            .filter(|h| !h.is_zero() && ends(at, *h))
            .ok_or_else(|| bad(format!("bad hold {hold:?}")))?;
        let mut any = false;
        for word in words {
            let input = match word.split_once('=') {
                Some((name, value)) => {
                    let (axis, low) =
                        axis(name).ok_or_else(|| bad(format!("unknown axis {name:?}")))?;
                    let value: f32 = value
                        .parse()
                        .ok()
                        .filter(|v| (low..=1.0).contains(v))
                        .ok_or_else(|| bad(format!("{name} needs {low} to 1, not {value:?}")))?;
                    Input::Axis(axis, value)
                }
                None => Input::Key(key(word).ok_or_else(|| bad(format!("unknown key {word:?}")))?),
            };
            steps.push(Step { at, hold, input });
            any = true;
        }
        if !any {
            return Err(bad("no input".into()));
        }
    }
    Ok(steps)
}

/// The inputs held at `now`: a step holds over `at..at + hold`. When two
/// steps set one axis at once, the one that started last wins (the file's
/// later line on a tie).
pub(super) fn held(steps: &[Step], now: Duration) -> Held {
    let mut held = Held::default();
    let mut since = [Duration::ZERO; 6];
    for step in steps
        .iter()
        .filter(|s| (s.at..s.at + s.hold).contains(&now))
    {
        match step.input {
            Input::Key(vk) => {
                if !held.keys.contains(&vk) {
                    held.keys.push(vk);
                }
            }
            Input::Axis(axis, value) => {
                let slot = axis as usize;
                if held.axes.0[slot].is_none() || step.at >= since[slot] {
                    held.axes.0[slot] = Some(value);
                    since[slot] = step.at;
                }
            }
        }
    }
    held
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(at: f64, hold: f64, input: Input) -> Step {
        Step {
            at: Duration::from_secs_f64(at),
            hold: Duration::from_secs_f64(hold),
            input,
        }
    }

    #[test]
    fn keys_keep_the_old_format_and_take_a_hold() {
        assert_eq!(
            parse_keys("20:Enter, 35:x,40:W:12.5"),
            [
                s(20.0, 0.2, Input::Key(0x0D)),
                s(35.0, 0.2, Input::Key(b'X')),
                s(40.0, 12.5, Input::Key(b'W')),
            ]
        );
    }

    #[test]
    fn keys_skip_what_does_not_parse() {
        assert_eq!(parse_keys(""), []);
        assert_eq!(
            parse_keys("x:Enter,5:Nope,-1:A,5:A:-2,5:A:1:2,1e19:A:1e19,7:B"),
            [s(7.0, 0.2, Input::Key(b'B'))]
        );
    }

    #[test]
    fn a_route_reads_keys_and_axes() {
        let route = "# warm-up\n\n130 40 ly=1 rx=0.3  # walk, turning\n170.5 0.2 Enter\n200 5 lt=1 ry=-0.25\n";
        assert_eq!(
            parse_route(route).unwrap(),
            [
                s(130.0, 40.0, Input::Axis(Axis::LeftY, 1.0)),
                s(130.0, 40.0, Input::Axis(Axis::RightX, 0.3)),
                s(170.5, 0.2, Input::Key(0x0D)),
                s(200.0, 5.0, Input::Axis(Axis::LeftTrigger, 1.0)),
                s(200.0, 5.0, Input::Axis(Axis::RightY, -0.25)),
            ]
        );
    }

    #[test]
    fn a_route_names_its_bad_line() {
        for (text, why) in [
            ("1 2 ly=1\nx 2 ly=1", "line 2: bad time"),
            ("1", "line 1: no hold"),
            ("1 -2 ly=1", "line 1: bad hold"),
            ("1 0 ly=1", "line 1: bad hold"),
            ("1e19 1e19 ly=1", "line 1: bad hold"),
            ("1 2", "line 1: no input"),
            ("1 2 zz=1", "line 1: unknown axis"),
            ("1 2 ly=1.5", "line 1: ly needs -1 to 1"),
            ("1 2 lt=-0.5", "line 1: lt needs 0 to 1"),
            ("1 2 ly=up", "line 1: ly needs"),
            ("1 2 Jump", "line 1: unknown key"),
        ] {
            let error = parse_route(text).unwrap_err();
            assert!(error.starts_with(why), "{text:?}: {error}");
        }
    }

    #[test]
    fn steps_hold_for_their_time_and_the_latest_axis_wins() {
        let steps = parse_route("10 10 ly=1 rx=0.3\n15 2 rx=-0.5 Enter\n15 2 Enter\n").unwrap();
        let at = |t: f64| held(&steps, Duration::from_secs_f64(t));
        assert_eq!(at(9.9), Held::default());
        let walking = at(12.0);
        assert_eq!(
            (
                walking.axes.get(Axis::LeftY),
                walking.axes.get(Axis::RightX)
            ),
            (Some(1.0), Some(0.3))
        );
        assert!(walking.keys.is_empty());
        let turned = at(16.0);
        assert_eq!(turned.axes.get(Axis::RightX), Some(-0.5));
        assert_eq!(turned.keys, [0x0D]);
        assert_eq!(at(17.5).axes.get(Axis::RightX), Some(0.3));
        assert_eq!(at(20.0), Held::default());
    }

    #[test]
    fn scripted_axes_replace_the_hosts() {
        let mut pad = PadState {
            left_x: 0.5,
            left_y: 0.5,
            right_trigger: 1.0,
            ..Default::default()
        };
        let held = held(&parse_route("0 1 ly=-1 rt=0").unwrap(), Duration::ZERO);
        held.axes.apply(&mut pad);
        assert_eq!(
            (pad.left_x, pad.left_y, pad.right_trigger),
            (0.5, -1.0, 0.0)
        );
    }
}
