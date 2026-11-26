//! Diagnostics: the card's time split by who asked for the work, the
//! title or VIctory (`VICTORY_GPU_SPANS=1`, diagnostics builds). A list's
//! commands are marked as they are recorded ([`Span`]); where the kind
//! changes a timestamp goes into the list, and when the list has run the
//! time between two timestamps goes to the kind marked at the first. Per
//! stats line: the milliseconds and marks per frame of each kind, in the
//! log and, with `VICTORY_GPU_SPANS_CSV=<path>`, as a row of a CSV file.
//!
//! Timestamps are taken at the bottom of the pipe, so a span holds the
//! work that finished between its two stamps: what overlaps a boundary
//! goes to the later side, and the stamps add a little time of their own.
//! Lists of ours that are not marked (the tile mapping runs, the tests'
//! lists) are not counted. A list is marked only while it records: a mark
//! on a closed list records nothing. Without the feature the markers are
//! no-ops: [`Span`], [`Guard`] and the marking calls exist but record
//! nothing.
//!
//! shortcut: a list a submission runs twice resolves its stamps into one
//! buffer, so its time is the last run's, counted once per run.

/// What the commands recorded from a mark on are, until the next mark.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Span {
    /// The title's draws (also mesh dispatches), dispatches and indirect
    /// executions, as the host call that runs each.
    TitleDraw,
    TitleDispatch,
    TitleIndirect,
    /// The title's copies, clears and resolves, and its buffer copies done
    /// on the card ahead of a submission (lifted reads).
    TitleCopy,
    /// The title's barriers.
    TitleBarrier,
    /// Ours: the indirect argument clamps, their barriers and budget words.
    /// Its marks are the clamp dispatches recorded (each noted as an op,
    /// [`crate::host::HostList::span_op`]), not the marks of its spans.
    Clamp,
    /// Ours: the hash and compare passes over copies in the card's memory.
    CardHash,
    /// Ours: copies that bring the card's copies up to date.
    Upload,
    /// Ours: copies of what shaders wrote down to the PC's memory.
    Fetch,
    /// Ours: depth twins (their copies and plane passes, with their state
    /// moves; a use that copies nothing moves states under what is around
    /// it).
    DepthTwin,
    /// Ours: texture uploads (detiled texels) and alias copies.
    Texture,
    /// Ours: the state transitions a list's prelude runs before it.
    Prelude,
    /// Ours: the pass that stands in for the title's upscaler.
    Upscale,
    /// Ours, not marked otherwise: binds and their barriers, buffer moves.
    Other,
    /// A list's commands past its last timestamp (`MAIN_STAMPS`).
    Untimed,
}

#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
impl Span {
    pub(crate) const ALL: [Span; 15] = [
        Span::TitleDraw,
        Span::TitleDispatch,
        Span::TitleIndirect,
        Span::TitleCopy,
        Span::TitleBarrier,
        Span::Clamp,
        Span::CardHash,
        Span::Upload,
        Span::Fetch,
        Span::DepthTwin,
        Span::Texture,
        Span::Prelude,
        Span::Upscale,
        Span::Other,
        Span::Untimed,
    ];

    /// The name in the log line and the CSV header.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Span::TitleDraw => "title_draw",
            Span::TitleDispatch => "title_dispatch",
            Span::TitleIndirect => "title_indirect",
            Span::TitleCopy => "title_copy",
            Span::TitleBarrier => "title_barrier",
            Span::Clamp => "clamp",
            Span::CardHash => "card_hash",
            Span::Upload => "upload",
            Span::Fetch => "fetch",
            Span::DepthTwin => "depth_twin",
            Span::Texture => "texture",
            Span::Prelude => "prelude",
            Span::Upscale => "upscale",
            Span::Other => "other",
            Span::Untimed => "untimed",
        }
    }

    /// Whether the title asked for the work (the rest is ours, or untimed).
    pub(crate) fn title(self) -> bool {
        matches!(
            self,
            Span::TitleDraw
                | Span::TitleDispatch
                | Span::TitleIndirect
                | Span::TitleCopy
                | Span::TitleBarrier
        )
    }
}

/// Marks a list's commands as `span` until it drops, then as what they
/// were before ([`crate::host::HostList::spanned`]).
pub(crate) struct Guard<'a> {
    #[cfg(feature = "diagnostics")]
    list: &'a crate::host::HostList,
    #[cfg(feature = "diagnostics")]
    before: Option<Span>,
    #[cfg(not(feature = "diagnostics"))]
    list: std::marker::PhantomData<&'a ()>,
}

#[cfg(feature = "diagnostics")]
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        if let Some(before) = self.before
            && let Some(stamps) = self.list.spans.main()
        {
            stamps.restore(&self.list.list, before);
        }
    }
}

/// The lists of ours a host list owns, whose stamps go with it
/// ([`crate::host::HostList::own_mark`]).
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
#[derive(Clone, Copy)]
pub(crate) enum Own {
    /// Its prelude (`HostList::prelude`).
    Prelude,
    /// The lifted reads at the head of a submission it is the first of
    /// (`HostList::lift_list`).
    Lifts,
}

impl crate::host::HostList {
    /// Marks the commands recorded on the list from here on as `span`,
    /// until the next mark (`VICTORY_GPU_SPANS=1`, diagnostics builds,
    /// [`crate::spans`]); does nothing otherwise, or on a closed list.
    pub(crate) fn span(&self, span: Span) {
        #[cfg(feature = "diagnostics")]
        if on()
            && let Some(stamps) = self.spans.main()
        {
            stamps.mark(&self.list, span, true);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = span;
    }

    /// Marks the title's own commands, as the host call that runs each
    /// (from [`Self::op`]); `crate::list::with_state` marks what
    /// follows as ours.
    pub(crate) fn span_op(&self, kind: crate::host::OpKind) {
        use crate::host::OpKind;
        match kind {
            OpKind::Draw | OpKind::Indexed | OpKind::Mesh => self.span(Span::TitleDraw),
            OpKind::Dispatch => self.span(Span::TitleDispatch),
            OpKind::Indirect => self.span(Span::TitleIndirect),
            // Ours, in spans the guards around them mark; a clamp's
            // dispatch is what the clamp kind counts.
            #[cfg(feature = "diagnostics")]
            OpKind::Clamp => {
                if on()
                    && let Some(stamps) = self.spans.main()
                {
                    stamps.count(&self.list, Span::Clamp);
                }
            }
            #[cfg(not(feature = "diagnostics"))]
            OpKind::Clamp => {}
            OpKind::Plane | OpKind::Upscale => {}
        }
    }

    /// [`Self::span`] until the guard drops; then the commands are of
    /// the kind before again.
    pub(crate) fn spanned(&self, span: Span) -> Guard<'_> {
        #[cfg(feature = "diagnostics")]
        {
            let before = on().then(|| {
                self.spans
                    .main()
                    .and_then(|stamps| stamps.mark(&self.list, span, true))
                    .unwrap_or(Span::Other)
            });
            Guard { list: self, before }
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = span;
            Guard {
                list: std::marker::PhantomData,
            }
        }
    }

    /// Marks nothing; when the guard drops the commands are of the kind
    /// marked now again: around a call that marks what it records with
    /// [`Self::span`] and may record nothing (`crate::indirect::done`).
    pub(crate) fn span_held(&self) -> Guard<'_> {
        #[cfg(feature = "diagnostics")]
        {
            let before = on().then(|| {
                self.spans
                    .main()
                    .and_then(Stamps::current)
                    .unwrap_or(Span::Other)
            });
            Guard { list: self, before }
        }
        #[cfg(not(feature = "diagnostics"))]
        Guard {
            list: std::marker::PhantomData,
        }
    }

    /// Marks the commands recorded on `list`, one of ours this host list
    /// owns (`own`), from here on as `span`. Only while `list` records
    /// (between its reset and its close): its first mark after
    /// [`Self::own_end`] starts a new recording.
    pub(crate) fn own_mark(
        &self,
        own: Own,
        list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
        span: Span,
    ) {
        #[cfg(feature = "diagnostics")]
        if on()
            && let Some(stamps) = self.spans.own(own, true)
        {
            stamps.recording();
            stamps.mark(list, span, true);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = (own, list, span);
    }

    /// Before that list closes (`Stamps::end`).
    pub(crate) fn own_end(
        &self,
        own: Own,
        list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    ) {
        #[cfg(feature = "diagnostics")]
        if let Some(stamps) = self.spans.own(own, false) {
            stamps.end(list);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = (own, list);
    }

    /// The list starts closing: the guards of what is recorded before its
    /// end ([`Self::spans_end`]) do not give the kind back, as nothing of
    /// another kind follows.
    pub(crate) fn spans_closing(&self) {
        #[cfg(feature = "diagnostics")]
        if on()
            && let Some(Some(stamps)) = self.spans.main.get()
        {
            stamps.closing();
        }
    }

    /// The list `own` is about to run with this host list's submission:
    /// one run more of what it recorded, taken with the host list's
    /// ([`Self::spans_ran`]).
    pub(crate) fn own_sent(&self, own: Own) {
        #[cfg(feature = "diagnostics")]
        if on() {
            self.spans.sent(own);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = own;
    }

    /// Before the list closes: its last timestamp (`Stamps::end`); no
    /// mark records anything from here until [`Self::spans_reopened`].
    pub(crate) fn spans_end(&self) {
        // Made here if not yet, so that stamps made later know the list
        // is closed.
        #[cfg(feature = "diagnostics")]
        if on()
            && let Some(stamps) = self.spans.main()
        {
            stamps.end(&self.list);
        }
    }

    /// The list was reset and records again: the marks of a recording that
    /// did not run go (`spans_ran` took those of one that did).
    pub(crate) fn spans_reopened(&self) {
        #[cfg(feature = "diagnostics")]
        {
            if let Some(Some(stamps)) = self.spans.main.get() {
                stamps.begin();
            }
            self.spans.unsent();
        }
    }

    /// The list's submission has run it `runs` times: the time of its
    /// spans goes to the sums, and that of its prelude and lifted reads
    /// as often as the submission sent them ([`Self::own_sent`]: a
    /// naming of the list whose prelude had nothing to do sent none).
    /// What was taken is not taken again.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn spans_ran(&self, runs: u64) {
        if !on() {
            return;
        }
        if let Some(Some(stamps)) = self.spans.main.get() {
            stamps.take(runs);
        }
        for own in [Own::Prelude, Own::Lifts] {
            let sent = self.spans.taken(own);
            if let Some(stamps) = self.spans.own(own, false) {
                stamps.take(sent);
            }
        }
    }

    /// A submission that was waited for has run `runs` (a list named
    /// twice ran twice): [`Self::spans_ran`] for each. Those sent ahead are
    /// taken when they are settled (`crate::list::recycle`).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn spans_waited(runs: &[&Self]) {
        if !on() {
            return;
        }
        for (i, &list) in runs.iter().enumerate() {
            if runs[..i].iter().any(|&l| std::ptr::eq(l, list)) {
                continue;
            }
            let n = runs.iter().filter(|&&l| std::ptr::eq(l, list)).count();
            list.spans_ran(n as u64);
        }
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) use on::*;

#[cfg(feature = "diagnostics")]
mod on {
    use super::Span;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Arc, Mutex, OnceLock};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::Interface;

    /// Timestamps of a title list. A list past them puts the rest of its
    /// time under [`Span::Untimed`].
    pub(crate) const MAIN_STAMPS: u32 = 8192;
    /// Timestamps of a list of ours (a prelude, lifted reads, a card list,
    /// a fetch).
    pub(crate) const OWN_STAMPS: u32 = 512;

    const KINDS: usize = Span::ALL.len();
    /// Nanoseconds and marks of each kind since the last line.
    static NS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
    static MARKS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
    /// Lists read since the last line, those that ran out of timestamps,
    /// and the frames presented.
    static LISTS: AtomicU64 = AtomicU64::new(0);
    static FULL: AtomicU64 = AtomicU64::new(0);
    static FRAMES: AtomicU64 = AtomicU64::new(0);
    /// When marking was first asked for: the CSV's time column.
    static STARTED: OnceLock<std::time::Instant> = OnceLock::new();

    #[cfg(test)]
    thread_local! {
        /// Tests: lists are marked on this thread whatever the switch says.
        static FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Whether lists are marked (`VICTORY_GPU_SPANS=1`).
    pub(crate) fn on() -> bool {
        #[cfg(test)]
        if FORCED.with(std::cell::Cell::get) {
            return true;
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var_os("VICTORY_GPU_SPANS").is_some_and(|v| v == "1");
            STARTED.get_or_init(std::time::Instant::now);
            if on {
                tracing::info!(
                    "GPU time split by kind of work in the stats line (VICTORY_GPU_SPANS=1); \
                     the timestamps add time of their own"
                );
            }
            on
        })
    }

    /// Ticks per second of the timestamps (at least 1).
    fn frequency() -> u64 {
        static FREQUENCY: OnceLock<u64> = OnceLock::new();
        *FREQUENCY.get_or_init(|| {
            // SAFETY: a plain query.
            let ticks = crate::host::host().map(|h| unsafe { h.queue().GetTimestampFrequency() });
            ticks.and_then(Result::ok).unwrap_or(1).max(1)
        })
    }

    /// A host list's timestamps ([`crate::host::HostList`]'s `spans`): its
    /// own, made at its first mark or close, and those of its prelude and
    /// lifted reads, made at their first mark. They go with the host list.
    #[derive(Default)]
    pub(crate) struct ListSpans {
        pub(super) main: OnceLock<Option<Stamps>>,
        prelude: OnceLock<Option<Stamps>>,
        lifts: OnceLock<Option<Stamps>>,
        /// How often the prelude and the lifted reads were sent since they
        /// were last taken.
        sent: [AtomicU64; 2],
    }

    impl ListSpans {
        /// The host list's own stamps, made if not yet. A new host list is
        /// open, and one that closes makes them ([`HostList::spans_end`]),
        /// so stamps made here belong to an open list.
        ///
        /// [`HostList::spans_end`]: crate::host::HostList::spans_end
        pub(crate) fn main(&self) -> Option<&Stamps> {
            self.main.get_or_init(|| Stamps::new(MAIN_STAMPS)).as_ref()
        }

        /// `own`'s list goes to the card once more.
        pub(super) fn sent(&self, own: super::Own) {
            self.sent[own as usize].fetch_add(1, Relaxed);
        }

        /// How often `own`'s list was sent since the last call.
        pub(super) fn taken(&self, own: super::Own) -> u64 {
            self.sent[own as usize].swap(0, Relaxed)
        }

        /// The list was reset: what was sent and did not run goes.
        pub(super) fn unsent(&self) {
            for sent in &self.sent {
                sent.store(0, Relaxed);
            }
        }

        /// The stamps of `own`'s list; made if not yet only with `make`.
        pub(super) fn own(&self, own: super::Own, make: bool) -> Option<&Stamps> {
            let cell = match own {
                super::Own::Prelude => &self.prelude,
                super::Own::Lifts => &self.lifts,
            };
            if make {
                cell.get_or_init(|| Stamps::new(OWN_STAMPS)).as_ref()
            } else {
                cell.get()?.as_ref()
            }
        }
    }

    /// A list's timestamps: a query heap, the buffer they are read back
    /// into, and the kind of each span recorded so far.
    pub(crate) struct Stamps {
        heap: ID3D12QueryHeap,
        back: ID3D12Resource,
        count: u32,
        marks: Mutex<Marks>,
        /// Tests: what these stamps added to the sums (marks by kind,
        /// lists, lists past their stamps).
        #[cfg(test)]
        added: Mutex<(Vec<u64>, u64, u64)>,
    }

    #[derive(Default)]
    struct Marks {
        /// The kind of the span that starts at each timestamp, and how many
        /// marks it took.
        spans: Vec<(Span, u32)>,
        /// The list records: set when it is made or reset
        /// ([`Stamps::begin`]), cleared before it closes ([`Stamps::end`]).
        /// A mark on a list that does not record records nothing.
        open: bool,
        /// The list closes ([`Stamps::closing`]): a guard's end gives the
        /// kind back no more.
        closing: bool,
        /// The list closed with its last timestamp resolved: to be read
        /// once it has run.
        closed: bool,
        /// The timestamps ran out: the last span is [`Span::Untimed`].
        full: bool,
    }

    impl Marks {
        /// Those of a list that starts recording.
        fn open() -> Marks {
            Marks {
                open: true,
                ..Marks::default()
            }
        }
    }

    impl Stamps {
        /// The stamps of an open list. `None` with no host or when the
        /// heap cannot be made.
        pub(crate) fn new(count: u32) -> Option<Stamps> {
            let device = &crate::host::host()?.device;
            let readback = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(count) * 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut heap: Option<ID3D12QueryHeap> = None;
            let mut back: Option<ID3D12Resource> = None;
            let query = D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: count,
                NodeMask: 0,
            };
            // SAFETY: plain object creation.
            unsafe {
                device.CreateQueryHeap(&query, &mut heap).ok()?;
                device
                    .CreateCommittedResource(
                        &readback,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut back,
                    )
                    .ok()?;
            }
            let back = back?;
            crate::host::label(&back, "span timestamp read-back", u64::from(count) * 8);
            Some(Stamps {
                heap: heap?,
                back,
                count,
                marks: Mutex::new(Marks::open()),
                #[cfg(test)]
                added: Mutex::new((vec![0; KINDS], 0, 0)),
            })
        }

        fn marks(&self) -> std::sync::MutexGuard<'_, Marks> {
            self.marks.lock().unwrap_or_else(|e| e.into_inner())
        }

        /// The list was reset and records again: the marks of a recording
        /// that did not run go.
        pub(crate) fn begin(&self) {
            *self.marks() = Marks::open();
        }

        /// For a list of ours, marked only while it records: if it does not
        /// ([`Stamps::end`] since), it was reset, and this is a new
        /// recording ([`Stamps::begin`]).
        pub(crate) fn recording(&self) {
            let mut marks = self.marks();
            if !marks.open {
                *marks = Marks::open();
            }
        }

        /// The kind marked last, while the list records.
        pub(crate) fn current(&self) -> Option<Span> {
            let marks = self.marks();
            marks.spans.last().filter(|_| marks.open).map(|s| s.0)
        }

        /// The end of a guard ([`super::Guard`]): back to `span`, not
        /// counted; nothing on a list not marked yet, or one that closes.
        pub(crate) fn restore(&self, list: &ID3D12GraphicsCommandList, span: Span) {
            let marks = self.marks();
            if !marks.spans.is_empty() && !marks.closing {
                drop(marks);
                self.mark(list, span, false);
            }
        }

        /// The list starts closing ([`Marks`]' `closing`).
        pub(crate) fn closing(&self) {
            self.marks().closing = true;
        }

        /// One more of `span`'s count, marked there if the commands are
        /// not of that kind yet (after them: their time stays with the
        /// kind before).
        pub(crate) fn count(&self, list: &ID3D12GraphicsCommandList, span: Span) {
            {
                let mut marks = self.marks();
                if !marks.open {
                    return;
                }
                let full = marks.full;
                if let Some(last) = marks.spans.last_mut()
                    && (last.0 == span || full)
                {
                    last.1 += 1;
                    return;
                }
            }
            self.mark_counted(list, span, true);
        }

        /// The commands `list` (the one these stamps belong to) records
        /// from here on are `span`; `counted`: the mark counts as one of
        /// the span's. The kind marked before, if any. Nothing while the
        /// list does not record (`Marks::open`).
        pub(crate) fn mark(
            &self,
            list: &ID3D12GraphicsCommandList,
            span: Span,
            counted: bool,
        ) -> Option<Span> {
            // What is not marked otherwise has no count of its own; a
            // clamp counts its dispatches ([`Stamps::count`]).
            self.mark_counted(
                list,
                span,
                counted && !matches!(span, Span::Other | Span::Clamp),
            )
        }

        fn mark_counted(
            &self,
            list: &ID3D12GraphicsCommandList,
            span: Span,
            counted: bool,
        ) -> Option<Span> {
            let mut marks = self.marks();
            if !marks.open {
                return None;
            }
            let before = marks.spans.last().map(|s| s.0);
            let at = marks.spans.len() as u32;
            // The last timestamp ends the list ([`Stamps::end`]); past the
            // others the last span takes the rest.
            if !marks.full && before != Some(span) && at + 1 >= self.count {
                marks.full = true;
            }
            if before == Some(span) || marks.full {
                if counted && let Some(last) = marks.spans.last_mut() {
                    last.1 += 1;
                }
                return before;
            }
            // SAFETY: an open list (it records) and our query heap.
            unsafe { list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            marks.spans.push((span, u32::from(counted)));
            before
        }

        /// Before `list` closes: its last timestamp, and all of them into
        /// the read-back buffer. Nothing is marked from here until the list
        /// records again.
        pub(crate) fn end(&self, list: &ID3D12GraphicsCommandList) {
            let mut marks = self.marks();
            let at = marks.spans.len() as u32;
            let open = std::mem::replace(&mut marks.open, false);
            if !open || at == 0 {
                return;
            }
            // SAFETY: an open list, our heap and read-back buffer (in the
            // copy destination state it was made in).
            unsafe {
                list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, at);
                list.ResolveQueryData(
                    &self.heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    0,
                    at + 1,
                    &self.back,
                    0,
                );
            }
            marks.closed = true;
        }

        /// The list has run `runs` times since it closed: what it resolved
        /// goes to the sums, once per run (the last run's stamps), and the
        /// marks are cleared until it records again.
        pub(crate) fn take(&self, runs: u64) {
            let marks = std::mem::take(&mut *self.marks());
            if runs == 0 || !marks.closed {
                return;
            }
            let n = marks.spans.len();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE {
                Begin: 0,
                End: (n + 1) * 8,
            };
            // SAFETY: a read-back buffer the list resolved into, and it ran.
            let stamps: Vec<u64> = unsafe {
                if self.back.Map(0, Some(&range), Some(&mut data)).is_err() {
                    return;
                }
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), n + 1).to_vec();
                self.back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
                stamps
            };
            let frequency = u128::from(frequency());
            for (i, &(span, count)) in marks.spans.iter().enumerate() {
                let kind = if marks.full && i + 1 == n {
                    Span::Untimed
                } else {
                    span
                };
                let ticks = u128::from(stamps[i + 1].saturating_sub(stamps[i]));
                let ns = (ticks * 1_000_000_000 / frequency) as u64;
                NS[kind as usize].fetch_add(ns * runs, Relaxed);
                MARKS[kind as usize].fetch_add(u64::from(count) * runs, Relaxed);
            }
            LISTS.fetch_add(runs, Relaxed);
            if marks.full {
                FULL.fetch_add(runs, Relaxed);
            }
            #[cfg(test)]
            {
                let mut added = self.added.lock().unwrap_or_else(|e| e.into_inner());
                for (i, &(span, count)) in marks.spans.iter().enumerate() {
                    let kind = if marks.full && i + 1 == n {
                        Span::Untimed
                    } else {
                        span
                    };
                    added.0[kind as usize] += u64::from(count) * runs;
                }
                added.1 += runs;
                added.2 += u64::from(marks.full) * runs;
            }
        }

        /// Tests: what these stamps added since the last call.
        #[cfg(test)]
        pub(crate) fn added(&self) -> (Vec<u64>, u64, u64) {
            let mut added = self.added.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::replace(&mut *added, (vec![0; KINDS], 0, 0))
        }
    }

    /// The stamps of the lists of ours that live as long as the host (the
    /// lists that bring the card's copies up to date and compare them, the
    /// fetches, the reference copies and the memory directory's), by the
    /// list's address. A pooled list's go when its ring lets it go
    /// ([`forget`]). A host list's own lists keep theirs in it
    /// ([`ListSpans`]).
    static OWN: Mutex<Option<HashMap<usize, Arc<Stamps>>>> = Mutex::new(None);

    /// The stamps of that list; made if not yet only with `make`.
    fn own(list: &ID3D12GraphicsCommandList, make: bool) -> Option<Arc<Stamps>> {
        let key = list.as_raw() as usize;
        let mut own = OWN.lock().unwrap_or_else(|e| e.into_inner());
        let own = own.get_or_insert_with(HashMap::new);
        if let Some(stamps) = own.get(&key) {
            return Some(stamps.clone());
        }
        if !make {
            return None;
        }
        let stamps = Arc::new(Stamps::new(OWN_STAMPS)?);
        own.insert(key, stamps.clone());
        Some(stamps)
    }

    /// Marks the commands of a list of ours that lives as long as the host,
    /// while it records ([`Stamps::recording`]).
    pub(crate) fn mark(list: &ID3D12GraphicsCommandList, span: Span) {
        if on()
            && let Some(stamps) = own(list, true)
        {
            stamps.recording();
            stamps.mark(list, span, true);
        }
    }

    /// Before that list closes ([`Stamps::end`]).
    pub(crate) fn end(list: &ID3D12GraphicsCommandList) {
        if on()
            && let Some(stamps) = own(list, false)
        {
            stamps.end(list);
        }
    }

    /// That list has run ([`Stamps::take`]).
    pub(crate) fn ran(list: &ID3D12GraphicsCommandList) {
        if on()
            && let Some(stamps) = own(list, false)
        {
            stamps.take(1);
        }
    }

    /// That list is let go of, its last run over: [`ran`], and its
    /// stamps with it.
    pub(crate) fn forget(list: &ID3D12GraphicsCommandList) {
        if !on() {
            return;
        }
        ran(list);
        let mut own = OWN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(own) = own.as_mut() {
            own.remove(&(list.as_raw() as usize));
        }
    }

    /// A frame was presented.
    pub(crate) fn presented() {
        FRAMES.fetch_add(1, Relaxed);
    }

    /// The stats line: per frame, the milliseconds and marks of each kind,
    /// the title's and ours, and our share; a row of the CSV file too.
    pub(crate) fn log() {
        if !on() {
            return;
        }
        let frames = FRAMES.swap(0, Relaxed);
        let ns: Vec<u64> = NS.iter().map(|a| a.swap(0, Relaxed)).collect();
        let marks: Vec<u64> = MARKS.iter().map(|a| a.swap(0, Relaxed)).collect();
        let (lists, full) = (LISTS.swap(0, Relaxed), FULL.swap(0, Relaxed));
        if frames == 0 {
            return;
        }
        let per = frames as f64;
        let ms = |ns: u64| ns as f64 / 1e6 / per;
        let mut text = String::new();
        let (mut title, mut ours) = (0.0, 0.0);
        for span in Span::ALL {
            let (t, n) = (ms(ns[span as usize]), marks[span as usize] as f64 / per);
            text.push_str(&format!(" {}={t:.2}ms/{n:.0}", span.name()));
            match span {
                Span::Untimed => {}
                s if s.title() => title += t,
                _ => ours += t,
            }
        }
        let share = if title + ours > 0.0 {
            100.0 * ours / (title + ours)
        } else {
            0.0
        };
        tracing::info!(
            "gpu spans per frame (frames={frames} lists={lists} full={full}):{text}; \
             title={title:.2}ms ours={ours:.2}ms ours_share={share:.1}%"
        );
        csv(frames, lists, full, &ns, &marks);
    }

    /// A row of `VICTORY_GPU_SPANS_CSV`: the interval's totals (ms, marks).
    fn csv(frames: u64, lists: u64, full: u64, ns: &[u64], marks: &[u64]) {
        use std::io::Write;
        static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
        let started = *STARTED.get_or_init(std::time::Instant::now);
        let file = FILE.get_or_init(|| {
            let path = std::env::var_os("VICTORY_GPU_SPANS_CSV").filter(|p| !p.is_empty())?;
            let path = std::path::PathBuf::from(path);
            match std::fs::File::create(&path) {
                Ok(mut file) => {
                    let mut header = String::from("t_s,frames,lists,full");
                    for span in Span::ALL {
                        header.push_str(&format!(",{0}_ms,{0}_n", span.name()));
                    }
                    let _ = writeln!(file, "{header}");
                    tracing::info!(path = %path.display(), "gpu spans CSV");
                    Some(Mutex::new(file))
                }
                Err(error) => {
                    tracing::warn!(path = %path.display(), %error, "no gpu spans CSV");
                    None
                }
            }
        });
        let Some(file) = file else { return };
        let mut row = format!(
            "{:.1},{frames},{lists},{full}",
            started.elapsed().as_secs_f64()
        );
        for span in Span::ALL {
            row.push_str(&format!(
                ",{:.3},{}",
                ns[span as usize] as f64 / 1e6,
                marks[span as usize]
            ));
        }
        let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(file, "{row}");
        let _ = file.flush();
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The spans tests one at a time: they mark on their own thread
        /// ([`Forced`]) and read what their own stamps added.
        static TEST_LOCK: Mutex<()> = Mutex::new(());

        /// Lists are marked on this thread until it drops, whatever
        /// `VICTORY_GPU_SPANS` says.
        struct Forced;

        impl Forced {
            fn on() -> Forced {
                FORCED.with(|f| f.set(true));
                Forced
            }
        }

        impl Drop for Forced {
            fn drop(&mut self) {
                FORCED.with(|f| f.set(false));
            }
        }

        /// On WARP (`VICTORY_D3D12_WARP=1`): a list's marks become spans
        /// once it ran; a recording that did not run adds nothing; a list
        /// past its timestamps puts the rest under untimed.
        #[test]
        fn marks_become_spans_when_the_list_ran() {
            let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let Some(host) = crate::host::host() else {
                return;
            };
            // What the stamps added: marks by kind, lists, lists past their
            // stamps.
            let run = |count: u32, kinds: &[Span], ran: bool| {
                let stamps = Stamps::new(count).expect("a timestamp heap");
                let list = host.new_list().unwrap();
                for &kind in kinds {
                    stamps.mark(&list.list, kind, true);
                }
                stamps.end(&list.list);
                list.close();
                if ran {
                    host.execute(&[&list]);
                }
                stamps.take(u64::from(ran));
                // Taken once: a second read adds nothing.
                stamps.take(u64::from(ran));
                stamps.added()
            };
            let (marks, lists, full) = run(
                8,
                &[
                    Span::TitleDraw,
                    Span::Upload,
                    Span::Upload,
                    Span::Clamp,
                    Span::Other,
                ],
                true,
            );
            assert_eq!((lists, full), (1, 0));
            assert_eq!(marks[Span::TitleDraw as usize], 1);
            assert_eq!(marks[Span::Upload as usize], 2);
            assert_eq!(
                marks[Span::Clamp as usize],
                0,
                "a clamp counts its dispatches"
            );
            assert_eq!(marks[Span::Other as usize], 0, "other has no count");
            assert_eq!(
                run(8, &[Span::TitleDraw], false).1,
                0,
                "a list that did not run"
            );
            // Three timestamps: two spans, the last one untimed.
            let (marks, lists, full) = run(
                3,
                &[Span::TitleDraw, Span::Upload, Span::Fetch, Span::Texture],
                true,
            );
            assert_eq!((lists, full), (1, 1));
            assert_eq!(marks[Span::TitleDraw as usize], 1);
            assert_eq!(marks[Span::Upload as usize], 0);
            assert_eq!(marks[Span::Untimed as usize], 3);
        }

        /// On WARP, with or without `VICTORY_GPU_SPANS` (best with
        /// `VICTORY_D3D12_DEBUG=1`): a host list is marked while it
        /// records, through its guards, close, a waited submission and
        /// reset; a mark on it closed (as the title's reset finds it)
        /// records nothing; a clamp counts its dispatches, not its marks;
        /// a list a submission names twice counts twice, and its prelude as
        /// often as it was sent (once: the second naming's has nothing to
        /// do); a recording reset without running adds nothing.
        #[test]
        fn a_host_list_is_marked_only_while_it_records() {
            use windows::Win32::Graphics::Direct3D12::*;
            use windows::Win32::Graphics::Dxgi::Common::{
                DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
            };
            let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let Some(host) = crate::host::host() else {
                return;
            };
            let _forced = Forced::on();
            use crate::host::OpKind;
            use crate::spans::Own;
            let list = host.new_list().unwrap();
            fn main(list: &crate::host::HostList) -> &Stamps {
                list.spans.main().unwrap()
            }
            let current = |list: &crate::host::HostList| main(list).current();
            list.span(Span::TitleDraw);
            {
                let _clamp = list.spanned(Span::Clamp);
                assert_eq!(current(&list), Some(Span::Clamp));
                // A clamp's dispatch, noted as its op.
                list.span_op(OpKind::Clamp);
                // Marks nothing; gives back the clamp.
                let _held = list.span_held();
            }
            assert_eq!(current(&list), Some(Span::TitleDraw), "the guard restored");
            // A texture in the render target state the list reads as a copy
            // source: its prelude moves it there.
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 16,
                Height: 16,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut raw: Option<ID3D12Resource> = None;
            // SAFETY: a plain 16x16 render target on the host device.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut raw,
                    )
                    .unwrap();
            }
            let texture = std::sync::Arc::new(crate::resource::HostTexture::new(
                raw.unwrap(),
                D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
            ));
            list.settle(&texture, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
            // Closing: a guard's end gives no kind back.
            list.spans_closing();
            drop(list.spanned(Span::Upload));
            assert_eq!(current(&list), Some(Span::Upload), "no kind given back");
            list.close();
            // Closed: as `crate::list`'s reset finds it.
            list.span(Span::Other);
            drop(list.spanned(Span::Clamp));
            assert_eq!(current(&list), None, "a closed list is not marked");
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            // Named twice; the waited submission takes the stamps, once per
            // run, and the prelude's once (`HostList::prelude`).
            host.execute(&[&list, &list]);
            let (marks, lists, full) = main(&list).added();
            assert_eq!((lists, full), (2, 0));
            assert_eq!(marks[Span::TitleDraw as usize], 2);
            assert_eq!(marks[Span::Clamp as usize], 2, "one dispatch a run");
            assert_eq!(marks[Span::Upload as usize], 2);
            let own = list
                .spans
                .own(Own::Prelude, false)
                .expect("the prelude was marked");
            let (prelude, sent, _) = own.added();
            assert_eq!(sent, 1, "the prelude ran once");
            assert_eq!(prelude[Span::Prelude as usize], 1);
            // Taken by the submission: a later take adds nothing.
            list.spans_ran(1);
            assert_eq!(main(&list).added().1, 0);
            list.reset();
            // A recording reset without running, then one that ran.
            list.span(Span::TitleCopy);
            list.close();
            list.reset();
            assert_eq!(current(&list), None, "a reset list has no marks");
            list.span(Span::TitleDispatch);
            assert_eq!(current(&list), Some(Span::TitleDispatch));
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            let (marks, lists, _) = main(&list).added();
            assert_eq!(lists, 1, "only the recording that ran");
            assert_eq!(marks[Span::TitleCopy as usize], 0);
            assert_eq!(marks[Span::TitleDispatch as usize], 1);
            assert_eq!(own.added().1, 0, "no prelude this time");
            list.reset();
        }
    }
}

/// Without the feature: marking a list of ours does nothing.
#[cfg(not(feature = "diagnostics"))]
pub(crate) fn mark(_: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList, _: Span) {}
#[cfg(not(feature = "diagnostics"))]
pub(crate) fn end(_: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList) {}
#[cfg(not(feature = "diagnostics"))]
pub(crate) fn ran(_: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList) {}
#[cfg(not(feature = "diagnostics"))]
pub(crate) fn forget(_: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList) {}
