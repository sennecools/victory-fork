//! An async call with no block queue and no process queue must fail with
//! `E_NO_TASK_QUEUE`, as the GDK documents for `XAsyncBlock.queue`.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use victory_hle::modules::xasync::{
    AsyncBlock, E_NO_TASK_QUEUE, ProviderData, XAsyncBegin, XTaskQueueSetCurrentProcessTaskQueue,
};

static PROVIDER_CALLS: AtomicUsize = AtomicUsize::new(0);

extern "system" fn provider(_: u32, _: *const ProviderData) -> i32 {
    PROVIDER_CALLS.fetch_add(1, Ordering::SeqCst);
    0
}

#[test]
fn begin_without_any_queue_reports_no_task_queue() {
    XTaskQueueSetCurrentProcessTaskQueue(std::ptr::null());
    let mut block = AsyncBlock {
        queue: std::ptr::null(),
        context: std::ptr::null_mut::<c_void>(),
        callback: None,
        internal: [0; 4],
    };

    assert_eq!(
        XAsyncBegin(
            &mut block,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            Some(provider)
        ),
        E_NO_TASK_QUEUE
    );
    assert_eq!(PROVIDER_CALLS.load(Ordering::SeqCst), 0);
}
