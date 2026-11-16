//! A thread to run a compilation on, and what a panic on it leaves behind.
//!
//! A compiler recurses deeply, so its work runs on a thread with a large stack ([run]). A panic on
//! that thread is recorded by a panic hook, so that the thread that started the work can report it
//! as a defect of the compiler, with what the worker had rendered before it.
//!
//! The hook is installed once in the process, by the first [run], and acts only on the threads
//! [run] started: it records the panic for the [WorkerError] the run answers with, and hands every
//! other panic to the hook installed before it. A host that replaces the hook after the first run
//! loses the location and backtrace of a panic on a worker (the error then holds the message
//! only), and the hook it installed sees that panic too. The first panic recorded is what went
//! wrong, and it may be one that a stage caught itself.
//!
//! When the process cannot unwind (a panic that cannot unwind, or `panic = "abort"`), the process
//! aborts after the hook, and nothing could write what the worker had rendered or report the
//! panic. The hook then writes to the process's standard error what the run's [SharedBuffer] holds
//! and the text `on_abort` makes of the panics, and calls the hook installed before it.
//!
//! When logging is enabled, a logger writes to the process's standard error from the worker thread
//! while the caller of [run] waits: a caller must not hold a lock on a standard stream across it.

use std::{
    cell::RefCell,
    io::{self, Write},
    sync::{Arc, Mutex, MutexGuard, Once, PoisonError},
};

/// The size of a worker's stack.
pub const STACK_SIZE: usize = 256 * 1024 * 1024;

/// Text two threads share: a worker writes what it renders for standard error as it renders it,
/// and the thread that started it writes the buffer out once the worker has ended, whether it
/// returned or panicked.
#[derive(Clone, Default)]
pub struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl SharedBuffer {
    /// Appends `text`.
    pub fn push(&self, text: &str) {
        self.bytes().extend_from_slice(text.as_bytes());
    }

    /// Removes and returns what the buffer holds.
    pub fn take(&self) -> Vec<u8> {
        core::mem::take(&mut *self.bytes())
    }

    /// The bytes, also after a panic while they were held.
    fn bytes(&self) -> MutexGuard<'_, Vec<u8>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A panic on a worker thread, as the panic hook recorded it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WorkerPanic {
    /// What the panic said.
    pub message: String,
    /// Where it happened, as `file:line:column`, when the hook saw it.
    pub location: Option<String>,
    /// The backtrace, when the environment asks for one.
    pub backtrace: Option<String>,
}

/// Why a worker gave no value.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerError {
    /// The thread could not be started.
    NotStarted(io::Error),
    /// The work panicked: the panics recorded, the first being what went wrong (at least one).
    Panicked(Vec<WorkerPanic>),
}

impl core::fmt::Display for WorkerPanic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.message)?;
        match &self.location {
            Some(location) => write!(f, " at {location}"),
            None => Ok(()),
        }
    }
}

impl core::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotStarted(error) => write!(f, "could not start a thread: {error}"),
            Self::Panicked(panics) => {
                f.write_str("the work panicked")?;
                if let Some(first) = panics.first() {
                    write!(f, ": {first}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for WorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotStarted(error) => Some(error),
            Self::Panicked(_) => None,
        }
    }
}

/// What makes the text that reports panics when the process aborts.
type AbortReport = dyn Fn(&[WorkerPanic]) -> String + Send + Sync;

/// What the hook needs to act for one run: where to record, and what to write when the process
/// cannot unwind.
#[derive(Clone)]
struct Slot {
    /// What the worker has written to standard error.
    stderr: SharedBuffer,
    /// The panics, in the order they happened.
    panics: Arc<Mutex<Vec<WorkerPanic>>>,
    /// The text that reports the panics when the process aborts.
    on_abort: Arc<AbortReport>,
}

std::thread_local! {
    /// Where a panic on this thread is recorded: set on a worker thread only.
    static WORKER: RefCell<Option<Slot>> = const { RefCell::new(None) };
}

/// The message of a panic.
fn panic_message(payload: &(dyn core::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        String::from(*text)
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        String::from("a panic without a message")
    }
}

/// Installs, once in the process, a panic hook that records a panic on a worker thread for the run
/// to report, and hands any other to the hook installed before it.
fn install_panic_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let slot = WORKER
                .try_with(|slot| slot.try_borrow().ok().and_then(|slot| slot.clone()))
                .ok()
                .flatten();
            let Some(slot) = slot else {
                return previous(info);
            };
            let panic = WorkerPanic {
                message: panic_message(info.payload()),
                location: info.location().map(ToString::to_string),
                backtrace: {
                    let backtrace = std::backtrace::Backtrace::capture();
                    (backtrace.status() == std::backtrace::BacktraceStatus::Captured)
                        .then(|| backtrace.to_string())
                },
            };
            let mut recorded = slot.panics.lock().unwrap_or_else(PoisonError::into_inner);
            if info.can_unwind() && !cfg!(panic = "abort") {
                // The first panic is the one that says what went wrong; a later one (a destructor
                // that panics while the first unwinds) is kept after it.
                recorded.push(panic);
            } else {
                // The process aborts after the hook (the panic cannot unwind, or the program is
                // built with `panic = "abort"`), and nothing would write what the worker had
                // rendered or the report.
                recorded.push(panic);
                let mut stderr = io::stderr();
                let _ = stderr.write_all(&slot.stderr.take());
                let _ = stderr.write_all((slot.on_abort)(&recorded).as_bytes());
                drop(recorded);
                let _ = stderr.flush();
                previous(info);
            }
        }));
    });
}

/// Runs `work` on a thread of its own named `thread_name`, with a stack of [STACK_SIZE], and
/// waits for it.
///
/// `work` may borrow from the caller. What it writes to `stderr` before it panics is still there
/// when this returns, for the caller to write out. `on_abort` makes the text that reports the
/// panics when the process aborts; see the [module](self) for when that is.
pub fn run<T: Send>(
    thread_name: &str,
    stderr: &SharedBuffer,
    on_abort: impl Fn(&[WorkerPanic]) -> String + Send + Sync + 'static,
    work: impl FnOnce() -> T + Send,
) -> Result<T, WorkerError> {
    install_panic_hook();
    let slot = Slot {
        stderr: stderr.clone(),
        panics: Arc::default(),
        on_abort: Arc::new(on_abort),
    };
    let joined = std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name(String::from(thread_name))
            .stack_size(STACK_SIZE)
            .spawn_scoped(scope, || {
                WORKER.with(|worker| *worker.borrow_mut() = Some(slot.clone()));
                work()
            })?;
        Ok::<_, io::Error>(worker.join())
    });
    match joined {
        Err(error) => Err(WorkerError::NotStarted(error)),
        Ok(Ok(value)) => Ok(value),
        Ok(Err(payload)) => {
            // The hook recorded the panic, unless a host replaced it: then its message is all
            // there is.
            let mut panics =
                core::mem::take(&mut *slot.panics.lock().unwrap_or_else(PoisonError::into_inner));
            if panics.is_empty() {
                panics.push(WorkerPanic {
                    message: panic_message(payload.as_ref()),
                    location: None,
                    backtrace: None,
                });
            }
            Err(WorkerError::Panicked(panics))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(_panics: &[WorkerPanic]) -> String {
        String::new()
    }

    #[test]
    fn a_panic_is_displayed_as_its_message_and_where_it_happened_when_known() {
        let mut panic = WorkerPanic {
            message: String::from("it broke"),
            location: None,
            backtrace: None,
        };
        assert_eq!(panic.to_string(), "it broke");
        panic.location = Some(String::from("src/a.rs:1:2"));
        assert_eq!(panic.to_string(), "it broke at src/a.rs:1:2");
    }

    #[test]
    fn a_worker_error_is_displayed_with_its_cause_and_is_an_error() {
        let panics = WorkerError::Panicked(vec![WorkerPanic {
            message: String::from("it broke"),
            location: Some(String::from("src/a.rs:1:2")),
            backtrace: None,
        }]);
        assert_eq!(panics.to_string(), "the work panicked: it broke at src/a.rs:1:2");
        let not_started = WorkerError::NotStarted(io::Error::other("no threads"));
        assert_eq!(not_started.to_string(), "could not start a thread: no threads");
        let error: &dyn std::error::Error = &not_started;
        assert!(error.source().is_some());
        let error: &dyn std::error::Error = &panics;
        assert!(error.source().is_none());
    }

    #[test]
    fn a_run_returns_the_value_of_its_work_which_may_borrow_from_the_caller() {
        let words = [String::from("a"), String::from("b")];
        let buffer = SharedBuffer::default();
        let joined = run("test worker", &buffer, quiet, || words.join("+"));
        assert_eq!(joined.unwrap(), "a+b");
        assert_eq!(words.len(), 2);
    }

    #[test]
    fn a_run_works_on_the_thread_it_names() {
        let buffer = SharedBuffer::default();
        let name = run("a named worker", &buffer, quiet, || {
            std::thread::current().name().map(str::to_owned)
        });
        assert_eq!(name.unwrap().as_deref(), Some("a named worker"));
    }

    #[test]
    fn a_panic_is_an_error_with_its_message_and_location_and_the_text_pushed_before_it_stays() {
        let buffer = SharedBuffer::default();
        let joined: Result<(), WorkerError> = run("test worker", &buffer, quiet, || {
            buffer.push("rendered before\n");
            panic!("the work panics on purpose");
        });
        let Err(WorkerError::Panicked(panics)) = joined else {
            panic!("the work panicked");
        };
        assert_eq!(panics.len(), 1, "{panics:?}");
        assert_eq!(panics[0].message, "the work panics on purpose");
        let location = panics[0].location.as_deref().expect("the hook saw the location");
        assert!(location.contains("worker.rs"), "{location}");
        assert_eq!(buffer.take(), b"rendered before\n");
    }

    #[test]
    fn a_run_after_a_run_that_panicked_is_unaffected() {
        let buffer = SharedBuffer::default();
        let first: Result<(), WorkerError> =
            run("test worker", &buffer, quiet, || panic!("the first run panics"));
        assert!(matches!(first, Err(WorkerError::Panicked(_))));
        let second = run("test worker", &buffer, quiet, || 7);
        assert_eq!(second.unwrap(), 7);
    }

    #[test]
    fn a_panic_on_a_thread_a_run_did_not_start_is_not_recorded_for_a_later_run() {
        let buffer = SharedBuffer::default();
        // The panic goes to the previous hook (the test process's standard error), not to a run.
        let _ = std::thread::spawn(|| panic!("a stray thread panics")).join();
        let second: Result<(), WorkerError> =
            run("test worker", &buffer, quiet, || panic!("the run panics"));
        let Err(WorkerError::Panicked(panics)) = second else {
            panic!("the work panicked");
        };
        assert_eq!(panics.len(), 1, "{panics:?}");
        assert_eq!(panics[0].message, "the run panics");
    }

    #[test]
    fn a_panic_on_another_thread_while_the_work_goes_on_does_not_fail_it() {
        let buffer = SharedBuffer::default();
        let joined = run("test worker", &buffer, quiet, || {
            let _ = std::thread::spawn(|| panic!("a stray thread panics")).join();
            "done"
        });
        assert_eq!(joined.unwrap(), "done");
    }
}
