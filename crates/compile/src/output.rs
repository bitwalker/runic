//! What a compilation delivered: the outputs that were requested, and writing a file.

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::CheckpointId;

/// One output a compilation delivered, tagged with the checkpoint it was rendered at.
///
/// A file is not read back: the entry says where it was written. Bytes bound for standard output
/// are kept and never printed by the harness, so whoever embeds the compiler decides what to do
/// with them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Emitted {
    /// The file at `path` was written.
    File {
        /// The checkpoint whose artifact the file holds.
        checkpoint: CheckpointId,
        /// Where the file was written.
        path: PathBuf,
    },
    /// The output meant for standard output.
    Stdout {
        /// The checkpoint whose artifact the bytes render.
        checkpoint: CheckpointId,
        /// The rendered artifact.
        bytes: Vec<u8>,
    },
    /// The output the caller asked to receive in memory.
    Memory {
        /// The checkpoint whose artifact the bytes render.
        checkpoint: CheckpointId,
        /// The rendered artifact.
        bytes: Vec<u8>,
    },
}

impl Emitted {
    /// The checkpoint the output was rendered at.
    pub fn checkpoint(&self) -> CheckpointId {
        match self {
            Self::File { checkpoint, .. }
            | Self::Stdout { checkpoint, .. }
            | Self::Memory { checkpoint, .. } => *checkpoint,
        }
    }

    /// The rendered bytes, or `None` for a file, which holds them instead.
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::File { .. } => None,
            Self::Stdout { bytes, .. } | Self::Memory { bytes, .. } => Some(bytes),
        }
    }
}

/// Writes `bytes` to `path`.
///
/// A regular file, or a path where nothing is yet, is written through a temporary file beside it
/// (named `<name>.runic-<pid>-<n>.tmp`, where `<n>` counts the writes of this process, so two
/// threads that write at once never share one), renamed into place once it is complete, so that a write that
/// fails part of the way leaves `path` as it was; the file keeps its permissions. A link is
/// followed, and what it points at is written, so the link stays. Anything else (a device such as
/// `/dev/null`, a pipe) is written directly. The directory of `path` must exist.
pub fn write_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let target = match fs::canonicalize(path) {
        Ok(target) => target,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // A link to nothing: writing through it creates what it points at.
            if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
                return write_directly(path, bytes);
            }
            path.to_path_buf()
        }
        Err(error) => return Err(error),
    };
    let permissions = match fs::metadata(&target) {
        Ok(metadata) if metadata.is_file() => Some(metadata.permissions()),
        Ok(_) => return write_directly(&target, bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let name = target
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the path names no file"))?;
    static WRITES: AtomicU64 = AtomicU64::new(0);
    // The temporary file is created anew, never opened if something is already there under its
    // name: a name that is taken is passed over.
    let mut attempts = 0;
    let (temporary, created) = loop {
        let write = WRITES.fetch_add(1, Ordering::Relaxed);
        let mut temporary = name.to_os_string();
        temporary.push(format!(".runic-{}-{write}.tmp", std::process::id()));
        let temporary = target.with_file_name(temporary);
        match fs::OpenOptions::new().write(true).create_new(true).open(&temporary) {
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists && attempts < 16 => {
                attempts += 1;
            }
            created => break (temporary, created),
        }
    };
    let written = created.and_then(|mut file| {
        file.write_all(bytes)?;
        if let Some(permissions) = permissions {
            file.set_permissions(permissions)?;
        }
        file.sync_all()
    });
    match written.and_then(|()| fs::rename(&temporary, &target)) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}

/// Writes `bytes` to `path` as it is, for what a temporary file cannot stand in for.
fn write_directly(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(bytes)?;
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_file_is_written_with_the_bytes_and_no_temporary_is_left() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.runic");
        write_file(&path, b"hello").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello");
        let names: Vec<_> =
            fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names, ["out.runic"]);
    }

    #[cfg(unix)]
    #[test]
    fn a_replaced_file_holds_the_new_bytes_and_keeps_its_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.runic");
        fs::write(&path, b"old, and longer").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        write_file(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o640);
    }

    #[cfg(unix)]
    #[test]
    fn a_write_that_fails_leaves_the_old_file_as_it_was() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.runic");
        fs::write(&path, b"old").unwrap();
        // Nothing can be created in the directory, so the temporary file cannot be.
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();
        let outcome = write_file(&path, b"new");
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(outcome.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1, "no temporary file is left");
    }

    #[test]
    fn a_missing_directory_is_a_not_found_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("out.runic");
        let error = write_file(&path, b"x").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(!path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_link_is_followed_and_stays_a_link() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.runic");
        let link = dir.path().join("link.runic");
        fs::write(&real, b"old").unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();
        write_file(&link, b"new").unwrap();
        assert!(fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
        assert_eq!(fs::read(&real).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn a_link_to_nothing_creates_what_it_points_at() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.runic");
        let link = dir.path().join("link.runic");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        write_file(&link, b"new").unwrap();
        assert!(fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
        assert_eq!(fs::read(&real).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn a_device_is_written_directly() {
        write_file(Path::new("/dev/null"), b"discarded").unwrap();
        let temporaries = fs::read_dir("/dev")
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("null.runic-"))
            .count();
        assert_eq!(temporaries, 0, "no temporary file is made beside a device");
    }

    #[test]
    fn an_emitted_output_reports_its_checkpoint_and_its_bytes() {
        let file = Emitted::File {
            checkpoint: CheckpointId::IR_LOWERED,
            path: PathBuf::from("a.runic"),
        };
        let stdout = Emitted::Stdout {
            checkpoint: CheckpointId::IR_OPTIMIZED,
            bytes: b"out".to_vec(),
        };
        let memory = Emitted::Memory {
            checkpoint: CheckpointId::IR_TRANSLATED,
            bytes: b"mem".to_vec(),
        };
        assert_eq!(file.checkpoint(), CheckpointId::IR_LOWERED);
        assert_eq!(stdout.checkpoint(), CheckpointId::IR_OPTIMIZED);
        assert_eq!(memory.checkpoint(), CheckpointId::IR_TRANSLATED);
        assert_eq!(file.bytes(), None);
        assert_eq!(stdout.bytes(), Some(&b"out"[..]));
        assert_eq!(memory.bytes(), Some(&b"mem"[..]));
    }
}
