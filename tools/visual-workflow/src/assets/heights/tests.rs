use super::*;
use std::io::{self, Cursor};

#[test]
fn file_decode_preserves_all_bits_across_chunk_boundaries() {
    let bits: Vec<u32> = (0..(CHUNK_BYTES / 4 + 11))
        .map(|i| match i % 6 {
            0 => 0x8000_0000,
            1 => 0x7fc0_0123,
            2 => 0x7f80_0000,
            3 => 1,
            _ => (i as f32 * -0.125).to_bits(),
        })
        .collect();
    let bytes: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heights.bin");
    std::fs::write(&path, &bytes).unwrap();
    let digest = hex::encode(Sha256::digest(&bytes));
    let actual = heights_verified_blocking(&path, bits.len(), &digest).unwrap();
    assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    assert!(heights_verified_blocking(&path, bits.len() - 1, &digest).is_err());
    assert!(heights_verified_blocking(&path, bits.len() + 1, &digest).is_err());
    assert!(heights_verified_blocking(&path, bits.len(), &"0".repeat(64)).is_err());
}

struct ShortReads(Cursor<Vec<u8>>);
impl Read for ShortReads {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let size = buffer.len().min(3);
        self.0.read(&mut buffer[..size])
    }
}

#[test]
fn short_reads_and_partial_values_are_handled() {
    let bytes = [1.5_f32, -2.25]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let digest = hex::encode(Sha256::digest(&bytes));
    let path = Path::new("short-heights.bin");
    let actual = decode_blocking(ShortReads(Cursor::new(bytes.clone())), path, 2, &digest).unwrap();
    assert_eq!(actual, [1.5, -2.25]);
    for size in 0..bytes.len() {
        let result = decode_blocking(Cursor::new(&bytes[..size]), path, 2, &digest);
        assert!(
            matches!(result, Err(Error::Io { source, .. }) if source.kind() == io::ErrorKind::UnexpectedEof)
        );
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert!(decode_blocking(Cursor::new(trailing), path, 2, &digest).is_err());
    assert!(decode_blocking(Cursor::new([]), path, usize::MAX, &digest).is_err());
}

#[test]
fn read_failure_keeps_path_and_source() {
    struct Fails;
    impl Read for Fails {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "test source",
            ))
        }
    }
    let path = Path::new("unreadable-heights.bin");
    let result = decode_blocking(Fails, path, 1, &"0".repeat(64));
    assert!(matches!(result, Err(Error::Io { path: actual, source, .. })
        if actual == path && source.kind() == io::ErrorKind::PermissionDenied));
}
