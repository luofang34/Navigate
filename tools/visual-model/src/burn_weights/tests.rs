use super::*;

#[test]
fn a_file_with_another_digest_is_rejected_before_it_is_read() {
    let error = prepare(Path::new("unused.ckpt"), b"not the checkpoint", true).unwrap_err();
    assert!(matches!(error, Error::Digest { expected, .. } if expected == CHECKPOINT_SHA256));
}
