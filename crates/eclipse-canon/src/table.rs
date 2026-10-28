//! The embedded tables: a 16-byte header, then fixed-size little-endian
//! records in catalogue order.

const HEADER_BYTES: usize = 16;

/// The records of `table` if its header has `magic` and `N`-byte records
/// and the body holds exactly the count it gives; otherwise none.
pub(crate) const fn records<'a, const N: usize>(table: &'a [u8], magic: &[u8; 8]) -> &'a [[u8; N]] {
    let Some((header, body)) = table.split_at_checked(HEADER_BYTES) else {
        return &[];
    };
    let mut i = 0;
    while i < 8 {
        if header[i] != magic[i] {
            return &[];
        }
        i += 1;
    }
    let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let size = u16::from_le_bytes([header[12], header[13]]) as usize;
    let (records, rest) = body.as_chunks::<N>();
    if size != N || !rest.is_empty() || records.len() != count {
        return &[];
    }
    records
}

/// `N` bytes of `record` from `offset`.
pub(crate) fn bytes<const M: usize, const N: usize>(record: &[u8; M], offset: usize) -> [u8; N] {
    record
        .get(offset..offset + N)
        .and_then(|slice| slice.try_into().ok())
        .unwrap_or([0; N])
}

pub(crate) fn u16_at<const M: usize>(record: &[u8; M], offset: usize) -> u16 {
    u16::from_le_bytes(bytes(record, offset))
}

pub(crate) fn i16_at<const M: usize>(record: &[u8; M], offset: usize) -> i16 {
    i16::from_le_bytes(bytes(record, offset))
}

pub(crate) fn i32_at<const M: usize>(record: &[u8; M], offset: usize) -> i32 {
    i32::from_le_bytes(bytes(record, offset))
}

pub(crate) fn u32_at<const M: usize>(record: &[u8; M], offset: usize) -> u32 {
    u32::from_le_bytes(bytes(record, offset))
}

/// A `u16` field with `u16::MAX` for an absent value.
pub(crate) fn optional_u16<const M: usize>(record: &[u8; M], offset: usize) -> Option<u16> {
    Some(u16_at(record, offset)).filter(|&value| value != u16::MAX)
}

/// The calendar date at bytes 2 to 5, which every record shares.
pub(crate) fn date<const M: usize>(record: &[u8; M]) -> crate::CalendarDate {
    let [month, day] = bytes(record, 4);
    crate::CalendarDate {
        year: i16_at(record, 2),
        month,
        day,
    }
}
