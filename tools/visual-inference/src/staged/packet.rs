//! Bounded feature packets use explicit little-endian scalar values.
use super::{Features, InferenceError, XFeatFeatures};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"NAVXFT01";
const HEADER: usize = 84;
const RECORD: usize = 284;

impl XFeatFeatures {
    /// Encode a checksummed packet without model weights or image pixels.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER + self.len() * RECORD + 32);
        bytes.extend(MAGIC);
        bytes.extend(self.contract);
        bytes.extend(self.image_digest);
        for value in [self.image_size[0], self.image_size[1], self.len() as u32] {
            bytes.extend(value.to_le_bytes());
        }
        for i in 0..self.len() {
            for p in self.features.pixels[i] {
                bytes.extend(p.to_le_bytes());
            }
            for p in &self.features.model_pixels[i * 2..i * 2 + 2] {
                bytes.extend(p.to_le_bytes());
            }
            bytes.extend(self.features.scores[i].to_le_bytes());
            for value in &self.features.descriptors[i * 64..i * 64 + 64] {
                bytes.extend(value.to_le_bytes());
            }
        }
        let checksum = Sha256::digest(&bytes);
        bytes.extend(checksum);
        bytes
    }

    /// Decode a bounded packet and check its checksum and scalar values.
    /// The backend must then validate the encoder and source image identities.
    ///
    /// # Errors
    /// Rejects malformed, corrupt, oversized, or non-finite feature data.
    pub fn decode(bytes: &[u8], max_features: usize) -> Result<Self, InferenceError> {
        let invalid = || InferenceError::Invalid("invalid XFeat feature packet".into());
        if !(32..=4096).contains(&max_features)
            || bytes.len() < HEADER + 32
            || bytes.len() > HEADER + max_features * RECORD + 32
        {
            return Err(invalid());
        }
        let (body, checksum) = bytes.split_at(bytes.len() - 32);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid());
        }
        let mut reader = Reader {
            bytes: body,
            position: 0,
        };
        if &reader.take::<8>()? != MAGIC {
            return Err(invalid());
        }
        let contract = reader.take()?;
        let image_digest = reader.take()?;
        let width = u32::from_le_bytes(reader.take()?);
        let height = u32::from_le_bytes(reader.take()?);
        let count = u32::from_le_bytes(reader.take()?) as usize;
        if !(16..=8192).contains(&width)
            || !(16..=8192).contains(&height)
            || count > max_features
            || body.len() != HEADER + count * RECORD
        {
            return Err(invalid());
        }
        let mut features = Features::empty(64);
        for _ in 0..count {
            read_feature(&mut reader, &mut features, [width, height])?;
        }
        Ok(Self::new(contract, image_digest, [width, height], features))
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], InferenceError> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| InferenceError::Invalid("packet offset overflow".into()))?;
        let value = self
            .bytes
            .get(self.position..end)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| InferenceError::Invalid("truncated XFeat packet".into()))?;
        self.position = end;
        Ok(value)
    }
    fn f32(&mut self) -> Result<f32, InferenceError> {
        Ok(f32::from_le_bytes(self.take()?))
    }
}
fn read_feature(
    reader: &mut Reader<'_>,
    features: &mut Features,
    size: [u32; 2],
) -> Result<(), InferenceError> {
    let pixels = [
        f64::from_le_bytes(reader.take()?),
        f64::from_le_bytes(reader.take()?),
    ];
    let model = [reader.f32()?, reader.f32()?];
    let score = reader.f32()?;
    let mut descriptor = [0.0_f32; 64];
    for value in &mut descriptor {
        *value = reader.f32()?;
    }
    let norm = descriptor.iter().map(|v| v * v).sum::<f32>();
    if pixels
        .iter()
        .zip(size)
        .any(|(p, n)| !p.is_finite() || *p < 0.0 || *p >= f64::from(n))
        || model
            .iter()
            .zip([800.0, 600.0])
            .any(|(p, n)| !p.is_finite() || *p < 0.0 || *p >= n)
        || !score.is_finite()
        || score <= 0.0
        || descriptor.iter().any(|v| !v.is_finite())
        || !norm.is_finite()
        || (norm - 1.0).abs() > 0.001
    {
        return Err(InferenceError::Invalid(
            "invalid feature coordinates, score, or descriptor".into(),
        ));
    }
    // Preserve the stored descriptor bits. Re-normalization would change cache results.
    features.pixels.push(pixels);
    features.model_pixels.extend(model);
    features.scores.push(score);
    features.descriptors.extend(descriptor);
    Ok(())
}
