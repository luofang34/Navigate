//! The storage classes of offline data (Pilotage ADR-0044).

/// Where a resource lives, by what the platform may do with it.
///
/// A host maps each class to a platform root. Code names a resource by class
/// and logical path, never by a platform path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum StorageClass {
    /// Data that the operator makes, such as missions. Included in backup.
    Data,
    /// Released data that the host can download again, such as navigation
    /// data, terrain, and imagery. Excluded from backup and never purged.
    Offline,
    /// Derived data that the host can make again. The platform can purge it.
    Cache,
    /// Settings.
    Config,
    /// Data for one operation.
    Temp,
}

impl StorageClass {
    /// Every storage class, in declaration order.
    pub const ALL: [Self; 5] = [
        Self::Data,
        Self::Offline,
        Self::Cache,
        Self::Config,
        Self::Temp,
    ];

    /// The URI scheme that names this class, for example `offline`.
    pub const fn scheme(self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::Offline => "offline",
            Self::Cache => "cache",
            Self::Config => "config",
            Self::Temp => "temp",
        }
    }

    /// The class that a URI scheme names, if any.
    pub fn from_scheme(scheme: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.scheme() == scheme)
    }
}

#[cfg(test)]
mod tests;
