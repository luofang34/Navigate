//! Argument builders and typed extraction for the clap builder API.
//!
//! The clap derives expand to `#[allow(clippy::restriction)]`, which the
//! crate's `forbid` lint levels reject, so commands are built explicitly.
use crate::BenchError;
use clap::{Arg, ArgAction, ArgMatches, builder::ValueParser, error::ErrorKind, value_parser};
use std::path::PathBuf;

pub(crate) fn positional(id: &'static str) -> Arg {
    Arg::new(id)
        .required(true)
        .value_parser(value_parser!(PathBuf))
}

pub(crate) fn path(id: &'static str) -> Arg {
    Arg::new(id).long(id).value_parser(value_parser!(PathBuf))
}

pub(crate) fn flag(id: &'static str) -> Arg {
    Arg::new(id).long(id).action(ArgAction::SetTrue)
}

pub(crate) fn value(
    id: &'static str,
    default: &'static str,
    parser: impl Into<ValueParser>,
) -> Arg {
    Arg::new(id)
        .long(id)
        .default_value(default)
        .value_parser(parser.into())
}

pub(crate) fn take<T: Clone + Send + Sync + 'static>(
    matches: &mut ArgMatches,
    id: &str,
) -> Result<T, BenchError> {
    matches.remove_one(id).ok_or_else(|| {
        BenchError::Arguments(clap::Error::raw(
            ErrorKind::MissingRequiredArgument,
            format!("missing argument {id}\n"),
        ))
    })
}
