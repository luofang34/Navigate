//! One navigator and one flight director, linked as trait objects.

use navigate_contract::{MonotonicNanos, NavigationSolution};

use crate::{DirectorOutput, FlightDirector, Navigator};

/// The onboard bundle.
pub struct Onboard<'a> {
    /// The navigator.
    pub navigator: Box<dyn Navigator + 'a>,
    /// The flight director.
    pub director: Box<dyn FlightDirector + 'a>,
}

impl<'a> Onboard<'a> {
    /// A bundle of `navigator` and `director`.
    pub fn new(navigator: Box<dyn Navigator + 'a>, director: Box<dyn FlightDirector + 'a>) -> Self {
        Self {
            navigator,
            director,
        }
    }

    /// The solution at `now` and the director step on it.
    pub fn direct(&mut self, now: MonotonicNanos) -> (Option<NavigationSolution>, DirectorOutput) {
        let solution = self.navigator.solution(now);
        let output = self.director.step(solution.as_ref(), now);
        (solution, output)
    }
}
