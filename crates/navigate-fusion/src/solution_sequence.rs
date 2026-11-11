//! The solution sequence of a filter: it advances only when the
//! published solution changes.
//!
//! A filter publishes a solution on each call of its publication
//! function. A second call for the same state at the same time gives the
//! same solution, and that solution keeps the sequence of the first one.
//! Thus a consumer can use the sequence to find a new solution, and two
//! reads of one state compare equal, the sequence included.

use navigate_contract::{CartesianSolution, NavigationSolution, WrappingSequence};

/// The last solution that a filter published, of either kind.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Published {
    Surface(NavigationSolution),
    Space(CartesianSolution),
}

/// Gives the sequence of each published solution within one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SolutionSequencer {
    next: WrappingSequence,
    last: Option<Published>,
}

impl SolutionSequencer {
    /// A sequencer whose first solution has the sequence zero.
    pub(crate) const fn new() -> Self {
        Self {
            next: WrappingSequence::new(0),
            last: None,
        }
    }

    /// Gives `solution` the sequence of the last published solution when
    /// the two are equal in every other field, and the next sequence
    /// otherwise. A solution with a NaN field never equals the last one,
    /// so it always gets the next sequence.
    pub(crate) fn surface(&mut self, mut solution: NavigationSolution) -> NavigationSolution {
        if let Some(Published::Surface(last)) = self.last {
            solution.stamp.sequence = last.stamp.sequence;
            if solution == last {
                return solution;
            }
        }
        solution.stamp.sequence = self.advance();
        self.last = Some(Published::Surface(solution));
        solution
    }

    /// [`Self::surface`] for a Cartesian solution. A Cartesian solution
    /// never equals a surface solution, so a change of kind advances the
    /// sequence.
    pub(crate) fn space(&mut self, mut solution: CartesianSolution) -> CartesianSolution {
        if let Some(Published::Space(last)) = self.last {
            solution.stamp.sequence = last.stamp.sequence;
            if solution == last {
                return solution;
            }
        }
        solution.stamp.sequence = self.advance();
        self.last = Some(Published::Space(solution));
        solution
    }

    fn advance(&mut self) -> WrappingSequence {
        let sequence = self.next;
        self.next = sequence.next();
        sequence
    }
}
