//! A snapshot of a navigator with a terrain camera carries the state of
//! the camera: the map, the frame order, and the map cells with a fix.

use super::fixes::{across_cells, belly, terrain};
use super::flight::{at, fly, load, navigator};
use super::world::{World, WorldSpec};
use crate::{Navigator, Outcome, SensorInput};

#[test]
fn a_fork_after_a_fix_gives_the_same_records_and_solutions_as_the_original() {
    let world = World::build(&WorldSpec::default());
    let flight = across_cells();
    let mut original = navigator(&flight, belly(), [-20.0, 30.0, -8.0], 40.0);
    load(&mut original, &world);
    let mut before = Vec::new();
    let each_half_second = |step: u64| step.is_multiple_of(25);
    fly(
        &mut original,
        &flight,
        &world,
        &belly(),
        1..=60,
        each_half_second,
        &mut before,
    )
    .unwrap();
    assert_eq!(
        terrain(&before).first().map(|r| &r.0),
        Some(&Outcome::Accepted)
    );
    let mut fork = original.snapshot();
    let kept = original.snapshot();

    // The same inputs for each copy: the third frame of the first cell,
    // then the first frame of the next cell.
    let mut inputs: Vec<SensorInput> = Vec::new();
    for step in 61..=110 {
        inputs.push(flight.increment(step));
        if each_half_second(step) {
            inputs.push(flight.frame(&world, &belly(), step, 0));
        }
    }
    let run = |navigator: &mut dyn Navigator| {
        let mut sink = Vec::new();
        for input in &inputs {
            navigator.ingest(input, &mut sink).unwrap();
        }
        (sink, navigator.solution(at(110)).unwrap())
    };
    let (records, solution) = run(&mut original);
    // The fork knows the fix of the first cell: its frame there shares the
    // map error, and the frame in the next cell is a new fix.
    assert_eq!(
        terrain(&records),
        [
            (Outcome::Unused("shared_map_error"), Some("render")),
            (Outcome::Accepted, Some("render"))
        ]
    );
    let (fork_records, fork_solution) = run(&mut *fork);
    assert_eq!(fork_records, records);
    assert_eq!(fork_solution, solution);

    // A restore puts the original back at the fork point.
    original.restore(&*kept).unwrap();
    let (again_records, again_solution) = run(&mut original);
    assert_eq!(again_records, records);
    assert_eq!(again_solution, solution);
}
