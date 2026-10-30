use super::check_encoder_contract;
use navigate_match::{CorridorLandmarkBank, DEFAULT_KASHIN_SEED, PrecomputedReferenceBank, XFeat};

const CATALOG: [u8; 32] = [1; 32];
const WEIGHTS: [u8; 32] = [2; 32];
const OTHER_WEIGHTS: [u8; 32] = [3; 32];

fn reference_bank(stamp: [u8; 32]) -> PrecomputedReferenceBank {
    let bank =
        PrecomputedReferenceBank::new(CATALOG, stamp, DEFAULT_KASHIN_SEED, Vec::new()).unwrap();
    PrecomputedReferenceBank::from_bytes(&bank.to_bytes()).unwrap()
}

fn corridor_bank(stamp: [u8; 32]) -> CorridorLandmarkBank {
    let bank =
        CorridorLandmarkBank::new(CATALOG, stamp, DEFAULT_KASHIN_SEED, 500.0, Vec::new()).unwrap();
    CorridorLandmarkBank::from_bytes(&bank.to_bytes()).unwrap()
}

#[test]
fn contract_stamped_banks_accept_any_weight_file() {
    let reference = reference_bank(XFeat::ENCODER_CONTRACT_V1);
    let corridor = corridor_bank(XFeat::ENCODER_CONTRACT_V1);
    for weights in [WEIGHTS, OTHER_WEIGHTS] {
        check_encoder_contract("precomputed bank", &reference.encoder_contract, &weights).unwrap();
        check_encoder_contract("corridor bank", &corridor.encoder_contract, &weights).unwrap();
    }
}

#[test]
fn legacy_weight_digest_banks_accept_only_that_file() {
    let reference = reference_bank(WEIGHTS);
    let corridor = corridor_bank(WEIGHTS);
    check_encoder_contract("precomputed bank", &reference.encoder_contract, &WEIGHTS).unwrap();
    check_encoder_contract("corridor bank", &corridor.encoder_contract, &WEIGHTS).unwrap();
    assert!(
        check_encoder_contract(
            "precomputed bank",
            &reference.encoder_contract,
            &OTHER_WEIGHTS
        )
        .is_err()
    );
    assert!(
        check_encoder_contract("corridor bank", &corridor.encoder_contract, &OTHER_WEIGHTS)
            .is_err()
    );
}

#[test]
fn unrelated_stamps_are_rejected_with_both_digests() {
    let corridor = corridor_bank([9; 32]);
    let error =
        check_encoder_contract("corridor bank", &corridor.encoder_contract, &WEIGHTS).unwrap_err();
    let message = error.to_string();
    assert!(message.contains(&"09".repeat(32)), "{message}");
    assert!(message.contains(&"02".repeat(32)), "{message}");
}
