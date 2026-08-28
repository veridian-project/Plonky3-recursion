mod common;

use p3_baby_bear::default_babybear_poseidon2_16;
use p3_circuit::ops::{generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit::test_utils::{FibonacciAir, generate_trace_rows};
use p3_circuit::{CircuitBuilder, CircuitError};
use p3_field::PrimeCharacteristicRing;
use p3_poseidon2_circuit_air::BabyBearD4Width16;
use p3_recursion::generation::{FriGenerationParams, generate_uni_fri_witness_context};
use p3_recursion::pcs::fri::{
    ExpandedFriMmcsPaths, FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs,
    expand_fri_mmcs_paths,
};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::public_inputs::StarkVerifierInputsBuilder;
use p3_recursion::{Poseidon2Config, VerificationError, verify_p3_uni_proof_circuit};
use p3_test_utils::baby_bear_params::*;
use p3_uni_stark::{prove, verify};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::common::InnerFriGeneric;

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;

fn init_logger() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
}

struct FibonacciTestSetup {
    config: MyConfig,
    perm: Perm,
    fri_verifier_params: FriVerifierParams,
    proof: p3_uni_stark::Proof<MyConfig>,
    pis: Vec<F>,
    air: FibonacciAir,
}

fn build_fibonacci_test_setup() -> FibonacciTestSetup {
    let n = 1 << 3;
    let x = 21;

    let trace = generate_trace_rows::<F>(0, 1, n);

    let config = make_test_config();
    // Same default permutation make_test_config uses, for the recursive verifier circuit.
    let perm = default_babybear_poseidon2_16();

    // Enable MMCS verification
    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        scalars.num_queries,
        Poseidon2Config::BABY_BEAR_D4_W16,
    );
    let pis = vec![F::ZERO, F::ONE, F::from_u64(x)];
    let air = FibonacciAir {};
    let proof = prove(&config, &air, trace, &pis);

    FibonacciTestSetup {
        config,
        perm,
        fri_verifier_params,
        proof,
        pis,
        air,
    }
}

fn expand_fibonacci_paths(
    setup: &FibonacciTestSetup,
    proof: &p3_uni_stark::Proof<MyConfig>,
    pis: &[F],
) -> Result<ExpandedFriMmcsPaths<F, DIGEST_ELEMS>, VerificationError> {
    let params = setup.fri_verifier_params;
    let witness = generate_uni_fri_witness_context(
        &setup.air,
        &setup.config,
        proof,
        pis,
        FriGenerationParams {
            log_final_height: params.log_blowup + params.log_final_poly_len,
            commit_pow_bits: params.commit_pow_bits,
            query_pow_bits: params.query_pow_bits,
            num_queries: params.num_queries,
        },
        None,
    )
    .map_err(|error| VerificationError::InvalidProofShape(error.to_string()))?;
    let hash = MyHash::new(setup.perm.clone());
    let compress = MyCompress::new(setup.perm.clone());
    expand_fri_mmcs_paths::<
        F,
        Challenge,
        MyMmcs,
        ChallengeMmcs,
        MyHash,
        MyCompress,
        MyHash,
        MyCompress,
        2,
        DIGEST_ELEMS,
    >(
        &proof.opening_proof,
        &witness.input_batches,
        witness.alpha,
        &witness.betas,
        &witness.query_indices,
        params.log_blowup,
        params.log_final_poly_len,
        &hash,
        &compress,
        0,
        &hash,
        &compress,
        0,
    )
    .map_err(|error| VerificationError::InvalidProofShape(error.to_string()))
}

fn run_recursive_verifier(
    setup: &FibonacciTestSetup,
    proof: &p3_uni_stark::Proof<MyConfig>,
    pis: &[F],
) -> Result<(), VerificationError> {
    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width16>,
        setup.perm.clone(),
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    // Allocate all targets
    let verifier_inputs = StarkVerifierInputsBuilder::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFri,
    >::allocate(&mut circuit_builder, proof, None, pis.len());

    // Add the verification circuit to the builder.
    let mmcs_op_ids = verify_p3_uni_proof_circuit::<
        FibonacciAir,
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        _,
        WIDTH,
        RATE,
    >(
        &setup.config,
        &setup.air,
        &mut circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &None,
        &setup.fri_verifier_params,
        Poseidon2Config::BABY_BEAR_D4_W16,
    )?;

    // Build the circuit.
    let circuit = circuit_builder.build()?;

    let mut runner = circuit.runner();

    // Pack values using the same builder
    let (public_inputs, private_inputs) = verifier_inputs.pack_values(pis, proof, &None);
    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    let expanded_paths = expand_fibonacci_paths(setup, proof, pis)?;
    set_fri_mmcs_private_data::<F, Challenge, DIGEST_ELEMS>(
        &mut runner,
        &mmcs_op_ids,
        &expanded_paths,
        Poseidon2Config::BABY_BEAR_D4_W16,
    )
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;

    runner.run().map_err(VerificationError::Circuit)?;

    Ok(())
}

fn assert_recursive_rejects_tampering(result: Result<(), VerificationError>) {
    match result {
        Err(VerificationError::InvalidProofShape(message))
            if message.contains("Witness check failed during challenge generation") => {}
        Err(VerificationError::Circuit(CircuitError::WitnessConflict { .. })) => {}
        Err(error) => panic!("unexpected recursive rejection: {error}"),
        Ok(()) => panic!("recursive verifier accepted a tampered proof"),
    }
}

#[test]
fn test_fibonacci_verifier() -> Result<(), VerificationError> {
    init_logger();
    let setup = build_fibonacci_test_setup();
    assert!(verify(&setup.config, &setup.air, &setup.proof, &setup.pis).is_ok());
    run_recursive_verifier(&setup, &setup.proof, &setup.pis)
}

/// A tampered trace commitment changes the Fiat-Shamir transcript. It must fail
/// either at the bound PoW witness or at a downstream circuit equality.
#[test]
fn test_tampered_trace_commitment() {
    let mut setup = build_fibonacci_test_setup();

    // The cap at height 0 contains a single digest; corrupt its first word.
    let mut roots = setup.proof.commitments.trace.into_roots();
    roots[0][0] += F::ONE;
    setup.proof.commitments.trace = roots.into();

    assert_recursive_rejects_tampering(run_recursive_verifier(&setup, &setup.proof, &setup.pis));
}

/// Flipping a coefficient in the FRI final polynomial breaks the low-degree test,
/// so either its bound PoW witness or the folding equations must reject it.
#[test]
fn test_tampered_fri_final_poly() {
    let mut setup = build_fibonacci_test_setup();

    setup.proof.opening_proof.final_poly[0] += Challenge::ONE;

    assert_recursive_rejects_tampering(run_recursive_verifier(&setup, &setup.proof, &setup.pis));
}

/// Wrong public inputs change the Fiat-Shamir transcript and the claimed AIR
/// statement, so either the bound PoW witness or a circuit equality must reject.
#[test]
fn test_wrong_public_inputs() {
    let setup = build_fibonacci_test_setup();

    let mut wrong_pis = setup.pis.clone();
    // Corrupt the claimed output value.
    wrong_pis[2] += F::ONE;

    assert_recursive_rejects_tampering(run_recursive_verifier(&setup, &setup.proof, &wrong_pis));
}

/// Modifying an OOD trace evaluation changes the quotient-consistency check
/// and transcript, so either the bound PoW witness or a circuit equality must reject.
#[test]
fn test_tampered_ood_evaluation() {
    let mut setup = build_fibonacci_test_setup();

    setup.proof.opened_values.trace_local[0] += Challenge::ONE;

    assert_recursive_rejects_tampering(run_recursive_verifier(&setup, &setup.proof, &setup.pis));
}

/// A proof with fewer query rounds than the circuit expects causes
/// `set_fri_mmcs_private_data` to report a shape mismatch, returned as
/// VerificationError::InvalidProofShape.
#[test]
fn test_truncated_fri_proof() {
    let setup = build_fibonacci_test_setup();

    assert!(
        setup
            .proof
            .opening_proof
            .input_openings
            .first()
            .is_some_and(|opening| !opening.opened_values.is_empty()),
        "need at least one query round to truncate"
    );

    // Build the circuit against the valid proof so op_ids match the full shape.
    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width16>,
        setup.perm.clone(),
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    // Allocate all targets
    let verifier_inputs = StarkVerifierInputsBuilder::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFri,
    >::allocate(&mut circuit_builder, &setup.proof, None, setup.pis.len());
    // Add the verification circuit to the builder.
    let mmcs_op_ids = verify_p3_uni_proof_circuit::<
        FibonacciAir,
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        _,
        WIDTH,
        RATE,
    >(
        &setup.config,
        &setup.air,
        &mut circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &None,
        &setup.fri_verifier_params,
        Poseidon2Config::BABY_BEAR_D4_W16,
    )
    .unwrap();
    // Build the circuit.
    let circuit = circuit_builder.build().unwrap();
    let mut runner = circuit.runner();
    // Pack values using the same builder
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&setup.pis, &setup.proof, &None);

    runner.set_public_inputs(&public_inputs).unwrap();
    runner.set_private_inputs(&private_inputs).unwrap();

    // Supply truncated expanded paths, giving fewer siblings than op_ids expects.
    let mut truncated_paths = expand_fibonacci_paths(&setup, &setup.proof, &setup.pis).unwrap();
    truncated_paths.input_paths.pop();
    truncated_paths.commit_phase_paths.pop();

    let result = set_fri_mmcs_private_data::<F, Challenge, DIGEST_ELEMS>(
        &mut runner,
        &mmcs_op_ids,
        &truncated_paths,
        Poseidon2Config::BABY_BEAR_D4_W16,
    )
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()));

    assert!(
        matches!(result, Err(VerificationError::InvalidProofShape(_))),
        "expected InvalidProofShape for a truncated FRI proof, got: {result:?}",
    );
}
