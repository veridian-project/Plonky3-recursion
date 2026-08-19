//! Production-shape compatibility test for Veridian's recursive wrapper.
//!
//! The inner proof uses Goldilocks^5, width-12/rate-6 Poseidon2 with
//! Veridian's exact round-constant seed, HidingFriPcs, and scalar salted
//! Merkle MMCSs. The test native-verifies that proof, verifies it inside a
//! recursive circuit (including salted Merkle paths), then proves and
//! native-verifies the resulting D5 verifier circuit.

use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{ProverData, StarkInstance, prove_batch, verify_batch};
use p3_challenger::DuplexChallenger;
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{
    GoldilocksD1Width12, Poseidon2Config, Poseidon2PermCall, generate_poseidon2_trace,
    generate_recompose_trace,
};
use p3_circuit_prover::batch_stark_prover::{
    poseidon2_air_builders_d5, poseidon2_table_provers_binomial_d5, recompose_air_builders,
};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor,
    RecomposePreprocessor, TablePacking,
};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_lookup::logup::LogUpGadget;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_recursion::pcs::fri::{
    FriVerifierParams, HidingFriProofTargets, InputProofTargets, MerkleCapTargets,
    RecExtensionValMmcs, RecValHidingScalarMmcs, Witness,
};
use p3_recursion::pcs::set_hiding_salted_fri_mmcs_private_data;
use p3_recursion::{BatchStarkVerifierInputsBuilder, VerificationError, verify_batch_circuit};
use p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation};
use p3_uni_stark::StarkConfig;
use rand::SeedableRng;
use rand::rngs::SmallRng;

const VERIDIAN_POSEIDON2_SEED: u64 = 0x0056_4552_4944_414e;
const WIDTH: usize = 12;
const RATE: usize = 6;
const DIGEST_ELEMS: usize = 6;
const SALT_ELEMS: usize = 4;

type F = Goldilocks;
type Challenge = BinomialExtensionField<F, 5>;
type Dft = Radix2DitParallel<F>;
type Perm = Poseidon2Goldilocks<WIDTH>;
type Hash = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_ELEMS>;
type Compress = TruncatedPermutation<Perm, 2, DIGEST_ELEMS, WIDTH>;
type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;

type HidingValMmcs =
    MerkleTreeHidingMmcs<F, F, Hash, Compress, SmallRng, 2, DIGEST_ELEMS, SALT_ELEMS>;
type HidingChallengeMmcs = ExtensionMmcs<F, Challenge, HidingValMmcs>;
type HidingPcs = HidingFriPcs<F, Dft, HidingValMmcs, HidingChallengeMmcs, SmallRng>;
type HidingConfig = StarkConfig<HidingPcs, Challenge, Challenger>;

type OuterValMmcs = MerkleTreeMmcs<F, F, Hash, Compress, 2, DIGEST_ELEMS>;
type OuterChallengeMmcs = ExtensionMmcs<F, Challenge, OuterValMmcs>;
type OuterPcs = TwoAdicFriPcs<F, Dft, OuterValMmcs, OuterChallengeMmcs>;
type OuterConfig = StarkConfig<OuterPcs, Challenge, Challenger>;

type RecursiveHidingValMmcs =
    RecValHidingScalarMmcs<F, DIGEST_ELEMS, SALT_ELEMS, Hash, Compress, SmallRng>;
type InnerFri = HidingFriProofTargets<
    F,
    Challenge,
    RecExtensionValMmcs<F, Challenge, DIGEST_ELEMS, RecursiveHidingValMmcs>,
    InputProofTargets<F, Challenge, RecursiveHidingValMmcs>,
    Witness<F>,
>;

#[derive(Clone)]
struct LiftBasePermutation<P, const W: usize> {
    inner: P,
    _marker: PhantomData<F>,
}

impl<P, const W: usize> LiftBasePermutation<P, W> {
    const fn new(inner: P) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<P, const W: usize> Permutation<[Challenge; W]> for LiftBasePermutation<P, W>
where
    P: Permutation<[F; W]>,
{
    fn permute(&self, input: [Challenge; W]) -> [Challenge; W] {
        let base: [F; W] = core::array::from_fn(|i| input[i].as_basis_coefficients_slice()[0]);
        let output = self.inner.permute(base);
        core::array::from_fn(|i| Challenge::from(output[i]))
    }
}

#[derive(Clone, Copy)]
struct AddAir;

impl<Val: Field> BaseAir<Val> for AddAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for AddAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        builder.assert_zero(row[0] + row[1] - row[2]);
    }
}

fn add_trace(rows: usize) -> RowMajorMatrix<F> {
    let mut values = F::zero_vec(rows * 3);
    for row in 0..rows {
        let a = F::from_usize(row);
        let b = F::from_usize(row + 1);
        values[row * 3] = a;
        values[row * 3 + 1] = b;
        values[row * 3 + 2] = a + b;
    }
    RowMajorMatrix::new(values, 3)
}

fn veridian_perm() -> Perm {
    let mut rng = SmallRng::seed_from_u64(VERIDIAN_POSEIDON2_SEED);
    Perm::new_from_rng_128(&mut rng)
}

fn hiding_config(seed: u64) -> (HidingConfig, FriParameters<HidingChallengeMmcs>) {
    let perm = veridian_perm();
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = HidingValMmcs::new(hash, compress, 0, SmallRng::seed_from_u64(seed + 1));
    let challenge_mmcs = HidingChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
    let pcs = HidingPcs::new(
        Dft::default(),
        val_mmcs,
        fri_params.clone(),
        4,
        SmallRng::seed_from_u64(seed + 2),
    );
    (HidingConfig::new(pcs, Challenger::new(perm)), fri_params)
}

fn outer_config() -> OuterConfig {
    let perm = veridian_perm();
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = OuterValMmcs::new(hash, compress, 0);
    let challenge_mmcs = OuterChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
    let pcs = OuterPcs::new(Dft::default(), val_mmcs, fri_params);
    OuterConfig::new(pcs, Challenger::new(perm))
}

#[test]
fn veridian_d5_w12_raw_compression_binds_every_input() {
    let mut circuit_builder = CircuitBuilder::<Challenge>::new();
    circuit_builder.enable_poseidon2_perm_base_width_12::<GoldilocksD1Width12, _>(
        generate_poseidon2_trace::<Challenge, GoldilocksD1Width12>,
        LiftBasePermutation::new(veridian_perm()),
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    circuit_builder.set_recompose_coeff_ctl_for_decompose_links(true);

    let input_targets = circuit_builder.alloc_private_inputs(WIDTH, "raw_compression_input");
    let base_input_targets: Vec<_> = input_targets
        .iter()
        .copied()
        .map(|target| {
            let coefficients = circuit_builder
                .decompose_ext_to_base_coeffs::<F>(target)
                .expect("decompose raw compression input");
            for coefficient in coefficients.iter().skip(1) {
                let squared = circuit_builder.mul(*coefficient, *coefficient);
                circuit_builder.assert_zero(squared);
            }
            coefficients[0]
        })
        .collect();
    let direction = circuit_builder.define_const(Challenge::ZERO);
    let (_, outputs) = circuit_builder
        .add_poseidon2_perm(&Poseidon2PermCall {
            config: Poseidon2Config::GOLDILOCKS_D1_W12,
            new_start: true,
            merkle_path: true,
            mmcs_bit: Some(direction),
            mmcs_bit2: None,
            inputs: base_input_targets.iter().copied().map(Some).collect(),
            out_ctl: vec![true; RATE],
            return_all_outputs: false,
            absorb_len: 0,
            mmcs_index_sum: None,
        })
        .expect("build the raw compression row");
    let expected_targets: Vec<_> = (0..RATE).map(|_| circuit_builder.public_input()).collect();
    for (output, expected) in outputs[..RATE]
        .iter()
        .map(|output| output.expect("raw compression output is missing"))
        .zip(expected_targets)
    {
        let difference = circuit_builder.sub(output, expected);
        let squared = circuit_builder.mul(difference, difference);
        circuit_builder.assert_zero(squared);
    }

    let circuit = circuit_builder
        .build()
        .expect("build raw compression circuit");
    let native_inputs: [F; WIDTH] = core::array::from_fn(|index| F::from_usize(index + 1));
    let native_outputs = veridian_perm().permute(native_inputs);
    let public_inputs: Vec<_> = native_outputs[..RATE]
        .iter()
        .copied()
        .map(Challenge::from)
        .collect();
    let private_inputs: Vec<_> = native_inputs.into_iter().map(Challenge::from).collect();
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&public_inputs)
        .expect("set raw compression outputs");
    runner
        .set_private_inputs(&private_inputs)
        .expect("set raw compression inputs");
    let traces = runner.run().expect("execute raw compression circuit");

    let packing = TablePacking::new(RATE, 8).with_exposed_public_inputs(RATE);
    let preprocessors: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::new(true)),
    ];
    let mut air_builders = poseidon2_air_builders_d5::<OuterConfig>();
    air_builders.extend(recompose_air_builders::<OuterConfig, 5>(1, true));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<OuterConfig, _, 5>(
            &circuit,
            &packing,
            &preprocessors,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .expect("lower raw compression circuit");
    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
    let config = outer_config();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(config).with_table_packing(packing);
    for table in poseidon2_table_provers_binomial_d5(Poseidon2Config::GOLDILOCKS_D1_W12) {
        prover.register_table_prover(table);
    }
    for table in
        p3_circuit_prover::batch_stark_prover::recompose_table_provers::<OuterConfig, 5>(1, true)
    {
        prover.register_table_prover(table);
    }
    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .expect("prove raw compression circuit");
    prover
        .verify_all_tables::<Challenge>(&proof)
        .expect("raw compression witness bus must balance");

    let mut tampered = private_inputs;
    tampered[0] += Challenge::ONE;
    let mut tampered_runner = circuit.runner();
    tampered_runner
        .set_public_inputs(&public_inputs)
        .expect("set tampered raw compression outputs");
    tampered_runner
        .set_private_inputs(&tampered)
        .expect("set tampered raw compression inputs");
    assert!(
        tampered_runner.run().is_err(),
        "changing one raw compression input must not preserve the claimed output"
    );
}

#[test]
fn veridian_d5_w12_hiding_proof_recurses_end_to_end() -> Result<(), VerificationError> {
    let air = AddAir;
    let trace = add_trace(1 << 6);
    let public_values = vec![vec![]];

    let (inner_config, common_fri) = hiding_config(10);
    let instance = StarkInstance {
        air: &air,
        trace: &trace,
        public_values: vec![],
    };
    let instances = vec![instance];
    let prover_data = ProverData::from_instances(&inner_config, &instances);
    let common = &prover_data.common;
    let inner_proof = prove_batch(&inner_config, &instances, &prover_data);
    verify_batch(&inner_config, &[air], &inner_proof, &public_values, common)
        .expect("native verifier rejected the production-shape hiding proof");

    let (recursive_config, _) = hiding_config(20);
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        common_fri.log_blowup,
        common_fri.log_final_poly_len,
        common_fri.commit_proof_of_work_bits,
        common_fri.query_proof_of_work_bits,
        common_fri.num_queries,
        Poseidon2Config::GOLDILOCKS_D1_W12,
    );

    let mut circuit_builder = CircuitBuilder::<Challenge>::new();
    let lifted = LiftBasePermutation::new(veridian_perm());
    circuit_builder.enable_poseidon2_perm_base_width_12::<GoldilocksD1Width12, _>(
        generate_poseidon2_trace::<Challenge, GoldilocksD1Width12>,
        lifted,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    circuit_builder.set_recompose_coeff_ctl_for_decompose_links(true);

    let lookup_gadget = LogUpGadget::new();
    let verifier_inputs = BatchStarkVerifierInputsBuilder::<
        HidingConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFri,
    >::allocate(&mut circuit_builder, &inner_proof, common, &[0]);
    let mmcs_op_ids = verify_batch_circuit::<_, _, _, _, _, _, _, WIDTH, RATE>(
        &recursive_config,
        &[air],
        &mut circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &fri_verifier_params,
        &verifier_inputs.common_data,
        &lookup_gadget,
        Poseidon2Config::GOLDILOCKS_D1_W12,
    )?;

    let verification_circuit = circuit_builder.build()?;
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&public_values, &inner_proof, common);
    let mut runner = verification_circuit.runner();
    runner.set_public_inputs(&public_inputs)?;
    runner.set_private_inputs(&private_inputs)?;
    assert!(
        !mmcs_op_ids.is_empty(),
        "the compatibility test must verify salted Merkle paths"
    );
    set_hiding_salted_fri_mmcs_private_data::<
        F,
        Challenge,
        HidingChallengeMmcs,
        HidingValMmcs,
        DIGEST_ELEMS,
    >(
        &mut runner,
        &mmcs_op_ids,
        &inner_proof.opening_proof,
        Poseidon2Config::GOLDILOCKS_D1_W12,
    )
    .expect("failed to load scalar salted-MMCS siblings");
    let verification_traces = runner.run()?;

    let verification_table_packing = TablePacking::new(1, 8);
    let preprocessors: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::new(true)),
    ];
    let mut air_builders = poseidon2_air_builders_d5::<OuterConfig>();
    air_builders.extend(recompose_air_builders::<OuterConfig, 5>(1, true));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<OuterConfig, _, 5>(
            &verification_circuit,
            &verification_table_packing,
            &preprocessors,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .expect("failed to lower the D5/W12 verifier circuit");
    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();

    let outer_config = outer_config();
    let outer_prover_data = ProverData::from_airs_and_degrees(&outer_config, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(outer_prover_data, primitive_columns, non_primitive_columns);
    let mut outer_prover =
        BatchStarkProver::new(outer_config).with_table_packing(verification_table_packing);
    for table in poseidon2_table_provers_binomial_d5(Poseidon2Config::GOLDILOCKS_D1_W12) {
        outer_prover.register_table_prover(table);
    }
    outer_prover.register_recompose_table::<5>(true);
    let outer_proof = outer_prover
        .prove_all_tables(&verification_traces, &circuit_prover_data)
        .expect("failed to prove the D5/W12 verifier circuit");
    outer_prover
        .verify_all_tables::<Challenge>(&outer_proof)
        .expect("native verifier rejected the D5/W12 recursive proof");

    Ok(())
}
