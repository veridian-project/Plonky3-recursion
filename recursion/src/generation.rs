use alloc::vec;
use alloc::vec::Vec;

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BaseAir};
use p3_batch_stark::symbolic::get_log_num_quotient_chunks as get_batch_log_num_quotient_chunks;
use p3_batch_stark::{BatchProof, BatchTranscript, CommonData};
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace};
use p3_field::{
    Algebra, BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField, TwoAdicField,
};
use p3_fri::{BatchMultiOpening, FriProof, HidingFriPcs, TwoAdicFriPcs};
use p3_lookup::symbolic::InteractionSymbolicBuilder;
use p3_lookup::{Lookup, LookupProtocol};
use p3_uni_stark::{
    Domain, PreprocessedVerifierKey, Proof as UniProof, StarkGenericConfig, SymbolicExpression,
    SymbolicExpressionExt, Val,
};
use thiserror::Error;

use crate::pcs::fri::FriInputMatrix;

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("Missing parameter for challenge generation")]
    MissingParameterError,

    #[error("The FRI batch randomization does not correspond to the ZK setting.")]
    RandomizationError,

    #[error("Witness check failed during challenge generation.")]
    InvalidPowWitness,

    #[error("Invalid proof shape: {0}")]
    InvalidProofShape(&'static str),

    #[error(
        "Invalid proof shape: instance metadata length mismatch (airs={airs}, opened_values={opened_values}, public_values={public_values}, degree_bits={degree_bits})"
    )]
    InstanceMetadataLengthMismatch {
        airs: usize,
        opened_values: usize,
        public_values: usize,
        degree_bits: usize,
    },

    #[error(
        "Invalid proof shape: hiding random opening round count mismatch (expected {expected}, got {got})"
    )]
    HidingRandomOpeningRoundCountMismatch { expected: usize, got: usize },

    #[error(
        "Invalid proof shape: hiding random opening matrix count mismatch in round {round} (expected {expected}, got {got})"
    )]
    HidingRandomOpeningMatrixCountMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },

    #[error(
        "Invalid proof shape: hiding random opening point count mismatch in round {round}, matrix {matrix} (expected {expected}, got {got})"
    )]
    HidingRandomOpeningPointCountMismatch {
        round: usize,
        matrix: usize,
        expected: usize,
        got: usize,
    },
}

/// Verifier-known FRI geometry needed to replay query sampling exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FriGenerationParams {
    /// Domain height after every commit-phase fold.
    pub log_final_height: usize,
    /// Number of commit-phase proof-of-work bits required by the verifier.
    pub commit_pow_bits: usize,
    /// Number of query proof-of-work bits required by the verifier.
    pub query_pow_bits: usize,
    /// Consensus query count. The proof cannot choose this value.
    pub num_queries: usize,
}

/// Typed Fiat-Shamir values needed to reconstruct the sole v1 shared FRI
/// multiproofs. Query indices are sampled by the verifier and never read from
/// the proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FriTranscriptChallenges<EF> {
    pub alpha: EF,
    pub betas: Vec<EF>,
    pub query_indices: Vec<usize>,
    pub public_values: Vec<EF>,
}

/// All verifier-derived values and verifier-known matrix metadata needed to
/// expand a batch STARK's shared FRI multiproofs.
#[derive(Clone, Debug)]
pub struct BatchFriWitnessContext<D, EF> {
    pub input_batches: Vec<Vec<FriInputMatrix<D, EF>>>,
    pub alpha: EF,
    pub betas: Vec<EF>,
    pub query_indices: Vec<usize>,
}

/// A type alias for a single opening point and its values.
type PointOpening<SC> = (
    <SC as StarkGenericConfig>::Challenge,
    Vec<<SC as StarkGenericConfig>::Challenge>,
);

/// A type alias for all openings within a specific domain.
type DomainOpenings<SC> = Vec<(Domain<SC>, Vec<PointOpening<SC>>)>;

/// A type alias for a commitment and its associated domain openings.
type CommitmentWithOpenings<SC> = (
    <<SC as StarkGenericConfig>::Pcs as Pcs<
        <SC as StarkGenericConfig>::Challenge,
        <SC as StarkGenericConfig>::Challenger,
    >>::Commitment,
    DomainOpenings<SC>,
);

/// The final type alias for a slice of commitments with their openings.
type ComsWithOpenings<SC> = [CommitmentWithOpenings<SC>];

/// Trait which defines the methods necessary
/// for a Pcs to generate challenge values.
pub trait PcsGeneration<SC: StarkGenericConfig, OpeningProof> {
    /// Reconstruct the exact opening claims observed by the native PCS verifier.
    /// Non-hiding PCS proofs use the public claims unchanged; hiding PCS proofs
    /// append their private random-codeword openings after validating every
    /// round, matrix, and point boundary.
    fn prepare_openings(
        &self,
        coms_to_verify: &ComsWithOpenings<SC>,
        _opening_proof: &OpeningProof,
    ) -> Result<Vec<CommitmentWithOpenings<SC>>, GenerationError> {
        Ok(coms_to_verify.to_vec())
    }

    fn generate_challenges(
        &self,
        config: &SC,
        challenger: &mut SC::Challenger,
        coms_to_verify: &ComsWithOpenings<SC>,
        opening_proof: &OpeningProof,
        // Depending on the `OpeningProof`, we might need additional parameters. For example, for a `FriProof`, we need the `log_max_height` to sample query indices.
        fri_params: Option<FriGenerationParams>,
    ) -> Result<FriTranscriptChallenges<SC::Challenge>, GenerationError>;

    fn num_challenges(
        opening_proof: &OpeningProof,
        fri_params: Option<FriGenerationParams>,
    ) -> Result<usize, GenerationError>;
}

fn replay_pow_witness<F, C>(
    challenger: &mut C,
    bits: usize,
    witness: F,
) -> Result<Option<F>, GenerationError>
where
    F: Field,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if bits == 0 {
        return Ok(None);
    }

    let mut sampled = challenger.clone();
    sampled.observe(witness);
    let sampled: F = sampled.sample();
    if !challenger.check_witness(bits, witness) {
        return Err(GenerationError::InvalidPowWitness);
    }
    Ok(Some(sampled))
}

/// Generates the challenges used in the verification of a batch-STARK proof.
pub fn generate_batch_challenges<SC: StarkGenericConfig, A, LG: LookupProtocol>(
    airs: &[A],
    config: &SC,
    proof: &BatchProof<SC>,
    public_values: &[Vec<Val<SC>>],
    fri_params: Option<FriGenerationParams>,
    common_data: &CommonData<SC>,
    lookup_gadget: &LG,
) -> Result<Vec<SC::Challenge>, GenerationError>
where
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>,
    SC::Pcs: PcsGeneration<SC, <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    generate_batch_challenge_data(
        airs,
        config,
        proof,
        public_values,
        fri_params,
        common_data,
        lookup_gadget,
        &common_data.lookups,
    )
    .map(|(challenges, _, _)| challenges)
}

/// Generate the public transcript values and the typed host context used to
/// expand shared FRI multiproofs for a two-adic PCS.
pub fn generate_batch_fri_witness_context<SC: StarkGenericConfig, A, LG: LookupProtocol>(
    airs: &[A],
    config: &SC,
    proof: &BatchProof<SC>,
    public_values: &[Vec<Val<SC>>],
    fri_params: FriGenerationParams,
    common_data: &CommonData<SC>,
    lookup_gadget: &LG,
    verifier_lookups: &[Vec<Lookup<Val<SC>>>],
) -> Result<
    (
        Vec<SC::Challenge>,
        BatchFriWitnessContext<Domain<SC>, SC::Challenge>,
    ),
    GenerationError,
>
where
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>,
    SC::Pcs: PcsGeneration<SC, <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let (challenges, fri, input_batches) = generate_batch_challenge_data(
        airs,
        config,
        proof,
        public_values,
        Some(fri_params),
        common_data,
        lookup_gadget,
        verifier_lookups,
    )?;
    let input_batches = input_batches
        .into_iter()
        .map(|batch| {
            batch
                .into_iter()
                .map(|(domain, points_and_values)| FriInputMatrix {
                    domain,
                    points_and_values,
                })
                .collect()
        })
        .collect();
    Ok((
        challenges,
        BatchFriWitnessContext {
            input_batches,
            alpha: fri.alpha,
            betas: fri.betas,
            query_indices: fri.query_indices,
        },
    ))
}

/// Replay a single-STARK verifier transcript and return the verifier-derived
/// context needed to expand its shared FRI multiproofs.
pub fn generate_uni_fri_witness_context<SC: StarkGenericConfig, A>(
    air: &A,
    config: &SC,
    proof: &UniProof<SC>,
    public_values: &[Val<SC>],
    fri_params: FriGenerationParams,
    preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
) -> Result<BatchFriWitnessContext<Domain<SC>, SC::Challenge>, GenerationError>
where
    A: BaseAir<Val<SC>>,
    SC::Pcs: PcsGeneration<SC, <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
{
    let pcs = config.pcs();
    let is_zk = config.is_zk();
    let base_degree_bits =
        proof
            .degree_bits
            .checked_sub(is_zk)
            .ok_or(GenerationError::InvalidProofShape(
                "degree bits are smaller than the ZK extension",
            ))?;
    let degree =
        1usize
            .checked_shl(proof.degree_bits as u32)
            .ok_or(GenerationError::InvalidProofShape(
                "trace degree exceeds the platform limit",
            ))?;
    let trace_domain = pcs.natural_domain_for_degree(degree);
    let preprocessed_width =
        preprocessed_vk.map_or_else(|| air.preprocessed_width(), |vk| vk.width);
    let preprocessed_commitment = match (preprocessed_width, preprocessed_vk) {
        (0, None) => None,
        (width, Some(vk)) if width == vk.width && vk.degree_bits == proof.degree_bits => {
            Some(vk.commitment.clone())
        }
        _ => {
            return Err(GenerationError::InvalidProofShape(
                "preprocessed verifier key does not match the proof",
            ));
        }
    };
    if public_values.len() != air.num_public_values() {
        return Err(GenerationError::InvalidProofShape(
            "public-value count does not match the AIR",
        ));
    }

    let quotient_chunks = proof.opened_values.quotient_chunks.len();
    let quotient_domain_size = trace_domain.size().checked_mul(quotient_chunks).ok_or(
        GenerationError::InvalidProofShape("quotient domain size overflow"),
    )?;
    let quotient_domain = trace_domain
        .try_create_disjoint_domain(quotient_domain_size)
        .ok_or(GenerationError::InvalidProofShape(
            "quotient domain is unavailable",
        ))?;
    let quotient_domains = quotient_domain.split_domains(quotient_chunks);
    let randomized_quotient_domains = quotient_domains
        .iter()
        .map(|domain| {
            domain
                .size()
                .checked_shl(is_zk as u32)
                .map(|size| pcs.natural_domain_for_degree(size))
                .ok_or(GenerationError::InvalidProofShape(
                    "randomized quotient domain size overflow",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut challenger = config.initialise_challenger();
    challenger.observe(Val::<SC>::from_usize(proof.degree_bits));
    challenger.observe(Val::<SC>::from_usize(base_degree_bits));
    challenger.observe(Val::<SC>::from_usize(preprocessed_width));
    challenger.observe(proof.commitments.trace.clone());
    if let Some(commitment) = &preprocessed_commitment {
        challenger.observe(commitment.clone());
    }
    challenger.observe_slice(public_values);
    let _: SC::Challenge = challenger.sample_algebra_element();
    challenger.observe(proof.commitments.quotient_chunks.clone());
    if let Some(commitment) = &proof.commitments.random {
        challenger.observe(commitment.clone());
    }
    let zeta: SC::Challenge = challenger.sample_algebra_element();
    let zeta_next = trace_domain
        .next_point(zeta)
        .ok_or(GenerationError::InvalidProofShape(
            "trace domain has no next point",
        ))?;

    let mut coms_to_verify = if let (Some(commitment), Some(values)) =
        (&proof.commitments.random, &proof.opened_values.random)
    {
        vec![(
            commitment.clone(),
            vec![(trace_domain, vec![(zeta, values.clone())])],
        )]
    } else if proof.commitments.random.is_none() && proof.opened_values.random.is_none() {
        Vec::new()
    } else {
        return Err(GenerationError::RandomizationError);
    };

    let mut trace_points = vec![(zeta, proof.opened_values.trace_local.clone())];
    if !air.main_next_row_columns().is_empty() {
        trace_points.push((
            zeta_next,
            proof
                .opened_values
                .trace_next
                .clone()
                .ok_or(GenerationError::InvalidProofShape(
                    "next-row trace opening is missing",
                ))?,
        ));
    }
    coms_to_verify.push((
        proof.commitments.trace.clone(),
        vec![(trace_domain, trace_points)],
    ));
    if randomized_quotient_domains.len() != proof.opened_values.quotient_chunks.len() {
        return Err(GenerationError::InvalidProofShape(
            "quotient opening count mismatch",
        ));
    }
    coms_to_verify.push((
        proof.commitments.quotient_chunks.clone(),
        randomized_quotient_domains
            .iter()
            .zip(&proof.opened_values.quotient_chunks)
            .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
            .collect(),
    ));
    if let Some(commitment) = preprocessed_commitment {
        let mut points = vec![(
            zeta,
            proof.opened_values.preprocessed_local.clone().ok_or(
                GenerationError::InvalidProofShape("preprocessed opening is missing"),
            )?,
        )];
        if !air.preprocessed_next_row_columns().is_empty() {
            points.push((
                zeta_next,
                proof.opened_values.preprocessed_next.clone().ok_or(
                    GenerationError::InvalidProofShape("next-row preprocessed opening is missing"),
                )?,
            ));
        }
        coms_to_verify.push((commitment, vec![(trace_domain, points)]));
    }

    let coms_to_verify = pcs.prepare_openings(&coms_to_verify, &proof.opening_proof)?;
    let fri = pcs.generate_challenges(
        config,
        &mut challenger,
        &coms_to_verify,
        &proof.opening_proof,
        Some(fri_params),
    )?;
    let input_batches = coms_to_verify
        .into_iter()
        .map(|(_, matrices)| {
            matrices
                .into_iter()
                .map(|(domain, points_and_values)| FriInputMatrix {
                    domain,
                    points_and_values,
                })
                .collect()
        })
        .collect();
    Ok(BatchFriWitnessContext {
        input_batches,
        alpha: fri.alpha,
        betas: fri.betas,
        query_indices: fri.query_indices,
    })
}

#[allow(clippy::type_complexity)]
fn generate_batch_challenge_data<SC: StarkGenericConfig, A, LG: LookupProtocol, L>(
    airs: &[A],
    config: &SC,
    proof: &BatchProof<SC>,
    public_values: &[Vec<Val<SC>>],
    fri_params: Option<FriGenerationParams>,
    common_data: &CommonData<SC>,
    lookup_gadget: &LG,
    verifier_lookups: &[L],
) -> Result<
    (
        Vec<SC::Challenge>,
        FriTranscriptChallenges<SC::Challenge>,
        Vec<DomainOpenings<SC>>,
    ),
    GenerationError,
>
where
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>,
    SC::Pcs: PcsGeneration<SC, <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
    L: AsRef<[Lookup<Val<SC>>]>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let all_lookups = verifier_lookups;

    let BatchProof {
        commitments,
        opened_values,
        opening_proof,
        lookup_terminals,
        degree_bits,
    } = proof;

    // Single-terminal layout: each AIR commits exactly one terminal iff it declares any lookup.
    all_lookups
        .iter()
        .zip(lookup_terminals)
        .try_for_each(|(lookups, terminal)| {
            if lookups.as_ref().is_empty() == terminal.is_some() {
                return Err(GenerationError::InvalidProofShape(
                    "Lookup terminal presence does not match the AIR's declared lookups",
                ));
            }
            Ok(())
        })?;

    let n_instances = airs.len();
    if n_instances == 0
        || opened_values.instances.len() != n_instances
        || public_values.len() != n_instances
        || degree_bits.len() != n_instances
    {
        return Err(GenerationError::InstanceMetadataLengthMismatch {
            airs: n_instances,
            opened_values: opened_values.instances.len(),
            public_values: public_values.len(),
            degree_bits: degree_bits.len(),
        });
    }

    // Check randomization consistency against the PCS ZK setting.
    if (opened_values
        .instances
        .iter()
        .any(|ov| ov.base_opened_values.random.is_some() != SC::Pcs::ZK))
        || (commitments.random.is_some() != SC::Pcs::ZK)
    {
        return Err(GenerationError::RandomizationError);
    }

    let pcs = config.pcs();
    let mut transcript = BatchTranscript::<SC>::new(config.initialise_challenger());

    transcript.observe_instance_count(n_instances);

    for inst in &opened_values.instances {
        if inst
            .base_opened_values
            .quotient_chunks
            .iter()
            .any(|c| c.len() != SC::Challenge::DIMENSION)
        {
            return Err(GenerationError::InvalidProofShape(
                "invalid quotient chunk length",
            ));
        }

        if inst
            .base_opened_values
            .random
            .as_ref()
            .is_some_and(|r_vals| r_vals.len() != SC::Challenge::DIMENSION)
        {
            return Err(GenerationError::RandomizationError);
        }
    }

    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    let mut log_quotient_degrees = Vec::with_capacity(n_instances);
    let mut quotient_degrees = Vec::with_capacity(n_instances);
    for (i, air) in airs.iter().enumerate() {
        let is_zk = config.is_zk();
        let base_db =
            degree_bits[i]
                .checked_sub(is_zk)
                .ok_or(GenerationError::InvalidProofShape(
                    "extended degree smaller than zk adjustment",
                ))?;
        let trace_len =
            1usize
                .checked_shl(base_db as u32)
                .ok_or(GenerationError::InvalidProofShape(
                    "base degree bits exceed the platform limit",
                ))?;
        let pre_w = common_data
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances[i].as_ref().map(|m| m.width))
            .unwrap_or(0);
        preprocessed_widths.push(pre_w);

        let batch_layout = AirLayout {
            preprocessed_width: pre_w,
            main_width: air.width(),
            num_public_values: air.num_public_values(),
            ..Default::default()
        };
        let log_qd = get_batch_log_num_quotient_chunks(
            air,
            batch_layout,
            trace_len,
            all_lookups[i].as_ref(),
            config.is_zk(),
            lookup_gadget,
        );
        let quotient_degree = 1 << (log_qd + config.is_zk());
        log_quotient_degrees.push(log_qd);
        quotient_degrees.push(quotient_degree);
    }

    for i in 0..n_instances {
        let ext_db = degree_bits[i];
        let base_db =
            ext_db
                .checked_sub(config.is_zk())
                .ok_or(GenerationError::InvalidProofShape(
                    "extended degree smaller than zk adjustment",
                ))?;

        transcript.observe_instance_binding(
            ext_db,
            base_db,
            A::width(&airs[i]),
            quotient_degrees[i],
        );
    }

    transcript.observe_main(&commitments.main, public_values);
    transcript.observe_preprocessed(&preprocessed_widths, common_data.preprocessed.as_ref());

    let is_lookup = commitments.permutation.is_some();

    // Sample the batch's single permutation challenge pair on the transcript challenger. This has
    // the same transcript effect as the native `sample_perm_challenges` (two `sample_algebra_element`
    // draws) while returning the raw pair the in-circuit verifier samples and connects to.
    let different_challenges = get_different_perm_challenges::<SC, LG, _>(
        &mut transcript.challenger,
        all_lookups,
        lookup_gadget,
    );

    // Then, observe the permutation tables, if any and sample the alpha challenge.
    let alpha = transcript
        .observe_perm_and_sample_alpha(commitments.permutation.as_ref(), lookup_terminals);

    transcript.observe_quotient_commitment(&commitments.quotient_chunks);
    if let Some(random_commit) = &commitments.random {
        transcript.observe_random_commitment(random_commit);
    }
    let zeta = transcript.sample_zeta();

    let trace_domains: Vec<_> = degree_bits
        .iter()
        .map(|&ext_db| {
            let base_db =
                ext_db
                    .checked_sub(config.is_zk())
                    .ok_or(GenerationError::InvalidProofShape(
                        "extended degree smaller than zk adjustment",
                    ))?;
            Ok(pcs.natural_domain_for_degree(1 << base_db))
        })
        .collect::<Result<Vec<_>, GenerationError>>()?;
    let ext_trace_domains: Vec<_> = degree_bits
        .iter()
        .map(|&ext_db| pcs.natural_domain_for_degree(1 << ext_db))
        .collect();

    // We have, in the typical lookup case, up to five rounds:
    // optional random, trace, quotient, optional preprocessed, and optional permutation.
    let mut coms_to_verify = Vec::with_capacity(5);

    if let Some(random_commit) = &commitments.random {
        let random_round = ext_trace_domains
            .iter()
            .zip(opened_values.instances.iter())
            .map(|(domain, inst)| {
                let random_vals = inst
                    .base_opened_values
                    .random
                    .as_ref()
                    .ok_or(GenerationError::RandomizationError)?;
                Ok((*domain, vec![(zeta, random_vals.clone())]))
            })
            .collect::<Result<Vec<_>, GenerationError>>()?;
        coms_to_verify.push((random_commit.clone(), random_round));
    }

    let trace_round = ext_trace_domains
        .iter()
        .zip(trace_domains.iter())
        .zip(opened_values.instances.iter())
        .map(|((ext_dom, trace_dom), inst)| {
            // The `zeta_next` opening is present only when the AIR accesses the next row
            // (mirrors the native prover's `main_next_row_columns` gating).
            let mut points = vec![(zeta, inst.base_opened_values.trace_local.clone())];
            if let Some(trace_next) = &inst.base_opened_values.trace_next {
                let zeta_next =
                    trace_dom
                        .next_point(zeta)
                        .ok_or(GenerationError::InvalidProofShape(
                            "trace domain lacks next point",
                        ))?;
                points.push((zeta_next, trace_next.clone()));
            }
            Ok((*ext_dom, points))
        })
        .collect::<Result<Vec<_>, GenerationError>>()?;
    coms_to_verify.push((commitments.main.clone(), trace_round));

    let quotient_domains: Vec<Vec<_>> = degree_bits
        .iter()
        .zip(ext_trace_domains.iter())
        .zip(log_quotient_degrees.iter())
        .map(
            |((&ext_db, ext_dom), &log_qd)| -> Result<Vec<_>, GenerationError> {
                let base_db = ext_db.checked_sub(config.is_zk()).ok_or(
                    GenerationError::InvalidProofShape(
                        "extended degree smaller than zk adjustment",
                    ),
                )?;
                let q_domain =
                    ext_dom.create_disjoint_domain(1 << (base_db + log_qd + config.is_zk()));
                Ok(q_domain.split_domains(1 << (log_qd + config.is_zk())))
            },
        )
        .collect::<Result<Vec<_>, GenerationError>>()?;

    let randomized_quotient_domains: Vec<Vec<_>> = quotient_domains
        .iter()
        .map(|domains| {
            domains
                .iter()
                .map(|domain| pcs.natural_domain_for_degree(domain.size() << config.is_zk()))
                .collect()
        })
        .collect();

    let mut quotient_round = Vec::with_capacity(
        randomized_quotient_domains
            .iter()
            .map(|domains| domains.len())
            .sum(),
    );
    for (domains, inst) in randomized_quotient_domains
        .iter()
        .zip(opened_values.instances.iter())
    {
        if inst.base_opened_values.quotient_chunks.len() != domains.len() {
            return Err(GenerationError::InvalidProofShape(
                "quotient chunk count mismatch",
            ));
        }
        for (domain, values) in domains
            .iter()
            .zip(inst.base_opened_values.quotient_chunks.iter())
        {
            quotient_round.push((*domain, vec![(zeta, values.clone())]));
        }
    }
    coms_to_verify.push((commitments.quotient_chunks.clone(), quotient_round));

    if let Some(global) = &common_data.preprocessed {
        let mut pre_round = Vec::with_capacity(global.matrix_to_instance.len());

        for (matrix_index, &inst_idx) in global.matrix_to_instance.iter().enumerate() {
            let pre_w = preprocessed_widths[inst_idx];
            if pre_w == 0 {
                return Err(GenerationError::InvalidProofShape(
                    "preprocessed width is zero but commitment exists",
                ));
            }

            let inst = &opened_values.instances[inst_idx];
            let local = inst.base_opened_values.preprocessed_local.as_ref().ok_or(
                GenerationError::InvalidProofShape("preprocessed local values should exist"),
            )?;
            let next = inst.base_opened_values.preprocessed_next.as_ref().ok_or(
                GenerationError::InvalidProofShape("preprocessed next values should exist"),
            )?;

            // Validate that the preprocessed data's degree metadata matches this instance.
            let ext_db = degree_bits[inst_idx];

            let meta =
                global.instances[inst_idx]
                    .as_ref()
                    .ok_or(GenerationError::InvalidProofShape(
                        "Missing preprocessed instance metadata",
                    ))?;
            if meta.matrix_index != matrix_index || meta.degree_bits != ext_db {
                return Err(GenerationError::InvalidProofShape(
                    "Preprocessed instance metadata mismatch",
                ));
            }

            let base_db = meta.degree_bits;
            let pre_domain = pcs.natural_domain_for_degree(1 << base_db);
            let zeta_next_i = trace_domains[inst_idx].next_point(zeta).ok_or(
                GenerationError::InvalidProofShape("Preprocessed domain lacks next point"),
            )?;

            pre_round.push((
                pre_domain,
                vec![(zeta, local.clone()), (zeta_next_i, next.clone())],
            ));
        }

        coms_to_verify.push((global.commitment.clone(), pre_round));
    }

    if is_lookup {
        let permutation_commit = commitments.permutation.clone().unwrap();
        let mut permutation_round = Vec::with_capacity(ext_trace_domains.len());
        for (i, (ext_dom, inst_opened_vals)) in ext_trace_domains
            .iter()
            .zip(opened_values.instances.iter())
            .enumerate()
        {
            if inst_opened_vals.permutation_local.len() != inst_opened_vals.permutation_next.len() {
                return Err(GenerationError::InvalidProofShape(
                    "Permutation opened values length mismatch",
                ));
            }
            if !inst_opened_vals.permutation_local.is_empty() {
                let zeta_next =
                    trace_domains[i]
                        .next_point(zeta)
                        .ok_or(GenerationError::InvalidProofShape(
                            "Missing preprocessed instance metadata",
                        ))?;
                permutation_round.push((
                    *ext_dom,
                    vec![
                        (zeta, inst_opened_vals.permutation_local.clone()),
                        (zeta_next, inst_opened_vals.permutation_next.clone()),
                    ],
                ));
            }
        }
        coms_to_verify.push((permutation_commit, permutation_round));
    }

    let coms_to_verify = pcs.prepare_openings(&coms_to_verify, opening_proof)?;
    let fri_challenges = pcs.generate_challenges(
        config,
        &mut transcript.challenger,
        &coms_to_verify,
        opening_proof,
        fri_params,
    )?;

    let mut challenges = Vec::with_capacity(2 + fri_challenges.public_values.len());
    challenges.extend(different_challenges);
    challenges.push(alpha);
    challenges.push(zeta);
    challenges.extend(fri_challenges.public_values.iter().copied());

    let input_batches = coms_to_verify
        .into_iter()
        .map(|(_, openings)| openings)
        .collect();
    Ok((challenges, fri_challenges, input_batches))
}

type InnerFriProof<SC, InputMmcs, FriMmcs> = FriProof<
    <SC as StarkGenericConfig>::Challenge,
    FriMmcs,
    Val<SC>,
    Vec<BatchMultiOpening<Val<SC>, InputMmcs>>,
>;

impl<SC: StarkGenericConfig, Dft, InputMmcs: Mmcs<Val<SC>>, FriMmcs: Mmcs<SC::Challenge>>
    PcsGeneration<SC, InnerFriProof<SC, InputMmcs, FriMmcs>>
    for TwoAdicFriPcs<Val<SC>, Dft, InputMmcs, FriMmcs>
where
    Val<SC>: TwoAdicField + PrimeField,
    SC::Challenger: FieldChallenger<Val<SC>>
        + GrindingChallenger<Witness = Val<SC>>
        + CanObserve<FriMmcs::Commitment>,
{
    fn generate_challenges(
        &self,
        _config: &SC,
        challenger: &mut SC::Challenger,
        coms_to_verify: &ComsWithOpenings<SC>,
        opening_proof: &InnerFriProof<SC, InputMmcs, FriMmcs>,
        fri_params: Option<FriGenerationParams>,
    ) -> Result<FriTranscriptChallenges<SC::Challenge>, GenerationError> {
        let params = fri_params.ok_or(GenerationError::MissingParameterError)?;
        let num_challenges =
            <Self as PcsGeneration<SC, InnerFriProof<SC, InputMmcs, FriMmcs>>>::num_challenges(
                opening_proof,
                Some(params),
            )?;
        let mut challenges = Vec::with_capacity(num_challenges);

        // Observe all openings.
        for (_, round) in coms_to_verify {
            for (_, mat) in round {
                for (_, point) in mat {
                    point
                        .iter()
                        .for_each(|&opening| challenger.observe_algebra_element(opening));
                }
            }
        }

        let alpha = challenger.sample_algebra_element();
        challenges.push(alpha);

        // Get `beta` challenges for the FRI rounds.
        let mut betas = Vec::with_capacity(opening_proof.commit_phase_commits.len());
        for (comm, pow_witness) in opening_proof
            .commit_phase_commits
            .iter()
            .zip(&opening_proof.commit_pow_witnesses)
        {
            // To match with the prover (and for security purposes),
            // we observe the commitment before sampling the challenge.
            challenger.observe(comm.clone());
            if let Some(rand_f) =
                replay_pow_witness(challenger, params.commit_pow_bits, *pow_witness)?
            {
                let rand_usize = rand_f
                    .as_canonical_biguint()
                    .to_u64_digits()
                    .first()
                    .copied()
                    .unwrap_or(0) as usize;
                challenges.push(SC::Challenge::from_usize(rand_usize));
            }

            let beta = challenger.sample_algebra_element();
            challenges.push(beta);
            betas.push(beta);
        }

        // Observe all coefficients of the final polynomial.
        opening_proof
            .final_poly
            .iter()
            .for_each(|x| challenger.observe_algebra_element(*x));

        // Bind the variable-arity schedule into the transcript before query grinding,
        // matching the native FRI verifier in Plonky3.
        for step in &opening_proof.commit_phase_openings {
            challenger.observe(Val::<SC>::from_usize(step.log_arity as usize));
        }

        // Check PoW witness.
        if let Some(rand_f) = replay_pow_witness(
            challenger,
            params.query_pow_bits,
            opening_proof.query_pow_witness,
        )? {
            let rand_usize = rand_f
                .as_canonical_biguint()
                .to_u64_digits()
                .first()
                .copied()
                .unwrap_or(0) as usize;
            challenges.push(SC::Challenge::from_usize(rand_usize));
        }

        let total_log_reduction: usize = opening_proof
            .commit_phase_openings
            .iter()
            .map(|opening| opening.log_arity as usize)
            .sum();
        let log_global_max_height = total_log_reduction + params.log_final_height;
        let mut query_indices = Vec::with_capacity(params.num_queries);
        for _ in 0..params.num_queries {
            // For each query proof, we start by generating the random index.
            let index = challenger.sample_bits(log_global_max_height);
            query_indices.push(index);
            challenges.push(SC::Challenge::from_usize(index));
        }

        Ok(FriTranscriptChallenges {
            alpha,
            betas,
            query_indices,
            public_values: challenges,
        })
    }

    fn num_challenges(
        opening_proof: &InnerFriProof<SC, InputMmcs, FriMmcs>,
        fri_params: Option<FriGenerationParams>,
    ) -> Result<usize, GenerationError> {
        let params = fri_params.ok_or(GenerationError::MissingParameterError)?;
        let commit_phases = opening_proof.commit_phase_commits.len();
        let num_challenges = 1
            + commit_phases
            + params.num_queries
            + usize::from(params.commit_pow_bits > 0) * commit_phases
            + usize::from(params.query_pow_bits > 0);

        Ok(num_challenges)
    }
}

type HidingInnerFriProof<SC, InputMmcs, FriMmcs> = (
    OpenedValues<<SC as StarkGenericConfig>::Challenge>,
    InnerFriProof<SC, InputMmcs, FriMmcs>,
);

impl<SC: StarkGenericConfig, Dft, InputMmcs: Mmcs<Val<SC>>, FriMmcs: Mmcs<SC::Challenge>, R>
    PcsGeneration<SC, HidingInnerFriProof<SC, InputMmcs, FriMmcs>>
    for HidingFriPcs<Val<SC>, Dft, InputMmcs, FriMmcs, R>
where
    Val<SC>: TwoAdicField + PrimeField,
    SC::Challenger: FieldChallenger<Val<SC>>
        + GrindingChallenger<Witness = Val<SC>>
        + CanObserve<FriMmcs::Commitment>,
{
    fn prepare_openings(
        &self,
        coms_to_verify: &ComsWithOpenings<SC>,
        opening_proof: &HidingInnerFriProof<SC, InputMmcs, FriMmcs>,
    ) -> Result<Vec<CommitmentWithOpenings<SC>>, GenerationError> {
        let random_openings = &opening_proof.0;
        if random_openings.len() != coms_to_verify.len() {
            return Err(GenerationError::HidingRandomOpeningRoundCountMismatch {
                expected: coms_to_verify.len(),
                got: random_openings.len(),
            });
        }

        let mut merged = Vec::with_capacity(coms_to_verify.len());
        for (round, ((commitment, matrices), random_matrices)) in
            coms_to_verify.iter().zip(random_openings).enumerate()
        {
            if random_matrices.len() != matrices.len() {
                return Err(GenerationError::HidingRandomOpeningMatrixCountMismatch {
                    round,
                    expected: matrices.len(),
                    got: random_matrices.len(),
                });
            }

            let mut merged_matrices = Vec::with_capacity(matrices.len());
            for (matrix, ((domain, points), random_points)) in
                matrices.iter().zip(random_matrices).enumerate()
            {
                if random_points.len() != points.len() {
                    return Err(GenerationError::HidingRandomOpeningPointCountMismatch {
                        round,
                        matrix,
                        expected: points.len(),
                        got: random_points.len(),
                    });
                }

                let merged_points = points
                    .iter()
                    .zip(random_points)
                    .map(|((point, values), random_values)| {
                        let mut merged_values = values.clone();
                        merged_values.extend_from_slice(random_values);
                        (*point, merged_values)
                    })
                    .collect();
                merged_matrices.push((*domain, merged_points));
            }
            merged.push((commitment.clone(), merged_matrices));
        }
        Ok(merged)
    }

    fn generate_challenges(
        &self,
        _config: &SC,
        challenger: &mut SC::Challenger,
        coms_to_verify: &ComsWithOpenings<SC>,
        opening_proof: &HidingInnerFriProof<SC, InputMmcs, FriMmcs>,
        fri_params: Option<FriGenerationParams>,
    ) -> Result<FriTranscriptChallenges<SC::Challenge>, GenerationError> {
        let inner_proof = &opening_proof.1;
        let params = fri_params.ok_or(GenerationError::MissingParameterError)?;
        let num_challenges = <Self as PcsGeneration<
            SC,
            HidingInnerFriProof<SC, InputMmcs, FriMmcs>,
        >>::num_challenges(opening_proof, Some(params))?;
        let mut challenges = Vec::with_capacity(num_challenges);

        for (_, round) in coms_to_verify {
            for (_, mat) in round {
                for (_, point) in mat {
                    point
                        .iter()
                        .for_each(|&opening| challenger.observe_algebra_element(opening));
                }
            }
        }

        let alpha = challenger.sample_algebra_element();
        challenges.push(alpha);

        let mut betas = Vec::with_capacity(inner_proof.commit_phase_commits.len());
        for (comm, pow_witness) in inner_proof
            .commit_phase_commits
            .iter()
            .zip(&inner_proof.commit_pow_witnesses)
        {
            challenger.observe(comm.clone());
            if let Some(rand_f) =
                replay_pow_witness(challenger, params.commit_pow_bits, *pow_witness)?
            {
                let rand_usize = rand_f
                    .as_canonical_biguint()
                    .to_u64_digits()
                    .first()
                    .copied()
                    .unwrap_or(0) as usize;
                challenges.push(SC::Challenge::from_usize(rand_usize));
            }
            let beta = challenger.sample_algebra_element();
            challenges.push(beta);
            betas.push(beta);
        }

        inner_proof
            .final_poly
            .iter()
            .for_each(|x| challenger.observe_algebra_element(*x));

        for step in &inner_proof.commit_phase_openings {
            challenger.observe(Val::<SC>::from_usize(step.log_arity as usize));
        }

        if let Some(rand_f) = replay_pow_witness(
            challenger,
            params.query_pow_bits,
            inner_proof.query_pow_witness,
        )? {
            let rand_usize = rand_f
                .as_canonical_biguint()
                .to_u64_digits()
                .first()
                .copied()
                .unwrap_or(0) as usize;
            challenges.push(SC::Challenge::from_usize(rand_usize));
        }

        let total_log_reduction: usize = inner_proof
            .commit_phase_openings
            .iter()
            .map(|opening| opening.log_arity as usize)
            .sum();
        let log_global_max_height = total_log_reduction + params.log_final_height;
        let mut query_indices = Vec::with_capacity(params.num_queries);
        for _ in 0..params.num_queries {
            let index = challenger.sample_bits(log_global_max_height);
            query_indices.push(index);
            challenges.push(SC::Challenge::from_usize(index));
        }

        Ok(FriTranscriptChallenges {
            alpha,
            betas,
            query_indices,
            public_values: challenges,
        })
    }

    fn num_challenges(
        opening_proof: &HidingInnerFriProof<SC, InputMmcs, FriMmcs>,
        fri_params: Option<FriGenerationParams>,
    ) -> Result<usize, GenerationError> {
        let inner_proof = &opening_proof.1;
        let params = fri_params.ok_or(GenerationError::MissingParameterError)?;
        let commit_phases = inner_proof.commit_phase_commits.len();
        Ok(1 + commit_phases
            + params.num_queries
            + usize::from(params.commit_pow_bits > 0) * commit_phases
            + usize::from(params.query_pow_bits > 0))
    }
}

/// Samples the batch's single permutation challenge pair on the transcript challenger and returns
/// it, so the generated challenge public values stay in the sampling order the in-circuit verifier
/// reproduces. Returns an empty vector when no AIR declares a lookup (no pair is drawn), matching
/// the native `sample_perm_challenges`.
pub fn get_different_perm_challenges<SC, LG, L>(
    challenger: &mut SC::Challenger,
    all_lookups: &[L],
    lookup_gadget: &LG,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    LG: LookupProtocol,
    L: AsRef<[Lookup<Val<SC>>]>,
{
    assert_eq!(
        lookup_gadget.num_challenges(),
        2,
        "single-pair bus-prefix challenge layout requires exactly two challenges per lookup"
    );

    if !all_lookups
        .iter()
        .any(|contexts| !contexts.as_ref().is_empty())
    {
        return Vec::new();
    }

    let alpha = challenger.sample_algebra_element::<SC::Challenge>();
    let beta = challenger.sample_algebra_element::<SC::Challenge>();
    vec![alpha, beta]
}
