//! Sound witness expansion for shared Merkle multi-openings.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use itertools::Itertools;
use p3_commit::Mmcs;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedValue, TwoAdicField,
    batch_multiplicative_inverse,
};
use p3_fri::{BatchMultiOpening, FriProof};
use p3_matrix::Dimensions;
use p3_merkle_tree::{MerkleTreeMmcs, PrunedMerklePaths};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use thiserror::Error;

/// Failures while expanding a verifier-bound pruned frontier into full paths.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum MerkleWitnessError {
    #[error("invalid Merkle geometry")]
    InvalidGeometry,
    #[error("query index {index} is outside the tallest matrix height {max_height}")]
    IndexOutOfBounds { max_height: usize, index: usize },
    #[error("opened query count mismatch: expected {expected}, got {got}")]
    QueryCountMismatch { expected: usize, got: usize },
    #[error("query {query} matrix count mismatch: expected {expected}, got {got}")]
    MatrixCountMismatch {
        query: usize,
        expected: usize,
        got: usize,
    },
    #[error("query {query} matrix {matrix} width mismatch: expected {expected}, got {got}")]
    WidthMismatch {
        query: usize,
        matrix: usize,
        expected: usize,
        got: usize,
    },
    #[error("duplicate queries for leaf slot {slot} carry different openings")]
    InconsistentDuplicateOpenings { slot: usize },
    #[error("queries merged at slot {slot} disagree on injected matrix {matrix}")]
    InconsistentGroupOpening { slot: usize, matrix: usize },
    #[error("pruned frontier sibling count mismatch: expected {expected}, got {got}")]
    SiblingCountMismatch { expected: usize, got: usize },
}

/// One verifier-known input matrix and all claimed evaluations at its opening points.
#[derive(Clone, Debug)]
pub struct FriInputMatrix<Domain, EF> {
    pub domain: Domain,
    pub points_and_values: Vec<(EF, Vec<EF>)>,
}

/// Host rows and verifier-derived indices needed to expand every FRI multiproof.
#[derive(Clone, Debug)]
pub struct ReconstructedFriRows<F, EF> {
    pub input_dimensions: Vec<Vec<Dimensions>>,
    pub input_indices: Vec<Vec<usize>>,
    pub commit_phase_dimensions: Vec<Vec<Dimensions>>,
    pub commit_phase_indices: Vec<Vec<usize>>,
    pub commit_phase_rows: Vec<Vec<Vec<Vec<EF>>>>,
    _phantom: core::marker::PhantomData<F>,
}

/// Failures while replaying the audited two-adic FRI arithmetic for witness rows.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum FriWitnessError {
    #[error("{0}")]
    InvalidShape(&'static str),
    #[error("input batch {batch} matrix {matrix} has no opening point")]
    MatrixWithoutOpeningPoint { batch: usize, matrix: usize },
    #[error("query {query} batch {batch} matrix {matrix} opening width mismatch")]
    OpeningWidthMismatch {
        query: usize,
        batch: usize,
        matrix: usize,
    },
    #[error("query {query} opening point equals the sampled domain point")]
    OpeningPointMatchesQueryPoint { query: usize },
}

/// Full per-query Merkle paths reconstructed from the sole v1 shared
/// multiproofs. Layout is `[query][input batch or commit phase][sibling]`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpandedFriMmcsPaths<F, const DIGEST_ELEMS: usize> {
    pub input_paths: Vec<Vec<Vec<[F; DIGEST_ELEMS]>>>,
    pub commit_phase_paths: Vec<Vec<Vec<[F; DIGEST_ELEMS]>>>,
}

impl<F, const DIGEST_ELEMS: usize> ExpandedFriMmcsPaths<F, DIGEST_ELEMS> {
    pub fn new(
        input_paths: Vec<Vec<Vec<[F; DIGEST_ELEMS]>>>,
        commit_phase_paths: Vec<Vec<Vec<[F; DIGEST_ELEMS]>>>,
    ) -> Result<Self, FriWitnessBuildError> {
        if input_paths.len() != commit_phase_paths.len() {
            return Err(FriWitnessBuildError::InvalidShape(
                "FRI input and commit-phase path query counts differ",
            ));
        }
        let input_batches = input_paths.first().map_or(0, Vec::len);
        let commit_phases = commit_phase_paths.first().map_or(0, Vec::len);
        if input_paths.iter().any(|paths| paths.len() != input_batches) {
            return Err(FriWitnessBuildError::InvalidShape(
                "FRI input path batch count differs across queries",
            ));
        }
        if commit_phase_paths
            .iter()
            .any(|paths| paths.len() != commit_phases)
        {
            return Err(FriWitnessBuildError::InvalidShape(
                "FRI commit-phase path count differs across queries",
            ));
        }
        Ok(Self {
            input_paths,
            commit_phase_paths,
        })
    }
}

/// Failures while converting the compact native proof into recursive witness
/// paths. Every variant is a proof-shape failure, never a recoverable fallback.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum FriWitnessBuildError {
    #[error("{0}")]
    InvalidShape(&'static str),
    #[error(transparent)]
    Arithmetic(#[from] FriWitnessError),
    #[error("input batch {batch}: {source}")]
    InputMerkle {
        batch: usize,
        source: MerkleWitnessError,
    },
    #[error("commit phase {phase}: {source}")]
    CommitPhaseMerkle {
        phase: usize,
        source: MerkleWitnessError,
    },
    #[error(
        "{location} query {query} matrix {matrix} salt width mismatch: expected {expected}, got {got}"
    )]
    SaltWidthMismatch {
        location: &'static str,
        query: usize,
        matrix: usize,
        expected: usize,
        got: usize,
    },
}

/// Expand every non-hiding input and commit-phase shared multiproof into the
/// complete paths consumed by the recursive MMCS verifier.
#[allow(clippy::too_many_arguments)]
pub fn expand_fri_mmcs_paths<
    F,
    EF,
    InputMmcs,
    FriMmcs,
    InputH,
    InputC,
    FriH,
    FriC,
    const N: usize,
    const DIGEST_ELEMS: usize,
>(
    proof: &FriProof<EF, FriMmcs, F, Vec<BatchMultiOpening<F, InputMmcs>>>,
    input_batches: &[Vec<FriInputMatrix<TwoAdicMultiplicativeCoset<F>, EF>>],
    alpha: EF,
    betas: &[EF],
    indices: &[usize],
    log_blowup: usize,
    log_final_poly_len: usize,
    input_hash: &InputH,
    input_compress: &InputC,
    input_cap_height: usize,
    fri_hash: &FriH,
    fri_compress: &FriC,
    fri_cap_height: usize,
) -> Result<ExpandedFriMmcsPaths<F, DIGEST_ELEMS>, FriWitnessBuildError>
where
    F: TwoAdicField + PackedValue<Value = F> + Default + Eq,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    InputMmcs: Mmcs<F, MultiProof = PrunedMerklePaths<F, DIGEST_ELEMS>>,
    FriMmcs: Mmcs<EF, MultiProof = PrunedMerklePaths<F, DIGEST_ELEMS>>,
    InputH: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Clone + Sync,
    InputC: PseudoCompressionFunction<[F; DIGEST_ELEMS], N> + Clone + Sync,
    FriH: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Clone + Sync,
    FriC: PseudoCompressionFunction<[F; DIGEST_ELEMS], N> + Clone + Sync,
{
    let reconstructed = reconstruct_two_adic_fri_rows(
        proof,
        input_batches,
        alpha,
        betas,
        indices,
        log_blowup,
        log_final_poly_len,
    )?;
    let mut input_paths = vec![Vec::with_capacity(proof.input_openings.len()); indices.len()];
    for (batch, opening) in proof.input_openings.iter().enumerate() {
        let paths = expand_pruned_merkle_paths::<F, InputH, InputC, N, DIGEST_ELEMS>(
            &reconstructed.input_dimensions[batch],
            &reconstructed.input_indices[batch],
            &opening.opened_values,
            &opening.opening_proof,
            input_hash,
            input_compress,
            input_cap_height,
        )
        .map_err(|source| FriWitnessBuildError::InputMerkle { batch, source })?;
        for (query, path) in paths.into_iter().enumerate() {
            input_paths[query].push(path);
        }
    }

    let mut commit_phase_paths =
        vec![Vec::with_capacity(proof.commit_phase_openings.len()); indices.len()];
    for (phase, opening) in proof.commit_phase_openings.iter().enumerate() {
        let dimensions =
            flatten_extension_dimensions::<F, EF>(&reconstructed.commit_phase_dimensions[phase]);
        let opened_values =
            flatten_extension_rows::<F, EF>(&reconstructed.commit_phase_rows[phase]);
        let paths = expand_pruned_merkle_paths::<F, FriH, FriC, N, DIGEST_ELEMS>(
            &dimensions,
            &reconstructed.commit_phase_indices[phase],
            &opened_values,
            &opening.opening_proof,
            fri_hash,
            fri_compress,
            fri_cap_height,
        )
        .map_err(|source| FriWitnessBuildError::CommitPhaseMerkle { phase, source })?;
        for (query, path) in paths.into_iter().enumerate() {
            commit_phase_paths[query].push(path);
        }
    }

    ExpandedFriMmcsPaths::new(input_paths, commit_phase_paths)
}

/// Hiding-MMCS counterpart of [`expand_fri_mmcs_paths`]. Native salts are
/// reattached before hashing, exactly as `MerkleTreeHidingMmcs` does.
#[allow(clippy::too_many_arguments)]
pub fn expand_hiding_fri_mmcs_paths<
    F,
    EF,
    InputMmcs,
    FriMmcs,
    InputH,
    InputC,
    FriH,
    FriC,
    const N: usize,
    const DIGEST_ELEMS: usize,
    const SALT_ELEMS: usize,
>(
    proof: &FriProof<EF, FriMmcs, F, Vec<BatchMultiOpening<F, InputMmcs>>>,
    input_batches: &[Vec<FriInputMatrix<TwoAdicMultiplicativeCoset<F>, EF>>],
    alpha: EF,
    betas: &[EF],
    indices: &[usize],
    log_blowup: usize,
    log_final_poly_len: usize,
    input_hash: &InputH,
    input_compress: &InputC,
    input_cap_height: usize,
    fri_hash: &FriH,
    fri_compress: &FriC,
    fri_cap_height: usize,
) -> Result<ExpandedFriMmcsPaths<F, DIGEST_ELEMS>, FriWitnessBuildError>
where
    F: TwoAdicField + PackedValue<Value = F> + Default + Eq,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    InputMmcs: Mmcs<F, MultiProof = (Vec<Vec<Vec<F>>>, PrunedMerklePaths<F, DIGEST_ELEMS>)>,
    FriMmcs: Mmcs<EF, MultiProof = (Vec<Vec<Vec<F>>>, PrunedMerklePaths<F, DIGEST_ELEMS>)>,
    InputH: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Clone + Sync,
    InputC: PseudoCompressionFunction<[F; DIGEST_ELEMS], N> + Clone + Sync,
    FriH: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Clone + Sync,
    FriC: PseudoCompressionFunction<[F; DIGEST_ELEMS], N> + Clone + Sync,
{
    let reconstructed = reconstruct_two_adic_fri_rows(
        proof,
        input_batches,
        alpha,
        betas,
        indices,
        log_blowup,
        log_final_poly_len,
    )?;
    let mut input_paths = vec![Vec::with_capacity(proof.input_openings.len()); indices.len()];
    for (batch, opening) in proof.input_openings.iter().enumerate() {
        let (salts, pruned) = &opening.opening_proof;
        let (dimensions, opened_values) = attach_salts::<F, SALT_ELEMS>(
            &reconstructed.input_dimensions[batch],
            &opening.opened_values,
            salts,
            "input batch",
        )?;
        let paths = expand_pruned_merkle_paths::<F, InputH, InputC, N, DIGEST_ELEMS>(
            &dimensions,
            &reconstructed.input_indices[batch],
            &opened_values,
            pruned,
            input_hash,
            input_compress,
            input_cap_height,
        )
        .map_err(|source| FriWitnessBuildError::InputMerkle { batch, source })?;
        for (query, path) in paths.into_iter().enumerate() {
            input_paths[query].push(path);
        }
    }

    let mut commit_phase_paths =
        vec![Vec::with_capacity(proof.commit_phase_openings.len()); indices.len()];
    for (phase, opening) in proof.commit_phase_openings.iter().enumerate() {
        let base_dimensions =
            flatten_extension_dimensions::<F, EF>(&reconstructed.commit_phase_dimensions[phase]);
        let base_rows = flatten_extension_rows::<F, EF>(&reconstructed.commit_phase_rows[phase]);
        let (salts, pruned) = &opening.opening_proof;
        let (dimensions, opened_values) =
            attach_salts::<F, SALT_ELEMS>(&base_dimensions, &base_rows, salts, "commit phase")?;
        let paths = expand_pruned_merkle_paths::<F, FriH, FriC, N, DIGEST_ELEMS>(
            &dimensions,
            &reconstructed.commit_phase_indices[phase],
            &opened_values,
            pruned,
            fri_hash,
            fri_compress,
            fri_cap_height,
        )
        .map_err(|source| FriWitnessBuildError::CommitPhaseMerkle { phase, source })?;
        for (query, path) in paths.into_iter().enumerate() {
            commit_phase_paths[query].push(path);
        }
    }

    ExpandedFriMmcsPaths::new(input_paths, commit_phase_paths)
}

fn flatten_extension_dimensions<F, EF>(dimensions: &[Dimensions]) -> Vec<Dimensions>
where
    F: Field,
    EF: ExtensionField<F>,
{
    dimensions
        .iter()
        .map(|dimensions| Dimensions {
            width: dimensions.width * EF::DIMENSION,
            height: dimensions.height,
        })
        .collect()
}

fn flatten_extension_rows<F, EF>(rows: &[Vec<Vec<EF>>]) -> Vec<Vec<Vec<F>>>
where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    rows.iter()
        .map(|query| {
            query
                .iter()
                .map(|row| {
                    row.iter()
                        .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn attach_salts<F: Field, const SALT_ELEMS: usize>(
    dimensions: &[Dimensions],
    rows: &[Vec<Vec<F>>],
    salts: &[Vec<Vec<F>>],
    location: &'static str,
) -> Result<(Vec<Dimensions>, Vec<Vec<Vec<F>>>), FriWitnessBuildError> {
    if rows.len() != salts.len() {
        return Err(FriWitnessBuildError::InvalidShape(
            "hiding salt query count mismatch",
        ));
    }
    let mut salted_rows = Vec::with_capacity(rows.len());
    for (query, (query_rows, query_salts)) in rows.iter().zip(salts).enumerate() {
        if query_rows.len() != dimensions.len() || query_salts.len() != dimensions.len() {
            return Err(FriWitnessBuildError::InvalidShape(
                "hiding salt matrix count mismatch",
            ));
        }
        let mut salted_query = Vec::with_capacity(dimensions.len());
        for (matrix, (row, salt)) in query_rows.iter().zip(query_salts).enumerate() {
            if salt.len() != SALT_ELEMS {
                return Err(FriWitnessBuildError::SaltWidthMismatch {
                    location,
                    query,
                    matrix,
                    expected: SALT_ELEMS,
                    got: salt.len(),
                });
            }
            let mut salted = Vec::with_capacity(row.len() + SALT_ELEMS);
            salted.extend_from_slice(row);
            salted.extend_from_slice(salt);
            salted_query.push(salted);
        }
        salted_rows.push(salted_query);
    }
    let salted_dimensions = dimensions
        .iter()
        .map(|dimensions| Dimensions {
            width: dimensions.width + SALT_ELEMS,
            height: dimensions.height,
        })
        .collect();
    Ok((salted_dimensions, salted_rows))
}

/// Reconstruct every commit-phase row using the same arithmetic as the audited
/// native two-adic FRI verifier. Authentication is performed separately by
/// [`expand_pruned_merkle_paths`].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn reconstruct_two_adic_fri_rows<F, EF, InputMmcs, FriMmcs>(
    proof: &FriProof<EF, FriMmcs, F, Vec<BatchMultiOpening<F, InputMmcs>>>,
    input_batches: &[Vec<FriInputMatrix<TwoAdicMultiplicativeCoset<F>, EF>>],
    alpha: EF,
    betas: &[EF],
    indices: &[usize],
    log_blowup: usize,
    log_final_poly_len: usize,
) -> Result<ReconstructedFriRows<F, EF>, FriWitnessError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    InputMmcs: Mmcs<F>,
    FriMmcs: Mmcs<EF>,
{
    if proof.input_openings.len() != input_batches.len() {
        return Err(FriWitnessError::InvalidShape(
            "input opening batch count mismatch",
        ));
    }
    if proof.commit_phase_openings.len() != proof.commit_phase_commits.len()
        || betas.len() != proof.commit_phase_commits.len()
    {
        return Err(FriWitnessError::InvalidShape("commit-phase count mismatch"));
    }
    if indices.is_empty() {
        return Err(FriWitnessError::InvalidShape(
            "FRI requires at least one query",
        ));
    }

    let log_arities: Vec<usize> = proof
        .commit_phase_openings
        .iter()
        .map(|opening| opening.log_arity as usize)
        .collect();
    if log_arities.iter().any(|&log_arity| log_arity == 0) {
        return Err(FriWitnessError::InvalidShape("zero FRI log arity"));
    }
    let total_log_reduction: usize = log_arities.iter().sum();
    let log_global_max_height = total_log_reduction + log_blowup + log_final_poly_len;

    let mut input_dimensions = Vec::with_capacity(input_batches.len());
    let mut input_indices = Vec::with_capacity(input_batches.len());
    for (batch, (opening, matrices)) in proof.input_openings.iter().zip(input_batches).enumerate() {
        if opening.opened_values.len() != indices.len() {
            return Err(FriWitnessError::InvalidShape(
                "input opening query count mismatch",
            ));
        }
        let mut dimensions = Vec::with_capacity(matrices.len());
        for (matrix, metadata) in matrices.iter().enumerate() {
            let width = metadata
                .points_and_values
                .first()
                .ok_or(FriWitnessError::MatrixWithoutOpeningPoint { batch, matrix })?
                .1
                .len();
            if metadata
                .points_and_values
                .iter()
                .any(|(_, values)| values.len() != width)
            {
                return Err(FriWitnessError::InvalidShape(
                    "opening-point evaluation widths disagree",
                ));
            }
            dimensions.push(Dimensions {
                width,
                height: metadata.domain.size() << log_blowup,
            });
        }
        for (query, rows) in opening.opened_values.iter().enumerate() {
            if rows.len() != matrices.len() {
                return Err(FriWitnessError::InvalidShape(
                    "input opening matrix count mismatch",
                ));
            }
            for (matrix, row) in rows.iter().enumerate() {
                if row.len() != dimensions[matrix].width {
                    return Err(FriWitnessError::OpeningWidthMismatch {
                        query,
                        batch,
                        matrix,
                    });
                }
            }
        }
        let max_height = dimensions
            .iter()
            .map(|dims| dims.height)
            .max()
            .ok_or(FriWitnessError::InvalidShape("empty input batch"))?;
        let bits_reduced = log_global_max_height
            .checked_sub(log2_strict_usize(max_height))
            .ok_or(FriWitnessError::InvalidShape(
                "input height exceeds global FRI height",
            ))?;
        input_indices.push(indices.iter().map(|&index| index >> bits_reduced).collect());
        input_dimensions.push(dimensions);
    }

    let mut reduced_by_query = Vec::with_capacity(indices.len());
    for (query, &index) in indices.iter().enumerate() {
        let mut reduced = alloc::collections::BTreeMap::<usize, (EF, EF)>::new();
        for (batch, (opening, matrices)) in
            proof.input_openings.iter().zip(input_batches).enumerate()
        {
            for (matrix, (row, metadata)) in opening.opened_values[query]
                .iter()
                .zip(matrices)
                .enumerate()
            {
                let log_height = log2_strict_usize(metadata.domain.size()) + log_blowup;
                let bits_reduced = log_global_max_height - log_height;
                let reversed = reverse_bits_len(index >> bits_reduced, log_height);
                let x = F::GENERATOR * F::two_adic_generator(log_height).exp_u64(reversed as u64);
                let (alpha_power, reduced_opening) =
                    reduced.entry(log_height).or_insert((EF::ONE, EF::ZERO));
                for (point, values_at_point) in &metadata.points_and_values {
                    if row.len() != values_at_point.len() {
                        return Err(FriWitnessError::OpeningWidthMismatch {
                            query,
                            batch,
                            matrix,
                        });
                    }
                    let denominator = *point - x;
                    if denominator.is_zero() {
                        return Err(FriWitnessError::OpeningPointMatchesQueryPoint { query });
                    }
                    let inverse = denominator.inverse();
                    for (&value_at_x, &value_at_point) in row.iter().zip(values_at_point) {
                        *reduced_opening += *alpha_power * (value_at_point - value_at_x) * inverse;
                        *alpha_power *= alpha;
                    }
                }
            }
        }
        if reduced
            .get(&log_blowup)
            .is_some_and(|(_, opening)| !opening.is_zero())
        {
            return Err(FriWitnessError::InvalidShape(
                "nonzero constant-polynomial reduced opening",
            ));
        }
        reduced_by_query.push(
            reduced
                .into_iter()
                .rev()
                .map(|(height, (_, opening))| (height, opening))
                .collect::<Vec<_>>(),
        );
    }

    let rounds = proof.commit_phase_openings.len();
    let mut commit_phase_indices = vec![Vec::with_capacity(indices.len()); rounds];
    let mut commit_phase_rows = vec![Vec::with_capacity(indices.len()); rounds];
    for (query, (&initial_index, reduced_openings)) in
        indices.iter().zip(reduced_by_query).enumerate()
    {
        let Some(&(first_height, mut folded_eval)) = reduced_openings.first() else {
            return Err(FriWitnessError::InvalidShape("missing reduced opening"));
        };
        if first_height != log_global_max_height {
            return Err(FriWitnessError::InvalidShape(
                "initial reduced opening height mismatch",
            ));
        }
        let mut reduced_cursor = 1usize;
        let mut domain_index = initial_index;
        let mut log_current_height = log_global_max_height;
        for (round, ((&beta, &log_arity), opening)) in betas
            .iter()
            .zip(&log_arities)
            .zip(&proof.commit_phase_openings)
            .enumerate()
        {
            if opening.sibling_values.len() != indices.len() {
                return Err(FriWitnessError::InvalidShape(
                    "commit-phase sibling query count mismatch",
                ));
            }
            let arity = 1usize << log_arity;
            let siblings = &opening.sibling_values[query];
            if siblings.len() != arity - 1 {
                return Err(FriWitnessError::InvalidShape(
                    "commit-phase sibling row width mismatch",
                ));
            }
            let index_in_group = domain_index % arity;
            let mut row = vec![EF::ZERO; arity];
            row[index_in_group] = folded_eval;
            let mut sibling = 0usize;
            for (position, value) in row.iter_mut().enumerate() {
                if position != index_in_group {
                    *value = siblings[sibling];
                    sibling += 1;
                }
            }

            let log_folded_height = log_current_height - log_arity;
            domain_index >>= log_arity;
            folded_eval = fold_row::<F, EF>(domain_index, log_folded_height, log_arity, beta, &row);
            commit_phase_indices[round].push(domain_index);
            commit_phase_rows[round].push(vec![row]);

            if reduced_openings
                .get(reduced_cursor)
                .is_some_and(|(height, _)| *height == log_folded_height)
            {
                let roll_in = reduced_openings[reduced_cursor].1;
                folded_eval += beta.exp_power_of_2(log_arity) * roll_in;
                reduced_cursor += 1;
            }
            log_current_height = log_folded_height;
        }
    }

    let mut log_current_height = log_global_max_height;
    let mut commit_phase_dimensions = Vec::with_capacity(rounds);
    for &log_arity in &log_arities {
        let log_folded_height = log_current_height - log_arity;
        commit_phase_dimensions.push(vec![Dimensions {
            width: 1usize << log_arity,
            height: 1usize << log_folded_height,
        }]);
        log_current_height = log_folded_height;
    }

    Ok(ReconstructedFriRows {
        input_dimensions,
        input_indices,
        commit_phase_dimensions,
        commit_phase_indices,
        commit_phase_rows,
        _phantom: core::marker::PhantomData,
    })
}

fn fold_row<F: TwoAdicField, EF: ExtensionField<F>>(
    index: usize,
    log_height: usize,
    log_arity: usize,
    beta: EF,
    evaluations: &[EF],
) -> EF {
    let arity = 1usize << log_arity;
    debug_assert_eq!(evaluations.len(), arity);
    let subgroup_start = F::two_adic_generator(log_height + log_arity)
        .exp_u64(reverse_bits_len(index, log_height) as u64);
    let mut points: Vec<F> = F::two_adic_generator(log_arity)
        .shifted_powers(subgroup_start)
        .take(arity)
        .collect();
    reverse_slice_index_bits(&mut points);
    lagrange_interpolate_at(&points, evaluations, beta)
}

fn lagrange_interpolate_at<F: TwoAdicField, EF: ExtensionField<F>>(
    points: &[F],
    evaluations: &[EF],
    point: EF,
) -> EF {
    if points.is_empty() {
        return EF::ZERO;
    }
    for (index, &x) in points.iter().enumerate() {
        if (point - x).is_zero() {
            return evaluations[index];
        }
    }
    let log_arity = log2_strict_usize(points.len());
    let coset_power = points[0].exp_power_of_2(log_arity);
    let weight_scale = (F::from_usize(points.len()) * coset_power).inverse();
    let differences: Vec<EF> = points.iter().map(|&x| point - x).collect();
    let inverses = batch_multiplicative_inverse(&differences);
    let vanishing = differences.iter().copied().product::<EF>();
    let mut result = EF::ZERO;
    for ((&x, &evaluation), &inverse) in points.iter().zip(evaluations).zip(&inverses) {
        result += evaluation * (x * weight_scale) * inverse;
    }
    result * vanishing
}

#[derive(Clone)]
struct FrontierNode<D, const DIGEST_ELEMS: usize> {
    index: usize,
    digest: [D; DIGEST_ELEMS],
    members: Vec<usize>,
}

/// Expand one shared pruned Merkle proof into complete authentication paths.
///
/// The returned paths are in the original query order. Every path is recomputed
/// from verifier-owned indices, dimensions, opened rows, and the actual hash and
/// compression functions. Boundary hashes are consumed exactly once in native
/// frontier wire order; a short or long proof is rejected.
#[allow(clippy::too_many_arguments)]
pub fn expand_pruned_merkle_paths<F, H, C, const N: usize, const DIGEST_ELEMS: usize>(
    dimensions: &[Dimensions],
    indices: &[usize],
    opened_values: &[Vec<Vec<F>>],
    proof: &PrunedMerklePaths<F, DIGEST_ELEMS>,
    hash: &H,
    compress: &C,
    cap_height: usize,
) -> Result<Vec<Vec<[F; DIGEST_ELEMS]>>, MerkleWitnessError>
where
    F: Field + PackedValue<Value = F> + Default + Eq,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Clone + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], N> + Clone + Sync,
{
    if opened_values.len() != indices.len() {
        return Err(MerkleWitnessError::QueryCountMismatch {
            expected: indices.len(),
            got: opened_values.len(),
        });
    }
    if dimensions.is_empty() || dimensions.iter().all(|dims| dims.height == 0) {
        return Err(MerkleWitnessError::InvalidGeometry);
    }

    for (query, rows) in opened_values.iter().enumerate() {
        if rows.len() != dimensions.len() {
            return Err(MerkleWitnessError::MatrixCountMismatch {
                query,
                expected: dimensions.len(),
                got: rows.len(),
            });
        }
        for (matrix, (row, dims)) in rows.iter().zip(dimensions).enumerate() {
            if row.len() != dims.width {
                return Err(MerkleWitnessError::WidthMismatch {
                    query,
                    matrix,
                    expected: dims.width,
                    got: row.len(),
                });
            }
        }
    }

    if indices.is_empty() {
        return if proof.sibling_hashes.is_empty() {
            Ok(Vec::new())
        } else {
            Err(MerkleWitnessError::SiblingCountMismatch {
                expected: 0,
                got: proof.sibling_hashes.len(),
            })
        };
    }

    let max_height = dimensions
        .iter()
        .map(|dims| dims.height)
        .max()
        .ok_or(MerkleWitnessError::InvalidGeometry)?;

    let schedule_mmcs = MerkleTreeMmcs::<F, F, H, C, N, DIGEST_ELEMS>::new(
        hash.clone(),
        compress.clone(),
        cap_height,
    );
    let arity_schedule = schedule_mmcs
        .proof_arity_schedule(dimensions)
        .map_err(|_| MerkleWitnessError::InvalidGeometry)?;

    let mut sorted_unique = indices.to_vec();
    sorted_unique.sort_unstable();
    sorted_unique.dedup();
    if let Some(&index) = sorted_unique.last()
        && index >= max_height
    {
        return Err(MerkleWitnessError::IndexOutOfBounds { max_height, index });
    }

    let mut representatives = vec![None; sorted_unique.len()];
    let mut original_slots = Vec::with_capacity(indices.len());
    for (query, &leaf) in indices.iter().enumerate() {
        let slot = sorted_unique
            .binary_search(&leaf)
            .expect("leaf came from sorted_unique");
        original_slots.push(slot);
        match representatives[slot] {
            None => representatives[slot] = Some(query),
            Some(representative) => {
                if opened_values[representative] != opened_values[query] {
                    return Err(MerkleWitnessError::InconsistentDuplicateOpenings { slot });
                }
            }
        }
    }
    let representatives: Vec<usize> = representatives
        .into_iter()
        .map(|representative| representative.expect("every unique leaf has a query"))
        .collect();

    let mut heights_tallest_first = dimensions
        .iter()
        .enumerate()
        .sorted_by_key(|(_, dims)| Reverse(dims.height))
        .peekable();
    let leaf_height_npt = max_height.next_power_of_two();
    let leaf_matrices: Vec<usize> = heights_tallest_first
        .peeking_take_while(|(_, dims)| dims.height.next_power_of_two() == leaf_height_npt)
        .map(|(matrix, _)| matrix)
        .collect();

    let mut nodes: Vec<FrontierNode<F, DIGEST_ELEMS>> = sorted_unique
        .iter()
        .enumerate()
        .map(|(slot, &index)| {
            let representative = representatives[slot];
            let digest = hash.hash_iter_slices(
                leaf_matrices
                    .iter()
                    .map(|&matrix| opened_values[representative][matrix].as_slice()),
            );
            FrontierNode {
                index,
                digest,
                members: vec![slot],
            }
        })
        .collect();

    let full_sibling_count: usize = arity_schedule.iter().map(|arity| arity - 1).sum();
    let mut paths = vec![Vec::with_capacity(full_sibling_count); sorted_unique.len()];
    let default_digest = [F::default(); DIGEST_ELEMS];
    let mut proof_cursor = 0usize;
    let mut curr_height_padded = padded_len(max_height, N);

    for &step in &arity_schedule {
        let mut parents = Vec::with_capacity(nodes.len());
        let mut i = 0usize;
        while i < nodes.len() {
            let parent_index = nodes[i].index / step;
            let group_start = parent_index * step;
            let mut j = i + 1;
            while j < nodes.len() && nodes[j].index / step == parent_index {
                j += 1;
            }

            let mut children = [default_digest; N];
            let mut present = [false; N];
            for node in &nodes[i..j] {
                let position = node.index - group_start;
                children[position] = node.digest;
                present[position] = true;
            }
            for position in 0..step {
                if !present[position] {
                    let Some(boundary) = proof.sibling_hashes.get(proof_cursor) else {
                        return Err(MerkleWitnessError::SiblingCountMismatch {
                            expected: proof_cursor + 1,
                            got: proof.sibling_hashes.len(),
                        });
                    };
                    children[position] = *boundary;
                    proof_cursor += 1;
                }
            }

            for node in &nodes[i..j] {
                let own_position = node.index - group_start;
                for &member in &node.members {
                    for (position, digest) in children.iter().enumerate().take(step) {
                        if position != own_position {
                            paths[member].push(*digest);
                        }
                    }
                }
            }

            let mut members = Vec::new();
            for node in &nodes[i..j] {
                members.extend_from_slice(&node.members);
            }
            parents.push(FrontierNode {
                index: parent_index,
                digest: compress.compress(children),
                members,
            });
            i = j;
        }
        nodes = parents;

        let logical_next = curr_height_padded / step;
        curr_height_padded = padded_len(logical_next, N);
        let logical_next_npt = logical_next.next_power_of_two();
        let next_height = heights_tallest_first
            .peek()
            .map(|(_, dims)| dims.height)
            .filter(|height| height.next_power_of_two() == logical_next_npt);
        if let Some(next_height) = next_height {
            let inject_matrices: Vec<usize> = heights_tallest_first
                .peeking_take_while(|(_, dims)| dims.height == next_height)
                .map(|(matrix, _)| matrix)
                .collect();

            for node in &mut nodes {
                let lead_slot = node.members[0];
                let lead_query = representatives[lead_slot];
                for &slot in node.members.iter().skip(1) {
                    let query = representatives[slot];
                    for &matrix in &inject_matrices {
                        if opened_values[query][matrix] != opened_values[lead_query][matrix] {
                            return Err(MerkleWitnessError::InconsistentGroupOpening {
                                slot,
                                matrix,
                            });
                        }
                    }
                }

                let injected = hash.hash_iter_slices(
                    inject_matrices
                        .iter()
                        .map(|&matrix| opened_values[lead_query][matrix].as_slice()),
                );
                let mut inputs = [default_digest; N];
                inputs[0] = node.digest;
                inputs[1] = injected;
                node.digest = compress.compress(inputs);
            }
        }
    }

    if proof_cursor != proof.sibling_hashes.len() {
        return Err(MerkleWitnessError::SiblingCountMismatch {
            expected: proof_cursor,
            got: proof.sibling_hashes.len(),
        });
    }
    if paths.iter().any(|path| path.len() != full_sibling_count) {
        return Err(MerkleWitnessError::InvalidGeometry);
    }

    Ok(original_slots
        .into_iter()
        .map(|slot| paths[slot].clone())
        .collect())
}

const fn padded_len(raw_len: usize, n: usize) -> usize {
    if raw_len <= 1 {
        raw_len
    } else if raw_len >= n {
        raw_len.div_ceil(n) * n
    } else {
        n
    }
}
