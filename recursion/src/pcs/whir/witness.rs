//! Sound witness expansion for WHIR shared Merkle multi-openings.

use alloc::vec;
use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PackedValue};
use p3_matrix::Dimensions;
use p3_merkle_tree::PrunedMerklePaths;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_whir::pcs::proof::{QueryOpenings, WhirProof};
use thiserror::Error;

use super::params::WhirVerifierParams;
use crate::pcs::fri::{MerkleWitnessError, expand_pruned_merkle_paths};
use crate::pcs::mmcs::ExpandedWhirMmcsPaths;

/// Failures while converting the sole v1 WHIR multiproofs into recursive paths.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum WhirWitnessBuildError {
    #[error("{0}")]
    InvalidShape(&'static str),
    #[error("WHIR round {round}: {source}")]
    Merkle {
        round: usize,
        source: MerkleWitnessError,
    },
}

/// Expand every WHIR shared multiproof into the complete paths consumed by the
/// recursive MMCS verifier.
///
/// `round_indices` and `final_indices` must be replayed from the verifier's
/// Fiat-Shamir transcript. The compact frontier is never trusted to choose its
/// own positions or geometry.
#[allow(clippy::too_many_arguments)]
pub fn expand_whir_mmcs_paths<F, EF, MT, H, C, const N: usize, const DIGEST_ELEMS: usize>(
    proof: &WhirProof<F, EF, MT>,
    params: &WhirVerifierParams<F>,
    round_indices: &[Vec<usize>],
    final_indices: &[usize],
    hash: &H,
    compress: &C,
    cap_height: usize,
) -> Result<ExpandedWhirMmcsPaths<F, DIGEST_ELEMS>, WhirWitnessBuildError>
where
    F: Field + PackedValue<Value = F> + Default + Eq + Send + Sync + Clone,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    MT: Mmcs<F, MultiProof = PrunedMerklePaths<F, DIGEST_ELEMS>>,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Clone + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], N> + Clone + Sync,
{
    if proof.rounds.len() != params.round_params.len()
        || round_indices.len() != params.round_params.len()
    {
        return Err(WhirWitnessBuildError::InvalidShape(
            "WHIR round count differs between proof, parameters, and transcript",
        ));
    }

    let mut paths = Vec::new();
    for (round, ((round_proof, round_params), indices)) in proof
        .rounds
        .iter()
        .zip(&params.round_params)
        .zip(round_indices)
        .enumerate()
    {
        let dimensions = [Dimensions {
            height: round_params.domain_size >> round_params.folding_factor,
            width: 1usize << round_params.folding_factor,
        }];
        let expanded = match (&round_proof.openings, round == 0) {
            (QueryOpenings::Base(opening), true) => {
                expand_base_opening::<F, H, C, N, DIGEST_ELEMS>(
                    &dimensions,
                    indices,
                    &opening.rows,
                    &opening.proof,
                    hash,
                    compress,
                    cap_height,
                )
            }
            (QueryOpenings::Extension(opening), false) => {
                expand_extension_opening::<F, EF, H, C, N, DIGEST_ELEMS>(
                    &dimensions,
                    indices,
                    &opening.rows,
                    &opening.proof,
                    hash,
                    compress,
                    cap_height,
                )
            }
            _ => {
                return Err(WhirWitnessBuildError::InvalidShape(
                    "WHIR opening field does not match its round",
                ));
            }
        }
        .map_err(|source| WhirWitnessBuildError::Merkle { round, source })?;
        paths.extend(expanded);
    }

    let final_round = params.round_params.len();
    let final_dimensions = [Dimensions {
        height: params.final_domain_size >> params.final_sumcheck_rounds,
        width: 1usize << params.final_sumcheck_rounds,
    }];
    let expanded = match (&proof.final_openings, params.round_params.is_empty()) {
        (QueryOpenings::Base(opening), true) => expand_base_opening::<F, H, C, N, DIGEST_ELEMS>(
            &final_dimensions,
            final_indices,
            &opening.rows,
            &opening.proof,
            hash,
            compress,
            cap_height,
        ),
        (QueryOpenings::Extension(opening), false) => {
            expand_extension_opening::<F, EF, H, C, N, DIGEST_ELEMS>(
                &final_dimensions,
                final_indices,
                &opening.rows,
                &opening.proof,
                hash,
                compress,
                cap_height,
            )
        }
        _ => {
            return Err(WhirWitnessBuildError::InvalidShape(
                "WHIR final opening field does not match the round count",
            ));
        }
    }
    .map_err(|source| WhirWitnessBuildError::Merkle {
        round: final_round,
        source,
    })?;
    paths.extend(expanded);

    Ok(ExpandedWhirMmcsPaths { paths })
}

#[allow(clippy::too_many_arguments)]
fn expand_base_opening<F, H, C, const N: usize, const DIGEST_ELEMS: usize>(
    dimensions: &[Dimensions],
    indices: &[usize],
    rows: &[Vec<F>],
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
    let opened_values: Vec<Vec<Vec<F>>> = rows.iter().map(|row| vec![row.clone()]).collect();
    expand_pruned_merkle_paths(
        dimensions,
        indices,
        &opened_values,
        proof,
        hash,
        compress,
        cap_height,
    )
}

#[allow(clippy::too_many_arguments)]
fn expand_extension_opening<F, EF, H, C, const N: usize, const DIGEST_ELEMS: usize>(
    dimensions: &[Dimensions],
    indices: &[usize],
    rows: &[Vec<EF>],
    proof: &PrunedMerklePaths<F, DIGEST_ELEMS>,
    hash: &H,
    compress: &C,
    cap_height: usize,
) -> Result<Vec<Vec<[F; DIGEST_ELEMS]>>, MerkleWitnessError>
where
    F: Field + PackedValue<Value = F> + Default + Eq,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Clone + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], N> + Clone + Sync,
{
    let flattened_dimensions: Vec<Dimensions> = dimensions
        .iter()
        .map(|dimensions| Dimensions {
            height: dimensions.height,
            width: dimensions.width * EF::DIMENSION,
        })
        .collect();
    let opened_values: Vec<Vec<Vec<F>>> = rows
        .iter()
        .map(|row| {
            vec![
                row.iter()
                    .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                    .collect(),
            ]
        })
        .collect();
    expand_pruned_merkle_paths(
        &flattened_dimensions,
        indices,
        &opened_values,
        proof,
        hash,
        compress,
        cap_height,
    )
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_commit::Mmcs;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Dimensions;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::PrunedMerklePaths;
    use p3_test_utils::koala_bear_params::{
        DIGEST_ELEMS, F, MyCompress, MyHash, MyMmcs, default_koalabear_poseidon2_16,
    };

    use super::expand_base_opening;
    use crate::pcs::fri::MerkleWitnessError;

    type Proof = PrunedMerklePaths<F, DIGEST_ELEMS>;

    fn base_fixture() -> (
        [Dimensions; 1],
        Vec<usize>,
        Vec<Vec<F>>,
        Proof,
        MyHash,
        MyCompress,
    ) {
        let permutation = default_koalabear_poseidon2_16();
        let hash = MyHash::new(permutation.clone());
        let compress = MyCompress::new(permutation);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);
        let matrix = RowMajorMatrix::new((0..64).map(F::from_u64).collect(), 4);
        let dimensions = [Dimensions {
            height: 16,
            width: 4,
        }];
        let indices = vec![1, 5, 9];
        let (_, prover_data) = mmcs.commit(vec![matrix]);
        let (opened_values, proof) = mmcs.open_multi_batch(&indices, &prover_data);
        let rows = opened_values
            .into_iter()
            .map(|mut query| query.remove(0))
            .collect();
        (dimensions, indices, rows, proof, hash, compress)
    }

    #[test]
    fn honest_shared_frontier_expands_to_one_full_path_per_query() {
        let (dimensions, indices, rows, proof, hash, compress) = base_fixture();
        let paths = expand_base_opening::<F, MyHash, MyCompress, 2, DIGEST_ELEMS>(
            &dimensions,
            &indices,
            &rows,
            &proof,
            &hash,
            &compress,
            0,
        )
        .expect("honest shared frontier must expand");
        assert_eq!(paths.len(), indices.len());
        assert!(paths.iter().all(|path| !path.is_empty()));
    }

    #[test]
    fn malformed_shared_frontier_lengths_are_rejected() {
        let (dimensions, indices, rows, proof, hash, compress) = base_fixture();

        let mut short = proof.clone();
        short.sibling_hashes.pop();
        assert!(matches!(
            expand_base_opening::<F, MyHash, MyCompress, 2, DIGEST_ELEMS>(
                &dimensions,
                &indices,
                &rows,
                &short,
                &hash,
                &compress,
                0,
            ),
            Err(MerkleWitnessError::SiblingCountMismatch { .. })
        ));

        let mut long = proof;
        long.sibling_hashes.push([F::ZERO; DIGEST_ELEMS]);
        assert!(matches!(
            expand_base_opening::<F, MyHash, MyCompress, 2, DIGEST_ELEMS>(
                &dimensions,
                &indices,
                &rows,
                &long,
                &hash,
                &compress,
                0,
            ),
            Err(MerkleWitnessError::SiblingCountMismatch { .. })
        ));
    }

    #[test]
    fn verifier_owned_query_and_row_geometry_is_enforced() {
        let (dimensions, indices, rows, proof, hash, compress) = base_fixture();

        let mut missing_query = rows.clone();
        missing_query.pop();
        assert_eq!(
            expand_base_opening::<F, MyHash, MyCompress, 2, DIGEST_ELEMS>(
                &dimensions,
                &indices,
                &missing_query,
                &proof,
                &hash,
                &compress,
                0,
            ),
            Err(MerkleWitnessError::QueryCountMismatch {
                expected: indices.len(),
                got: missing_query.len(),
            })
        );

        let mut wrong_width = rows.clone();
        wrong_width[0].pop();
        assert!(matches!(
            expand_base_opening::<F, MyHash, MyCompress, 2, DIGEST_ELEMS>(
                &dimensions,
                &indices,
                &wrong_width,
                &proof,
                &hash,
                &compress,
                0,
            ),
            Err(MerkleWitnessError::WidthMismatch {
                query: 0,
                matrix: 0,
                ..
            })
        ));

        let mut out_of_bounds = indices;
        out_of_bounds[0] = dimensions[0].height;
        assert!(matches!(
            expand_base_opening::<F, MyHash, MyCompress, 2, DIGEST_ELEMS>(
                &dimensions,
                &out_of_bounds,
                &rows,
                &proof,
                &hash,
                &compress,
                0,
            ),
            Err(MerkleWitnessError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn inconsistent_duplicate_query_rows_are_rejected() {
        let (dimensions, _, rows, proof, hash, compress) = base_fixture();
        let indices = vec![1, 1];
        let mut duplicate_rows = vec![rows[0].clone(), rows[0].clone()];
        duplicate_rows[1][0] += F::ONE;
        assert_eq!(
            expand_base_opening::<F, MyHash, MyCompress, 2, DIGEST_ELEMS>(
                &dimensions,
                &indices,
                &duplicate_rows,
                &proof,
                &hash,
                &compress,
                0,
            ),
            Err(MerkleWitnessError::InconsistentDuplicateOpenings { slot: 0 })
        );
    }
}
