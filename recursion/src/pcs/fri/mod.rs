//! FRI for recursive verification.

mod params;
mod targets;
mod verifier;
mod witness;

pub use params::FriVerifierParams;
pub use targets::{
    BatchMultiOpeningTargets, CommitPhaseMultiStepTargets, FriProofTargets, HashProofTargets,
    HidingFriProofTargets, HidingHashProofTargets, HidingOpenedValuesTargets,
    HidingPrunedMerklePathsTargets, InputProofTargets, MerkleCapTargets, MmcsMultiProofTargets,
    MmcsProofTargets, PrunedMerklePathsTargets, RecExtensionValMmcs, RecExtensionValMmcsArity4,
    RecValHidingMmcs, RecValHidingScalarMmcs, RecValMmcs, RecValMmcsArity4, TwoAdicFriProofTargets,
    Witness,
};
pub use verifier::verify_fri_circuit;
pub use witness::{
    ExpandedFriMmcsPaths, FriInputMatrix, FriWitnessBuildError, FriWitnessError,
    MerkleWitnessError, ReconstructedFriRows, expand_fri_mmcs_paths, expand_hiding_fri_mmcs_paths,
    expand_pruned_merkle_paths, reconstruct_two_adic_fri_rows,
};
