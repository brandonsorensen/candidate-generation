use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecommendError {
  #[cfg(feature = "annoy")]
  #[error("database unreachable")]
  DatabaseError(#[from] heed::Error),
  #[cfg(feature = "annoy")]
  #[error("could not search index")]
  AnnoyError(#[from] arroy::Error),
  #[error("incompatible ID type for operatation")]
  IncompatibleId,
  #[error("vector not found")]
  NotFound
}
