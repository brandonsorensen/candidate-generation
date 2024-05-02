use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecommendError {
  #[error("database unreachable")]
  DatabaseError(#[from] heed::Error),
  #[error("could not search index")]
  AnnoyError(#[from] arroy::Error),
  #[error("incompatible ID type for operatation")]
  IncompatibleId,
  #[error("vector not found")]
  NotFound
}
