use super::{
  Recommender,
  RecommendError,
  RecommendationList
};

pub struct IdMappingRecommender<M, R> {
  mapper: M,
  recommender: R,
}

impl<M, R> IdMappingRecommender<M, R> {
  pub fn new(mapper: M, recommender: R) -> Self {
    IdMappingRecommender { mapper, recommender }
  }
}

impl<M, R, InputKey, MappedKey, Rec> Recommender<InputKey, Rec> for IdMappingRecommender<M, R>
  where R: Recommender<MappedKey, Rec>,
        M: Fn(&InputKey) -> Option<&MappedKey> {
  fn recommend(&self, item_id: &InputKey, n_items: u16)
        -> Result<RecommendationList<Rec>, RecommendError> {
    (self.mapper)(item_id)
      .ok_or(RecommendError::NotFound)
      .and_then(|key| self.recommender.recommend(key, n_items))
  }
}
