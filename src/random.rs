use std::iter::repeat_with;

use rand::prelude::Rng;

use super::{
  Recommender,
  Recommendation,
  RecommendationList,
  RecommendError
};

#[derive(Builder)]
pub struct RandomRecommender<Provider, Rec>
  where Provider: Fn() -> Rec {
  id_provider: Provider,
  #[builder(default = "0.2")]
  empty_rate: f32
}

impl<Provider, Rec> RandomRecommender<Provider, Rec>
  where Provider: Fn() -> Rec + Clone,
        Rec: Clone {
  pub fn builder() -> RandomRecommenderBuilder<Provider, Rec> {
    RandomRecommenderBuilder::default()
  }
}

impl<Provider, Rec> RandomRecommender<Provider, Rec>
  where Provider: Fn() -> Rec {
  // Can be instantiated with builder
  #[allow(dead_code)]
  pub fn new(id_provider: Provider, empty_rate: f32) -> Self {
    Self { id_provider, empty_rate }
  }
}

impl<Provider, Key, Rec> Recommender<Key, Rec> for RandomRecommender<Provider, Rec>
  where Provider: Fn() -> Rec,
        Rec: PartialEq {
  fn recommend(&self, _subject_id: &Key, n_recommendations: u16)
    -> Result<RecommendationList<Rec>, RecommendError> {
    let mut rng = rand::thread_rng();
    if rng.gen::<f32>() < self.empty_rate {
      return Err(RecommendError::NotFound)
    }
    let n_recs = n_recommendations as usize;
    let recs = repeat_with(|| (self.id_provider)())
      .zip(repeat_with(move || rng.gen::<f32>()))
      .map(Recommendation::from)
      .take(n_recs);
    Ok(RecommendationList::from_iter_with_sort(recs))
  }
}
