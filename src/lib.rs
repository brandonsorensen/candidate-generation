#[cfg(feature = "annoy")]
pub mod annoy_recommender;
pub mod error;
#[cfg(feature = "hnsw")]
pub mod hnsw_recommender;
pub mod list;
pub mod mapping;
#[cfg(feature = "random_recommender")]
pub mod random;
#[cfg(feature = "space")]
pub mod spatial;
pub mod types;

#[cfg(any(feature = "annoy", feature = "hnsw"))]
#[macro_use]
extern crate derive_builder;

#[cfg(feature = "random_recommender")]
pub use random::RandomRecommender;
#[cfg(feature = "annoy")]
pub use annoy_recommender::AnnoyRecommender;
#[cfg(feature = "hnsw")]
pub use hnsw_recommender::HnswRecommender;
pub use list::RecommendationList;
pub use error::RecommendError;
pub use types::Recommendation;

pub trait Recommender<K, R> {
  fn recommend(&self, item_id: &K, n_items: u16)
      -> Result<RecommendationList<R>, RecommendError>;
}

pub trait VectorProvider<K>: ExactSizeIterator<Item = KeyedVector<K>> {
  fn vector_dimensions(&self) -> u16;
}

pub struct KeyedVector<K> {
  pub key: K,
  pub vector: Vec<f32>
}

impl<K> KeyedVector<K> {
  pub fn new(key: K, vector: Vec<f32>) -> Self {
    KeyedVector { key, vector }
  }
}

impl<K> From<KeyedVector<K>> for (K, Vec<f32>) {
  fn from(value: KeyedVector<K>) -> Self {
    (value.key, value.vector)
  }
}


// impl<K, I> Recommender<K> for I
//   where I: NavigableIndex<Key = K>,
//         K: Clone + PartialEq {
//   fn recommend(&self, item_id: &K, n_items: u16)
//       -> Result<impl IntoIterator<Item = Recommendation<K>>, RecommendError<K>> {
//     self.get_point(item_id)
//       .ok_or_else(|| RecommendError::NotFound(item_id.clone()))
//       .map(|point| {
//         self.search(&point, n_items)
//           .into_iter()
//           .filter(|dist| dist.item_id != (*item_id))
//           .map(Recommendation::from)
//       })
//   }
// }
