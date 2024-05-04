pub mod annoy_recommender;
pub mod error;
pub mod list;
pub mod mapping;
#[cfg(feature = "random_recommender")]
pub mod random;
#[cfg(feature = "space")]
pub mod spatial;
pub mod types;

#[macro_use]
extern crate derive_builder;

#[cfg(feature = "random_recommender")]
pub use random::RandomRecommender;
pub use annoy_recommender::AnnoyRecommender;
pub use list::RecommendationList;
pub use error::RecommendError;
pub use types::Recommendation;

pub trait Recommender<K, R> {
  fn recommend(&self, item_id: &K, n_items: u16)
      -> Result<RecommendationList<R>, RecommendError>;
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
