use dashmap::DashMap;
use hnsw_rs::{
  dist::Distance as HnswDistance,
  hnsw::{Hnsw, Neighbour}
};
use ndarray::prelude::{Array2, s};
use tracing::{Level, span, debug, trace};

use super::{
  Recommendation,
  Recommender,
  RecommendError,
  RecommendationList
};

#[cfg(feature = "space")]
use std::{
  iter::Map,
  vec::IntoIter
};
#[cfg(feature = "space")]
use super::spatial::{Distance, NavigableIndex};

pub use hnsw_rs::dist;

struct HnswRecommender<'a, D>
  where D: HnswDistance<f32> + Send + Sync {
  index: Hnsw<'a, f32, D>,
  id_to_vector: DashMap<usize, usize>,
  vectors: Array2<f32>
}

impl<'a, T, D> Recommender<T, usize> for HnswRecommender<'a, D>
  where D: HnswDistance<f32> + Send + Sync,
        T: TryInto<usize> + Clone {
  fn recommend(&self, item_id: &T, n_items: u16)
      -> Result<RecommendationList<usize>, RecommendError> {
    let span = span!(Level::DEBUG, "hnsw-recommend");
    let _guard = span.enter();
    debug!("Converting ID to usize");
    let converted: usize = item_id.clone().try_into()
      .map_err(|_| RecommendError::IncompatibleId)?;
    self.get_point(&converted)
      .ok_or(RecommendError::NotFound)
      .map(|point| self.search(&point, n_items).map(Recommendation::from))
      .map(|neighbors| RecommendationList::new_with_subject(&converted, neighbors))
  }
}

#[cfg(feature = "space")]
impl<'a, D> NavigableIndex for HnswRecommender<'a, D>
  where D: HnswDistance<f32> + Send + Sync {
  type Key = usize;
  type Point = Vec<f32>;
  type Neighbors = Map<IntoIter<Neighbour>, fn(Neighbour) -> Distance<usize>>;

  fn search(&self, subject: &Self::Point, n_items: u16) -> Self::Neighbors {
    trace!("Searching for point in index");
    self.index.search(subject.as_slice(), n_items as usize, 1)
      .into_iter()
      .map(Distance::from)
  }

  fn get_point(&self, key: &Self::Key) -> Option<Self::Point> {
    trace!("Retrieving point");
    self.id_to_vector.get(key)
      .map(|val| self.vectors.slice(s!(*val, ..)))
      // TODO
      .map(|arr| arr.into_owned())
      .map(|owned| owned.into_raw_vec())
  }
}

#[cfg(feature = "space")]
impl From<Neighbour> for Distance<usize> {
  fn from(value: hnsw_rs::hnsw::Neighbour) -> Self {
    super::spatial::Distance::new(
      value.d_id,
      value.distance
    )
  }
}
