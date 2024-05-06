use dashmap::DashMap;
use hnsw_rs::hnsw::{Hnsw, Neighbour};
use ndarray::prelude::{Array1, Array2, s};
use tracing::{Level, span, debug, trace};

use super::{
  Recommendation,
  Recommender,
  RecommendError,
  RecommendationList,
  VectorProvider
};

#[cfg(feature = "space")]
use std::{
  iter::Map,
  vec::IntoIter
};
#[cfg(feature = "space")]
use super::spatial::{Distance, NavigableIndex};

pub use hnsw_rs::dist;
pub use hnsw_rs::dist::Distance as HnswDistance;

pub struct HnswRecommender<'a, D>
  where D: HnswDistance<f32> + Send + Sync {
  index: Hnsw<'a, f32, D>,
  id_to_vector: DashMap<usize, usize>,
  vectors: Array2<f32>
}

impl<'a, D> HnswRecommender<'a, D>
  where D: HnswDistance<f32> + Send + Sync {
  fn new(index: Hnsw<'a, f32, D>, id_to_vector: DashMap<usize, usize>, vectors: Array2<f32>) -> Self {
    Self { index, id_to_vector, vectors }
  }

  pub fn builder<P>() -> HnswRecommenderBuilder<P, D>
    where P: VectorProvider<usize>{
    HnswRecommenderBuilder::default()
  }
}

impl<'a, T, D, Rec> Recommender<T, Rec> for HnswRecommender<'a, D>
  where D: HnswDistance<f32> + Send + Sync,
        T: TryInto<usize> + Clone,
        Rec: From<usize> + PartialEq + PartialEq<T> + PartialEq<usize> {
  fn recommend(&self, item_id: &T, n_items: u16)
      -> Result<RecommendationList<Rec>, RecommendError> {
    let span = span!(Level::DEBUG, "hnsw-recommend");
    let _guard = span.enter();
    debug!("Converting ID to usize");
    let converted: usize = item_id.clone().try_into()
      .map_err(|_| RecommendError::IncompatibleId)?;
    self.get_point(&converted)
      .ok_or(RecommendError::NotFound)
      .map(|point| {
        self.search(&point, n_items)
          .map(|distance| (distance.item_id, distance.distance))
          .map(Recommendation::from)
      })
      .map(|neighbors| RecommendationList::new_with_subject(&converted, neighbors))
  }
}

#[derive(Builder)]
#[builder(name = "HnswRecommenderBuilder", pattern="owned", public, build_fn(skip))]
#[allow(dead_code)]
pub struct HnswRecommenderArguments<P, D>
  where P: VectorProvider<usize>,
        D: HnswDistance<f32> {
  max_connections: usize,
  n_layers: usize,
  ef_coef: usize,
  metric: D,
  vector_provider: P
}

impl<P, D> HnswRecommenderBuilder<P, D>
  where P: VectorProvider<usize>,
        D: HnswDistance<f32> + Send + Sync {
  pub fn build(self) -> Result<HnswRecommender<'static, D>, HnswRecommenderBuilderError> {
    let span  = span!(Level::DEBUG, "hnws-init");
    let _guard = span.enter();
    let provider = Self::unwrap_field(self.vector_provider, "vector_provider")?;
    let index = Hnsw::new(
      Self::unwrap_field(self.max_connections, "max_connections")?,
      provider.len(),
      Self::unwrap_field(self.n_layers, "n_layers")?,
      Self::unwrap_field(self.ef_coef, "ef_coefficient")?,
      Self::unwrap_field(self.metric, "distance_metric")?
    );
    let shape = (provider.len(), provider.vector_dimensions() as usize);
    let mut target_array = ndarray::Array2::<f32>::uninit(shape);
    let index_mapping = dashmap::DashMap::<usize, usize>::with_capacity(provider.len());
    for (i, keyed_vector) in provider.enumerate() {
      index_mapping.insert(keyed_vector.key, i);
      let as_array = Array1::from_vec(keyed_vector.vector);
      as_array.move_into_uninit(&mut target_array.slice_mut(s!(i, ..)));
    }
    unsafe {
      let out = target_array.assume_init();
      Ok(HnswRecommender::new(index, index_mapping, out))
    }
  }

  fn unwrap_field<T>(val: Option<T>, name: &'static str) -> Result<T, HnswRecommenderBuilderError> {
    val.ok_or(HnswRecommenderBuilderError::UninitializedField(name))
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
