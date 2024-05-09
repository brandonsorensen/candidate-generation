use dashmap::DashMap;
use hnsw_rs::hnsw::{Hnsw, Neighbour};
use ndarray::prelude::{Array1, Array2, s};
use tap::Tap;
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

pub(crate) struct KeyedVectorCache<K> {
  id_to_vector: DashMap<usize, K>,
  vectors: Array2<f32>
}

impl<K> KeyedVectorCache<K> {
  pub fn new(id_to_vector: DashMap<usize, K>, vectors: Array2<f32>) -> Self {
    Self { id_to_vector, vectors }
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
    let span  = span!(Level::DEBUG, "hnsw-init");
    let _guard = span.enter();
    let provider = Self::unwrap_field(self.vector_provider, "vector_provider")?;
    let shape = (provider.len(), provider.vector_dimensions() as usize);
    let mut uninitialized = Array2::<f32>::uninit(shape);
    let index_mapping = DashMap::<usize, usize>::with_capacity(provider.len());
    let mut order = Vec::<usize>::with_capacity(provider.len());
    debug!("Pre-init: Consuming vector provider");
    for (i, keyed_vector) in provider.enumerate() {
      order.push(keyed_vector.key);
      index_mapping.insert(keyed_vector.key, i);
      let as_array = Array1::from_vec(keyed_vector.vector);
      as_array.move_into_uninit(&mut uninitialized.slice_mut(s!(i, ..)));
    }
    debug!("Initializing index");
    let mut index = Hnsw::new(
      Self::unwrap_field(self.max_connections, "max_connections")?,
      shape.0,
      Self::unwrap_field(self.n_layers, "n_layers")?,
      Self::unwrap_field(self.ef_coef, "ef_coefficient")?,
      Self::unwrap_field(self.metric, "distance_metric")?
    );
    index.set_extend_candidates(false);
    unsafe {
      use ndarray::parallel::prelude::*;
      use ndarray::Axis;
      debug!("Initialized 2D array");
      let out = uninitialized.assume_init();
      out.axis_iter(Axis(0))
        .into_par_iter()
        .zip(order.into_par_iter())
        .inspect(|(_, key)| trace!("Inserting key \"{}\" into the index", key))
        .try_for_each(|(row, key)| {
          row.as_slice()
            .map(|unwrapped| index.insert_slice((unwrapped, key)))
            .ok_or_else(|| HnswRecommenderBuilderError::ValidationError("coudn't init index".to_string()))
        })?;
      debug!("Index initialized");
      // insert into index
      index.set_searching_mode(true);
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

  fn get_point(&self, key: &Self::Key) -> Option<Self::Point> {
    trace!("Retrieving point");
    self.id_to_vector.get(key)
      .map(|val| self.vectors.slice(s!(*val, ..)))
      // TODO
      .map(|arr| arr.into_owned())
      .map(|owned| owned.into_raw_vec())
  }

  fn search(&self, subject: &Self::Point, n_items: u16) -> Self::Neighbors {
    trace!("Searching for point in index");
    self.index.search(subject.as_slice(), n_items as usize, 20)
      .tap(|results| trace!("Searched returned {} results", results.len()))
      .into_iter()
      .map(Distance::from)
  }
}

#[cfg(feature = "space")]
impl From<Neighbour> for Distance<usize> {
  fn from(value: Neighbour) -> Self {
    Distance::new(
      value.d_id,
      value.distance
    )
  }
}
