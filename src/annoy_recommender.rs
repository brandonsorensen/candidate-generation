use std::{
  fs::File,
  io::BufReader,
  path::PathBuf
};

use anyhow::Result;
use arroy::{
  Database as ArroyDatabase,
  Reader,
  Writer,
  distances::DotProduct
};
use heed::{
  Env,
  EnvOpenOptions,
};
use rand::{
  SeedableRng,
  rngs::StdRng
};
use tracing::{Level, span, debug, trace};

use super::{
  Recommender,
  RecommendationList,
  error::RecommendError
};

pub struct AnnoyRecommender<D> {
  pub db: ArroyDatabase<D>,
  pub env: Env
}

impl<D> AnnoyRecommender<D> {
  pub fn new(db: ArroyDatabase<D>, env: Env) -> Self {
    Self { db, env }
  }

  pub fn builder<P, T>() -> AnnoyRecommenderBuilder<P, T>
    where P: IntoIterator<Item = T> + Clone,
            T: Into<u32> + Clone {
    AnnoyRecommenderBuilder::default()
  }
}

#[derive(Builder)]
#[builder(name = "AnnoyRecommenderBuilder", public, build_fn(skip))]
pub struct AnnoyRecommenderArguments<P, T>
  where P: IntoIterator<Item = T>,
        T: Into<u32> {
  map_size: usize,
  max_dbs: usize,
  path: PathBuf,
  vector_provider: Option<P>
}

impl<P, T> AnnoyRecommenderBuilder<P, T>
  where P: IntoIterator<Item = T>,
        T: Into<u32> {
  fn build(&self) -> Result<AnnoyRecommender<DotProduct>, AnnoyRecommenderBuilderError>
    {
    let span  = span!(Level::DEBUG, "annoy-init");
    let _guard = span.enter();
    debug!("Initializing heed environment");
    let env = EnvOpenOptions::new()
      .map_size(self.map_size.unwrap() as usize)
      .max_dbs(self.max_dbs.unwrap() as u32)
      .open(self.path.unwrap())?;
    let db = match &self.vector_provider.unwrap() {
      Some(init_config) => Self::init_db(&env, init_config),
      None => Self::open_existing_db(&env)
    }?;
    Ok(Self::new(db, env))
  }

  fn init_db(&self, env: &Env, config: &DatabaseInitConfig) -> Result<ArroyDatabase<DotProduct>> {
    debug!("Initializing new heed DB");
    let mut wrtx = env.write_txn()?;
    let db = env.create_database(&mut wrtx, Some("listing-db"))?;
    let listing_file = File::open(&config.listing_vectors)?;
    // let vec_iter = Unpacker::new(BufReader::new(listing_file), config.vector_dimensions)?;
    let writer = Writer::<DotProduct>::new(db, 0, config.vector_dimensions);
    let n_elements = vec_iter.n_elements;
    debug!("Loading {} vectors", n_elements);
    for (i, listing) in vec_iter.into_iter().enumerate() {
      let id: u32 = listing.key.try_into()?;
      trace!("Inserting vector {}/{} with ID \"{}\"", i, n_elements, id);
      writer.add_item(&mut wrtx, id, &listing.vector)?;
    }
    debug!("Committing initialize transaction");
    let mut rng = StdRng::from_entropy();
    writer.build(&mut wrtx, &mut rng, None)?;
    wrtx.commit()?;
    Ok(db)
  }

  fn open_existing_db(env: &Env) -> Result<ArroyDatabase<DotProduct>> {
    let rtx = env.read_txn()?;
    let db = env.open_database(&rtx, Some("listing-db"))?
      .ok_or_else(|| anyhow::anyhow!("Couldn't open existing heed DB"));
    let _ = rtx.commit();
    db
  }
}

pub struct DatabaseInitConfig {
  pub listing_vectors: PathBuf,
  pub vector_dimensions: usize
}

// impl<D> NavigableIndex for AnnoyRecommender<D>
//   where D: arroy::Distance {
//   type Key = ListingId;
//   type Point = Vec<f32>;
//   type Neighbors = std::vec::IntoIter<Distance<Self::Key>>;

//   fn get_point(&self, key: &Self::Key) -> Option<Self::Point> {
//     let rtx = self.env.read_txn()
//       .inspect_err(|e| error!("couldn't open read transaction: {}", e))
//       .ok()?;
//     trace!("creating reader");
//     let reader = Reader::open(&rtx, 0, self.db)
//       .inspect_err(|e| error!("couldn't create reader: {}", e))
//       .ok()?;
//     trace!("converting id");
//     let item_id: u32 = (*key).try_into().ok()?;
//     trace!("locating subject vector");
//     reader.item_vector(&rtx, item_id)
//       .inspect_err(|e| error!("couldn't read vector: {}", e))
//       .unwrap()
//   }

//   #[allow(unused)]
//   fn search(&self, subject: &Self::Point, n_items: u16) -> Self::Neighbors {
//     todo!()
//     let rtx = self.env.read_txn()
//       .inspect_err(|e| error!("couldn't open read transaction: {}", e))
//       .ok()?;
//     trace!("creating reader");
//     let reader = Reader::open(&rtx, 0, self.db)
//       .inspect_err(|e| error!("couldn't create reader: {}", e))
//       .ok()?;
//     trace!("locating subject vector");
//     search_with_reader(
//       subject.into(), n_items, rtc, reader
//     )
//   }
// }

// #[allow(dead_code)]
// fn search_with_reader<'a, D>(
//   subject: &[f32], n_items: u16,
//   rtx: &'a RoTxn, reader: Reader<'a, D>
// ) -> impl Iterator<Item = Distance<ListingId>>
//   where D: arroy::Distance {
//   // TODO: Error handling
//   reader.nns_by_vector(
//     rtx, subject, n_items as usize,
//     None, None
//   ).unwrap()
//    .into_iter()
//    .map(Distance::from)
// }

impl<D, Key, Rec> Recommender<Key, Rec> for AnnoyRecommender<D>
  where D: arroy::Distance,
        Key: TryInto<u32> + PartialEq + std::fmt::Debug + Clone,
        Rec: From<u32> + PartialEq + PartialEq<Key> {
    fn recommend(&self, subject_id: &Key, n_recommendations: u16)
      -> Result<RecommendationList<Rec>, RecommendError> {
    let span = span!(Level::TRACE, "arroy-recommend");
    debug!("Traversing annoy graph");
    let _guard = span.enter();
    trace!("Creating read transaction");
    let rtx = self.env.read_txn().map_err(RecommendError::DatabaseError)?;
    trace!("Creating reader");
    let reader = Reader::open(&rtx, 0, self.db)?;
    trace!("Converting input Id {:?}", subject_id);
    let converted_id: u32 = subject_id.clone().try_into()
      .map_err(|_| RecommendError::IncompatibleId)?;
    trace!("Locating subject vector");
    let subject_vector= reader.item_vector(&rtx, converted_id)?
      .ok_or(RecommendError::NotFound)?;
    let recs = RecommendationList::new_with_subject(
      subject_id, reader.nns_by_vector(
        &rtx, &subject_vector, n_recommendations as usize,
        None, None
      )?
    );
    trace!("Returning {} recommendations", recs.0.len());
    Ok(recs)
  }
}
