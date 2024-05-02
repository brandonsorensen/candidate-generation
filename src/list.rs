use serde::Serialize;

use super::Recommendation;

#[derive(Debug, Serialize)]
pub struct RecommendationList<K>(pub Vec<Recommendation<K>>);

impl<K> RecommendationList<K> {

  pub fn new_with_sort(mut recs: Vec<Recommendation<K>>) -> Self {
    recs.sort_by(|this, other| {
        other.score.partial_cmp(&this.score).unwrap()
      }
    );
    Self(recs)
  }

  pub fn from_iter<I>(value: I) -> Self
    where I: IntoIterator,
          I::Item: Into<Recommendation<K>> {
    Self(value.into_iter()
      .map(|item| item.into())
      .collect::<Vec<Recommendation<K>>>())
  }

  pub fn from_iter_with_sort<I>(value: I) -> Self
    where I: IntoIterator,
          I::Item: Into<Recommendation<K>>,
          K: PartialEq {
    let recs = value.into_iter()
      .map(|item| item.into())
      .collect::<Vec<Recommendation<K>>>();
    Self::new_with_sort(recs)
  }

  pub fn new_with_subject<I, O>(subject_id: &K, recommendations: I) -> RecommendationList<O>
    where I: IntoIterator,
          Recommendation<O>: From<<I as IntoIterator>::Item>,
          K: PartialEq,
          O: PartialEq<K> + PartialEq {
    RecommendationList::from_iter_with_sort(
      recommendations.into_iter()
        .map(Recommendation::<O>::from)
        .filter(|rec| rec.item_id != (*subject_id))
    )
  }
}

impl<K> From<RecommendationList<K>> for Vec<Recommendation<K>> {
  fn from(value: RecommendationList<K>) -> Self {
    value.0
  }
}

impl<I, K> From<I> for RecommendationList<K>
  where I: IntoIterator,
        I::Item: Into<Recommendation<K>>,
        K: PartialEq {
  fn from(value: I) -> Self {
    let recs = value.into_iter()
      .map(|v| v.into())
      .collect::<Vec<Recommendation<K>>>();
    RecommendationList::new_with_sort(recs)
  }
}
