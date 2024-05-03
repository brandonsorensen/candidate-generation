use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Recommendation<T> {
  pub item_id: T,
  pub score: f32
}

impl<T> Recommendation<T> {
  pub fn new(item_id: T, score: f32) -> Self {
    Self { item_id, score }
  }
}

impl<IntoId, Id> From<(IntoId, f32)> for Recommendation<Id>
  where IntoId: Into<Id> {
  fn from(value: (IntoId, f32)) -> Self {
    Recommendation::new(value.0.into(), value.1)
  }
}

#[cfg(feature = "space")]
impl<T> From<crate::spatial::Distance<T>> for Recommendation<T> {
  fn from(value: crate::spatial::Distance<T>) -> Self {
    Recommendation::new(value.item_id, 1f32 - value.distance)
  }
}

