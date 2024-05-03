pub struct Distance<T> {
    pub item_id: T,
    pub distance: f32,
}

impl<T> Distance<T> {
    pub fn new(item_id: T, distance: f32) -> Self {
        Distance { item_id, distance }
    }
}

impl<IntoId, Id> From<(IntoId, f32)> for Distance<Id>
where
    IntoId: Into<Id>,
{
    fn from(value: (IntoId, f32)) -> Self {
        Distance::new(value.0.into(), value.1)
    }
}

#[allow(dead_code)]
pub trait NavigableIndex {
    type Key;
    type Point;
    type Neighbors: IntoIterator<Item = Distance<Self::Key>>;

    /// Return an interable of the nearest points without the distance metrics.
    fn get_neighbors(
        &self,
        subject: &Self::Point,
        n_items: u16,
    ) -> impl Iterator<Item = Self::Key> {
        self.search(subject, n_items)
            .into_iter()
            .map(|dist| dist.item_id)
    }

    /// Get an item from the index
    fn get_point(&self, key: &Self::Key) -> Option<Self::Point>;

    /// Return an interable of the nearest points in the space.
    fn search(&self, subject: &Self::Point, n_items: u16) -> Self::Neighbors;
}
