pub trait Storage {
    fn save<T>(&self, value: T);
    fn search(&self, query: &str, limit: usize, threshold: f32);
    fn reset(&self);
}
