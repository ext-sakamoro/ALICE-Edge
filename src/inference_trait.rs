//! `EdgeInference` trait.

pub trait EdgeInference {
    /// 推論結果の型
    type Output;
    /// f32 特徴量ベクトルからの推論（クラスIDと信頼度を返す）
    fn infer(&mut self, features: &[f32]) -> (usize, f32);
    /// モデルの重みメモリ使用量（バイト）
    fn model_size_bytes(&self) -> usize;
}
