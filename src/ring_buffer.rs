// SPDX-License-Identifier: MIT
//! Fixed-size ring buffer for `no_std` environments
//!
//! `MaybeUninit<T>` ベースの固定長リングバッファ。ヒープ不要。
//! センサーデータのストリーミングウィンドウに使用。
//!
//! Author: Moroya Sakamoto

use core::mem::MaybeUninit;

/// `固定長リングバッファ（no_std` 対応）
///
/// `N` 要素の固定サイズバッファ。満杯時は最古の要素を上書き。
pub struct RingBuffer<T, const N: usize> {
    buf: [MaybeUninit<T>; N],
    head: usize,
    len: usize,
}

impl<T: Copy, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> RingBuffer<T, N> {
    /// 空のリングバッファを作成
    #[must_use]
    pub const fn new() -> Self {
        Self {
            // SAFETY: MaybeUninit<T> の配列は初期化不要。assume_init() は
            // MaybeUninit の配列に対して安全（内部値は未使用のまま）。
            buf: unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() },
            head: 0,
            len: 0,
        }
    }

    /// 要素を追加（満杯時は最古を上書き）
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        let idx = (self.head + self.len) % N;
        self.buf[idx] = MaybeUninit::new(value);
        if self.len < N {
            self.len += 1;
        } else {
            self.head = (self.head + 1) % N;
        }
    }

    /// 格納されている要素数
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// バッファが空かどうか
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// バッファが満杯かどうか
    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// バッファ容量
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// インデックスでアクセス（0 = 最古の要素）
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let actual = (self.head + index) % N;
        // SAFETY: index < self.len で、push 時に MaybeUninit::new() で初期化済み。
        Some(unsafe { self.buf[actual].assume_init_ref() })
    }

    /// 最新の要素を取得
    #[inline(always)]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        self.get(self.len - 1)
    }

    /// 最古の要素を取得
    #[inline(always)]
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// バッファをクリア
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// イテレータ（最古→最新の順）
    pub fn iter(&self) -> RingBufferIter<'_, T, N> {
        RingBufferIter {
            buf: self,
            index: 0,
        }
    }
}

/// リングバッファのイテレータ
pub struct RingBufferIter<'a, T, const N: usize> {
    buf: &'a RingBuffer<T, N>,
    index: usize,
}

impl<'a, T: Copy, const N: usize> IntoIterator for &'a RingBuffer<T, N> {
    type Item = &'a T;
    type IntoIter = RingBufferIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Copy, const N: usize> Iterator for RingBufferIter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.buf.get(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buf.len() - self.index;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let rb: RingBuffer<i32, 4> = RingBuffer::new();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 4);
    }

    #[test]
    fn test_push_and_get() {
        let mut rb: RingBuffer<i32, 4> = RingBuffer::new();
        rb.push(10);
        rb.push(20);
        rb.push(30);
        assert_eq!(rb.len(), 3);
        assert_eq!(*rb.get(0).unwrap(), 10);
        assert_eq!(*rb.get(2).unwrap(), 30);
    }

    #[test]
    fn test_wrap_around() {
        let mut rb: RingBuffer<i32, 3> = RingBuffer::new();
        rb.push(1);
        rb.push(2);
        rb.push(3);
        assert!(rb.is_full());
        rb.push(4); // 1 が上書きされる
        assert_eq!(rb.len(), 3);
        assert_eq!(*rb.first().unwrap(), 2);
        assert_eq!(*rb.last().unwrap(), 4);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let mut rb: RingBuffer<u8, 2> = RingBuffer::new();
        rb.push(1);
        assert!(rb.get(0).is_some());
        assert!(rb.get(1).is_none());
    }

    #[test]
    fn test_clear() {
        let mut rb: RingBuffer<i32, 4> = RingBuffer::new();
        rb.push(1);
        rb.push(2);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_iter() {
        let mut rb: RingBuffer<i32, 4> = RingBuffer::new();
        for i in 0..6 {
            rb.push(i);
        }
        // buf contains [2, 3, 4, 5]
        let mut iter = rb.iter();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), Some(&5));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_first_last() {
        let mut rb: RingBuffer<i32, 3> = RingBuffer::new();
        assert!(rb.first().is_none());
        assert!(rb.last().is_none());
        rb.push(10);
        assert_eq!(*rb.first().unwrap(), 10);
        assert_eq!(*rb.last().unwrap(), 10);
        rb.push(20);
        assert_eq!(*rb.first().unwrap(), 10);
        assert_eq!(*rb.last().unwrap(), 20);
    }

    #[test]
    fn test_single_element_buffer() {
        let mut rb: RingBuffer<i32, 1> = RingBuffer::new();
        rb.push(1);
        assert_eq!(rb.len(), 1);
        rb.push(2);
        assert_eq!(rb.len(), 1);
        assert_eq!(*rb.get(0).unwrap(), 2);
    }
}
