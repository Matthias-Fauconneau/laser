#![allow(non_camel_case_types)]
pub trait atomic_from_mut<T> where Self:Sized { fn get_mut_slice(this: &mut [Self]) -> &mut [T] ; }
impl atomic_from_mut<f32> for atomic_float::AtomicF32 { fn get_mut_slice(this: &mut [Self]) -> &mut [f32] { unsafe { &mut *(this as *mut [Self] as *mut [f32]) } } }

use vector::xyz;
pub type uint3 = xyz<u32>;
pub type size = uint3;
pub struct Volume<D> {
    pub data : D,
    pub size : size,
}
impl<D> Volume<D> {
    #[track_caller] pub fn index(&self, xyz{x,y,z}: uint3) -> usize { assert!(x < self.size.x && y < self.size.y && z < self.size.z, "{x} {y} {z} {:?}", self.size); (((z * self.size.y + y) * self.size.x) + x) as usize }
    pub fn new<T>(size : size, data: D) -> Self where D:AsRef<[T]> { assert_eq!(data.as_ref().len(), (size.z*size.y*size.x) as usize); Self{data, size} }
    pub fn as_mut<T>(&mut self) -> Volume<&mut [T]> where D:AsMut<[T]> { Volume{data: self.data.as_mut(), size: self.size} }
}

impl<T, D:std::ops::Deref<Target=[T]>> std::ops::Index<usize> for Volume<D> {
    type Output=T;
    fn index(&self, i:usize) -> &Self::Output { &self.data[i] }
}
impl<T, D:std::ops::DerefMut<Target=[T]>> std::ops::IndexMut<usize> for Volume<D> {
    fn index_mut(&mut self, i:usize) -> &mut Self::Output { &mut self.data[i] }
}

impl<D> std::ops::Index<uint3> for Volume<D> where Self: std::ops::Index<usize> {
    type Output = <Self as std::ops::Index<usize>>::Output;
    fn index(&self, i:uint3) -> &Self::Output { &self[self.index(i)] }
}
impl<D> std::ops::IndexMut<uint3> for Volume<D> where Self: std::ops::IndexMut<usize> {
    fn index_mut(&mut self, i:uint3) -> &mut Self::Output { let i = self.index(i); &mut self[i] }
}

impl<'t, T> Volume<&'t mut [T]> {
    pub fn take_mut<'s>(&'s mut self, mid: u32) -> Volume<&'t mut[T]> {
        assert!(mid <= self.size.z);
        self.size.z -= mid;
        Volume{size: xyz{x: self.size.x, y: self.size.y, z: mid}, data: self.data.take_mut(..(mid*self.size.y*self.size.x) as usize).unwrap()}
    }
}

impl<T> Volume<Box<[T]>> {
    pub fn from_iter<I:IntoIterator<Item=T>>(size : size, iter : I) -> Self { Self::new(size, iter.into_iter().take((size.z*size.y*size.x) as usize).collect()) }
}
impl<T:Default> Volume<Box<[T]>> {
    pub fn default(size: size) -> Self { Self::from_iter(size, std::iter::repeat_with(|| T::default())) }
}
