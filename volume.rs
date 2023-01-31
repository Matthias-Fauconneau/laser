#![allow(non_camel_case_types)]
pub trait get_mut<T> where Self:Sized { fn get_mut_slice(this: &mut [Self]) -> &mut [T] ; }
impl get_mut<f32> for atomic_float::AtomicF32 { fn get_mut_slice(this: &mut [Self]) -> &mut [f32] { unsafe { &mut *(this as *mut [Self] as *mut [f32]) } } }
impl get_mut<f64> for atomic_float::AtomicF64 { fn get_mut_slice(this: &mut [Self]) -> &mut [f64] { unsafe { &mut *(this as *mut [Self] as *mut [f64]) } } }

use vector::xyz;
pub type uint3 = xyz<u16>;
//pub fn prod(z: u16, y: u16, x: u16) -> u32 { z as u32 * y as u32 * x as u32 }
pub fn product(v: uint3) -> u32 { v.z as u32 * v.y as u32 * v.x as u32 }
pub type size = uint3;
pub struct Volume<D> {
    pub data : D,
    pub size : size,
}
impl<D> Volume<D> {
    #[track_caller] pub fn index(&self, xyz{x,y,z}: uint3) -> usize {
        assert!(x < self.size.x && y < self.size.y && z < self.size.z, "{x} {y} {z} {:?}", self.size);
        (((z as u32 * self.size.y as u32 + y as u32) * self.size.x as u32) + x as u32) as usize
    }
    pub fn new<T>(size : size, data: D) -> Self where D:AsRef<[T]> { assert_eq!(data.as_ref().len(), product(size) as usize); Self{data, size} }
    pub fn as_ref<T>(&self) -> Volume<&[T]> where D:AsRef<[T]> { Volume{data: self.data.as_ref(), size: self.size} }
    pub fn as_mut<T>(&mut self) -> Volume<&mut [T]> where D:AsMut<[T]> { Volume{data: self.data.as_mut(), size: self.size} }
    pub fn len(&self) -> usize { (self.size.z as u32 * self.size.y as u32 * self.size.x as u32) as usize }
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
    pub fn take_mut<'s>(&'s mut self, mid: u16) -> Volume<&'t mut[T]> {
        assert!(mid <= self.size.z);
        self.size.z -= mid;
        let size = xyz{x: self.size.x, y: self.size.y, z: mid};
        Volume{size, data: self.data.take_mut(..product(size) as usize).unwrap()}
    }
}

impl<T> Volume<Box<[T]>> {
    pub fn from_iter<I:IntoIterator<Item=T>>(size : size, iter : I) -> Self { Self::new(size, iter.into_iter().take((product(size)) as usize).collect()) }
    pub fn repeat(size: size, f: impl Fn()->T) -> Self { Self::from_iter(size, std::iter::repeat_with(f)) }
}
impl<T:Default> Volume<Box<[T]>> {
    pub fn default(size: size) -> Self { Self::repeat(size, || T::default()) }
}

//impl Volume<&mut [atomic_float::AtomicF32]> { pub fn get_mut(&mut self) -> Volume<&mut [f32]> { Volume::new(self.size, atomic_float::AtomicF32::get_mut_slice(&mut self.data)) } }
impl Volume<&mut [atomic_float::AtomicF64]> { pub fn get_mut(&mut self) -> Volume<&mut [f64]> { Volume::new(self.size, atomic_float::AtomicF64::get_mut_slice(&mut self.data)) } }
//impl Volume<Box<[atomic_float::AtomicF32]>> { pub fn get_mut(&mut self) -> Volume<&mut [f32]> { Volume::new(self.size, atomic_float::AtomicF32::get_mut_slice(&mut self.data)) } }
impl Volume<Box<[atomic_float::AtomicF64]>> { pub fn get_mut(&mut self) -> Volume<&mut [f64]> { Volume::new(self.size, atomic_float::AtomicF64::get_mut_slice(&mut self.data)) } }