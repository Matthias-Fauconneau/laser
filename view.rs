pub struct Grid<T>(pub T);
impl<T> ui::Widget for Grid<T> where for<'t> &'t mut T: IntoIterator<Item=&'t mut dyn ui::Widget> {
    #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, size: ui::size, _offset: ui::int2) {
        let mut widgets = self.0.into_iter();
        //let start = std::time::Instant::now();
        let (w, h) = (2, 2);
        for y in 0..h { for x in 0..w {
            let ref mut target = target.slice_mut(xy{x: x*size.x/w, y: y*size.y/h}, size/xy{x: w, y: h});
            let size = target.size;
            if let Some(widget) = widgets.next() { widget.paint(target, size, 0.into())?; } else { break; }
        }}
        //let elapsed = start.elapsed();
        //println!("plot: {}ms", elapsed.as_millis());
    }
}
#[macro_export] macro_rules! derive_IntoIterator { {struct $name:ident { $($field_name:ident: $field_type:ty),*}} => {
    struct $name { $($field_name: $field_type,)* }
    impl<'t> IntoIterator for &'t mut $name {
        type Item = &'t mut dyn ui::Widget;
        type IntoIter = std::array::IntoIter<Self::Item, ${count(field_name)} >;
        fn into_iter(self) -> Self::IntoIter { [$(&mut self.$field_name as &mut dyn ui::Widget),*].into_iter() }
    }
}}

use ui::plot::list;
#[derive(Debug)] pub struct Plot { pub title: &'static str, pub axis_label: xy<&'static str>, pub x_scale: f32, pub keys: Box<[String]>, pub values: Box<[Vec<f32>]> }
impl ui::Widget for Plot { #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, size: ui::size, offset: ui::int2) {
    ui::Plot::new(self.title.to_string(), self.axis_label.map(|s| s.to_string()), &[&list(self.keys.iter().map(|s| s.as_ref()))], list((0..self.values[0].len()).map(|i| ((self.x_scale*(i as f32)) as f64, Box::from([list(self.values.iter().map(|values| values[i] as f64))])))).as_ref()).paint(target, size, offset)?;
} }

use {vector::xy, image::Image};
pub fn rgb10(target: &mut Image<&mut [u32]>, source: Image<&[f32]>) {
    let max = source.iter().copied().reduce(f32::max).unwrap();
    if max == 0. { return; }
    let (num, den) = if source.size.x*target.size.y > source.size.y*target.size.x { (source.size.x, target.size.x) } else { (source.size.y, target.size.y) };
    for y in 0..std::cmp::min(source.size.y*den/num, target.size.y) {
        for x in 0..std::cmp::min(source.size.x*den/num, target.size.x) {
            let w = (source[xy{x: x*num/den, y: y*num/den}]/max * ((1<<10)-1) as f32) as u32;
            target[xy{x,y}] = w | w<<10 | w<<20;
        }
    }
}
pub struct ImageView(pub Image<Box<[f32]>>);
impl ui::Widget for ImageView { #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) {
    image::fill(target, 0);
    rgb10(target, self.0.as_ref()) }
}

//use image::Image;
pub fn write_avif(path: impl AsRef<std::path::Path>, image: Image<Box<[u32]>>) {
    #[cfg(not(feature="avif"))] println!("Built without AVIF support: {} {}", path.as_ref().display(), image.size);
    #[cfg(feature="avif")] {
        use ravif::*;
        let EncodedImage { avif_file, .. } = Encoder::new().encode_raw_planes_10_bit(image.size.x as usize, image.size.y as usize,
            image.iter().map(|rgb| [(rgb&0b1111111111) as u16, ((rgb>>10)&0b1111111111) as u16, (rgb>>20) as u16]),
            None::<[_; 0]>, rav1e::color::PixelRange::Full, MatrixCoefficients::Identity).unwrap(); // FIXME: PQ
        std::fs::write(path, avif_file).unwrap();
    }
}

use ui::Widget;
pub fn write_image(path: impl AsRef<std::path::Path>, view: &mut impl Widget) {
    let mut target = Image::zero(xy{x: 3840, y: 2400});
    let size = target.size;
    view.paint(&mut target.as_mut(), size, 0.into()).unwrap();
    write_avif(path, target);
}