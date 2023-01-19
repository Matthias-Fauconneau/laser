use ui::{prelude::*, Widget, size, int2, EventContext};

#[macro_export] macro_rules! derive_IntoIterator { {pub struct $name:ident { $(pub $field_name:ident: $field_type:ty),*}} => {
    pub struct $name { $(pub $field_name: $field_type,)* }
    impl<'t> IntoIterator for &'t mut $name {
        type Item = &'t mut dyn ui::Widget;
        type IntoIter = std::array::IntoIter<Self::Item, ${count(field_name)} >;
        fn into_iter(self) -> Self::IntoIter { [$(&mut self.$field_name as &mut dyn ui::Widget),*].into_iter() }
    }
}}

pub struct Linear<T>(pub T);
impl<T> Widget for Linear<T> where for<'t> &'t mut T: IntoIterator<Item=&'t mut dyn Widget> {
    #[throws] fn paint(&mut self, target: &mut ui::Target, size: size, _offset: int2) {
        let mut widgets = self.0.into_iter();
        let len = 2;
        let mut pen = 0;
        for _ in 0..len { if let Some(widget) = widgets.next() {
            let size = widget.size(size-xy{x: 0, y: pen});
            let ref mut target = target.slice_mut(xy{x: 0, y: pen}, size);
            let size = target.size;
            widget.paint(target, size, 0.into())?;
            pen += size.y;
        } else { break; }}
        if pen < size.y {
            let size = size-xy{x: 0, y: pen};
            let ref mut target = target.slice_mut(xy{x: 0, y: pen}, size);
            image::fill(target, 0);
        }
    }
    fn event(&mut self, size: size, context: &mut ui::widget::EventContext, event: &ui::widget::Event) -> Result<bool> { for w in &mut self.0 { w.event(size, context, event)?; } Ok(false) }
}
pub type VBox<T> = Linear<T>;

pub struct Grid<T>(pub T);
impl<T> Widget for Grid<T> where for<'t> &'t mut T: IntoIterator<Item=&'t mut dyn Widget> {
    #[throws] fn paint(&mut self, target: &mut ui::Target, size: size, _offset: int2) {
        let mut widgets = self.0.into_iter();
        let (w, h) = (2, 2);
        for y in 0..h { for x in 0..w {
            let ref mut target = target.slice_mut(xy{x: x*size.x/w, y: y*size.y/h}, size/xy{x: w, y: h});
            let size = target.size;
            if let Some(widget) = widgets.next() { widget.paint(target, size, 0.into())?; } else { break; }
        }}
    }
    fn event(&mut self, size: size, context: &mut ui::widget::EventContext, event: &ui::widget::Event) -> Result<bool> { for w in &mut self.0 { w.event(size, context, event)?; } Ok(false) }
}

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

type ImageF = Image<Box<[f32]>>;
pub struct ImageView(pub ImageF);
impl Widget for ImageView {
    fn size(&mut self, size: size) -> size {
        let ref source = self.0;
        let (num, den) = if source.size.x*size.y > source.size.y*size.x { (source.size.x, size.x) } else { (source.size.y, size.y) };
        xy{x: std::cmp::min(source.size.x*den/num, size.x), y: std::cmp::min(source.size.y*den/num, size.y)}
    }
    #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) { rgb10(target, self.0.as_ref()) }
}

pub struct Fill<T>{ widget: T, fresh: bool }
impl<T> Fill<T> { fn new(widget: T) -> Self { Self{widget, fresh: false} } }
impl<T:Widget> Widget for Fill<T> { fn paint(&mut self, target: &mut ui::Target, size: ui::size, offset: ui::int2) -> ui::Result {
    if self.fresh { return Ok(()); } self.fresh = true;
    image::fill(target, ui::background.into());
    self.widget.paint(target, size, offset)
}
fn event(&mut self, _: size, _: &mut ui::EventContext, _: &ui::Event) -> Result<bool> { self.fresh = false; Ok(true) }
fn size(&mut self, size: ui::size) -> ui::size { self.widget.size(size) }
}

derive_IntoIterator! { pub struct LabelImage { pub label: Fill<ui::text::Text>, pub image: ImageView } }
pub type LabeledImage = VBox<LabelImage>;
impl LabeledImage { pub fn new(label: &'static str, image: ImageF) -> Self { Self(LabelImage{label: Fill::new(ui::text::text(label, &ui::text::bold)), image: ImageView(image)}) } }

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

//use ui::Widget;
pub fn write_image(path: impl AsRef<std::path::Path>, view: &mut impl Widget) {
    let mut target = Image::zero(xy{x: 3840, y: 2400});
    let size = target.size;
    view.event(size, &mut EventContext{modifiers_state: Default::default(), cursor: None}, &ui::Event::Stale).unwrap();
    view.paint(&mut target.as_mut(), size, 0.into()).unwrap();
    write_avif(path, target);
}
