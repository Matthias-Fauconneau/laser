use ui::{prelude::*, Widget, Target, size, int2, Event::{self, Key}, EventContext};

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
    #[throws] fn paint(&mut self, target: &mut Target, size: size, _offset: int2) {
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
            image::fill(target, background().into());
        }
    }
    fn event(&mut self, size: size, context: &mut Option<EventContext>, event: &Event) -> Result<bool> { for w in &mut self.0 { w.event(size, context, event)?; } Ok(false) }
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
    fn event(&mut self, size: size, context: &mut Option<ui::widget::EventContext>, event: &ui::widget::Event) -> Result<bool> { for w in &mut self.0 { w.event(size, context, event)?; } Ok(false) }
}

use {num::lerp, vector::{xy, minmax, MinMax}, image::{Image, fill, /*PQ10*/sRGB8, bgr}, ui::{background, text::text}};
pub fn rgb/*10*/(target: &mut Image<&mut [u32]>, source: Image<&[f32]>, unit: impl Fn(f32)->String) {
    let MinMax{min,max} = minmax(source.data.into_iter().copied()).unwrap();
    if min == max { return; }
    /*let mut histogram = vec![0; target.size.x as usize];
    for v in source.data { histogram[((v-min)/(max-min)*(target.size.x-1) as f32) as usize] += 1; }
    let max_i = histogram.len()-1-histogram.iter().rev().scan(0, |sum, &h| { *sum += h; Some(*sum) }).position(|sum| sum>32).unwrap();
    let max = min+(max_i as f32)/(target.size.x as f32)*(max-min);*/
    //let MinMax{min,max} = minmax(source.data.into_iter().filter(|&v| histogram[((v-min)/(max-min)*(target.size.x-1) as f32) as usize]>source.size.y/2).copied()).unwrap();
    let [num, den] = if source.size.x*target.size.y > source.size.y*target.size.x { [source.size.x, target.size.x] } else { [source.size.y, target.size.y] };
    for y in 0..std::cmp::min(source.size.y*den/num, target.size.y) {
        for x in 0..std::cmp::min(source.size.x*den/num, target.size.x) {
            let v = source[xy{x: x*num/den, y: y*num/den}];
            target[xy{x,y}] = if v >= 0. { bgr::from(/*PQ10*/sRGB8(f32::min(v/max, 1.))) } else { bgr{b: /*PQ10*/sRGB8(f32::min(-v/-min, 1.)), g:0, r:0} }.into()
        }
    }
    let mut histogram = vec![0; target.size.x as usize];
    for v in source.data { histogram[(f32::clamp((v-min)/(max-min), 0., 1.)*(target.size.x-1) as f32) as usize] += 1; }
    let histogram_max = *histogram.iter().max().unwrap();
    {
        let mut target = target.slice_mut(xy{x: 0, y: target.size.y/2}, xy{x: target.size.x, y:target.size.y/4});
        fill(&mut target, background().into());
        for x in 0..target.size.x {
            let v = min+(x as f32)/(target.size.x as f32)*(max-min);
            let c = if v >= 0. { bgr::from(/*PQ10*/sRGB8(v/max)) } else { bgr{b: /*PQ10*/sRGB8(-v/-min), g:0, r:0} }.into();
            //for y in lerp(histogram[x as usize] as f32 / histogram_max as f32, target.size.y, 0)..target.size.y { target[xy{x,y}] = c; }
            if histogram[x as usize] > 0 {
                let y0 = lerp(f32::ln(histogram[x as usize] as f32)/f32::ln(histogram_max as f32), target.size.y, 0);
                for y in y0..target.size.y { target[xy{x,y}] = c; }
            }
        }
    }
    for (i, &v) in [min,max].iter().enumerate() {
        let mut target = target.slice_mut(xy{x: i as u32*target.size.x/2, y: target.size.y*3/4}, xy{x: target.size.x/2, y:target.size.y/4});
        fill(&mut target, background().into());
        let text = format!("{}", unit(v));
        let mut text = ui::text(&text);
        let size = target.size;
        let offset = xy{x: [0,(target.size.x/text.scale(target.size)-text.size().x) as i32][i], y: 0};
        text.paint_fit(&mut target, size, offset);
    }
}

type ImageF = Image<Box<[f32]>>;
pub struct ImageView { pub image: ImageF, pub unit: Box<dyn Fn(f32)->String>}
impl Widget for ImageView {
    fn size(&mut self, size: size) -> size {
        let ref source = self.image;
        let (num, den) = if source.size.x*size.y > source.size.y*size.x { (source.size.x, size.x) } else { (source.size.y, size.y) };
        xy{x: std::cmp::min(source.size.x*den/num, size.x), y: std::cmp::min(source.size.y*den/num, size.y)*2}
    }
    #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) { rgb/*10*/(target, self.image.as_ref(), &self.unit) }
}

pub struct Fill<T>{ widget: T, fresh: bool }
impl<T> Fill<T> { fn new(widget: T) -> Self { Self{widget, fresh: false} } }
impl<T:Widget> Widget for Fill<T> { fn paint(&mut self, target: &mut ui::Target, size: ui::size, offset: ui::int2) -> ui::Result {
    if self.fresh { return Ok(()); } self.fresh = true;
    fill(target, background().into());
    self.widget.paint(target, size, offset)
}
fn event(&mut self, _: size, _: &mut Option<ui::EventContext>, _: &ui::Event) -> Result<bool> { self.fresh = false; Ok(true) }
fn size(&mut self, size: ui::size) -> ui::size { self.widget.size(size) }
}

derive_IntoIterator! { pub struct LabelImage { pub label: Fill<ui::Text>, pub image: ImageView } }
pub type LabeledImage = VBox<LabelImage>;
impl LabeledImage { pub fn new(label: &'static str, image: ImageF, unit: Box<dyn Fn(f32)->String>) -> Self {
    Self(LabelImage{label: Fill::new(text(label)), image: ImageView{image, unit}}) } }

pub struct App<'i, 'a, 'f, S, W> {
    pub state: S,
    pub widget: W,
    pub actions: &'a [(char,&'f dyn Fn(&mut Self))],
}
impl<S, W: Widget> Widget for App<'_, '_, '_, S, W> {
    fn paint(&mut self, target: &mut Target, size: size, offset: int2) -> Result { self.widget.paint(target, size, offset) }
    fn event(&mut self, size: size, context: &mut Option<EventContext>, event: &ui::Event) -> Result<bool> {
        if self.widget.event(size, context, event)? { Ok(true) }
        else {
            match event {
                &Key(key) => { for action in self.actions { if action.0 == key && {(action.1)(self); true} { return Ok(true); } } Ok(false)},
                _=> Ok(false)
            }
        }
    }
}

pub struct Idle<'t, A> {
    pub app: A,
    pub idle: &'t mut dyn FnMut(&mut A) -> Result<bool>,
}
impl<A:Widget> Widget for Idle<'_, A> {
    fn paint(&mut self, target: &mut Target, size: size, offset: int2) -> Result { self.app.paint(target, size, offset) }
    fn event(&mut self, size: size, context: &mut Option<EventContext>, event: &ui::Event) -> Result<bool> {
        if self.app.event(size, context, event)? { Ok(true) }
        else {
            match event {
                ui::Event::Idle => (self.idle)(&mut self.app),
                _=> Ok(false)
            }
        }
    }
}

//use image::Image;
pub fn write_avif(path: impl AsRef<std::path::Path>, image: Image<Box<[u32]>>) {
    #[cfg(not(feature="avif"))] println!("Built without AVIF support: {} {}", path.as_ref().display(), image.size);
    #[cfg(feature="avif")] {
        use ravif::*;
        let EncodedImage { avif_file, .. } = Encoder::new().
            //encode_raw_planes_10_bit(image.size.x as usize, image.size.y as usize, image.iter().map(|rgb| [(rgb&((1<<10)-1)) as u16, ((rgb>>10)&((1<<10)-1)) as u16, (rgb>>20) as u16]),
            encode_raw_planes_8_bit(image.size.x as usize, image.size.y as usize, image.iter().map(|rgb| [(rgb&0xFF) as u8, ((rgb>>10)&0xFF) as u8, (rgb>>20) as u8]),
            None::<[_; 0]>, rav1e::color::PixelRange::Full, MatrixCoefficients::Identity).unwrap(); // FIXME: PQ
        std::fs::write(path, avif_file).unwrap();
    }
}

//use ui::Widget;
pub fn write_image(path: impl AsRef<std::path::Path>, view: &mut impl Widget) {
    let mut target = Image::fill(xy{x: 3840, y: 2400}, background().into());
    let size = target.size;
    view.event(size, &mut None, &ui::Event::Stale).unwrap();
    view.paint(&mut target.as_mut(), size, 0.into()).unwrap();
    write_avif(path, target);
}
