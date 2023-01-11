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
pub struct View(pub Image<Box<[f32]>>);
impl ui::Widget for View { #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) { rgb10(target, self.0.as_ref()) } }
