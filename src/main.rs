#![feature(thread_spawn_unchecked)]
fn main() -> ui::Result {
    #![allow(non_camel_case_types,non_snake_case)]
    fn sq(x: f32) -> f32 { x*x }

    struct Material {
        id: u8,
        absorption: f32, // μa [mm¯¹]
        scattering: f32, // μs [mm¯¹]
        anisotropy: f32, // g (mean cosine of the deflection angle) [Henyey-Greenstein]
        #[allow(dead_code)] refractive_index: f32 // n
    }
    let air = Material{id: 0, absorption: 0., scattering: 0., anisotropy: 1., refractive_index: 1.};
    let bone = Material{id: 1, absorption: 0.019, scattering: 7.800, anisotropy: 0.89, refractive_index: 1.37};
    let cerebrospinal_fluid = Material{id: 2, absorption: 0.004, scattering: 0.009, anisotropy: 0.89, refractive_index: 1.37};
    let gray_matter = Material{id: 3, absorption: 0.020, scattering: 9.00, anisotropy: 0.89, refractive_index: 1.37};
    let white_matter = Material{id: 4, absorption: 0.080, scattering: 40.9, anisotropy: 0.84, refractive_index: 1.37};

    use vector::{xyz, vec3};
    pub type uint3 = xyz<u32>;
    pub type size = uint3;
    pub struct Volume<D> {
        pub data : D,
        pub size : size,
    }
    impl<D> Volume<D> {
        pub fn index(&self, xyz{x,y,z}: uint3) -> usize { assert!(x < self.size.x && y < self.size.y && z < self.size.z); (((z * self.size.y + y) * self.size.x) + x) as usize }
        pub fn new<T>(size : size, data: D) -> Self where D:AsRef<[T]> { assert_eq!(data.as_ref().len(), (size.z*size.y*size.x) as usize); Self{data, size} }
    }

    impl<T, D:std::ops::Deref<Target=[T]>> std::ops::Index<usize> for Volume<D> {
        type Output=T;
        fn index(&self, i:usize) -> &Self::Output { &self.data[i] }
    }
    impl<D> std::ops::Index<uint3> for Volume<D> where Self: std::ops::Index<usize> {
        type Output = <Self as std::ops::Index<usize>>::Output;
        fn index(&self, i:uint3) -> &Self::Output { &self[self.index(i)] }
    }

    impl<T> Volume<Box<[T]>> {
        pub fn from_iter<I:IntoIterator<Item=T>>(size : size, iter : I) -> Self { Self::new(size, iter.into_iter().take((size.z*size.y*size.x) as usize).collect()) }
    }
    let size = xyz{x: 513, y: 513, z: 512};
    let z = |std::ops::Range{start, end}| (start*size.z/12)*size.y*size.x..(end*size.z/12)*size.y*size.x;
    let volume = Volume::from_iter(size,
                   z(0..5).map(|_| bone.id)
        .chain(z(5..6).map(|_| cerebrospinal_fluid.id))
        .chain(z(6..7).map(|_| gray_matter.id))
        .chain(z(7..12).map(|_| white_matter.id))
    );

    #[derive(Clone,Copy)] struct Ray { position: vec3, direction: vec3 }
    trait Source { fn sample(&self) -> Ray; }

    struct Pencil {
        position: vec3,
        direction: vec3,
    }
    impl Source for Pencil {
        fn sample(&self) -> Ray {let &Self{position, direction}=self; Ray{position, direction}}
    }

    let source = Pencil{position: xyz{x: size.x as f32/2., y: size.y as f32/2., z: 0.5}, direction: xyz{x: 0., y: 0., z: 1.}};
    //let wavelength = 650e-9;

    let materials = [air, bone, cerebrospinal_fluid, gray_matter, white_matter];
    for (table_index, material) in materials.iter().enumerate() { assert!(material.id == table_index as u8); }

    use vector::xy;
    use std::sync::atomic::{AtomicU16, Ordering::Relaxed};
    struct View(image::Image<Box<[AtomicU16]>>);
    impl ui::Widget for View { #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) {
        let ref source = self.0;
        let max = source.iter().map(|v| v.load(Relaxed)).max().unwrap() as u32;
        if max == 0 { return; }
        for y in 0..target.size.y {
            for x in 0..target.size.x {
                let w = (source[xy{x: x*source.size.x/target.size.x, y: y*source.size.y/target.size.y}].load(Relaxed) as u32) * ((1<<10)-1) / max;
                target[xy{x,y}] = w | w<<10 | w<<20;
            }
        }
    } }

    use {rand_xoshiro::rand_core::SeedableRng, rand::Rng};
    let mut rng = rand_xoshiro::Xoshiro128Plus::seed_from_u64(0);
    ui::run(&mut View(image::Image::zero(volume.size.yz())), &mut |View(ref mut image):&mut View| -> ui::Result<bool> {
        const N : usize = 8192;
        const workers : usize = 8;
        let task = |mut rng : rand_xoshiro::Xoshiro128Plus|{
            for _ in 0..N/workers {
                let Ray{mut position, mut direction} = source.sample();
                //let R = 1.; // m
                //let particle_diameter = 100e-9; // collagen
                //let rayleigh = f32::powi(PI,4)/4 * 1/(R*R) * f32::powi(diameter,6)/f32::pow(wavelength,4) ((n2-1)/(n2+2))^2
                //let scatter = rayleigh * (1.+cos2)/2.;
                //let polarizability = (n2-1)/(n2+2) * particle_radius; // Clausius-Mossotti (Lorentz-Lorenz)
                //let scattering_cross_section = f32::powi(2.,7)*f32::powi(PI,5)/(3.*f32::powi(wavelength,4))*polarizability //Cs
                //scattering_coefficient = N(a) x Cs(a)
                // let number_density_of_particles //N
                //let scattering_coefficient = number_density_of_particles * scattering_cross_section; // homogeneous (uniform a)
                //let mut radiance = 1.;
                loop {
                    {let xyz{x,y,z}=position; if x < 0. || x >= volume.size.x as f32 || y < 0. || y >= volume.size.y as f32 || z < 0. || z >= volume.size.z as f32 { break; }}
                    let id = volume[{let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}}];
                    let Material{absorption,scattering,anisotropy: g,..} = materials[id as usize];
                    let length = 1./(volume.size.z as f32);
                    if rng.gen::<f32>() < absorption * length {
                        assert!(image[{let xyz{y,z,..}=position; xy{x: y as u32, y: z as u32}}].fetch_add(1, Relaxed) != 0xFFFF);
                        break;
                    } // Absorption
                    //radiance *= f32::exp(-absorption * length);
                    //let minimum_intensity = 0.1;
                    //if radiance < minimum_intensity { break; }
                    //let optical_length = scattering * length;
                    if rng.gen::<f32>() < scattering * length {
                        //let R = optical_length;
                        let ξ = rng.gen::<f32>();
                        let cosθ = -1./(2.*g)*(1.+g*g-sq((1.-g*g)/(1.+g-2.*g*ξ)));// Henyey-Greenstein: 1/(4pi)*(1-g*g)/pow(1+g*g-2*g*cos, 2./3.)
                        //let cosθ = {let u = -2.*f32::cbrt(2.*(ξ1-1.)+f32::sqrt(4.*sq(2.*ξ-1.)+1.)); u-1./u; // Rayleigh: 1/4pi*3/4*(1+cos²θ)
                        let sinθ = 1. - cosθ*cosθ;
                        use std::f32::consts::PI;
                        let φ = 2.*PI*rng.gen::<f32>();
                        pub fn cross(a: vec3, b: vec3) -> vec3 { xyz{x: a.y*b.z - a.z*b.y, y: a.z*b.x - a.x*b.z, z: a.x*b.y - a.y*b.x} }
                        fn tangent_space(n@xyz{x,y,z}: vec3) -> (vec3, vec3) { let t = if x > y { xyz{x: -z, y: 0., z: x} } else { xyz{x: 0., y: z, z: -y} }; (t, cross(n, t)) }
                        let (T, B) = tangent_space(direction);
                        direction = sinθ*f32::cos(φ)*T + sinθ*f32::sin(φ)*B + cosθ*direction;
                    }
                    position = position + /*length **/ direction;
                }
            }
        };
        let start = std::time::Instant::now();
        for thread in [();workers].map(|_| unsafe{std::thread::Builder::new().spawn_unchecked(|| { task(rng.clone()); rng.jump() }).unwrap()}) { thread.join().unwrap(); }
        let elapsed = start.elapsed();
        //println!("{} rays {}ms {}μs", N, elapsed.as_millis(), elapsed.as_micros()/(N as u128));
        Ok(true)
    })
}
