#![feature(slice_take)]
fn main() -> ui::Result {
    #![allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]
    fn sq(x: f32) -> f32 { x*x }

    trait atomic_from_mut<T> where Self:Sized { fn get_mut_slice(this: &mut [Self]) -> &mut [T] ; }
    impl atomic_from_mut<f32> for AtomicF32 { fn get_mut_slice(this: &mut [Self]) -> &mut [f32] { unsafe { &mut *(this as *mut [Self] as *mut [f32]) } } }

    use vector::{xyz, vec3};
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
        pub fn default(size: size) -> Self { Self::from_iter(size, std::iter::from_fn(|| Some(T::default()))) }
    }

    use {vector::xy, image::Image};
    fn rgb10(target: &mut Image<&mut [u32]>, source: Image<&[f32]>) {
        let max = source.iter().copied().reduce(f32::max).unwrap();
        if max == 0. { return; }
        for y in 0..target.size.y {
            for x in 0..target.size.x {
                let w = (source[xy{x: x*source.size.x/target.size.x, y: y*source.size.y/target.size.y}]/max * ((1<<10)-1) as f32) as u32;
                target[xy{x,y}] = w | w<<10 | w<<20;
            }
        }
    }
    struct View(Image<Box<[f32]>>);
    impl ui::Widget for View { #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) { rgb10(target, self.0.as_ref()) } }

    #[derive(PartialEq)] struct Material {
        density: f32, // ρ [kg/m³]
        // => refractive_index: f32 // n
        specific_heat_capacity: f32, // c [J/(kg·K)]
        thermal_conductivity: f32,  // k [W/(m·K)]
        absorption: f32, // μa [m¯¹]
        scattering: f32, // μs [m¯¹]
    }
    let anisotropy = 0.9; // g (mean cosine of the deflection angle) [Henyey-Greenstein]
    let scattering = |reduced_scattering_coefficient| reduced_scattering_coefficient / anisotropy;
    let ref glue = Material{
        density: 895.,
        specific_heat_capacity: 3353.5 /*+ 5.245 * T*/,
        thermal_conductivity: 0.3528 /*+ 0.001645 * T*/,
        absorption: 519. /*- 0.5 * T*/,
        scattering: scattering(1.),
    };
    let ref tissue = Material{
        density: 1030.,
        specific_heat_capacity: 3595.,
        thermal_conductivity: 0.49,
        absorption: 7.540 /*@750nm*/,
        scattering: scattering(9.99),
        //blood_volume_fraction: 0.0093,
        //hemoglobin_oxygen_saturation: 0.8,
        //water_volume_fraction: 0.5,
        //scattering_parameter_a:  3670,
        //scattering_parameter_b: 1.27,
    };
    // surface_emissivity: 0.95
    let material_list = [glue, tissue];
    let id = |material| material_list.iter().position(|&o| o == material).unwrap();

    let size = xyz{x: 513, y: 513, z: 512};

    let (height, material_volume) = if true {
        let glue_height = 0.2e-2;
        let tissue_height = 4e-2;
        let height = glue_height + tissue_height;
        let z = |start, end| (start*size.z as f32/height) as u32*size.y*size.x..(end*size.z as f32/height) as u32*size.y*size.x;
        let material_volume = Volume::from_iter(size, z(0., glue_height).map(|_| id(glue)).chain(z(glue_height, height).map(|_| id(tissue))));
        (height, material_volume)
    } else {
        let _height = 4e-2;
        panic!("cancer: 5mm diameter sphere, 1mm+r below surface, absorbance = 1")
    };

    let δx = height / material_volume.size.x as f32;

    let material_list = material_list.map(|&Material{density,specific_heat_capacity,thermal_conductivity,absorption,scattering}|
        Material{density, specific_heat_capacity, thermal_conductivity, absorption: absorption*δx, scattering: scattering*δx}); // TODO

    use {rand_xoshiro::rand_core::SeedableRng, rand::Rng};
    let mut rng = rand_xoshiro::Xoshiro128Plus::seed_from_u64(0);
    let mut next = move |mut temperature: Volume<&mut [AtomicF32]>, mut next_temperature: Volume<&mut [AtomicF32]>| -> Vec<String>{
        let mut report = Vec::new();
        // Light propagation
        const samples : usize = 8192;
        const threads : usize = 8;
        let task = |mut rng : rand_xoshiro::Xoshiro128Plus|{
            for _ in 0..samples/threads {
                let laser_position = xyz{x: size.x as f32/2., y: size.y as f32/2., z: 0.5};
                let laser_direction = xyz{x: 0., y: 0., z: 1.};
                let mut position = laser_position; // TODO: Spotsize diameter: 0.8 cm (circular, top-hat or gaussian)
                let mut direction = laser_direction; // TODO: low divergence
                //let wavelength = 750e-9;
                //let power = 1.5-2 W

                loop {
                    {let xyz{x,y,z}=position; if x < 0. || x >= material_volume.size.x as f32 || y < 0. || y >= material_volume.size.y as f32 || z < 0. || z >= material_volume.size.z as f32 { break; }}
                    let id = material_volume[{let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}}];
                    let Material{absorption,scattering,..} = material_list[id as usize];
                    if rng.gen::<f32>() < absorption {
                        temperature[{let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}}].fetch_add(1., Relaxed);
                        break;
                    } // Absorption
                    if rng.gen::<f32>() < scattering {
                        let ξ = rng.gen::<f32>();
                        let g = anisotropy;
                        let cosθ = -1./(2.*g)*(1.+g*g-sq((1.-g*g)/(1.+g-2.*g*ξ))); // Henyey-Greenstein
                        let sinθ = 1. - cosθ*cosθ;
                        use std::f32::consts::PI;
                        let φ = 2.*PI*rng.gen::<f32>();
                        pub fn cross(a: vec3, b: vec3) -> vec3 { xyz{x: a.y*b.z - a.z*b.y, y: a.z*b.x - a.x*b.z, z: a.x*b.y - a.y*b.x} }
                        fn tangent_space(n@xyz{x,y,z}: vec3) -> (vec3, vec3) { let t = if x > y { xyz{x: -z, y: 0., z: x} } else { xyz{x: 0., y: z, z: -y} }; (t, cross(n, t)) }
                        let (T, B) = tangent_space(direction);
                        direction = sinθ*f32::cos(φ)*T + sinθ*f32::sin(φ)*B + cosθ*direction;
                    }
                    position = position + direction;
                }
            }
        };
        let start = std::time::Instant::now();
        std::thread::scope(|s| for thread in [();threads].map(|_| {
            let task_rng = rng.clone();
            let thread = std::thread::Builder::new().spawn_scoped(s, move || task(task_rng)).unwrap();
            rng.jump();
            thread
        }) { thread.join().unwrap(); });
        let elapsed = start.elapsed(); report.push(format!("{} samples {}ms {}μs", samples, elapsed.as_millis(), elapsed.as_micros()/(samples as u128)));

        {// Heat diffusion
            let start = std::time::Instant::now();

            let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
            let mut next_temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut next_temperature.data));

            // Boundary conditions: constant temperature (Dirichlet): T_boundary=0
            let task = |z0, z1, mut next_temperature_chunk: Volume<&mut[f32]>| for z in z0..z1 { for y in 1..size.y-1 { for x in 1..size.x-1 {
                let specific_heat_capacity = 4000.; // c [J/(kg·K)]
                let mass_density = 1000.; // ρ [kg/m³]
                // dt(Q) = c ρ dt(T) : heat energy
                // dt(Q) = - dx(q): heat flow (positive outgoing)
                // => dt(T) = - 1/(cρ) dx(q)
                let thermal_conductivity = 0.5; // k [W/(m·K)]
                // q = -k∇T (Fourier conduction)
                // Finite difference cartesian first order laplacian
                let T = |dx,dy,dz| temperature[xyz{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32, z: (z as i32+dz) as u32}];
                let dxxT = ( T(-1, 0, 0) - 2. * T(0, 0, 0) + T(1, 0, 0) ) / sq(δx);
                let dyyT = ( T(0, -1, 0) - 2. * T(0, 0, 0) + T(0, 1, 0) ) / sq(δx);
                let dzzT = ( T(0, 0, -1) - 2. * T(0, 0, 0) + T(0, 0, 1) ) / sq(δx);
                let thermal_conduction = dxxT + dyyT + dzzT; // Cartesian: ΔT = dxx(T) + dyy(T) + dzz(T)
                let thermal_diffusivity = thermal_conductivity / (specific_heat_capacity * mass_density); // dt(T) = k/(cρ) ΔT = α ΔT (α≡k/(cρ))
                let dtT = thermal_diffusivity * thermal_conduction; // dt(T) = αΔT
                let δt = 0.1 / (thermal_diffusivity / sq(δx)); // Time step
                let C = thermal_diffusivity / sq(δx) * δt;
                assert!(C < 1./2., "{C}"); // Courant–Friedrichs–Lewy condition
                // Explicit time step (First order: Euler): T[t+1]  = T(t) + δt·dt(T)
                next_temperature_chunk[xyz{x, y, z: z-z0}] = T(0,0,0) + δt * dtT;
            }}};
            let range = 1..size.z-1;
            next_temperature.take_mut(range.start);
            std::thread::scope(|s| for thread in std::array::from_fn::<_, threads, _>(move |thread| {
                let z0 = range.start + (range.end-range.start)*thread as u32/threads as u32;
                let z1 = range.start + (range.end-range.start)*(thread as u32+1)/threads as u32;
                let next_temperature_chunk = next_temperature.take_mut(z1-z0);
                std::thread::Builder::new().spawn_scoped(s, move || task(z0, z1, next_temperature_chunk)).unwrap()
            }) { thread.join().unwrap(); });
            let points = size.z*size.y*size.x;
            let elapsed = start.elapsed(); report.push(format!("{}M points {}ms {}ns", points/1000000, elapsed.as_millis(), elapsed.as_nanos()/(points as u128)));
        }
        report
    };

    use {atomic_float::AtomicF32, std::sync::atomic::Ordering::Relaxed};
    let mut temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);
    let mut next_temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);

    let mut step = 0;
    ui::run(&mut View(Image::zero(temperature.size.yz())), &mut |View(ref mut image):&mut View| -> ui::Result<bool> {
        let report = next(temperature.as_mut(), next_temperature.as_mut());
        std::mem::swap(&mut temperature, &mut next_temperature);

        use itertools::Itertools;
        println!("{step} {}", report.iter().format(" "));
        step += 1;

        let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
        for image_y in 0..image.size.y { for image_x in 0..image.size.x {
            image[xy{x: image_x, y: image_y}] = (0..temperature.size.x).map(|volume_x| temperature[xyz{x: volume_x, y: image_x, z: image_y}]).sum::<f32>();
        }}

        #[cfg(feature="avif")] if step%32 == 0 {
            let mut target = Image::zero(image.size);
            rgb10(&mut target.as_mut(), image.as_ref());
            use ravif::*;
            let EncodedImage { avif_file, .. } = Encoder::new().encode_raw_planes_10_bit(target.size.x as usize, target.size.y as usize, target.iter().map(|rgb| [(rgb&0b1111111111) as u16, ((rgb>>10)&0b1111111111) as u16, (rgb>>20) as u16]), None::<[_; 0]>, rav1e::color::PixelRange::Full, MatrixCoefficients::Identity)?;
            std::fs::write(format!("{step}.avif"), avif_file)?;
        }

        Ok(true)
    })
}
