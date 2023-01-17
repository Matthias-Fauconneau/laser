#![feature(slice_take, macro_metavar_expr, atomic_from_mut)]
mod volume;
mod view;
fn main() -> ui::Result {
    #![allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]
    use {num::sq, vector::{xy, xyz, vec3}, std::sync::atomic::{AtomicU16, Ordering::Relaxed}, atomic_float::AtomicF32, ui::plot::list, volume::{Volume, atomic_from_mut as _}, rand::{Rng as Random, SeedableRng as SeedableRandom}, rand_xoshiro::Xoshiro128Plus as ParallelRandom};
    const threads : usize = 8;

    //let size = xyz{x: 512, y: 512, z: 513};
    //let size = xyz{x: 513, y: 513, z: 257};
    let size = xyz{x: 257, y: 257, z: 129};

    #[derive(PartialEq,Clone)] struct Material {
        density: f32, // ρ [kg/m³]
        specific_heat_capacity: f32, // c [J/(kg·K)]
        thermal_conductivity: f32,  // k [W/(m·K)]
        absorbance: f32, // μa [m¯¹]
        scattering: f32, // μs [m¯¹]
    }
    const anisotropy : f32 = 0.9; // g (mean cosine of the deflection angle) [Henyey-Greenstein]
    let scattering = |reduced_scattering_coefficient| reduced_scattering_coefficient / anisotropy;
    let ref tissue = Material{
        density: 1030.,
        specific_heat_capacity: 3595.,
        thermal_conductivity: 0.49,
        absorbance: 7.540, scattering: scattering(999.), //@750nm
    };
    let ref cancer = Material{absorbance: 10., ..tissue.clone()};
    let T = 273.15 + 36.85; // K
    let ref glue = Material{
        density: 895.,
        specific_heat_capacity: 3353.5 + 5.245 * T,
        thermal_conductivity: 0.3528 + 0.001645 * T,
        absorbance: 15.84/*519. - 0.5 * T*/, scattering: scattering(1.),
    };
    let material_list = [tissue, cancer, glue];

    let id = |material| material_list.iter().position(|&o| o == material).unwrap() as u8;
    let (height, material_volume) = match "tissue" {
        "tissue" => {
            let height = 2e-2;
            let z = |start, end| (start*size.z as f32/height) as u32*size.y*size.x..(end*size.z as f32/height) as u32*size.y*size.x;
            let material_volume = Volume::from_iter(size, z(0., height).map(|_| id(tissue)));
            (height, material_volume)
        },
            "glue" => {
            let glue_height = 0.2e-2;
            let tissue_height = 2e-2;
            let height = glue_height + tissue_height;
            let z = |start, end| (start*size.z as f32/height) as u32*size.y*size.x..(end*size.z as f32/height) as u32*size.y*size.x;
            let material_volume = Volume::from_iter(size, z(0., glue_height).map(|_| id(glue)).chain(z(glue_height, height).map(|_| id(tissue))));
            (height, material_volume)
        }
        "cancer" => {
            let height = 4e-2;
            let mut material_volume = Volume::from_iter(size, std::iter::from_fn(|| Some(id(tissue))));
            let diameter = 5e-3;
            let center = diameter/2. + 1e-3;
            for z in ((center-diameter/2.) * size.z as f32 / height) as u32 ..= ((center+diameter/2.) * size.z as f32 / height) as u32 {
                for y in size.y/2-(diameter/2. * size.z as f32 / height) as u32 ..= size.y/2+(diameter/2. * size.z as f32 / height) as u32 {
                    for x in size.x/2-(diameter/2. * size.z as f32 / height) as u32 ..= size.x/2+(diameter/2. * size.z as f32 / height) as u32 {
                        let p = xyz{x: x as f32,y: y as f32, z: z as f32};
                        if vector::sq(p - xyz{x: size.x as f32 / 2., y: size.y as f32 / 2., z: center * size.z as f32 / height}) < sq(diameter/2. * size.z as f32 / height) {
                            material_volume[xyz{x,y,z}] = id(cancer);
                        }
                    }
                }
            }
            (height, material_volume)
        }
        _ => unreachable!()
    };

    let δx = height / material_volume.size.z as f32;

    let specific_heat_capacity = 4000.; // c [J/(kg·K)]
    let mass_density = 1000.; // ρ [kg/m³]
    let thermal_conductivity = 0.5; // k [W/(m·K)]
    let thermal_diffusivity = thermal_conductivity / (specific_heat_capacity * mass_density); // dt(T) = k/(cρ) ΔT = α ΔT (α≡k/(cρ))
    let δt = 0.1 / (thermal_diffusivity / sq(δx)); // Time step
    let C = thermal_diffusivity / sq(δx) * δt;
    assert!(C <= 1./2., "{C}"); // Courant–Friedrichs–Lewy condition

    let ref material_list = material_list.map(|&Material{density,specific_heat_capacity,thermal_conductivity,absorbance,scattering}|
        Material{density, specific_heat_capacity, thermal_conductivity, absorbance: absorbance*δx, scattering: scattering*δx});

    struct Laser {
        diameter: f32,
        position: vec3,
        direction: vec3,
    }
    impl Laser {
        fn sample(&self, ref mut random : &mut impl Random) -> (vec3, vec3) {
            let diameter = self.diameter;
            let position = self.position + {
                let xy{x,y} = if true { // Approximate Airy disc using gaussian
                    xy{x: diameter/2. * random.sample::<f32,_>(rand_distributions::StandardNormal), y: diameter/2. * random.sample::<f32,_>(rand_distributions::StandardNormal)}
                } else {
                    use rand::distributions::{Distribution, Uniform};
                    let square = Uniform::new_inclusive(-diameter/2., diameter/2.);
                    loop { let p = xy{x: Distribution::sample(&square, random), y: Distribution::sample(&square, random)}; if vector::sq(p) <= sq(diameter/2.) { break p; } }
                };
                xyz{x,y, z: 0.}
            };
            let direction = self.direction; // TODO: divergence
            //let wavelength = 750e-9;
            //let power = 1.5-2 W
            //let peak_intensity = power * pi*sq(diameter)/4 / (sq(wavelength)*sq(focal_length)) // W/m²
            (position, direction)
        }
    }

    let ref laser = Laser{
        diameter: 0.8e-2 / δx,
        position: xyz{x: size.x as f32/2., y: size.y as f32/2., z: 0.5},
        direction: xyz{x: 0., y: 0., z: 1.}
    };

    fn light_propagation(ref mut random : &mut ParallelRandom, laser: &Laser, (material_list, material_volume): (&[Material], &Volume<&[u8]>),
                                            absorption: Option<&Volume<&mut [AtomicU16]>>,
                                            temperature: &Volume<&mut [AtomicF32]>) -> String {
        const samples : usize = 8192*4;
        fn task(ref mut random : impl Random, laser: &Laser, (material_list, material_volume): (&[Material], &Volume<&[u8]>), /*mut*/ temperature: &Volume<&mut [AtomicF32]>,
                        /*mut*/ absorption: Option<&Volume<&mut [AtomicU16]>>) {
            for _ in 0..samples/threads {
                let (mut position, mut direction) = laser.sample(random);

                loop {
                    let xyz{x,y,z} = position;
                    if x < 0. || x >= material_volume.size.x as f32 || y < 0. || y >= material_volume.size.y as f32 || z < 0. || z >= material_volume.size.z as f32 { break; }

                    let id = material_volume[{let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}}];
                    let Material{absorbance,scattering,..} = material_list[id as usize];

                    // Absorption
                    if random.gen::<f32>() < absorbance {
                        let index = {let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}};
                        if let Some(ref absorption) = absorption { absorption[index].fetch_add(1, Relaxed); }
                        temperature[index].fetch_add(1., Relaxed); // TODO: {specific_heat_capacity , laser_power} => delta_T_scale
                        break;
                    }

                    // Scattering
                    if random.gen::<f32>() < scattering {
                        let ξ = random.gen::<f32>();
                        let g = anisotropy;
                        let cosθ = -1./(2.*g)*(1.+g*g-sq((1.-g*g)/(1.+g-2.*g*ξ))); // Henyey-Greenstein
                        let sinθ = f32::sqrt(1. - cosθ*cosθ);
                        use std::f32::consts::PI;
                        let φ = 2.*PI*random.gen::<f32>();
                        let (T, B) = vector::tangent_space(direction);
                        let next_direction = sinθ*(f32::cos(φ)*T + f32::sin(φ)*B) + cosθ*direction;
                        direction = vector::normalize(next_direction);
                    }

                    position = position + direction;
                }
            }
        }
        let start = std::time::Instant::now();
        std::thread::scope(|s| for thread in [();threads].map(|_| {
            let task_random = random.clone();
            let thread = std::thread::Builder::new().spawn_scoped(s, || task(task_random, laser, (material_list, material_volume), temperature, absorption)).unwrap();
            random.jump();
            thread
        }) { thread.join().unwrap(); });
        let elapsed = start.elapsed();
        format!("{} samples {}ms {}μs", samples, elapsed.as_millis(), elapsed.as_micros()/(samples as u128))
    }

    fn next(ref mut random : &mut ParallelRandom, laser: &Laser, (material_list, ref material_volume): (&[Material], Volume<&[u8]>),
                    absorption: Option<Volume<&mut [AtomicU16]>>, ref mut temperature: Volume<&mut [AtomicF32]>, mut next_temperature: Volume<&mut [AtomicF32]>, C: f32) -> Vec<String> {
        let mut report = Vec::new();
        report.push( light_propagation(random, laser, (material_list, material_volume), absorption.as_ref(), temperature) );
        // Heat diffusion
        let start = std::time::Instant::now();
        let size = temperature.size;
        let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
        let mut next_temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut next_temperature.data));
        { // Boundary conditions: constant temperature (Dirichlet): T_boundary=0 except top: adiabatic dz(T)_boundary=0 (Neumann) using ghost points
            let task = |z0, z1, mut next_temperature_chunk: Volume<&mut[f32]>| for z in z0..z1 { for y in 1..size.y-1 { for x in 1..size.x-1 {
                // dt(Q) = c ρ dt(T) : heat energy
                // dt(Q) = - dx(q): heat flow (positive outgoing)
                // => dt(T) = - 1/(cρ) dx(q)
                // q = -k∇T (Fourier conduction)
                // Finite difference cartesian first order laplacian
                let T = |dx,dy,dz| temperature[xyz{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32, z: (z as i32+dz) as u32}];
                let dxxT = T(-1, 0, 0) - 2. * T(0, 0, 0) + T(1, 0, 0);
                let dyyT = T(0, -1, 0) - 2. * T(0, 0, 0) + T(0, 1, 0);
                let dzzT = T(0, 0, -1) - 2. * T(0, 0, 0) + T(0, 0, 1);
                let thermal_conduction = dxxT + dyyT + dzzT; // Cartesian: ΔT = dxx(T) + dyy(T) + dzz(T)
                // Explicit time step (First order: Euler): T[t+1]  = T(t) + δt·dt(T)
                next_temperature_chunk[xyz{x, y, z: z-z0}] = T(0,0,0) + C * thermal_conduction; // dt(T) = αΔT
            }}};
            let mut next_temperature = next_temperature.as_mut();
            let range = 1..size.z-1;
            next_temperature.take_mut(range.start);
            std::thread::scope(|s| for thread in std::array::from_fn::<_, threads, _>(move |thread| {
                let z0 = range.start + (range.end-range.start)*thread as u32/threads as u32;
                let z1 = range.start + (range.end-range.start)*(thread as u32+1)/threads as u32;
                let next_temperature_chunk = next_temperature.take_mut(z1-z0);
                std::thread::Builder::new().spawn_scoped(s, move || task(z0, z1, next_temperature_chunk)).unwrap()
            }) { thread.join().unwrap(); });
        }
        for y in 0..size.y { for x in 0..size.x { next_temperature[xyz{x, y, z: 0}] = next_temperature[xyz{x, y, z: 1}]; } } // Sets ghost points to temperature of below point
        let points = size.z*size.y*size.x;
        let elapsed = start.elapsed(); report.push(format!("{}M points {}ms {}ns", points/1000000, elapsed.as_millis(), elapsed.as_nanos()/(points as u128)));
        report
    }

    let mut step = 0;
    let ref mut random = rand_xoshiro::Xoshiro128Plus::seed_from_u64(0);

    let mut absorption : Volume<Box<[AtomicU16]>> = Volume::default(size);
    let mut temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);
    let mut next_temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);

    use {image::Image, view::{Plot, ImageView, Grid, write_image}};

    let Iz_radius = (1e-3 / δx) as u32;
    let Tt_z = [1, 8, 16];
    let mut I0 = list(std::iter::repeat(0.).take((size.x/2-1) as usize));

    let mm = δx*1e3;
    let Tt = Plot{keys: Box::from(Tt_z.map(|z| format!("{}mm", f32::round(mm*(z as f32)) as u32))), values: Box::from(Tt_z.map(|_| Vec::new())), x_scale: δt};
    let Iz = Plot{keys: Box::from(["I(z)".to_owned()]), values: Box::from([Vec::new()]), x_scale: mm};
    let Ir = Plot{keys: Box::from(["I(r)".to_owned(),"I0(r)".to_owned()]), values: Box::from([Vec::new(),Vec::new()]), x_scale: mm};
    let Tyz = ImageView(Image::zero(xy{x: temperature.size.y, y: temperature.size.z-1}));

    derive_IntoIterator! { struct Plots { Tt: Plot, Iz: Plot, Ir: Plot, Tyz: ImageView } }
    ui::run(&mut Grid(Plots{Tt, Iz, Ir, Tyz}), &mut move |grid: &mut _| -> ui::Result<bool> {
        let _report = next(random, laser, (material_list, material_volume.as_ref()), Some(absorption.as_mut()), temperature.as_mut(), next_temperature.as_mut(), C);
        std::mem::swap(&mut temperature, &mut next_temperature);
        //use itertools::Itertools; println!("{step} {}s {}", step as f32*δt, report.iter().format(" "));
        step += 1;

        // T(t) at z={probes}
        let Grid(Plots{Tt, Iz, Ir, Tyz}) = grid;
        let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
        for (i, &z) in Tt_z.iter().enumerate() {
            Tt.values[i].push( temperature[xyz{x: size.x/2, y: size.y/2, z}] );
        }

        // axial: I(z)
        let absorption = Volume::new(size, AtomicU16::get_mut_slice(&mut absorption.data));
        Iz.values[0] = (0..size.z).map(|z| {
            let radius = Iz_radius;
            let mut sum = 0;
            for y in -(radius as i32) ..= radius as i32 {
                for x in -(radius as i32) ..= radius as i32 {
                    if vector::sq(xy{x,y}) as u32 <= sq(radius) { sum += absorption[xyz{x: ((size.x/2) as i32+x) as u32, y: ((size.y/2) as i32+y) as u32, z}]; }
                }
            }
            sum as f32
        }).collect();

        // radial: I(r)
        Ir.values[0] = (0..size.x/2-1).map(|radius| {
            let z = (2e-3 / δx) as u32;
            let mut sum = 0.;
            let mut count = 0;
            for y in -((radius+1) as i32) ..= (radius+1) as i32 {
                for x in -((radius+1) as i32)..= (radius+1) as i32 {
                    let r2 = vector::sq(xy{x,y}) as u32;
                    if sq(radius) <= r2 && r2 < sq(radius+1) {
                        sum += temperature[xyz{x: ((size.x/2) as i32+x) as u32, y: ((size.y/2) as i32+y) as u32, z}];
                        count += 1;
                    }
                }
            }
            sum / count as f32 // ~r2
        }).collect();
        for position in std::iter::repeat_with(|| { let (position, _) = laser.sample(random); position }).take((size.x*size.y) as usize) {
            let xyz{x,y,..} = position;
            let xy{x,y} = xy{x: x-(size.x/2) as f32, y: y-(size.y/2) as f32};
            let r2 = vector::sq(xy{x,y}) ;
            if r2 < sq(size.x/2-1) as f32 {
                I0[f32::sqrt(r2) as usize] += 1. / f32::sqrt(r2 as f32);
            }
        }
        let norm_Ir = Ir.values[0].iter().sum::<f32>();
        let norm_I0 = I0.iter().sum::<f32>();
        Ir.values[1] = list(I0.iter().map(|&I| I * norm_Ir / norm_I0)).into();

        // T(y,z) (x sum)
        let ref mut Tyz = Tyz.0;
        for image_y in 0..Tyz.size.y { for image_x in 0..Tyz.size.x {
            Tyz[xy{x: image_x, y: image_y}] = (0..size.x).map(|volume_x| temperature[xyz{x: volume_x, y: image_x, z: 1+image_y}]).sum::<f32>();
        }}

        let stop = 64;
        if [stop].contains(&step) {
            let path = format!("out/a={a},s={s},d={d},t={step}", a=f32::round(tissue.absorbance) as u32, s=f32::round(tissue.scattering) as u32, d=laser.diameter);
            std::fs::write(&path, format!("Tt: {Tt:?}\nIz: {Iz:?}\nIr: {Ir:?}\nTyz: ({Tyz:?}, {:?})", Tyz.data)).unwrap();
            write_image(path+".avif", grid);
        }
        Ok(step <= stop)
    })
}