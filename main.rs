#![feature(slice_take, macro_metavar_expr, atomic_from_mut)]#![allow(confusable_idents)]
#[allow(non_snake_case)] mod SI; use SI::*;
#[derive(PartialEq,Clone)] struct Material<S: System> {
    mass_density: MassDensity, // ρ [kg/m³]
    specific_heat_capacity: SpecificHeatCapacity, // c [J/(kg·K)]
    thermal_conductivity: ThermalConductivity,  // k [W/(m·K)]
    absorption_coefficient: S::Scalar<ByLength>, // μa [m¯¹]
    scattering_coefficient: S::Scalar<ByLength>, // μs [m¯¹]
}

mod volume;
mod view;
fn main() -> ui::Result {
    #![allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]
    use {num::sq, vector::{xy, xyz, vec3}, std::sync::atomic::{AtomicU16, Ordering::Relaxed}, atomic_float::{/*AtomicF32,*/AtomicF64}, ui::plot::list, volume::{atomic_from_mut as _, size, Volume}};
    use {rand::{Rng as Random, SeedableRng as SeedableRandom}, rand_xoshiro::Xoshiro128Plus as ParallelRandom};
    const threads : usize = 8;

    //let size = xyz{x: 512, y: 512, z: 513};
    //let size = xyz{x: 513, y: 513, z: 257};
    let size = xyz{x: 257, y: 257, z: 129};

    const anisotropy : f32 = 0.9; // g (mean cosine of the deflection angle) [Henyey-Greenstein]
    let scattering = |reduced_scattering_coefficient| reduced_scattering_coefficient / (anisotropy as f64);
    type DMaterial = self::Material<Dimensionalized>;
    let ref tissue = DMaterial{
        mass_density: 1030.|kg_m3,
        specific_heat_capacity: 3595.|J_K·kg,
        thermal_conductivity: 0.49|W_m·K,
        absorption_coefficient: 7.54|_m, scattering_coefficient: scattering(999.|_m), //@750nm
    };
    let ref cancer = DMaterial{absorption_coefficient: 10. |_m, ..tissue.clone()};
    let T = (273.15 + 36.85)|K;
    let ref glue = DMaterial{
        mass_density: 895.|kg_m3,
        specific_heat_capacity: (3353.5|J_K·kg) + (5.245|J_K2·kg) * T, // <~5000
        thermal_conductivity: (0.3528|W_m·K) + (0.001645|W_m·K2) * T, // <~1
        absorption_coefficient: 15.84|_m/*519.|_m - 0.5 |_Km * T*/, scattering_coefficient: scattering(1.|_m),
    };
    let material_list = [tissue, cancer, glue];

    let id = |material| material_list.iter().position(|&o| o == material).unwrap() as u8;
    fn map(size: size, height: Length, x: Length) -> u32 { (f64::from(x/height)*(size.z as f64)) as u32 }
    fn z(size: size, height: Length, start: Length, end: Length) -> std::ops::Range<u32> { map(size, height, start)*size.y*size.x .. map(size, height, end)*size.y*size.x }
    let (height, material_volume) : (Length,_) = match "tissue" {
        "tissue" => {
            let height = 2.|cm;
            let material_volume = Volume::from_iter(size, z(size, height, 0.|m, height).map(|_| id(tissue)));
            (height, material_volume)
        },
        "glue" => {
            let glue_height = 2.|mm;
            let tissue_height = 2.|cm;
            let height = glue_height + tissue_height;
            let z = |start, end| z(size, height, start, end);
            let material_volume = Volume::from_iter(size, z(0.|m, glue_height).map(|_| id(glue)).chain(z(glue_height, height).map(|_| id(tissue))));
            (height, material_volume)
        }
        "cancer" => {
            let height = 4.|cm;
            let mut material_volume = Volume::from_iter(size, std::iter::from_fn(|| Some(id(tissue))));
            let diameter = 5.|mm;
            let center_z = diameter/2. + (1e-3|m);
            let map = |x| map(size, height, x);
            let radius = diameter/2.;
            for z in map(center_z-radius) ..= map(center_z+radius) {
                for y in size.y/2-map(radius) ..= size.y/2+map(radius) {
                    for x in size.x/2-map(radius) ..= size.x/2+map(radius) {
                        let p = xyz{x: x as f32,y: y as f32, z: z as f32};
                        if vector::sq(p - xyz{x: size.x as f32 / 2., y: size.y as f32 / 2., z: map(center_z) as f32}) < sq(map(radius) as f32) {
                            material_volume[xyz{x,y,z}] = id(cancer);
                        }
                    }
                }
            }
            (height, material_volume)
        }
        _ => unreachable!()
    };

    let δx : Length = height / material_volume.size.z as f64;

    let mass_density = 1000. |kg_m3; // ρ
    let specific_heat_capacity = 4000. |J_K·kg; // c
    let volumetric_heat_capacity : VolumetricHeatCapacity = mass_density * specific_heat_capacity; // J/K·m³
    let thermal_conductivity = 0.5 |W_m·K; // k
    let thermal_diffusivity = thermal_conductivity / volumetric_heat_capacity; // dt(T) = k/(cρ) ΔT = α ΔT (α≡k/(cρ)) ~ 0.1 mm²/s
    let δt = 0.1 / (thermal_diffusivity / sq(δx)); // Time step ~ 20ms

    let C : Unitless = thermal_diffusivity / sq(δx) * δt;
    assert!(f64::from(C) <= 1./2., "{C}"); // Courant–Friedrichs–Lewy condition

    let ref material_list = material_list.map(|&DMaterial{mass_density, specific_heat_capacity, thermal_conductivity, absorption_coefficient, scattering_coefficient}| {
        Material{mass_density, specific_heat_capacity, thermal_conductivity, absorption_coefficient: (absorption_coefficient*δx).into(), scattering_coefficient: (scattering_coefficient*δx).into()}
    });
    type Material = self::Material<NonDimensionalized>;

    struct Laser {
        diameter: f32,
        position: vec3,
        direction: vec3,
        power: Power,
        //let wavelength = 750e-9;
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
            (position, direction)
        }
        fn sample_power(&self, samples_per_step: usize) -> Power { self.power / samples_per_step as f64 }
        //peak_intensity = power * pi*sq(diameter)/4 / (sq(wavelength)*sq(focal_length)) // W/m²
    }

    let ref laser = Laser{
        diameter: ((8.|mm) / δx).into(),
        position: xyz{x: size.x as f32/2., y: size.y as f32/2., z: 0.5},
        direction: xyz{x: 0., y: 0., z: 1.},
        power: 1.|W,
    };
    const laser_samples_per_step : usize = 8192*4; // if ! samples << voxels: atomic add u16 energy, and then have a temperature+float conversion pass might be more efficient than atomic float
    type float = f64;
    type AtomicF = AtomicF64;
    fn light_propagation(ref mut random : &mut ParallelRandom, (material_list, material_volume): (&[Material], &Volume<&[u8]>), δx: Length, δt: Time, laser: &Laser,
                                            absorption: Option<&Volume<&mut [AtomicU16]>>, temperature: &Volume<&mut [AtomicF]>) -> String {
        fn task(ref mut random : impl Random, (material_list, material_volume): (&[Material], &Volume<&[u8]>), δx: Length, δt: Time, laser: &Laser,
                        absorption: Option<&Volume<&mut [AtomicU16]>>, temperature: &Volume<&mut [AtomicF]>) {
            for _ in 0..laser_samples_per_step/threads {
                let (mut position, mut direction) = laser.sample(random);

                loop {
                    let xyz{x,y,z} = position;
                    if x < 0. || x >= material_volume.size.x as f32 || y < 0. || y >= material_volume.size.y as f32 || z < 0. || z >= material_volume.size.z as f32 { break; }

                    let id = material_volume[{let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}}];
                    let Material{mass_density: density, specific_heat_capacity, absorption_coefficient, scattering_coefficient,..} = material_list[id as usize];

                    // Absorption
                    if random.gen::<f32>() < absorption_coefficient /*·δx*/ {
                        let index = {let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}};
                        if let Some(ref absorption) = absorption { absorption[index].fetch_add(1, Relaxed); }
                        let volume = δx*δx*δx;
                        let heat_capacity : HeatCapacity = density * volume * specific_heat_capacity; // J/K
                        let δT : Temperature = δt * laser.sample_power(laser_samples_per_step) / heat_capacity;
                        temperature[index].fetch_add(δT.K(), Relaxed);
                        break;
                    }

                    // Scattering
                    if random.gen::<f32>() < scattering_coefficient /*·δx*/ {
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
            let thread = std::thread::Builder::new().spawn_scoped(s, || task(task_random, (material_list, material_volume), δx, δt, laser, absorption, temperature)).unwrap();
            random.jump();
            thread
        }) { thread.join().unwrap(); });
        let elapsed = start.elapsed();
        format!("{} samples {}ms {}μs", laser_samples_per_step, elapsed.as_millis(), elapsed.as_micros()/(laser_samples_per_step as u128))
    }

    fn next(ref mut random : &mut ParallelRandom, (material_list, ref material_volume): (&[Material], Volume<&[u8]>), δx: Length, δt: Time, laser: &Laser,
                    absorption: Option<Volume<&mut [AtomicU16]>>, ref mut temperature: Volume<&mut [AtomicF]>, mut next_temperature: Volume<&mut [AtomicF]>) -> Vec<String> {
        let mut report = Vec::new();
        report.push( light_propagation(random, (material_list, material_volume), δx, δt, laser, absorption.as_ref(), temperature) );
        // Heat diffusion
        let start = std::time::Instant::now();
        let size = temperature.size;
        let temperature = Volume::new(size, AtomicF::get_mut_slice(&mut temperature.data));
        let mut next_temperature = Volume::new(size, AtomicF::get_mut_slice(&mut next_temperature.data));
        { // Boundary conditions: constant temperature (Dirichlet): T_boundary=0 except top: adiabatic dz(T)_boundary=0 (Neumann) using ghost points
            let task = |z0, z1, mut next_temperature_chunk: Volume<&mut[float]>| for z in z0..z1 { for y in 1..size.y-1 { for x in 1..size.x-1 {
                let id = material_volume[xyz{x: x as u32, y: y as u32, z: z as u32}];
                let Material{mass_density: density, specific_heat_capacity,thermal_conductivity,..} = material_list[id as usize];
                let volumetric_heat_capacity : VolumetricHeatCapacity = density * specific_heat_capacity; // J/K·m³
                let thermal_diffusivity : ThermalDiffusivity = thermal_conductivity / volumetric_heat_capacity; // dt(T) = k/(cρ) ΔT = α ΔT (α≡k/(cρ)) [m²/s]
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
                let α = thermal_diffusivity / sq(δx) * δt;
                next_temperature_chunk[xyz{x, y, z: z-z0}] = T(0,0,0) + α * thermal_conduction; // dt(T) = αΔT
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
    let mut temperature : Volume<Box<[AtomicF]>> = Volume::default(size);
    let mut next_temperature : Volume<Box<[AtomicF]>> = Volume::default(size);

    use {image::Image, ui::Plot, view::{LabeledImage, Grid, write_image}};

    let Tt_z = [7, 14, 21];
    let Iz_radius = f32::from((1.|mm) / δx) as u32;
    let Ir_z = 2.|mm;

    //x_scale: δt.unwrap()
    let Tt = Plot::new("Temperature over time for probes on the axis", xy{x: "Time ($s)", y: "ΔTemperature ($K)"}, Box::from(Tt_z.map(|z| ((z as f64)*δx).to_string())));
    //x_scale: δx.unwrap(),
    let Iz = Plot::new("Laser intensity over depth (on the axis)",  xy{x: "Depth ($m)", y: "Intensity ($W/m²)"}, Box::from(["I(z)".to_string()]));
    //x_scale: δx.unwrap(),
    let Ir = Plot::new("Laser intensity at the surface (radial plot)", xy{x: "Radius ($m)", y: "Intensity ($W/m²)"}, Box::from([format!("I(r) at {Ir_z}"), "I0(r)".to_string()]));

    let Tyz = LabeledImage::new("Temperature over y,z (x projected by summing)", Image::zero(xy{x: temperature.size.y, y: temperature.size.z-1}));

    derive_IntoIterator! { pub struct Plots { pub Tt: Plot, pub Iz: Plot, pub Ir: Plot, pub Tyz: LabeledImage} }
    ui::run(&mut Grid(Plots{Tt, Iz, Ir, Tyz}), &mut move |grid: &mut _| -> ui::Result<bool> {
        let _report = next(random, (material_list, material_volume.as_ref()), δx, δt, laser, Some(absorption.as_mut()), temperature.as_mut(), next_temperature.as_mut());
        std::mem::swap(&mut temperature, &mut next_temperature);
        //use itertools::Itertools; println!("{step} {}s {}", step as f32*δt, report.iter().format(" "));
        step += 1;

        // T(t) at z={probes}
        let Grid(Plots{Tt, Iz, Ir, Tyz}) = grid;
        let temperature = Volume::new(size, AtomicF::get_mut_slice(&mut temperature.data));
        Tt.x_values.push( δt.unwrap() * step as f64 );
        for (i, &z) in Tt_z.iter().enumerate() {
            let p = xyz{x: size.x/2, y: size.y/2, z};
            Tt.sets[i].push( temperature[p] );
        }

        // axial: I(z)
        let absorption = Volume::new(size, AtomicU16::get_mut_slice(&mut absorption.data));
        Iz.x_values = list((0..size.z).map(|z| z as f64 * δx.unwrap())).into();
        Iz.sets[0] = (0..size.z).map(|z| {
            let radius = Iz_radius;
            let mut sum = 0;
            let mut count = 0;
            for y in -(radius as i32) ..= radius as i32 {
                for x in -(radius as i32) ..= radius as i32 {
                    if vector::sq(xy{x,y}) as u32 <= sq(radius) {
                        sum += absorption[xyz{x: ((size.x/2) as i32+x) as u32, y: ((size.y/2) as i32+y) as u32, z}];
                        count += 1;
                    }
                }
            }
            let power : Power = (sum as f64) * laser.sample_power(laser_samples_per_step);
            let area : Area = count as f64 * δx * δx;
            let intensity : Intensity = power / area;
            intensity.unwrap()
        }).collect();
        Iz.need_update();

        // radial: I(r)
        let Ir_z = f32::from(Ir_z / δx) as u32;
        Ir.x_values = list((0..size.x/2-1).map(|r| (r as f64) * δx.unwrap())).into();
        Ir.sets[0] = (0..size.x/2-1).map(|r| {
            let mut sum = 0;
            let mut count = 0;
            for y in -((r+1) as i32) ..= (r+1) as i32 {
                for x in -((r+1) as i32)..= (r+1) as i32 {
                    let r2 = vector::sq(xy{x,y}) as u32;
                    if sq(r) <= r2 && r2 < sq(r+1) {
                        sum += absorption[xyz{x: ((size.x/2) as i32+x) as u32, y: ((size.y/2) as i32+y) as u32, z: Ir_z}];
                        count += 1;
                    }
                }
            }
            let power : Power = (sum as f64) * laser.sample_power(laser_samples_per_step);
            let area : Area = count as f64 * δx * δx; // Ring
            let intensity : Intensity = power / area;
            intensity.unwrap()
        }).collect();
        let ref mut I0 = Ir.sets[1];
        if I0.is_empty() { *I0 = [0.].repeat((size.x/2-1) as usize); }
        for position in std::iter::repeat_with(|| { let (position, _) = laser.sample(random); position }).take(laser_samples_per_step) {
            let xyz{x,y,..} = position;
            let xy{x,y} = xy{x: x-(size.x/2) as f32, y: y-(size.y/2) as f32};
            let r = f64::sqrt(vector::sq(xy{x,y}) as f64);
            let power : Power = laser.sample_power(laser_samples_per_step);
            let area : Area = δx * r * δx; // Ring
            let intensity : Intensity = power / area;
            let r = r as usize;
            if r < I0.len() { I0[r] += intensity.unwrap(); }
        }
        /*let norm_Ir = Ir.sets[0].iter().sum::<f64>();
        let norm_I0 = I0.iter().sum::<f64>();
        Ir.sets[1] = list(I0.iter().map(|&I| I * norm_Ir / norm_I0)).into();*/
        Ir.need_update();

        // T(y,z) (x sum)
        let ref mut Tyz = Tyz.0.image.0;
        for image_y in 0..Tyz.size.y { for image_x in 0..Tyz.size.x {
            Tyz[xy{x: image_x, y: image_y}] = (0..size.x).map(|volume_x| temperature[xyz{x: volume_x, y: image_x, z: 1+image_y}]).sum::<float>() as f32;
        }}

        let stop = 64;
        if [stop].contains(&step) {
            let path = format!("out/a={a},s={s},d={d},t={step}", a=tissue.absorption_coefficient, s=tissue.scattering_coefficient, d=laser.diameter);
            std::fs::write(&path, format!("Tt: {Tt:?}\nIz: {Iz:?}\nIr: {Ir:?}\nTyz: ({Tyz:?}, {:?})", Tyz.data)).unwrap();
            write_image(path+".avif", grid);
        }
        Ok(step <= 128*10)
    })
}