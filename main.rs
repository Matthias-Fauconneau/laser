#![feature(slice_take)]
mod volume;
mod view;
fn main() -> ui::Result {
    #![allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]
    use {num::sq, vector::{xy, xyz}, volume::{Volume, atomic_from_mut as _}};

    #[derive(PartialEq,Clone)] struct Material {
        density: f32, // ρ [kg/m³]
        specific_heat_capacity: f32, // c [J/(kg·K)]
        thermal_conductivity: f32,  // k [W/(m·K)]
        absorption: f32, // μa [m¯¹]
        scattering: f32, // μs [m¯¹]
    }
    let anisotropy = 0.9; // g (mean cosine of the deflection angle) [Henyey-Greenstein]
    let scattering = |reduced_scattering_coefficient| reduced_scattering_coefficient / anisotropy;
    /*let T = 273.15 + 36.85; // K
    let ref glue = Material{
        density: 895.,
        specific_heat_capacity: 3353.5 + 5.245 * T,
        thermal_conductivity: 0.3528 + 0.001645 * T,
        absorption: 15.84/*519. - 0.5 * T*/, scattering: scattering(1.),
    };*/
    let ref tissue = Material{
        density: 1030.,
        specific_heat_capacity: 3595.,
        thermal_conductivity: 0.49,
        absorption: /*7.540*/0., scattering: scattering(100./*999.*/), //@750nm
    };
    //let ref cancer = Material{absorption: 10., ..tissue.clone()};
    //panic!("{} {} {}", glue.absorption, tissue.absorption, glue.absorption/tissue.absorption);
    //let material_list = [glue, tissue, cancer];
    let material_list = [tissue];
    let id = |material| material_list.iter().position(|&o| o == material).unwrap();

    //let size = xyz{x: 512, y: 512, z: 513};
    //let size = xyz{x: 256, y: 256, z: 257};
    let size = xyz{x: 257, y: 257, z: 129};

    let (height, material_volume) = {
        let height = 2e-2;
        let z = |start, end| (start*size.z as f32/height) as u32*size.y*size.x..(end*size.z as f32/height) as u32*size.y*size.x;
        let material_volume = Volume::from_iter(size, z(0., height).map(|_| id(tissue)));
        (height, material_volume)
    };
    /*let (height, material_volume) = if true {
        let glue_height = 0.2e-2;
        let tissue_height = 2e-2;
        let height = glue_height + tissue_height;
        let z = |start, end| (start*size.z as f32/height) as u32*size.y*size.x..(end*size.z as f32/height) as u32*size.y*size.x;
        let material_volume = Volume::from_iter(size, z(0., glue_height).map(|_| id(glue)).chain(z(glue_height, height).map(|_| id(tissue))));
        (height, material_volume)
    } else {
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
    };*/

    let δx = height / material_volume.size.z as f32;

    let specific_heat_capacity = 4000.; // c [J/(kg·K)]
    let mass_density = 1000.; // ρ [kg/m³]
    let thermal_conductivity = 0.5; // k [W/(m·K)]
    let thermal_diffusivity = thermal_conductivity / (specific_heat_capacity * mass_density); // dt(T) = k/(cρ) ΔT = α ΔT (α≡k/(cρ))
    let _δt = 0.1 / (thermal_diffusivity / sq(δx)); // Time step
                
    let material_list = material_list.map(|&Material{density,specific_heat_capacity,thermal_conductivity,absorption,scattering}|
        Material{density, specific_heat_capacity, thermal_conductivity, absorption: absorption*δx, scattering: scattering*δx}); // TODO

    const threads : usize = 1;

    use {rand_xoshiro::rand_core::SeedableRng, rand::Rng};
    let mut rng = rand_xoshiro::Xoshiro128Plus::seed_from_u64(0);
    let light_propagation = |ref mut rng : &mut rand_xoshiro::Xoshiro128Plus, /*mut*/ temperature: Volume<&mut [AtomicF32]>|{
        const samples : usize = 8; //8192
        let task = |ref mut rng : rand_xoshiro::Xoshiro128Plus|{
            for _ in 0..samples/threads {
                let laser_position = xyz{x: size.x as f32/2., y: size.y as f32/2., z: 0.5};
                let laser_direction = xyz{x: 0., y: 0., z: 1.};
                let diameter = 0.8e-2 / δx;
                let mut position = laser_position + {
                    let xy{x,y} = if true { // Approximate Airy disc using gaussian
                        xy{x: diameter/2. * rng.sample::<f32,_>(rand_distributions::StandardNormal), y: diameter/2. * rng.sample::<f32,_>(rand_distributions::StandardNormal)}
                    } else {
                        use rand::distributions::{Distribution, Uniform};
                        let square = Uniform::new_inclusive(-diameter/2., diameter/2.);
                        loop { let p = xy{x: Distribution::sample(&square, rng), y: Distribution::sample(&square, rng)}; if vector::sq(p) <= sq(diameter/2.) { break p; } }
                    };
                    xyz{x,y, z: 0.}
                };
                let mut direction = laser_direction; // TODO: divergence
                //let wavelength = 750e-9;
                //let power = 1.5-2 W
                //let peak_intensity = power * pi*sq(diameter)/4 / (sq(wavelength)*sq(focal_length)) // W/m²

                loop {
                    {let xyz{x,y,z}=position; if x < 0. || x >= material_volume.size.x as f32 || y < 0. || y >= material_volume.size.y as f32 || z < 0. || z >= material_volume.size.z as f32 { break; }}
                    let id = material_volume[{let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}}];
                    let Material{absorption,scattering,..} = material_list[id as usize];

                    if rng.gen::<f32>() < absorption {
                        temperature[{let xyz{x,y,z}=position; xyz{x: x as u32, y: y as u32, z: z as u32}}].fetch_add(1., Relaxed); // TODO: specific_heat_capacity
                        break;
                    } // Absorption
                    if rng.gen::<f32>() < scattering {
                        let ξ = rng.gen::<f32>();
                        let g = anisotropy;
                        let cosθ = -1./(2.*g)*(1.+g*g-sq((1.-g*g)/(1.+g-2.*g*ξ))); // Henyey-Greenstein
                        let sinθ = f32::sqrt(1. - cosθ*cosθ);
                        use std::f32::consts::PI;
                        let φ = 2.*PI*rng.gen::<f32>();
                        let (T, B) = vector::tangent_space(direction);
                        assert!(f32::abs(1.-vector::norm(direction))<2e-7, "{:e}", f32::abs(1.-vector::norm(direction)));
                        assert!(f32::abs(1.-vector::norm(T))<2e-7, "{:?} {:e}", T, 1.-vector::norm(T));
                        assert!(f32::abs(1.-vector::norm(B))<2e-7, "{:e} {:e} {:e}", f32::abs(1.-vector::norm(direction)), f32::abs(1.-vector::norm(T)), f32::abs(1.-vector::norm(B)));
                        assert!(f32::abs(1.-(sinθ*sinθ + cosθ*cosθ))<2e-7, "{:e} {} {}", 1.-(sinθ*sinθ + cosθ*cosθ), cosθ, sinθ);
                        assert!(f32::abs(1.-vector::norm(f32::cos(φ)*T + f32::sin(φ)*B))<2e-7, "{:e} {:e} {:e}", f32::abs(1.-vector::norm(f32::cos(φ)*T + f32::sin(φ)*B)), f32::abs(1.-vector::norm(T)), f32::abs(1.-vector::norm(B)), );
                        assert!(f32::abs(vector::dot(f32::cos(φ)*T + f32::sin(φ)*B, direction)) < 1e-7, "{:e}", vector::dot(f32::cos(φ)*T + f32::sin(φ)*B, direction));
                        let next_direction = sinθ*(f32::cos(φ)*T + f32::sin(φ)*B) + cosθ*direction;
                        assert!(f32::abs(1.-vector::norm(next_direction))<2e-7, "{:e} {:e}", f32::abs(1.-vector::norm(direction)), f32::abs(1.-vector::norm(next_direction)));
                        direction = vector::normalize(next_direction);
                        assert!(f32::abs(1.-vector::norm(direction))<2e-7, "{:e}", f32::abs(1.-vector::norm(direction)));
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
        let elapsed = start.elapsed(); 
        format!("{} samples {}ms {}μs", samples, elapsed.as_millis(), elapsed.as_micros()/(samples as u128))
    };

    /*let mut next = move |mut temperature: Volume<&mut [AtomicF32]>, mut next_temperature: Volume<&mut [AtomicF32]>| -> Vec<String>{
        let mut report = Vec::new();
        report.push(
            light_propagation(&mut rng, temperature.as_mut())
        );
        // Heat diffusion
        let start = std::time::Instant::now();
        let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
        let mut next_temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut next_temperature.data));
        if false {
            {
                // Boundary conditions: constant temperature (Dirichlet): T_boundary=0 except top: adiabatic dz(T)_boundary=0 (Neumann) using ghost points
                let task = |z0, z1, mut next_temperature_chunk: Volume<&mut[f32]>| for z in z0..z1 { for y in 1..size.y-1 { for x in 1..size.x-1 {
                    // dt(Q) = c ρ dt(T) : heat energy
                    // dt(Q) = - dx(q): heat flow (positive outgoing)
                    // => dt(T) = - 1/(cρ) dx(q)
                    // q = -k∇T (Fourier conduction)
                    // Finite difference cartesian first order laplacian
                    let T = |dx,dy,dz| temperature[xyz{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32, z: (z as i32+dz) as u32}];
                    let dxxT = ( T(-1, 0, 0) - 2. * T(0, 0, 0) + T(1, 0, 0) ) / sq(δx);
                    let dyyT = ( T(0, -1, 0) - 2. * T(0, 0, 0) + T(0, 1, 0) ) / sq(δx);
                    let dzzT = ( T(0, 0, -1) - 2. * T(0, 0, 0) + T(0, 0, 1) ) / sq(δx);
                    let thermal_conduction = dxxT + dyyT + dzzT; // Cartesian: ΔT = dxx(T) + dyy(T) + dzz(T)
                    let dtT = thermal_diffusivity * thermal_conduction; // dt(T) = αΔT
                    let C = thermal_diffusivity / sq(δx) * δt;
                    assert!(C < 1./2., "{C}"); // Courant–Friedrichs–Lewy condition
                    // Explicit time step (First order: Euler): T[t+1]  = T(t) + δt·dt(T)
                    next_temperature_chunk[xyz{x, y, z: z-z0}] = T(0,0,0) + δt * dtT;
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
        } else {
            for z in 0..size.z { for y in 0..size.y { for x in 0..size.x {  next_temperature[xyz{x, y, z}] =  temperature[xyz{x, y, z}]; } } }
        }
        let points = size.z*size.y*size.x;
        let elapsed = start.elapsed(); report.push(format!("{}M points {}ms {}ns", points/1000000, elapsed.as_millis(), elapsed.as_nanos()/(points as u128)));
        report
    };*/

    use {atomic_float::AtomicF32, std::sync::atomic::Ordering::Relaxed};
    /*if false {
        let mut temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);
        let mut next_temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);
        let mut step = 0;
        use {view::View, image::Image};
        ui::run(&mut View(Image::zero(xy{x: temperature.size.y, y: temperature.size.z-1})), &mut |View(ref mut image):&mut View| -> ui::Result<bool> {
            let report = next(temperature.as_mut(), next_temperature.as_mut());
            std::mem::swap(&mut temperature, &mut next_temperature);

            use itertools::Itertools;
            println!("{step} {}s {}", step as f32*δt, report.iter().format(" "));
            step += 1;

            let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
            for image_y in 0..image.size.y { for image_x in 0..image.size.x {
                image[xy{x: image_x, y: image_y}] = (0..temperature.size.x).map(|volume_x| temperature[xyz{x: volume_x, y: image_x, z: 1+image_y}]).sum::<f32>();
            }}

            #[cfg(feature="avif")] if let 10|100|1000 = step {
                let mut target = Image::zero(image.size);
                view::rgb10(&mut target.as_mut(), image.as_ref());
                use ravif::*;
                let EncodedImage { avif_file, .. } = Encoder::new().encode_raw_planes_10_bit(target.size.x as usize, target.size.y as usize, 
                    target.iter().map(|rgb| [(rgb&0b1111111111) as u16, ((rgb>>10)&0b1111111111) as u16, (rgb>>20) as u16]), 
                    None::<[_; 0]>, rav1e::color::PixelRange::Full, MatrixCoefficients::Identity)?;
                std::fs::write(format!("out/a={},s={},t={step}.avif", f32::round(tissue.absorption) as u32, f32::round(tissue.scattering) as u32), avif_file)?;
            }

            Ok(step <= 1000)
        })
    } else*/ {
        let mut temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);
        struct View(Box<[f32]>);
        impl ui::Widget for View { #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, size: ui::size, offset: ui::int2) { 
            let start = std::time::Instant::now();
            ui::Plot::new(&[&[""]], ui::plot::list(self.0.iter().enumerate().map(|(t,&y)| (t as f64, [[y as f64].into()].into()))).as_ref()).paint(target, size, offset)?; 
            let elapsed = start.elapsed();
            println!("{}ms", elapsed.as_millis());
        } }
        let mut step = 0;
        let radius = 8;
        println!("{}mm", 1000. * radius as f32 * height / size.z as f32);
        ui::run(&mut View([].into()), &mut |view: &mut View| -> ui::Result<bool> {
            let report = light_propagation(&mut rng, temperature.as_mut());
            println!("{step} {}", report);
            step += 1;
            let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
            view.0 = (0..size.z).map(|z| {
                //temperature[xyz{x: size.x/2, y: size.y/2, z}]
                let mut sum = 0.;
                for y in -radius as i32 ..= radius {
                    for x in -radius as i32..= radius {
                        if vector::sq(xy{x,y}) <= sq(radius) { sum += temperature[xyz{x: ((size.x/2) as i32+x) as u32, y: ((size.y/2) as i32+y) as u32, z}]; }
                    }
                }
                sum
            }).collect();
            
            #[cfg(feature="avif")] if let 512 = step {
                let mut target = image::Image::zero(xy{x: 3840, y: 2400});
                let size = target.size;
                use ui::Widget;
                view.paint(&mut target.as_mut(), size, 0.into())?;
                use ravif::*;
                let EncodedImage { avif_file, .. } = Encoder::new().encode_raw_planes_10_bit(target.size.x as usize, target.size.y as usize, 
                    target.iter().map(|rgb| [(rgb&0b1111111111) as u16, ((rgb>>10)&0b1111111111) as u16, (rgb>>20) as u16]), 
                    None::<[_; 0]>, rav1e::color::PixelRange::Full, MatrixCoefficients::Identity)?; // FIXME: PQ
                std::fs::write(format!("out/a={},s={},t={step}.plot.avif", f32::round(tissue.absorption) as u32, f32::round(tissue.scattering) as u32), avif_file)?;
            }

            Ok(step <= 512)
        })
    }
}