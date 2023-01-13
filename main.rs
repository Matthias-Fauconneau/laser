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
        absorption: 7.540, scattering: scattering(999.), //@750nm
    };
    //let ref cancer = Material{absorption: 10., ..tissue.clone()};
    //panic!("{} {} {}", glue.absorption, tissue.absorption, glue.absorption/tissue.absorption);
    //let material_list = [glue, tissue, cancer];
    let material_list = [tissue];
    let id = |material| material_list.iter().position(|&o| o == material).unwrap();

    //let size = xyz{x: 512, y: 512, z: 513};
    //let size = xyz{x: 513, y: 513, z: 257};
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
    let δt = 0.1 / (thermal_diffusivity / sq(δx)); // Time step
                
    let material_list = material_list.map(|&Material{density,specific_heat_capacity,thermal_conductivity,absorption,scattering}|
        Material{density, specific_heat_capacity, thermal_conductivity, absorption: absorption*δx, scattering: scattering*δx}); // TODO

    let laser_diameter = 0.8e-2 / δx;
    //let laser_diameter = 0.08e-2 / δx;
        
    let laser = |ref mut rng : &mut rand_xoshiro::Xoshiro128Plus| {
        let laser_position = xyz{x: size.x as f32/2., y: size.y as f32/2., z: 0.5};
        let laser_direction = xyz{x: 0., y: 0., z: 1.};
        let diameter = laser_diameter;
        let position = laser_position + {
            let xy{x,y} = if true { // Approximate Airy disc using gaussian
                xy{x: diameter/2. * rng.sample::<f32,_>(rand_distributions::StandardNormal), y: diameter/2. * rng.sample::<f32,_>(rand_distributions::StandardNormal)}
            } else {
                use rand::distributions::{Distribution, Uniform};
                let square = Uniform::new_inclusive(-diameter/2., diameter/2.);
                loop { let p = xy{x: Distribution::sample(&square, rng), y: Distribution::sample(&square, rng)}; if vector::sq(p) <= sq(diameter/2.) { break p; } }
            };
            xyz{x,y, z: 0.}
        };
        let direction = laser_direction; // TODO: divergence
        //let wavelength = 750e-9;
        //let power = 1.5-2 W
        //let peak_intensity = power * pi*sq(diameter)/4 / (sq(wavelength)*sq(focal_length)) // W/m²
        (position, direction)
    };

    const threads : usize = 8;

    let light_propagation = |ref mut rng : &mut rand_xoshiro::Xoshiro128Plus, /*mut*/ temperature: Volume<&mut [AtomicF32]>|{
        const samples : usize = 8192*4;
        let task = |ref mut rng : rand_xoshiro::Xoshiro128Plus|{
            for _ in 0..samples/threads {
                let (mut position, mut direction) = laser(rng);

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
                        let next_direction = sinθ*(f32::cos(φ)*T + f32::sin(φ)*B) + cosθ*direction;
                        direction = vector::normalize(next_direction);
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

    let next = move |ref mut rng : &mut rand_xoshiro::Xoshiro128Plus, mut temperature: Volume<&mut [AtomicF32]>, mut next_temperature: Volume<&mut [AtomicF32]>| -> Vec<String>{
        let mut report = Vec::new();
        report.push(
            light_propagation(rng, temperature.as_mut())
        );
        // Heat diffusion
        let start = std::time::Instant::now();
        let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
        let mut next_temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut next_temperature.data));
        if true {
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
                    assert!(C <= 1./2., "{C}"); // Courant–Friedrichs–Lewy condition
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
    };

    use image::Image;
    fn write_avif(path: impl AsRef<std::path::Path>, image: Image<Box<[u32]>>) {
        #[cfg(not(feature="avif"))] println!("Built without AVIF support: {} {}", path.as_ref().display(), image.size);
        #[cfg(feature="avif")] {
            use ravif::*;
            let EncodedImage { avif_file, .. } = Encoder::new().encode_raw_planes_10_bit(image.size.x as usize, image.size.y as usize, 
                image.iter().map(|rgb| [(rgb&0b1111111111) as u16, ((rgb>>10)&0b1111111111) as u16, (rgb>>20) as u16]), 
                None::<[_; 0]>, rav1e::color::PixelRange::Full, MatrixCoefficients::Identity).unwrap(); // FIXME: PQ
            std::fs::write(path, avif_file).unwrap();
        }
    }
    fn write_image(path: impl AsRef<std::path::Path>, view: &mut impl ui::Widget) {
        let mut target = Image::zero(xy{x: 3840, y: 2400});
        let size = target.size;
        view.paint(&mut target.as_mut(), size, 0.into()).unwrap();    
        write_avif(path, target);
    }
    let name = |step| format!("out/a={},s={},d={laser_diameter},t={step}", f32::round(tissue.absorption) as u32, f32::round(tissue.scattering) as u32);
    
    use {atomic_float::AtomicF32, std::sync::atomic::Ordering::Relaxed};
    use {rand_xoshiro::rand_core::SeedableRng, rand::Rng};
    let ref mut rng = rand_xoshiro::Xoshiro128Plus::seed_from_u64(0);
    let mut temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);
    let mut next_temperature : Volume<Box<[AtomicF32]>> = Volume::default(size);
    let mut step = 0;
    if false {
        use view::View;
        ui::run(&mut View(Image::zero(xy{x: temperature.size.y, y: temperature.size.z-1})), &mut |View(ref mut image):&mut View| -> ui::Result<bool> {
            let report = next(rng, temperature.as_mut(), next_temperature.as_mut());
            std::mem::swap(&mut temperature, &mut next_temperature);

            use itertools::Itertools;
            println!("{step} {}s {}", step as f32*δt, report.iter().format(" "));
            step += 1;

            let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
            for image_y in 0..image.size.y { for image_x in 0..image.size.x {
                image[xy{x: image_x, y: image_y}] = (0..temperature.size.x).map(|volume_x| temperature[xyz{x: volume_x, y: image_x, z: 1+image_y}]).sum::<f32>();
            }}

            if let 10|100|1000 = step {
                let mut target = Image::zero(image.size);
                view::rgb10(&mut target.as_mut(), image.as_ref());
                write_avif(name(step), target);
            }

            Ok(step <= 1000)
        })
    } else {
        use ui::plot::list;
        struct Plot { keys: Box<[String]>, values: Box<[Vec<f32>]> }
        impl ui::Widget for Plot { #[fehler::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, size: ui::size, offset: ui::int2) { 
            ui::Plot::new(&[&list(self.keys.iter().map(|s| s.as_ref()))], list((0..self.values[0].len()).map(|t| (t as f64, Box::from([list(self.values.iter().map(|values| values[t] as f64))])))).as_ref()).paint(target, size, offset)?; 
        } }

        #[fehler::throws(ui::Error)] fn grid(widgets: &mut [&mut dyn ui::Widget], target: &mut ui::Target, size: ui::size, _offset: ui::int2) { 
            //let start = std::time::Instant::now();
            assert!(widgets.len() <= 4);
            let (w, h) = (2, 2);
            for y in 0..h { for x in 0..w {
                let i = (y*w+x) as usize;
                if i >= widgets.len() { break; }
                let ref mut target = target.slice_mut(xy{x: x*size.x/w, y: y*size.y/h}, size/xy{x: w, y: h});
                let size = target.size;
                widgets[i].paint(target, size, 0.into())?;
            }}
            //let elapsed = start.elapsed();
            //println!("plot: {}ms", elapsed.as_millis());
        }
         
        let radius = 8;
        let mm = |x:u32| (x as f32)*δx*1e3;
        println!("{}mm", mm(radius));
        
        let probes = [1,8,16];
        
        let Tt = Plot{keys: Box::from(probes.map(|z| format!("{}mm", f32::round(mm(z)) as u32))), values: Box::from(probes.map(|_| Vec::new()))};
        let Iz = Plot{keys: Box::from(["I(z)".to_owned()]), values: Box::from([Vec::new()])};
        let Ir = Plot{keys: Box::from(["I(r)".to_owned(),"I0(r)".to_owned()]), values: Box::from([Vec::new(),Vec::new()])};
        struct Grid {
            Tt: Plot,
            Iz: Plot,
            Ir: Plot,
        }
        impl ui::Widget for Grid { 
            fn paint(&mut self, target: &mut ui::Target, size: ui::size, offset: ui::int2) -> ui::Result { grid(&mut [&mut self.Tt, &mut self.Iz, &mut self.Ir], target, size, offset) } 
        }
        
        let mut I0 = list(std::iter::repeat(0.).take((size.x/2-1) as usize));

        ui::run(&mut Grid{Tt,Iz,Ir}, &mut |grid: &mut Grid| -> ui::Result<bool> {
            let _report = if true {
                let report = next(rng, temperature.as_mut(), next_temperature.as_mut());
                std::mem::swap(&mut temperature, &mut next_temperature);
                report
            } else {
                [light_propagation(rng, temperature.as_mut())].into()
            };

            //use itertools::Itertools;
            //println!("{step} {}s {}", step as f32*δt, report.iter().format(" "));
            step += 1;

            let Grid{Tt,Iz,Ir} = grid;
            let temperature = Volume::new(size, AtomicF32::get_mut_slice(&mut temperature.data));
            for (i, &z) in probes.iter().enumerate() {
                Tt.values[i].push( temperature[xyz{x: size.x/2, y: size.y/2, z}] );
            }
            //println!("{}", plot.values.iter().map(|probe| probe.last().unwrap()).format(" "));
            //assert!(plot.values[0].len() == step);
            // axial: I(z)
            Iz.values[0] = (0..size.z).map(|z| {
                let mut sum = 0.;
                for y in -(radius as i32) ..= radius as i32 {
                    for x in -(radius as i32) ..= radius as i32 {
                        if vector::sq(xy{x,y}) as u32 <= sq(radius) { sum += temperature[xyz{x: ((size.x/2) as i32+x) as u32, y: ((size.y/2) as i32+y) as u32, z}]; }
                    }
                }
                sum
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
            for position in std::iter::repeat_with(|| { let (position, _) = laser(rng); position }).take((size.x*size.y) as usize) {
                let xyz{x,y,..} = position;
                let xy{x,y} = xy{x: x-(size.x/2) as f32, y: y-(size.y/2) as f32};
                let r2 = vector::sq(xy{x,y}) ;
                if r2 < sq(size.x/2-1) as f32 {
                    I0[f32::sqrt(r2) as usize] += 1. / f32::sqrt(r2 as f32);
                }
            }
            let norm_Ir = Ir.values[0].iter().sum::<f32>();
            //let max_Ir = *Ir.values[0].iter().max_by(|a,b| f32::total_cmp(a,b)).unwrap();
            let norm_I0 = I0.iter().sum::<f32>();
            //let max_I0 = *I0.iter().max_by(|a,b| f32::total_cmp(a,b)).unwrap();
            Ir.values[1] = list(I0.iter().map(|&I| I * norm_Ir / norm_I0)).into();
            //Ir.values[1] = list(I0.iter().map(|&I| I * max_Ir / max_I0)).into();
            if step == 512 { write_image(name(step), grid); }
            Ok(step <= 512)
        })
    }
}