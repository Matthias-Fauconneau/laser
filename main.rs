#![feature(slice_take, macro_metavar_expr, atomic_from_mut, array_methods, generic_const_exprs, generic_arg_infer, default_free_fn, const_trait_impl, const_fn_floating_point_arithmetic, associated_type_bounds)]
#![allow(confusable_idents, incomplete_features, non_camel_case_types, non_snake_case, non_upper_case_globals, uncommon_codepoints)]
use std::{default::default, mem::swap, ops::Range, iter, array::from_fn, f64::consts::PI as π, f32::consts::PI, thread, sync::atomic::{AtomicU32, Ordering::Relaxed}, time::Instant};
mod SI; use SI::*;
#[derive(PartialEq,Clone)] struct Material<S: System> {
    mass_density: MassDensity, // ρ [kg/m³]
    specific_heat_capacity: SpecificHeatCapacity, // c [J/(kg·K)]
    thermal_conductivity: ThermalConductivity,  // k [W/(m·K)]
    absorption_coefficient: S::Scalar<ByLength>, // μa [m¯¹]
    scattering_coefficient: S::Scalar<ByLength>, // μs [m¯¹]
}

mod volume; use {ui::Result, num::{sq,cb}, vector::{xy, xyz, vec3}, atomic_float::AtomicF32, volume::{product, size, Volume}};
mod view;

struct RayCell<const R: usize, const C: usize>(Box<[u8; R*C/8]>) where [(); R*C/8]:;
impl<const R: usize, const C: usize> Default for RayCell<R,C> where [(); R*C/8]: { fn default() -> Self { Self(Box::new([0; R*C/8])) } }
impl<const R: usize, const C: usize> RayCell<R,C> where [(); R*C/8]: {
    fn get(&self, ray: u8, cell: u8) -> bool { self.0[((ray as usize)*C+cell as usize)/8] & (1<<(cell%8)) != 0 }
    fn set(&mut self, ray: u8, cell: u8) { assert!(cell < 128); self.0[((ray as usize)*C+cell as usize)/8] |= 1<<(cell%8) }
    fn count(&self, cell: u8) -> u8 { assert!(R<=256); (0..R).filter(|&ray| self.get(ray as u8, cell)).count() as u8 }
}

fn main() -> Result {
    let size = xyz{x: 257, y: 257, z: 65};

    // Environnment
    const background_temperature : Temperature = 18.|C;

    // Body (Blood, Skin)
    const volumetric_rate_of_mass_perfusion : VolumetricMassRate = 0.5|kg_m3s;
    const heat_loss_through_skin_air_interface : HeatFluxDensity = 50.|W_m2; // evaporation, radiation, convection, conduction

    const anisotropy : f32 = 0.9; // g (mean cosine of the deflection angle) [Henyey-Greenstein]
    let scattering = |reduced_scattering_coefficient| reduced_scattering_coefficient / (anisotropy as f64);
    type DMaterial = self::Material<Dimensionalized>;
    let ref tissue = DMaterial{
        mass_density: 1030.|kg_m3,
        specific_heat_capacity: 3595.|J_K·kg,
        thermal_conductivity: 0.49|W_m·K,
        absorption_coefficient: 7.54|_cm,
        scattering_coefficient: scattering(999.|_m), //@750nm
    };
    let ref cancer = DMaterial{absorption_coefficient: 10. |_m, ..tissue.clone()};
    const initial_blood_temperature : Temperature = 36.85|C;
    let T = initial_blood_temperature;
    let ref glue = DMaterial{
        mass_density: 895.|kg_m3,
        specific_heat_capacity: (3353.5|J_K·kg) + (5.245|J_K2·kg) * T, // <~5000
        thermal_conductivity: (0.3528|W_m·K) + (0.001645|W_m·K2) * T, // <~1
        absorption_coefficient: 15.84|_m/*519.|_m - 0.5 |_Km * T*/, scattering_coefficient: scattering(1.|_m),
    };
    let material_list = [tissue, cancer, glue];

    let id = |material| material_list.iter().position(|&o| o == material).unwrap() as u8;
    fn map(size: size, height: Length, x: Length) -> u16 { ((x/height).unitless()*(size.z as f64)) as u16 }
    fn z(size: size, height: Length, start: Length, end: Length) -> Range<u32> { product(xyz{z: map(size, height, start), ..size}) .. product(xyz{z: map(size, height, end), ..size}) }
    let (height, material_volume) : (Length,_) = match "tissue" {
        "tissue" => {
            let height = 1.|cm;
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
            let mut material_volume = Volume::from_iter(size, iter::from_fn(|| Some(id(tissue))));
            let cancer_diameter = 5.|mm;
            let center_z = cancer_diameter/2. + (1e-3|m);
            let map = |x| map(size, height, x);
            let radius = cancer_diameter/2.;
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

    let Courant_Friedrichs_Lewy_condition = thermal_diffusivity / sq(δx) * δt;
    assert!(Courant_Friedrichs_Lewy_condition < 1., "{Courant_Friedrichs_Lewy_condition}");

    let ref material_list = material_list.map(|&DMaterial{mass_density, specific_heat_capacity, thermal_conductivity, absorption_coefficient, scattering_coefficient}| {
        Material{mass_density, specific_heat_capacity, thermal_conductivity, absorption_coefficient: (absorption_coefficient*δx).f32(), scattering_coefficient: (scattering_coefficient*δx).f32()}
    });
    type Material = self::Material<NonDimensionalized>;

    use {rand::{Rng as Random, SeedableRng as SeedableRandom}, rand_xoshiro::Xoshiro128Plus as ParallelRandom};

    struct Laser {
        diameter: f32,
        position: vec3,
        direction: vec3,
        power: Power,
        //wavelength: Length,
        //focal_length: Length,
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
        //fn peak_intensity(&self) -> Intensity { self.power * π * sq(self.diameter)/4. / (sq(self.wavelength)*sq(self.focal_length)) } // W/m²
    }

    let ref laser = Laser{
        diameter: ((0.6|mm) / δx).f32(),
        position: xyz{x: size.x as f32/2., y: size.y as f32/2., z: 0.5},
        direction: xyz{x: 0., y: 0., z: 1.},
        power: 1.|W,
        //wavelength: 750.|nm,
        //focal_length: 0.|m,
    };

    const ray_per_thread : usize = 256; // atomic add u16 energy, and then have a temperature+float conversion pass might be more efficient than atomic float
    const threads : usize = 8;
    const ray_per_step : usize = ray_per_thread * threads;

    type float = f32;
    type AtomicF = AtomicF32;

    const intensity_cells: usize = 128;
    type RayCell = self::RayCell<ray_per_thread, intensity_cells>;
    #[derive(Default)] struct IntensityCells { z: RayCell, r: RayCell, z_radius : f32, r_z: u16 }
    /*#[derive(Default)]*/ struct IntensitySum { z: [AtomicU32; intensity_cells], r: [AtomicU32; intensity_cells], z_radius: f32, r_z: u16 }
    impl Default for IntensitySum { fn default() -> Self { Self{z: from_fn(|_| default()), r: from_fn(|_| default()), z_radius: default(), r_z: default()}}}

    fn light_propagation(ref mut random : &mut ParallelRandom, (material_list, material_volume): (&[Material], &Volume<&[u8]>), δx: Length, δt: Time, laser: &Laser,
        intensity_sum: Option<&mut IntensitySum>, temperature: &Volume<&mut [AtomicF]>) -> String {
        fn task(ref mut random : impl Random, (material_list, material_volume): (&[Material], &Volume<&[u8]>), δx: Length, δt: Time, laser: &Laser,
                        intensity_sum: Option<&IntensitySum>, temperature: &Volume<&mut [AtomicF]>) {
            let size : vec3 = (material_volume.size-xyz{x: 0, y: 0, z: 1}).into();
            let mut intensity = intensity_sum.map(|I| IntensityCells{z_radius: I.z_radius, r_z: I.r_z, ..default()});
            for ray in 0..ray_per_thread {
                let (mut position, mut direction) = laser.sample(random);

                loop {
                    let xyz{x,y,z} = position;
                    if x < 0. || x >= size.x || y < 0. || y >= size.y || z < 0. || z >= size.z as f32 { break; }
                    let index = {let xyz{x,y,z}=position; xyz{x: x as u16, y: y as u16, z: z as u16}};

                    let id = material_volume[index];
                    let Material{mass_density: density, specific_heat_capacity, absorption_coefficient, scattering_coefficient,..} = material_list[id as usize];

                    // Absorption
                    if random.gen::<f32>() < absorption_coefficient /*·δx*/ {
                        let volume = δx*δx*δx;
                        let heat_capacity : HeatCapacity = density * volume * specific_heat_capacity; // J/K
                        let δT : Temperature = δt * laser.sample_power(ray_per_step) / heat_capacity;
                        temperature[index].fetch_add(δT.K() as f32, Relaxed);
                        break;
                    }

                    // Intensity
                    if let Some(intensity) = intensity.as_mut() {
                        let r2 = vector::sq(xy{x: x-size.x/2., y: y-size.y/2.});
                        if r2 <= sq(intensity.z_radius) { intensity.z.set(ray as u8, index.z as u8); } // Each ray can only contribute once per intensity disk
                        if index.z == intensity.r_z {
                            let r = f32::sqrt(r2) as u8; // FIXME: table or index by r2
                            if r < 128 { intensity.r.set(ray as u8, r); }
                        }
                    }

                    // Scattering
                    if random.gen::<f32>() < scattering_coefficient /*·δx*/ {
                        let ξ = random.gen::<f32>();
                        let g = anisotropy;
                        let cosθ = -1./(2.*g)*(1.+g*g-sq((1.-g*g)/(1.+g-2.*g*ξ))); // Importance sample with Henyey-Greenstein phase function
                        let sinθ = f32::sqrt(1. - cosθ*cosθ);
                        let φ = 2.*PI*random.gen::<f32>();
                        let (T, B) = vector::tangent_space(direction);
                        let next_direction = sinθ*(f32::cos(φ)*T + f32::sin(φ)*B) + cosθ*direction;
                        direction = vector::normalize(next_direction);
                    }

                    position = position + direction;
                }
            }
            if let Some(intensity_sum) = intensity_sum.as_ref() { // Intensity sum over rays
                let intensity = intensity.as_mut().unwrap();
                for cell in 0..intensity_cells {
                    intensity_sum.z[cell].fetch_add(intensity.z.count(cell as u8) as u32, Relaxed);
                    intensity_sum.r[cell].fetch_add(intensity.r.count(cell as u8) as u32, Relaxed);
                }
            }
        }
        let start = Instant::now();
        thread::scope(|s| for thread in from_fn::<_, threads, _>(|_| {
            let task_random = random.clone();
            let intensity_sum = intensity_sum.as_deref();
            let thread = thread::Builder::new().spawn_scoped(s, move || task(task_random, (material_list, material_volume), δx, δt, laser, intensity_sum, temperature)).unwrap();
            random.jump();
            thread
        }) { thread.join().unwrap(); });
        let elapsed = start.elapsed();
        format!("{} samples {}ms {}μs", ray_per_step, elapsed.as_millis(), elapsed.as_micros()/(ray_per_step as u128))
    }

    fn heat_diffusion((material_list, ref material_volume): (&[Material], &Volume<&[u8]>), δx: Length, δt: Time,
                    ref temperature: Volume<&[float]>, mut next_temperature: Volume<&mut [float]>) -> String {
        let start = Instant::now();
        let size = temperature.size;
        {
            let task = |z0, z1, mut next_temperature_chunk: Volume<&mut[float]>| for z in z0..z1 { for y in 1..size.y-1 { for x in 1..size.x-1 {
                let id = material_volume[xyz{x, y , z}];
                let Material{mass_density: density, specific_heat_capacity, thermal_conductivity,..} = material_list[id as usize];
                let volumetric_heat_capacity = density * specific_heat_capacity; // J/K·m³
                let thermal_diffusivity = thermal_conductivity / volumetric_heat_capacity; // dt(T) = k/(cρ) ΔT = α ΔT (α≡k/(cρ)) [m²/s]
                // dt(Q) = c ρ dt(T) : heat energy
                // dt(Q) = - dx(q): heat flow (positive outgoing)
                // => dt(T) = - 1/(cρ) dx(q)
                // q = -k∇T (Fourier conduction)
                // Finite difference cartesian first order laplacian
                let T = |dx,dy,dz| temperature[xyz{x: (x as i16+dx) as u16, y: (y as i16+dy) as u16, z: (z as i16+dz) as u16}];
                let dxxT = T(-1, 0, 0) - 2. * T(0, 0, 0) + T(1, 0, 0);
                let dyyT = T(0, -1, 0) - 2. * T(0, 0, 0) + T(0, 1, 0);
                let dzzT = T(0, 0, -1) - 2. * T(0, 0, 0) + T(0, 0, 1);
                let thermal_conduction = dxxT + dyyT + dzzT; // Cartesian: ΔT = dxx(T) + dyy(T) + dzz(T) {=> /δx²}
                let α = thermal_diffusivity / sq(δx) * δt;
                // Blood flow
                let blood_specific_heat_capacity = 3595.|J_K·kg;
                let volumetric_rate_of_heat_capacity_perfusion : VolumetricPowerCapacity = volumetric_rate_of_mass_perfusion * blood_specific_heat_capacity;
                let metabolic_heat = δt * ((1000.|W_m3) / volumetric_heat_capacity);
                // Explicit time step (First order: Euler): T[t+1]  = T[t] + δt·dt(T) {=> δt}
                let β = (δt * (volumetric_rate_of_heat_capacity_perfusion / volumetric_heat_capacity)).unitless();
                next_temperature_chunk[xyz{x, y, z: z-z0}] = (1.-β) as f32 * T(0,0,0) + α.unitless() as f32 * thermal_conduction + metabolic_heat.K() as f32;
            }}};
            let mut next_temperature = next_temperature.as_mut();
            let range = 1..size.z-1;
            next_temperature.take_mut(range.start);
            thread::scope(|s| for thread in from_fn::<_, threads, _>(move |thread| {
                let z0 = range.start + (range.end-range.start)*thread as u16/threads as u16;
                let z1 = range.start + (range.end-range.start)*(thread as u16+1)/threads as u16;
                let next_temperature_chunk = next_temperature.take_mut(z1-z0);
                thread::Builder::new().spawn_scoped(s, move || task(z0, z1, next_temperature_chunk)).unwrap()
            }) { thread.join().unwrap(); });
        }
        // Boundary conditions: adiabatic dz(T)_boundary=q (Neumann) using ghost points
        for y in 0..size.y { for x in 0..size.x { // Top: Sets ghost points to yield constant flux from points below
            let k = 1.380649e-23|J_K;
            const c : Speed = 299_792_458.|m_s;
            const h : PlanckConstant = 6.62607015e-34 |J_Hz;
            let σ = π*(c/(4.*π))*2.*1./8.*4.*π*cb(2./(h*c))*(k*k*k*k)*π.powi(4)/15.;
            let q = σ*(sq(sq(initial_blood_temperature+(temperature[xyz{x, y, z: 1}] as f64|K))) - sq(sq(background_temperature)));
            //let q = heat_loss_through_skin_air_interface + σ*((initial_blood_temperature+δT.K()).powi(4) - initial_blood_temperature.powi(4)); // T_blood⁴ - T_air⁴ is part of constant loss
            let δT = δx*(q/material_list[0].thermal_conductivity); // Temperature difference yielding the equivalent flux only with the simulated heat
            next_temperature[xyz{x, y, z: 0}] = next_temperature[xyz{x, y, z: 1}] - δT.K()  as f32;
             // could have set adiabatic and directly offset T instead but this way is more higher order compatible
        } }
        for y in 0..size.y { for x in 0..size.x { next_temperature[xyz{x, y, z: size.z-1}] = next_temperature[xyz{x, y, z: size.z-2}]; } } // Bottom: Sets ghost points to temperature of point above
        for z in 0..size.z { for x in 0..size.x { next_temperature[xyz{x, y : 0, z}] = next_temperature[xyz{x, y: 1, z}]; } } // Front: Sets ghost points to temperature of point behind
        for z in 0..size.z { for x in 0..size.x { next_temperature[xyz{x, y: size.y-1, z}] = next_temperature[xyz{x, y: size.y-2, z}]; } } // Back: Sets ghost points to temperature of point before
        for z in 0..size.z { for y in 0..size.y { next_temperature[xyz{x: 0, y, z}] = next_temperature[xyz{x: 1, y, z}]; } } // Left: Sets ghost points to temperature of point beside
        for z in 0..size.z { for y in 0..size.y { next_temperature[xyz{x: size.x-1, y, z}] = next_temperature[xyz{x: size.x-2, y, z}]; } } // Right: Sets ghost points to temperature of point beside
        let elapsed = start.elapsed();
        format!("{}M points {}ms {}ns", temperature.len()/1000000, elapsed.as_millis(), elapsed.as_nanos()/(temperature.len() as u128))
    }

    fn next(ref mut random : &mut ParallelRandom, (material_list, ref material_volume): (&[Material], Volume<&[u8]>), δx: Length, δt: Time, laser: &Laser,
                    intensity_sum: Option<&mut IntensitySum>, ref mut temperature: Volume<&mut [AtomicF]>, mut next_temperature: Volume<&mut [AtomicF]>) -> [String; 2] {
        [
            light_propagation(random, (material_list, material_volume), δx, δt, laser, intensity_sum, temperature),
            heat_diffusion((material_list, material_volume), δx, δt, temperature.get_mut().as_ref(), next_temperature.get_mut())
        ]
    }

    let mut step = 0;
    let ref mut random = rand_xoshiro::Xoshiro128Plus::seed_from_u64(0);

    let mut laser_profile = [0.|W_m2].repeat((size.x/2-1) as usize);
    let mut temperature = Volume::<Box<[AtomicF]>>::default(size);
    let mut next_temperature = Volume::<Box<[AtomicF]>>::default(size);
    let mut time_averaged_temperature = Volume::<Box<[f32]>>::default(size);
    let mut thermal_dose = Volume::<Box<[f32]>>::default(size);

    let Ir_z = 2.|mm;
    let mut intensity = IntensitySum{z_radius: ((1.|mm) / δx).f32(), r_z: (Ir_z / δx).f32() as u16,  ..default()};

    let Tt_z = [7, 14, 21];

    use {image::Image, ui::{list, Plot}, view::*};

    let Tt = Plot::new("Temperature over time for probes on the axis", xy{x: "Time ($s)", y: "ΔTemperature ($K)"}, Box::from(Tt_z.map(|z| format!("{} deep", (z as f64)*δx))));
    let Iz = Plot::new("Laser intensity over depth (on the axis)",  xy{x: "Depth ($m)", y: "Intensity ($W/m²)"}, Box::from(["I(z)".to_string()]));
    let Ir = Plot::new("Laser intensity at the surface (radial plot)", xy{x: "Radius ($m)", y: "Intensity ($W/m²)"}, Box::from([format!("I(r) at {Ir_z}"), "I0(r)".to_string()]));

    let _Tyz = LabeledImage::new("Temperature difference over y,z (average over x)", Image::zero(image::size::from(temperature.size.xz())-xy{x: 0, y: 1}), Box::new(|T| (T as f64|K).to_string()));
    let Tdyz = LabeledImage::new("Thermal dose over y,z (maximum over x)", Image::zero(image::size::from(temperature.size.xz())-xy{x: 0, y: 1}), Box::new(|Td| format!("{Td}")));

    derive_IntoIterator! { pub struct Plots { pub Tt: Plot, pub Iz: Plot, pub Ir: Plot, pub Tyz: LabeledImage} }
    struct State { stop: usize }
    let ref mut idle = move |app: &mut _| -> Result<bool> {
        let _report = next(random, (material_list, material_volume.as_ref()), δx, δt, laser, Some(&mut intensity), temperature.as_mut(), next_temperature.as_mut());
        swap(&mut temperature, &mut next_temperature);
        //use itertools::Itertools; println!("{step} {}s {}", step as f32*δt, report.iter().format(" "));
        step += 1;

        let temperature = temperature.get_ref();
        for (time_averaged_temperature, temperature) in time_averaged_temperature.data.iter_mut().zip(temperature.data) {
            let α = (δt/(60.|sec)).unitless() as f32;
            *time_averaged_temperature = (1.-α)* (*time_averaged_temperature) + α*temperature;
        }
        for (thermal_dose, temperature) in thermal_dose.data.iter_mut().zip(&*time_averaged_temperature.data) {
            let T = initial_blood_temperature + (*temperature as f64|K);
            if T > 43.|C { *thermal_dose += (δt/(60.|sec)).unitless() as f32 * f32::powf(2., (T - (43.|C)).K() as f32); }
        }

        let App{widget: Grid(Plots{Tt, Iz, Ir, Tyz: Tdyz}), state: State{stop},..} = app;

        // T(t) at z={probes}
        Tt.x_values.push( δt.s() * step as f64 );
        for (i, &z) in Tt_z.iter().enumerate() {
            let p = xyz{x: size.x/2, y: size.y/2, z};
            Tt.sets[i].push( temperature[p] as f64 );
        }

        // axial: I(z)
        Iz.x_values = list((0..size.z-1).map(|z| z as f64 * δx.m())).into();
        let count = {
            let r = f32::ceil(intensity.z_radius) as i32;
            (-r ..= r).map(|y| (-r ..= r).filter(move |&x| vector::sq(xy{x,y}) as f32 <= sq(intensity.z_radius))).flatten().count()
        };
        Iz.sets[0] = (0..size.z-1).map(|z| {
            let power : Power = (intensity.z[z as usize].load(Relaxed) as f64) * laser.sample_power(step * ray_per_step);
            let area : Area = count as f64 * sq(δx);
            //assert_eq!(area, π*sq(intensity.z_radius as f64)*sq(δx));
            let intensity : EnergyFluxDensity = power / area;
            intensity.W_m2()
        }).collect();
        Iz.need_update();

        // radial: I(r)
        Ir.x_values = list((0..size.x/2-1).map(|r| (r as f64) * δx.m())).into();
        Ir.sets[0] = (0..size.x/2-1).map(|r| {
            let mut count = 0;
            for y in -((r+1) as i16) ..= (r+1) as i16 {
                for x in -((r+1) as i16)..= (r+1) as i16 {
                    let r2 = vector::sq(xy{x,y}) as u16;
                    if sq(r) <= r2 && r2 < sq(r+1) {
                        count += 1;
                    }
                }
            }
            let power : Power = (intensity.r[r as usize].load(Relaxed) as f64) * laser.sample_power(step * ray_per_step);
            let area : Area = count as f64 * δx * δx; // Ring (discretized)
            //assert_eq!(sq(δx) * 2.*π * (r+1) as f64, area);
            let intensity : EnergyFluxDensity = power / area;
            intensity.W_m2()
        }).collect();
        for position in iter::repeat_with(|| { let (position, _) = laser.sample(random); position }).take(ray_per_step) {
            let xyz{x,y,..} = position;
            let xy{x,y} = xy{x: x-(size.x/2) as f32, y: y-(size.y/2) as f32};
            let r = f64::sqrt(vector::sq(xy{x,y}) as f64);
            let power : Power = laser.sample_power(ray_per_step);
            let area : Area = sq(δx) * 2.*π * r; // Ring
            let intensity : EnergyFluxDensity = power / area;
            let r = r as usize;
            if r < laser_profile.len() { laser_profile[r] += intensity; }
        }
        Ir.sets[1] = list(laser_profile.iter().map(|&I| I.W_m2() / (step as f64))).into();
        Ir.need_update();

        /*// T(y,z) (mean x)
        let ref mut Tyz = Tyz.0.image.image;
        for image_y in 0..Tyz.size.y { for image_x in 0..Tyz.size.x {
            fn mean<I:IntoIterator<IntoIter:ExactSizeIterator>,S:std::iter::Sum<I::Item>+std::ops::Div>(iter: I) -> S::Output where u32:Into<S> { let iter = iter.into_iter(); let len = iter.len(); iter.sum::<S>() / (len as u32).into() }
            Tyz[xy{x: image_x, y: image_y}] = mean::<_,f64>((1..size.x-1).map(|volume_x| temperature[xyz{x: volume_x, y: image_x as u16, z: 1+image_y as u16}])) as f32;
        }}*/

        // Td(y,z) (max x)
        let ref mut Tdyz = Tdyz.0.image.image;
        for image_y in 0..Tdyz.size.y { for image_x in 0..Tdyz.size.x {
            Tdyz[xy{x: image_x, y: image_y}] = (1..size.x-1).map(|volume_x| thermal_dose[xyz{x: volume_x, y: image_x as u16, z: 1+image_y as u16}]).reduce(f32::max).unwrap();
        }}

        let stop = *stop;
        if step == stop {
            let path = format!("out/a={a},s={s},d={d},q={heat_loss_through_skin_air_interface},w={volumetric_rate_of_mass_perfusion},t={step}", a=tissue.absorption_coefficient, s=tissue.scattering_coefficient, d=laser.diameter);
            //std::fs::write(&path, format!("Tt: {Tt:?}\nIz: {Iz:?}\nIr: {Ir:?}\nTdyz: ({Tdyz:?}, {:?})", Tdyz.data)).unwrap();
            write_image(path+".avif", app);
        }
        Ok(step <= stop)
    };
    let actions = [(' ', |app:&mut App<State,_>| app.state.stop *= 2)];
    let ref actions = actions.each_ref().map(|(key,closure)| (*key, closure as &_));
    ui::run("Laser", &mut Idle{app: App{state: State{stop: 1024}, widget: Grid(Plots{Tt, Iz, Ir, Tyz: Tdyz}), actions}, idle})
}
