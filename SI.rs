#![allow(dead_code,non_upper_case_globals)]

pub fn fmt(unit: &str, value: f64, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if value == 0. { write!(f,"0") } else {
        let log10 = f64::log10(f64::abs(value) as f64);
        let floor1000 = f64::floor(log10/3.); // submultiple
        let part1000 = num::exp10(log10 - floor1000*3.); // remaining magnitude part within the submultiple: x / 1000^⌊log1000(x)⌋
        let submagnitude = if part1000 < 1. { format!("{:.1}", part1000) } else { (f64::round(part1000) as u32).to_string() };
        assert!(f64::clamp(-3., floor1000, 3.) == floor1000, "{floor1000}");
        let submultiple = ["n","µ","m","","k","M","G"][(3+(floor1000 as i8)) as usize];
        write!(f, "{}{}{}{unit}", if value<0. {"-"} else {""}, submagnitude, submultiple)
    }
}

/// Quantities and units operators

#[const_trait] pub trait Float {
    fn unwrap(self) -> f64;
    fn wrap(value : f64) -> Self;
}

type int = i8;
#[derive(PartialEq,Clone,Copy,Debug)] pub struct Quantity<const A0 : int, const A1 : int, const A2 : int, const A3 : int>(f64);
/*impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::fmt::Display for Quantity<A0,A1,A2,A3> { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", self)
} }*/
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> const Float for Quantity<A0,A1,A2,A3>{
    fn unwrap(self) -> f64 { self.0 }
    fn wrap(value : f64) -> Self { Self(value) }
}

impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Add for Quantity<A0,A1,A2,A3> {
    type Output = Self;
    fn add(self, b: Self) -> Self::Output { Self(self.0.add(b.0)) }
}
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::AddAssign for Quantity<A0,A1,A2,A3> { fn add_assign(&mut self, b: Self) { self.0.add_assign(b.0); } }

impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> /*const*/ std::ops::Sub for Quantity<A0,A1,A2,A3> {
    type Output = Self;
    fn sub(self, b: Self) -> Self::Output { Self(self.0.sub(b.0)) }
}

impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::cmp::PartialOrd for Quantity<A0,A1,A2,A3> { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { self.0.partial_cmp(&other.0) } }

pub struct Unit<Q>(std::marker::PhantomData<Q>);
pub const fn unit<Q>() -> Unit<Q> { Unit(std::marker::PhantomData) }
impl<Q:~const Float> /*const*/ std::ops::BitOr<Unit<Q>> for f64 { type Output = Q; fn bitor(self, _: Unit<Q>) -> Self::Output { Q::wrap(self) } }

// quantity · quantity
#[const_trait] pub trait Mul<Q> { type Output : Float; }
impl<Q:Float> Mul<Q> for Quantity<0,0,0,0> { type Output = Q; } // 1 · quantity
pub trait NotDimensionless {}
impl<Q:Float+NotDimensionless> Mul<Quantity<0,0,0,0>> for Q { type Output = Q; } // quantity · 1 // Need NotDimensionless to disambiguate 1 · 1
macro_rules! impl_Mul { ([$a0:literal,$a1:literal,$a2:literal,$a3:literal], [$b0:literal,$b1:literal,$b2:literal,$b3:literal]) => {
    impl const Mul<Quantity<$b0,$b1,$b2,$b3>> for Quantity<$a0,$a1,$a2,$a3> { type Output = Quantity<{$a0+$b0},{$a1+$b1},{$a2+$b2},{$a3+$b3}>; } } }
impl_Mul!{[-1,0,0,0], [1,0,0,0]} // 1/T·T
impl_Mul!{[1,0,0,0], [-1,0,0,0]}// T·1/T
impl_Mul!{[0,-1,0,0], [0,1,0,0]}// 1/L·L
impl_Mul!{[0,0,0,1], [0,0,0,-1]}// 1/Θ·Θ
impl_Mul!{[0,1,0,0], [0,1,0,0]}// L·L
impl_Mul!{[0,2,0,0], [0,1,0,0]}// L·L²
impl_Mul!{[0,1,0,0], [-1,-1,0,0]}// Length·1/LT=Rate
//impl_Mul!{[-1,2,0,0], [-1,2,0,0]}// Diffusivity²
//impl_Mul!{[-2,1,0,0], [0,3,0,0]}// Volume·Acceleration
impl_Mul!{[1,0,0,0], [-1,0,0,1]}// T·Θ/T=Θ
impl_Mul!{[0,-3,1,0], [0,3,0,0]}// MassDensity·Volume
impl_Mul!{[0,1,0,0], [0,-1,0,1]}// Length*TemperatureGradient=Temperature
impl_Mul!{[0,0,1,0], [-2,2,0,-1]}// Mass·SpecificHeatCapacity
impl_Mul!{[0,-3,1,0], [-2,2,0,-1]}// MassDensity·SpecificHeatCapacity=VolumetricHeatCapacity
impl_Mul!{[-2,2,0,-2], [0,0,0,1]}// SpecificHeatCapacity/Θ·Θ
impl_Mul!{[-3,1,1,-2], [0,0,0,1]}// ThermalConductivity/Θ·Θ
impl_Mul!{[1,0,0,0], [-3,2,1,0]}// Time·Power
impl_Mul!{[-1,-3,1,0],[-2,2,0,-1]} // VolumetricMassRate·SpecificHeatCapacity=VolumetricPowerCapacity
impl_Mul!{[-3,0,1,-4], [0,0,0,4]} // σ·Θ⁴=EnergyFluxDensity
macro_rules! impl_Mul_ { ([$a0:literal,$a1:literal,$a2:literal,$a3:literal], [$b0:literal,$b1:literal,$b2:literal,$b3:literal]) => {
    impl_Mul!{[$a0,$a1,$a2,$a3], [$b0,$b1,$b2,$b3]}
    impl NotDimensionless for Quantity<{$a0+$b0},{$a1+$b1},{$a2+$b2},{$a3+$b3}> {}
}}
impl_Mul_!{[-1,2,1,0],[-1,1,0,0]} // h·c
impl_Mul_!{[2,-3,-1,0], [2,-3,-1,0]}// 1/(h·c)·1/(h·c)
impl_Mul_!{[4,-6,-2,0], [2,-3,-1,0]}// (1/(h·c))²·1/(h·c)
impl_Mul_!{[-1,1,0,0], [6,-9,-3,0]}// c·(1/(h·c))³
impl_Mul_!{[-2,2,1,-1],[-2,2,1,-1]} // k·k
impl_Mul_!{[-4,4,2,-2],[-2,2,1,-1]} // k²·k
impl_Mul_!{[-6,6,3,-3],[-2,2,1,-1]} // k³·k
impl_Mul_!{[5,-8,-3,0], [-8,8,4,-4]}// (c·(1/(h·c))³)·k⁴
impl_Mul_!{[0,0,0,1], [0,0,0,1]} // Θ·Θ
impl_Mul_!{[0,0,0,2], [0,0,0,1]} // Θ²·Θ
impl_Mul_!{[0,0,0,2], [0,0,0,2]} // Θ²·Θ²

impl<B: Float, const A0: int, const A1: int, const A2: int, const A3: int> std::ops::Mul<B> for Quantity<A0,A1,A2,A3> where Self:Mul<B> {
    type Output = <Self as Mul<B>>::Output;
    fn mul(self, b: B) -> Self::Output { Self::Output::wrap(self.0*b.unwrap()) }
}

// quantity / quantity
#[const_trait] pub trait Div<Q> { type Output : ~const Float; }
impl<Q> const Div<Q> for Q { type Output = Quantity<0,0,0,0>; } // Q/Q=1
impl<Q:~const Float+NotDimensionless> const Div<Quantity<0,0,0,0>> for Q { type Output = Q; } // Q/1=Q
//impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> Div<Quantity<A0,A1,A2,A3>> for Quantity<0,0,0,0> { type Output = Quantity<{-A0},{-A1},{-A2},{-A3}>; } // 1/Q conflicts with Q/Q, Q/1 for Q=1
macro_rules! impl_Div { ([$a0:literal,$a1:literal,$a2:literal,$a3:literal], [$b0:literal,$b1:literal,$b2:literal,$b3:literal]) => {
    impl const Div<Quantity<$b0,$b1,$b2,$b3>> for Quantity<$a0,$a1,$a2,$a3> { type Output = Quantity<{$a0-$b0},{$a1-$b1},{$a2-$b2},{$a3-$b3}>; } } }
impl_Div!{[0,0,0,0], [0,0,0,1]}// 1/Temperature
impl_Div!{[0,0,0,0], [-1,0,0,0]}// 1/(1/Time)
impl_Div!{[0,0,0,0], [-2,3,1,0]}// 1/(h·c)
impl_Div!{[0,2,0,0], [-1,2,0,0]}// Length²/Diffusivity=Time
impl_Div!{[-1,2,0,0], [0,2,0,0]} // Diffusivity/Length²=1/Time
impl_Div!{[-1,-1,1,0], [0,-3,1,0]}// DynamicViscosity/MassDensity=Diffusivity
impl_Div!{[-2,2,1,0], [-2,2,1,-1]}// Energy/HeatCapacity=Temperature
impl_Div!{[-3,1,1,-1], [-2,-1,1,-1]}// ThermalConductivity/VolumetricHeatCapacity=Diffusivity
impl_Div!{[-3,2,1,0], [0,2,0,0]}// Power/Area=EnergyFluxDensity
impl_Div!{[-3,0,1,0], [-3,1,1,-1]}// EnergyFluxDensity/ThermalConductivity=TemperatureGradient
impl_Div!{[-3,-1,1,-1], [-2,-1,1,-1]}// VolumetricPowerCapacity/VolumetricHeatCapacity
impl_Div!{[-3,-1,1,0], [-2,-1,1,-1]}// VolumetricPowerDensity/VolumetricHeatCapacity=Θ/T
//impl_Div!{[-1,2,1,0],[-2,2,0,0]} // h/c²
//impl_Div!{[-2,2,1,-1],[-1,2,1,0]} // k/h

impl<B:~const Float, const A0: int, const A1: int, const A2: int, const A3: int> /*const*/ std::ops::Div<B> for Quantity<A0,A1,A2,A3> where Self:Div<B> {
    type Output = <Self as Div<B>>::Output;
    fn div(self, b: B) -> Self::Output { Self::Output::wrap(self.0/b.unwrap()) }
}

pub type Dimensionless = Quantity<0,0,0,0>;
impl Dimensionless {
    pub fn unitless(self) -> f64 { self.0 }
    pub fn f32(self) -> f32 { self.0 as f32 }
}
impl<B> std::cmp::PartialEq<B> for Dimensionless where f64:std::cmp::PartialEq<B> { fn eq(&self, other: &B) -> bool { self.0.eq(other) }}
impl<B> std::cmp::PartialOrd<B> for Dimensionless where f64:std::cmp::PartialOrd<B> { fn partial_cmp(&self, other: &B) -> Option<std::cmp::Ordering> { self.0.partial_cmp(other) }}
impl std::fmt::Display for Dimensionless { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) } }

// Dimensionless · quantity
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Mul<Quantity<A0,A1,A2,A3>> for f64 where Quantity<A0,A1,A2,A3>:NotDimensionless {
    type Output = Quantity<A0,A1,A2,A3>;
    fn mul(self, b: Self::Output) -> Self::Output { Dimensionless::wrap(self)*b }
}

// quantity · Dimensionless
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Mul<f64> for Quantity<A0,A1,A2,A3> where Quantity<A0,A1,A2,A3>:NotDimensionless {
    type Output = Quantity<A0,A1,A2,A3>;
    fn mul(self, b: f64) -> Self::Output { self*Dimensionless::wrap(b) }
}

// quantity / Dimensionless
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> /*const*/ std::ops::Div<f64> for Quantity<A0,A1,A2,A3> where Self:NotDimensionless {
    type Output = Self;
    fn div(self, b: f64) -> Self { self/Dimensionless::wrap(b) }
}

// Dimensionless / quantity
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Div<Quantity<A0,A1,A2,A3>> for f64 where Dimensionless:Div<Quantity<A0,A1,A2,A3>> {
    type Output = <Dimensionless as Div<Quantity<A0,A1,A2,A3>>>::Output;
    fn div(self, b: Quantity<A0,A1,A2,A3>) -> Self::Output { Dimensionless::wrap(self)/b }
}

// f64 · Dimensionless
impl std::ops::Mul<Dimensionless> for f64 { type Output = f64; fn mul(self, b: Dimensionless) -> Self::Output { self*b.0 } }
//  Dimensionless · f64
impl std::ops::Mul<f64> for Dimensionless { type Output = f64; fn mul(self, b: f64) -> Self::Output { self.0*b } }
// Dimensionless / f64
impl std::ops::Div<f64> for Dimensionless { type Output = f64; fn div(self, b: f64) -> Self::Output { self.0/b } }

macro_rules! quantity_unit { ( [ $($dimensions:expr),+ ] $unit:ident $quantity:ident  ) => {
        #[allow(non_camel_case_types)] pub type $quantity = Quantity<$($dimensions),+>;
        impl NotDimensionless for $quantity {}
        #[allow(dead_code,non_upper_case_globals)] pub const $unit : Unit<$quantity> = unit();
        impl $quantity { #[allow(non_snake_case)] pub fn $unit(self) -> f64 { self.0 } }
        impl std::fmt::Display for $quantity { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { fmt(stringify!($unit), self.0, f) } }
} }

// time [T], length [L], mass [M], temperature [θ]
quantity_unit!([1,0,0,0] sec Time);
quantity_unit!([0,1,0,0] m Length );
quantity_unit!([0,0,1,0] kg Mass);
quantity_unit!([0,0,0,1] K Temperature);
quantity_unit!([0,-1,0,0] _m ReciprocalLength );
quantity_unit!([0, 2,0,0] m2 Area);
quantity_unit!([0, -2,0,0] _m2 ByArea);
quantity_unit!([0, 3,0,0] m3 Volume);
quantity_unit!([-1,1,0,0] m_s Speed);
quantity_unit!([-2,1,0,0] m_s2 Acceleration);
quantity_unit!([-1,2,0,0] m2_s Diffusivity);
quantity_unit!([-1,3,0,0] m3_s FlowRate);
quantity_unit!([0,-3,1,0] kg_m3 MassDensity);
quantity_unit!([-1,-3,1,0] kg_m3s VolumetricMassRate);
quantity_unit!([-1,0,0,1] K_s TemperatureRate);
quantity_unit!([-1,2,1,0] J_Hz/*J·s*/ PlanckConstant); // h [T⁻¹L²M]
quantity_unit!([-2,-1,1,0] Pa Pressure); // T⁻²L⁻¹M
quantity_unit!([-2,2,1,0] J Energy); // T⁻²L²M
quantity_unit!([-3,2,1,0] W Power);
quantity_unit!([-3,0,1,0] W_m2 EnergyFluxDensity);
quantity_unit!([-3,0,1,-1] W_m2·K HeatTransferCoefficient);
quantity_unit!([-3,-1,1,0] W_m3 VolumetricPowerDensity);
quantity_unit!([-2,2,1,-1] J_K HeatCapacity); // cp, k
quantity_unit!([-2,2,0,-1] J_K·kg SpecificHeatCapacity);
quantity_unit!([-2,-1,1,-1] J_K·m3 VolumetricHeatCapacity);
quantity_unit!([-3,1,1,-1] W_m·K ThermalConductivity);
quantity_unit!([-1,-1,1,0] Pa·s DynamicViscosity); //kg/m/s
quantity_unit!([0,0,0,-1] _K ThermalExpansion);
quantity_unit!([-3,-1,1,-1] W_m3·K VolumetricPowerCapacity);

//pub type ThermalDiffusivity = Diffusivity; // m²/s
pub type KinematicViscosity = Diffusivity; // m²/s
pub type HeatFluxDensity = EnergyFluxDensity;
pub type HeatFlux = HeatFluxDensity;
pub type PerfusionRate = VolumetricMassRate; // kg/m³/s

quantity_unit!([-2,2,0,-2] J_K2·kg SpecificHeatCapacityPerK);
quantity_unit!([-3,1,1,-2] W_m·K2 ThermalConductivityPerK);

impl Time { #[allow(non_snake_case)] pub fn s(self) -> f64 { self.sec() } }

pub struct MegaUnit<Q>(std::marker::PhantomData<Q>);
pub const fn mega_unit<Q>() -> MegaUnit<Q> { MegaUnit(std::marker::PhantomData) }
impl<Q:~const Float> /*const*/ std::ops::BitOr<MegaUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: MegaUnit<Q>) -> Self::Output { Q::wrap(self*1e6) } }
pub const _mm2 : MegaUnit<ByArea> = mega_unit();
pub struct HectoUnit<Q>(std::marker::PhantomData<Q>);
pub const fn hecto_unit<Q>() -> HectoUnit<Q> { HectoUnit(std::marker::PhantomData) }
impl<Q:~const Float> /*const*/ std::ops::BitOr<HectoUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: HectoUnit<Q>) -> Self::Output { Q::wrap(self*1e2) } }
pub const _cm : HectoUnit<ReciprocalLength> = hecto_unit();
pub struct CentiUnit<Q>(std::marker::PhantomData<Q>);
pub const fn centi_unit<Q>() -> CentiUnit<Q> { CentiUnit(std::marker::PhantomData) }
impl<Q:~const Float> /*const*/ std::ops::BitOr<CentiUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: CentiUnit<Q>) -> Self::Output { Q::wrap(self*1e-2) } }
pub const cm : CentiUnit<Length> = centi_unit();
pub struct MilliUnit<Q>(std::marker::PhantomData<Q>);
pub const fn milli_unit<Q>() -> MilliUnit<Q> { MilliUnit(std::marker::PhantomData) }
impl<Q:~const Float> /*const*/ std::ops::BitOr<MilliUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: MilliUnit<Q>) -> Self::Output { Q::wrap(self*1e-3) } }
pub const mm : MilliUnit<Length> = milli_unit();
pub const mm_s : MilliUnit<Speed> = milli_unit();
pub struct MicroUnit<Q>(std::marker::PhantomData<Q>);
pub const fn micro_unit<Q>() -> MicroUnit<Q> { MicroUnit(std::marker::PhantomData) }
impl<Q:~const Float> /*const*/ std::ops::BitOr<MicroUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: MicroUnit<Q>) -> Self::Output { Q::wrap(self*1e-6) } }
pub const µm : MicroUnit<Length> = micro_unit();
pub struct NanoUnit<Q>(std::marker::PhantomData<Q>);
pub const fn nano_unit<Q>() -> NanoUnit<Q> { NanoUnit(std::marker::PhantomData) }
impl<Q:~const Float> /*const*/ std::ops::BitOr<NanoUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: NanoUnit<Q>) -> Self::Output { Q::wrap(self*1e-9) } }
pub const nm : NanoUnit<Length> = nano_unit();
pub struct PicoUnit<Q>(std::marker::PhantomData<Q>);
pub const fn pico_unit<Q>() -> PicoUnit<Q> { PicoUnit(std::marker::PhantomData) }
impl<Q:~const Float> /*const*/ std::ops::BitOr<PicoUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: PicoUnit<Q>) -> Self::Output { Q::wrap(self*1e-12) } }
pub const µm2 : PicoUnit<Area> = pico_unit();

pub struct Celsius;
impl /*const*/ std::ops::BitOr<Celsius> for f64 { type Output = Temperature; fn bitor(self, _: Celsius) -> Self::Output { Temperature::wrap(273.15+self) } }
pub const C : Celsius = Celsius;

pub trait System { type Scalar<T: PartialEq+Clone> : PartialEq+Clone; }
#[derive(PartialEq,Clone)] pub struct Dimensionalized; //FIXME: derive should not be required here
impl System for Dimensionalized { type Scalar<T: PartialEq+Clone> = T; }
#[derive(PartialEq,Clone)] pub struct NonDimensionalized; //FIXME: derive should not be required here
impl System for NonDimensionalized { type Scalar<T: PartialEq+Clone> = f32; }