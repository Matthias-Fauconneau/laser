#![allow(dead_code,non_upper_case_globals)]
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

impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Sub for Quantity<A0,A1,A2,A3> {
    type Output = Self;
    fn sub(self, b: Self) -> Self::Output { Self(self.0.sub(b.0)) }
}

pub struct Unit<Q>(std::marker::PhantomData<Q>);
pub const fn unit<Q>() -> Unit<Q> { Unit(std::marker::PhantomData) }
impl<Q:~const Float> const std::ops::BitOr<Unit<Q>> for f64 { type Output = Q; fn bitor(self, _: Unit<Q>) -> Self::Output { Q::wrap(self) } }

// quantity · quantity
pub trait Mul<Q> { type Output : Float; }
impl<Q:Float> Mul<Quantity<0,0,0,0>> for Q { type Output = Q; }
impl<Q:Float+NotUnitless> Mul<Q> for Quantity<0,0,0,0> { type Output = Q; }
impl Mul<Quantity<1,0,0,0>> for Quantity<-1,0,0,0> { type Output = Quantity<0,0,0,0>; } // 1/T·T
impl Mul<Quantity<-1,0,0,0>> for Quantity<1,0,0,0> { type Output = Quantity<0,0,0,0>; } // T·1/T
impl Mul<Quantity<0,1,0,0>> for Quantity<0,-1,0,0> { type Output = Quantity<0,0,0,0>; } // 1/L·L
impl Mul<Quantity<0,1,0,0>> for Quantity<0,1,0,0> { type Output = Quantity<0,2,0,0>; } // L·L
impl Mul<Quantity<0,1,0,0>> for Quantity<0,2,0,0> { type Output = Quantity<0,3,0,0>; } // L·L²
impl Mul<Quantity<-1,-1,0,0>> for Quantity<0,1,0,0> { type Output = Quantity<-1,0,0,0>; } // Length·1/LT=Rate
//impl Mul<Quantity<-1,2,0,0>> for Quantity<-1,2,0,0> { type Output = Quantity<-2,4,0,0>; } // Diffusivity²
//impl Mul<Quantity<0,3,0,0>> for Quantity<-2,1,0,0> { type Output = Quantity<-2,4,0,0>; } // Volume·Acceleration
impl Mul<Quantity<0,0,0,-1>> for Quantity<0,0,0,1> { type Output = Quantity<0,0,0,0>; } // 1/K·K
impl Mul<Quantity<0,3,0,0>> for Quantity<0,-3,1,0> { type Output = Quantity<0,0,1,0>; } // MassDensity·Volume
impl Mul<Quantity<0,-1,0,1>> for Quantity<0,1,0,0> { type Output = Quantity<0,0,0,1>; } // Length*TemperatureGradient=Temperature
impl Mul<Quantity<-2,2,0,-1>> for Quantity<0,0,1,0> { type Output = Quantity<-2,2,1,-1>; } // Mass·SpecificHeatCapacity
impl Mul<Quantity<-2,2,0,-1>> for Quantity<0,-3,1,0> { type Output = Quantity<-2,-1,1,-1>; } // MassDensity·SpecificHeatCapacity=VolumetricHeatCapacity
impl Mul<Quantity<0,0,0,1>> for Quantity<-2,2,0,-2> { type Output = Quantity<-2,2,0,-1>; } // SpecificHeatCapacity/K·K
impl Mul<Quantity<0,0,0,1>> for Quantity<-3,1,1,-2> { type Output = Quantity<-3,1,1,-1>; } // ThermalConductivity/K·K
impl Mul<Quantity<-3,2,1,0>> for Quantity<1,0,0,0> { type Output = Quantity<-2,2,1,0>; } // Power·Time
impl Mul<Quantity<0,2,0,0>> for Quantity<-1,1,0,0> { type Output = Quantity<-1,3,0,0>; } // Speed·Area=FlowRate
impl Mul<Quantity<0,-2,0,0>> for Quantity<-1,3,0,0> { type Output = Quantity<-1,1,0,0>; } // FlowRate·ByArea=Speed
impl Mul<Quantity<-1,1,0,0>> for Quantity<-2,-1,1,-1> { type Output = Quantity<-3,0,1,-1>; } // VolumetricHeatCapacity·Speed=HeatFluxDensity/Temperature
impl Mul<Quantity<0,0,0,1>> for Quantity<-3,0,1,-1> { type Output = Quantity<-3,0,1,0>; } // (HeatFluxDensity/Temperature)·Temperature=HeatFluxDensity
impl Mul<Quantity<-1,-1,0,1>> for Quantity<0,1,0,0> { type Output = Quantity<-1,0,0,1>; } // Length·Θ/LT=TemperatureRate
impl Mul<Quantity<-1,0,0,1>> for Quantity<1,0,0,0> { type Output = Quantity<0,0,0,1>; } // Time·TemperatureRate=Temperature

impl<B: Float, const A0: int, const A1: int, const A2: int, const A3: int> std::ops::Mul<B> for Quantity<A0,A1,A2,A3> where Self:Mul<B> {
    type Output = <Self as Mul<B>>::Output;
    fn mul(self, b: B) -> Self::Output { Self::Output::wrap(self.0*b.unwrap()) }
}

// quantity / quantity
pub trait Div<Q> { type Output : Float; }
impl<Q> Div<Q> for Q { type Output = Quantity<0,0,0,0>; } // Q/Q=1
impl<Q:Float+NotUnitless> Div<Quantity<0,0,0,0>> for Q { type Output = Q; } // Q/1=Q
//impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> Div<Quantity<A0,A1,A2,A3>> for Quantity<0,0,0,0> { type Output = Quantity<{-A0},{-A1},{-A2},{-A3}>; } // 1/Q
impl Div<Quantity<0,0,0,1>> for Quantity<0,0,0,0> { type Output = Quantity<0,0,0,-1>; } // 1/Temperature
impl Div<Quantity<-1,0,0,0>> for Quantity<0,0,0,0> { type Output = Quantity<1,0,0,0>; } // 1/(1/Time)
impl Div<Quantity<-1,2,0,0>> for Quantity<0,2,0,0> { type Output = Quantity<1,0,0,0>; } // Length²/Diffusivity=Time
impl Div<Quantity<0,2,0,0>> for Quantity<-1,2,0,0> { type Output = Quantity<-1,0,0,0>; } // Diffusivity/Length²=1/Time
impl Div<Quantity<0,-3,1,0>> for Quantity<-1,-1,1,0> { type Output = Quantity<-1,2,0,0>; } // DynamicViscosity/MassDensity=Diffusivity
impl Div<Quantity<-2,2,1,-1>> for Quantity<-2,2,1,0> { type Output = Quantity<0,0,0,1>; } // Energy/HeatCapacity=Temperature
impl Div<Quantity<-2,-1,1,-1>> for Quantity<-3,1,1,-1> { type Output = Quantity<-1,2,0,0>; } // ThermalConductivity/VolumetricHeatCapacity=Diffusivity
impl Div<Quantity<0,2,0,0>> for Quantity<-3,2,1,0> { type Output = Quantity<-3,0,1,0>; } // Power/Area=EnergyFluxDensity
impl Div<Quantity<-3,1,1,-1>> for Quantity<-3,0,1,0> { type Output = Quantity<0,-1,0,1>; } // EnergyFluxDensity/ThermalConductivity=TemperatureGradient
//impl Div<Quantity<-2,-1,1,-1>> for Quantity<-3,0,1,0> { type Output = Quantity<-1,-1,0,1>; } // HeatFluxDensity/VolumetricHeatCapacity=Θ/LT
impl Div<Quantity<-2,-1,1,-1>> for Quantity<-3,0,1,-1> { type Output = Quantity<-1,-1,0,0>; } // (HeatFluxDensity/Θ)/VolumetricHeatCapacity=1/LT

impl<B:Float, const A0: int, const A1: int, const A2: int, const A3: int> std::ops::Div<B> for Quantity<A0,A1,A2,A3> where Self:Div<B> {
    type Output = <Self as Div<B>>::Output;
    fn div(self, b: B) -> Self::Output { Self::Output::wrap(self.0/b.unwrap()) }
}

pub type Unitless = Quantity<0,0,0,0>;
impl Unitless{
    pub fn unitless(self) -> f64 { self.0 }
    pub fn f32(self) -> f32 { self.0 as f32 }
}
impl<B> std::cmp::PartialEq<B> for Unitless where f64:std::cmp::PartialEq<B> { fn eq(&self, other: &B) -> bool { self.0.eq(other) }}
impl<B> std::cmp::PartialOrd<B> for Unitless where f64:std::cmp::PartialOrd<B> { fn partial_cmp(&self, other: &B) -> Option<std::cmp::Ordering> { self.0.partial_cmp(other) }}
impl std::fmt::Display for Unitless { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) } }

// unitless · quantity
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Mul<Quantity<A0,A1,A2,A3>> for f64 where Quantity<A0,A1,A2,A3>:NotUnitless {
    type Output = Quantity<A0,A1,A2,A3>;
    fn mul(self, b: Quantity<A0,A1,A2,A3>) -> Self::Output { Unitless::wrap(self)*b }
}

// quantity · unitless
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Mul<f64> for Quantity<A0,A1,A2,A3> where Quantity<A0,A1,A2,A3>:NotUnitless {
    type Output = Quantity<A0,A1,A2,A3>;
    fn mul(self, b: f64) -> Self::Output { self*Unitless::wrap(b) }
}

// quantity / unitless
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Div<f64> for Quantity<A0,A1,A2,A3> where Self:NotUnitless {
    type Output = Self;
    fn div(self, b: f64) -> Self { self/Unitless::wrap(b) }
}

// unitless / quantity
impl<const A0 : int, const A1 : int, const A2 : int, const A3 : int> std::ops::Div<Quantity<A0,A1,A2,A3>> for f64 where Unitless:Div<Quantity<A0,A1,A2,A3>> {
    type Output = <Unitless as Div<Quantity<A0,A1,A2,A3>>>::Output;
    fn div(self, b: Quantity<A0,A1,A2,A3>) -> Self::Output { Unitless::wrap(self)/b }
}

// f64 · unitless
impl std::ops::Mul<Unitless> for f64 { type Output = f64; fn mul(self, b: Unitless) -> Self::Output { self*b.0 } }
//  unitless · f64
impl std::ops::Mul<f64> for Unitless { type Output = f64; fn mul(self, b: f64) -> Self::Output { self.0*b } }
// unitless / f64
impl std::ops::Div<f64> for Unitless { type Output = f64; fn div(self, b: f64) -> Self::Output { self.0/b } }

pub trait NotUnitless {}
macro_rules! quantity_unit { ( [ $($dimensions:expr),+ ] $unit:ident $quantity:ident  ) => {
        #[allow(non_camel_case_types)] pub type $quantity = Quantity<$($dimensions),+>;
        impl NotUnitless for $quantity {}
        #[allow(dead_code,non_upper_case_globals)] pub const $unit : Unit<$quantity> = unit();
        impl $quantity { #[allow(non_snake_case)] pub fn $unit(self) -> f64 { self.0 } }
        impl std::fmt::Display for $quantity { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if self.0 == 0. { write!(f,"0") } else {
                let log10 = f64::log10(f64::abs(self.0) as f64);
                let floor1000 = f64::floor(log10/3.); // submultiple
                let part1000 = num::exp10(log10 - floor1000*3.); // remaining magnitude part within the submultiple: x / 1000^⌊log1000(x)⌋
                let submagnitude = if part1000 < 1. { format!("{:.1}", part1000) } else { (f64::round(part1000) as u32).to_string() };
                assert!(f64::clamp(-3., floor1000, 3.) == floor1000, "{floor1000}");
                let submultiple = ["n","µ","m","","k","M","G"][(3+(floor1000 as i8)) as usize];
                write!(f, concat!("{}{}{}", stringify!($unit)), if self.0<0. {"-"} else {""}, submagnitude, submultiple)
            }
    }}
} }

// time [T], length [L], mass [M], temperature [θ]
quantity_unit!([1,0,0,0] second Time);
quantity_unit!([0,1,0,0] m Length );
quantity_unit!([0,0,1,0] kg Mass);
quantity_unit!([0,0,0,1] K Temperature);
quantity_unit!([0,-1,0,0] _m ByLength );
quantity_unit!([0, 2,0,0] m2 Area);
quantity_unit!([0, -2,0,0] _m2 ByArea);
quantity_unit!([0, 3,0,0] m3 Volume);
quantity_unit!([-1,1,0,0] m_s Speed);
quantity_unit!([-2,1,0,0] m_s2 Acceleration);
quantity_unit!([-1,2,0,0] m2_s Diffusivity);
quantity_unit!([-1,3,0,0] m3_s FlowRate);
quantity_unit!([0,-3,1,0] kg_m3 MassDensity);
quantity_unit!([-2,2,1,0] J Energy); //T⁻²L²M
quantity_unit!([-3,2,1,0] W Power); // J/s
quantity_unit!([-3,0,1,0] W_m2 EnergyFluxDensity);
quantity_unit!([-2,2,1,-1] J_K HeatCapacity);
quantity_unit!([-2,2,0,-1] J_K·kg SpecificHeatCapacity);
quantity_unit!([-2,-1,1,-1] J_K·m3 VolumetricHeatCapacity);
quantity_unit!([-3,1,1,-1] W_m·K ThermalConductivity);
quantity_unit!([-1,-1,1,0] Pa·s DynamicViscosity); //kg/m/s
quantity_unit!([0,0,0,-1] _K ThermalExpansion);

pub type ThermalDiffusivity = Diffusivity; // m²/s
//pub type KinematicViscosity = Diffusivity; // m²/s
pub type FluxDensity = Speed;
pub type HeatFluxDensity = EnergyFluxDensity;

quantity_unit!([-2,2,0,-2] J_K2·kg SpecificHeatCapacity_K);
quantity_unit!([-3,1,1,-2] W_m·K2 ThermalConductivity_K);

pub struct MegaUnit<Q>(std::marker::PhantomData<Q>);
pub const fn mega_unit<Q>() -> MegaUnit<Q> { MegaUnit(std::marker::PhantomData) }
impl<Q:Float> std::ops::BitOr<MegaUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: MegaUnit<Q>) -> Self::Output { Q::wrap(self*1e6) } }
pub const _mm2 : MegaUnit<ByArea> = mega_unit();
pub struct HectoUnit<Q>(std::marker::PhantomData<Q>);
pub const fn hecto_unit<Q>() -> HectoUnit<Q> { HectoUnit(std::marker::PhantomData) }
impl<Q:Float> std::ops::BitOr<HectoUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: HectoUnit<Q>) -> Self::Output { Q::wrap(self*1e2) } }
pub const _cm : HectoUnit<ByLength> = hecto_unit();
pub struct CentiUnit<Q>(std::marker::PhantomData<Q>);
pub const fn centi_unit<Q>() -> CentiUnit<Q> { CentiUnit(std::marker::PhantomData) }
impl<Q:Float> std::ops::BitOr<CentiUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: CentiUnit<Q>) -> Self::Output { Q::wrap(self*1e-2) } }
pub const cm : CentiUnit<Length> = centi_unit();
pub struct MilliUnit<Q>(std::marker::PhantomData<Q>);
pub const fn milli_unit<Q>() -> MilliUnit<Q> { MilliUnit(std::marker::PhantomData) }
impl<Q:Float> std::ops::BitOr<MilliUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: MilliUnit<Q>) -> Self::Output { Q::wrap(self*1e-3) } }
pub const mm : MilliUnit<Length> = milli_unit();
pub const mm_s : MilliUnit<Speed> = milli_unit();
pub struct MicroUnit<Q>(std::marker::PhantomData<Q>);
pub const fn micro_unit<Q>() -> MicroUnit<Q> { MicroUnit(std::marker::PhantomData) }
impl<Q:Float> std::ops::BitOr<MicroUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: MicroUnit<Q>) -> Self::Output { Q::wrap(self*1e-6) } }
pub const µm : MicroUnit<Length> = micro_unit();
pub struct NanoUnit<Q>(std::marker::PhantomData<Q>);
pub const fn nano_unit<Q>() -> NanoUnit<Q> { NanoUnit(std::marker::PhantomData) }
impl<Q:Float> std::ops::BitOr<NanoUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: NanoUnit<Q>) -> Self::Output { Q::wrap(self*1e-9) } }
pub const nm : NanoUnit<Length> = nano_unit();
pub struct PicoUnit<Q>(std::marker::PhantomData<Q>);
pub const fn pico_unit<Q>() -> PicoUnit<Q> { PicoUnit(std::marker::PhantomData) }
impl<Q:Float> std::ops::BitOr<PicoUnit<Q>> for f64 { type Output = Q; fn bitor(self, _: PicoUnit<Q>) -> Self::Output { Q::wrap(self*1e-12) } }
pub const µm2 : PicoUnit<Area> = pico_unit();

pub struct Celsius;
impl const std::ops::BitOr<Celsius> for f64 { type Output = Temperature; fn bitor(self, _: Celsius) -> Self::Output { Temperature::wrap(273.15+self) } }
pub const C : Celsius = Celsius;

pub trait System { type Scalar<T: PartialEq+Clone> : PartialEq+Clone; }
#[derive(PartialEq,Clone)] pub struct Dimensionalized; //FIXME: derive should not be required here
impl System for Dimensionalized { type Scalar<T: PartialEq+Clone> = T; }
#[derive(PartialEq,Clone)] pub struct NonDimensionalized; //FIXME: derive should not be required here
impl System for NonDimensionalized { type Scalar<T: PartialEq+Clone> = f32; }