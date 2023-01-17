/// Quantities and units operators

pub trait F32 {
    fn unwrap(self) -> f32;
    fn wrap(value : f32) -> Self;
}

#[derive(PartialEq,Clone,Copy,Debug)] pub struct Quantity<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32>(f32);
/*impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::fmt::Display for Quantity<A0,A1,A2,A3> { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", self)
} }*/
impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> F32 for Quantity<A0,A1,A2,A3>{
    fn unwrap(self) -> f32 { self.0 }
    fn wrap(value : f32) -> Self { Self(value) }
}

impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::ops::Add for Quantity<A0,A1,A2,A3> {
    type Output = Self;
    fn add(self, b: Self) -> Self::Output { Self::Output::wrap(self.unwrap().add(b.unwrap())) }
}

impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::ops::Sub for Quantity<A0,A1,A2,A3> {
    type Output = Self;
    fn sub(self, b: Self) -> Self::Output { Self::Output::wrap(self.unwrap().sub(b.unwrap())) }
}

pub struct Unit<Q>(std::marker::PhantomData<Q>);
pub const fn unit<Q>() -> Unit<Q> { Unit(std::marker::PhantomData) }
impl<Q:F32> std::ops::BitOr<Unit<Q>> for f32 { type Output = Q; fn bitor(self, _: Unit<Q>) -> Self::Output { Q::wrap(self) } }

// quantity · quantity
pub trait Mul<Q> { type Output : F32; }
impl<Q:F32> Mul<Quantity<0,0,0,0>> for Q { type Output = Q; }
impl<Q:F32+NotUnitless> Mul<Q> for Quantity<0,0,0,0> { type Output = Q; }
impl Mul<Quantity<1,0,0,0>> for Quantity<-1,0,0,0> { type Output = Quantity<0,0,0,0>; } // 1/T·T
//impl Mul<Quantity<-1,0,0,0>> for Quantity<1,0,0,0> { type Output = Quantity<0,0,0,0>; } // T·1/T
impl Mul<Quantity<0,1,0,0>> for Quantity<0,-1,0,0> { type Output = Quantity<0,0,0,0>; } // 1/L·L
impl Mul<Quantity<0,1,0,0>> for Quantity<0,1,0,0> { type Output = Quantity<0,2,0,0>; } // L·L
impl Mul<Quantity<0,1,0,0>> for Quantity<0,2,0,0> { type Output = Quantity<0,3,0,0>; } // L·L²
//impl Mul<Quantity<-1,2,0,0>> for Quantity<-1,2,0,0> { type Output = Quantity<-2,4,0,0>; } // Diffusivity²
//impl Mul<Quantity<0,3,0,0>> for Quantity<-2,1,0,0> { type Output = Quantity<-2,4,0,0>; } // Volume·Acceleration
impl Mul<Quantity<0,0,0,-1>> for Quantity<0,0,0,1> { type Output = Quantity<0,0,0,0>; } // 1/K·K
impl Mul<Quantity<0,3,0,0>> for Quantity<0,-3,1,0> { type Output = Quantity<0,0,1,0>; } // MassDensity·Volume
impl Mul<Quantity<-2,2,0,-1>> for Quantity<0,0,1,0> { type Output = Quantity<-2,2,1,-1>; } // Mass·SpecificHeatCapacity
impl Mul<Quantity<-2,2,0,-1>> for Quantity<0,-3,1,0> { type Output = Quantity<-2,-1,1,-1>; } // MassDensity·SpecificHeatCapacity=VolumetricHeatCapacity
impl Mul<Quantity<0,0,0,1>> for Quantity<-2,2,0,-2> { type Output = Quantity<-2,2,0,-1>; } // SpecificHeatCapacity/K·K
impl Mul<Quantity<0,0,0,1>> for Quantity<-3,1,1,-2> { type Output = Quantity<-3,1,1,-1>; } // ThermalConductivity/K·K
impl Mul<Quantity<-3,2,1,0>> for Quantity<1,0,0,0> { type Output = Quantity<-2,2,1,0>; } // Power·Time

impl<B : F32, const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::ops::Mul<B> for Quantity<A0,A1,A2,A3> where Self:Mul<B> {
    type Output = <Self as Mul<B>>::Output;
    fn mul(self, b: B) -> Self::Output { Self::Output::wrap(self.unwrap()*b.unwrap()) }
}

// quantity / quantity
pub trait Div<Q> { type Output : F32; }
impl<Q> Div<Q> for Q { type Output = Quantity<0,0,0,0>; } // Q/Q=1
impl<Q:F32+NotUnitless> Div<Quantity<0,0,0,0>> for Q { type Output = Q; } // Q/1=Q
//impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> Div<Quantity<A0,A1,A2,A3>> for Quantity<0,0,0,0> { type Output = Quantity<{-A0},{-A1},{-A2},{-A3}>; } // 1/Q
impl Div<Quantity<0,0,0,1>> for Quantity<0,0,0,0> { type Output = Quantity<0,0,0,-1>; } // 1/Temperature
impl Div<Quantity<-1,0,0,0>> for Quantity<0,0,0,0> { type Output = Quantity<1,0,0,0>; } // 1/(1/Time)
impl Div<Quantity<-1,2,0,0>> for Quantity<0,2,0,0> { type Output = Quantity<1,0,0,0>; } // Length²/Diffusivity=Time
impl Div<Quantity<0,2,0,0>> for Quantity<-1,2,0,0> { type Output = Quantity<-1,0,0,0>; } // Diffusivity/Length²=1/Time
impl Div<Quantity<0,-3,1,0>> for Quantity<-1,-1,1,0> { type Output = Quantity<-1,2,0,0>; } // DynamicViscosity/MassDensity=Diffusivity
impl Div<Quantity<-2,2,1,-1>> for Quantity<-2,2,1,0> { type Output = Quantity<0,0,0,1>; } // Energy/HeatCapacity=Temperature
impl Div<Quantity<-2,-1,1,-1>> for Quantity<-3,1,1,-1> { type Output = Quantity<-1,2,0,0>; } // ThermalConductivity/VolumetricHeatCapacity=Diffusivity

impl<B:F32, const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::ops::Div<B> for Quantity<A0,A1,A2,A3> where Self:Div<B> {
    type Output = <Self as Div<B>>::Output;
    fn div(self, b: B) -> Self::Output { Self::Output::wrap(self.unwrap()/b.unwrap()) }
}

pub type Unitless = Quantity<0,0,0,0>;
impl From<Unitless> for f32 { fn from(v: Unitless) -> Self { v.unwrap() } }
impl std::fmt::Display for Unitless { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.unwrap()) } }

// unitless · quantity
impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::ops::Mul<Quantity<A0,A1,A2,A3>> for f32 where Quantity<A0,A1,A2,A3>:NotUnitless {
    type Output = Quantity<A0,A1,A2,A3>;
    fn mul(self, b: Quantity<A0,A1,A2,A3>) -> Self::Output { Unitless::wrap(self)*b }
}

// quantity · unitless
impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::ops::Mul<f32> for Quantity<A0,A1,A2,A3> where Quantity<A0,A1,A2,A3>:NotUnitless {
    type Output = Quantity<A0,A1,A2,A3>;
    fn mul(self, b: f32) -> Self::Output { self*Unitless::wrap(b) }
}

// quantity / unitless
impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::ops::Div<f32> for Quantity<A0,A1,A2,A3> where Self:NotUnitless {
    type Output = Self;
    fn div(self, b: f32) -> Self { self/Unitless::wrap(b) }
}

// unitless / quantity
impl<const A0 : i32, const A1 : i32, const A2 : i32, const A3 : i32> std::ops::Div<Quantity<A0,A1,A2,A3>> for f32 where Unitless:Div<Quantity<A0,A1,A2,A3>> {
    type Output = <Unitless as Div<Quantity<A0,A1,A2,A3>>>::Output;
    fn div(self, b: Quantity<A0,A1,A2,A3>) -> Self::Output { Unitless::wrap(self)/b }
}

// f32 · unitless
impl std::ops::Mul<Unitless> for f32 { type Output = f32; fn mul(self, b: Unitless) -> Self::Output { self*b.unwrap() } }
//  unitless · f32
impl std::ops::Mul<f32> for Unitless { type Output = f32; fn mul(self, b: f32) -> Self::Output { self.unwrap()*b } }
// unitless / f32
impl std::ops::Div<f32> for Unitless { type Output = f32; fn div(self, b: f32) -> Self::Output { self.unwrap()/b } }

pub trait NotUnitless {}
macro_rules! quantity_unit { ( [ $($dimensions:expr),+ ] $unit:ident $quantity:ident  ) => {
        #[allow(non_camel_case_types)] pub type $quantity = Quantity<$($dimensions),+>;
        impl NotUnitless for $quantity {}
        #[allow(dead_code,non_upper_case_globals)] pub const $unit : Unit<$quantity> = unit();
        impl $quantity { #[allow(non_snake_case)] pub fn $unit(self) -> f32 { self.unwrap() } }
        impl std::fmt::Display for $quantity { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, concat!("{}", stringify!($unit)), self.unwrap())
        } }
} }

// time [T], length [L], mass [M], temperature [θ]
quantity_unit!([1,0,0,0] second Time);
quantity_unit!([0,1,0,0] m Length );
quantity_unit!([0,0,1,0] kg Mass);
quantity_unit!([0,0,0,1] K Temperature);
quantity_unit!([0,-1,0,0] _m ByLength );
quantity_unit!([0, 2,0,0] m2 Area);
quantity_unit!([0, 3,0,0] m3 Volume);
quantity_unit!([-2,1,0,0] m_s2 Acceleration);
quantity_unit!([-1,2,0,0] m2_s Diffusivity);
quantity_unit!([0,-3,1,0] kg_m3 MassDensity);
quantity_unit!([-2,2,1,0] J Energy); //T⁻²L²M
quantity_unit!([-3,2,1,0] W Power); // J/s
quantity_unit!([-2,2,1,-1] J_K HeatCapacity);
quantity_unit!([-2,2,0,-1] J_K·kg SpecificHeatCapacity);
quantity_unit!([-2,-1,1,-1] J_K·m3 VolumetricHeatCapacity);
quantity_unit!([-3,1,1,-1] W_m·K ThermalConductivity);
quantity_unit!([-1,-1,1,0] Pa·s DynamicViscosity); //kg/m/s
quantity_unit!([0,0,0,-1] _K ThermalExpansion);

pub type ThermalDiffusivity = Diffusivity; // m²/s
//pub type KinematicViscosity = Diffusivity; // m²/s

quantity_unit!([-2,2,0,-2] J_K2·kg SpecificHeatCapacity_K);
quantity_unit!([-3,1,1,-2] W_m·K2 ThermalConductivity_K);

pub struct CentiUnit<Q>(std::marker::PhantomData<Q>);
pub const fn centi_unit<Q>() -> CentiUnit<Q> { CentiUnit(std::marker::PhantomData) }
impl<Q:F32> std::ops::BitOr<CentiUnit<Q>> for f32 { type Output = Q; fn bitor(self, _: CentiUnit<Q>) -> Self::Output { Q::wrap(self*1e-2) } }
#[allow(dead_code,non_upper_case_globals)] pub const cm : CentiUnit<Length> = centi_unit();
pub struct MilliUnit<Q>(std::marker::PhantomData<Q>);
pub const fn milli_unit<Q>() -> MilliUnit<Q> { MilliUnit(std::marker::PhantomData) }
impl<Q:F32> std::ops::BitOr<MilliUnit<Q>> for f32 { type Output = Q; fn bitor(self, _: MilliUnit<Q>) -> Self::Output { Q::wrap(self*1e-3) } }
#[allow(dead_code,non_upper_case_globals)] pub const mm : MilliUnit<Length> = milli_unit();

pub trait System { type Scalar<T: PartialEq+Clone> : PartialEq+Clone; }
#[derive(PartialEq,Clone)] pub struct Dimensionalized; //FIXME: derive should not be required here
impl System for Dimensionalized { type Scalar<T: PartialEq+Clone> = T; }
#[derive(PartialEq,Clone)] pub struct NonDimensionalized; //FIXME: derive should not be required here
impl System for NonDimensionalized { type Scalar<T: PartialEq+Clone> = f32; }