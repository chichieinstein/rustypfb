use num::Complex;
extern "C"
{
    pub fn bessel_func(inp: f32) -> f32;
    pub fn filter_apply(lhs: *mut Complex<f32>, rhs: *mut f32, prod: *mut Complex<f32>, nsamples: usize);
}

// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
