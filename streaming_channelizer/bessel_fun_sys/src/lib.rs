use num::Complex;
extern "C" {
    pub fn bessel_func(inp: f32) -> f32;
    pub fn filter_apply(
        lhs: *mut Complex<f32>,
        rhs: *mut f32,
        prod: *mut Complex<f32>,
        nsamples: usize,
    );
    pub fn multiply_complex_intrinsic(
        lhs: *mut Complex<f32>,
        rhs: *mut f32,
        prod: *mut Complex<f32>,
        nsamples: i32,
    );
    pub fn strided_copy(
        lhs: *mut Complex<f32>,
        rhs: *mut Complex<f32>,
        channels: i32,
        samples: i32,
    );
    pub fn convolve(
        lhs: *mut Complex<f32>,
        filter: *mut f32,
        output: *mut Complex<f32>,
        scratch: *mut Complex<f32>,
        progression: i32,
        slice: i32,
    );
    pub fn transpose(
        lhs: *mut Complex<f32>,
        rhs: *mut Complex<f32>,
        rows: i32,
        cols: i32,
    );
    pub fn plain_multiply(
        lhs: *mut Complex<f32>,
        rhs: *mut Complex<f32>,
        prod: *mut Complex<f32>,
        nsamples: i32, 
    );
}

// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = unsafe { bessel_func(3.0 as f32) };
    }
}
