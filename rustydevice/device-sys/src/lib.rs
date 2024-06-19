use std::ffi::c_void;
extern "C" {
    pub fn memory_allocate_device(inp: i32) -> *mut c_void;
    pub fn memory_deallocate_device(inp: *mut c_void);
    pub fn transfer(dst: *mut c_void, src: *mut c_void, count: i32);
    pub fn bessel_func(inp: f32)->f32;
}
#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn it_works() {
    //     let result = add(2, 2);
    //     assert_eq!(result, 4);
    // }
}
