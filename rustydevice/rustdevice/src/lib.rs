extern crate device_sys;
use device_sys::{memory_allocate_device, memory_deallocate_device, transfer, bessel_func};
use num_complex::Complex;
use std::ffi::c_void;
use std::mem::size_of;
use std::{clone::Clone, default::Default, fmt::Display};

pub struct DevicePtr<T: Display + Default + Clone> {
    pub ptr: *mut T,
    pub size: i32,
}

impl<T: Display + Default + Clone> DevicePtr<T> {
    pub fn new(sz: i32) -> Self {
        // println!("Memory getting allocated in Rust\n");
        Self {
            ptr: unsafe { memory_allocate_device(sz * (size_of::<T>() as i32)) } as *mut T,
            size: sz,
        }
    }

    pub fn display(&self, count: i32) {
        let right_num = {
            if count < self.size {
                count
            } else {
                self.size
            }
        };
        unsafe {
            let mut z = vec![T::default(); right_num as usize];
            transfer(
                z.as_mut_ptr() as *mut c_void,
                self.ptr as *mut c_void,
                right_num,
            );
            for ind in 0..(right_num as usize) {
                println!("{}\n", z[ind]);
            }
        };
    }

    pub fn dump(&self, output: &mut [T])
    {
        unsafe{transfer(output.as_mut_ptr() as *mut c_void, self.ptr as *mut c_void, self.size * (size_of::<T>() as i32))}
    }
}

impl<T: Display + Default + Clone> Drop for DevicePtr<T> {
    fn drop(&mut self) {
        // println!("Memory getting deallocated in Rust");
        unsafe { memory_deallocate_device(self.ptr as *mut c_void) };
    }
}

pub fn compute_bessel(inp: f32) -> f32
{
    unsafe{bessel_func(inp)}
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut device_ptr = DevicePtr::<Complex<f64>>::new(100);
        device_ptr.display(10);
    }

    fn others() {
        todo!();
    }
}
