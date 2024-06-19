use cc;
fn main() {
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .cuda(true)
        .cudart("shared")
        .file("../device/src/device_C_interface.cu")
        .flag("-gencode")
        .flag("arch=compute_75, code=sm_80")
        .compile("libgpu.a");
    println!("cargo:rustc-link-search=native=/usr/local/cuda-11.8/lib64");
    println!("cargo:rustc-link-lib=dylib=cufft");
}