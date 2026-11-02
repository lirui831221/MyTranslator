fn main() {
    #[cfg(target_os = "macos")]
    {
        let project = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let whisper = project.join("bin/vad-dist");
        println!("cargo:rustc-link-search=native={}", whisper.display());
        // Main binary, test binaries, and the packaged .app have different
        // executable-relative locations. Supplying all three keeps direct
        // whisper VAD linkage portable without relying on DYLD_LIBRARY_PATH.
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../../bin/vad-dist");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../../../bin/vad-dist");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Resources/bin/vad-dist");
        println!(
            "cargo:rerun-if-changed={}",
            whisper.join("libwhisper.dylib").display()
        );
    }
    tauri_build::build()
}
