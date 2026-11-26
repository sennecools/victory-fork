//! Embeds the executable's manifest (MSVC linker only).

// The segment heap backs the process heap and every heap made with
// HeapCreate, the title's included (its C runtime and heap calls are the
// host's). The default NT heap searched one long free list for its large
// blocks: 23 % of all CPU time in a play session.
const MANIFEST: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly manifestVersion="1.0" xmlns="urn:schemas-microsoft-com:asm.v1">
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <heapType xmlns="http://schemas.microsoft.com/SMI/2020/WindowsSettings">SegmentHeap</heapType>
    </windowsSettings>
  </application>
</assembly>
"#;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    let out = std::path::Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("victory.manifest");
    std::fs::write(&out, MANIFEST).expect("the build's own output folder is writable");
    // link.exe merges this with its default manifest (the UAC section).
    println!("cargo:rustc-link-arg-bins=/MANIFEST:EMBED");
    println!("cargo:rustc-link-arg-bins=/MANIFESTINPUT:{}", out.display());
}
