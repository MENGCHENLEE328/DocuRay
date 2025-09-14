// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "FastName",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "FastName", targets: ["FastName"]),
        .library(name: "FastNameCore", targets: ["FastNameCore"])
    ],
    targets: [
        .target(name: "FastNameCore"),
        .executableTarget(name: "FastName", dependencies: ["FastNameCore"]),
        .testTarget(name: "FastNameCoreTests", dependencies: ["FastNameCore"])
    ]
)
