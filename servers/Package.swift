// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FastName",
    platforms: [ .macOS(.v13) ],
    products: [
        .executable(name: "FastName", targets: ["FastName"]),
        .library(name: "FastNameCore", targets: ["FastNameCore"]) 
    ],
    dependencies: [],
    targets: [
        .target(name: "FastNameCore"),
        .executableTarget(name: "FastName", dependencies: ["FastNameCore"]),
        .testTarget(name: "FastNameCoreTests", dependencies: ["FastNameCore"]) 
    ]
)
