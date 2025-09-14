// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "FastName",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "FastName", targets: ["FastName"]),
        .executable(name: "FastNameDaemon", targets: ["FastNameDaemon"]),
        .library(name: "FastNameCore", targets: ["FastNameCore"]),
        .library(name: "SpotSearchCore", targets: ["SpotSearchCore"]) // New module for next-gen search
    ],
    targets: [
        .target(name: "FastNameCore"),
        .target(name: "SpotSearchCore"),
        .executableTarget(name: "FastName", dependencies: ["FastNameCore"]),
        .executableTarget(name: "FastNameDaemon", dependencies: ["FastNameCore"]),
        .testTarget(name: "FastNameCoreTests", dependencies: ["FastNameCore"]),
        .testTarget(name: "SpotSearchCoreTests", dependencies: ["SpotSearchCore"]) // New tests
    ]
)
