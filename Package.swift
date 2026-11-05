// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXTAESD",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "MLXTAESD", targets: ["MLXTAESD"]),
    ],
    dependencies: [
        // The only dependency: MLX Swift (MLX + MLXNN).
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.6"),
    ],
    targets: [
        .target(
            name: "MLXTAESD",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXTAESDTests",
            dependencies: ["MLXTAESD"]
        ),
    ]
)
