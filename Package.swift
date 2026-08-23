// swift-tools-version: 6.0
import PackageDescription

// Built in the Swift 6 language mode (full strict data-race safety), opting in early to upcoming
// features that become defaults in a later mode, so the code is already compliant. Unknown flags are
// ignored with a warning by older compilers, so this is safe to carry.
let upcoming: [SwiftSetting] = [
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("MemberImportVisibility"),
]

let package = Package(
    name: "model-lint",
    // Deliberately low. The tool reads file headers and JSON; there is no reason it should demand a
    // recent OS, and its whole point is running where a model runtime cannot.
    platforms: [.macOS(.v13)],
    products: [
        // The library is a product, not an implementation detail: the findings are more useful to a
        // converter or a model host embedding them than they are as terminal output.
        .library(name: "ModelLint", targets: ["ModelLint"]),
        .executable(name: "model-lint", targets: ["ModelLintCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.4.0"),
    ],
    targets: [
        // No dependencies at all — Foundation only. That is the property worth protecting: it is what
        // lets the tool audit bundles it could never load, on machines without the RAM to load them.
        .target(
            name: "ModelLint",
            swiftSettings: upcoming
        ),
        .executableTarget(
            name: "ModelLintCLI",
            dependencies: [
                "ModelLint",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: upcoming
        ),
        .testTarget(
            name: "ModelLintTests",
            dependencies: ["ModelLint"],
            swiftSettings: upcoming
        ),
    ]
)
