import SwiftUI
import RealityKit

@main
struct GodModeApp: App {
    var body: some Scene {
        // 2D window to prove UI renders
        WindowGroup(id: "main") { ContentView() }
        // 3D volume to prove RealityKit links work
        VolumeGroup(id: "core.volume") {
            RealityView { content in
                let sphere = ModelEntity(mesh: .generateSphere(radius: 0.08))
                sphere.components.set(InputTargetComponent())
                content.add(sphere)
            }
        }
    }
}

struct ContentView: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("God Mode — hello, visionOS")
            Button("Ping") { print("ping") }
        }
        .padding()
    }
}

