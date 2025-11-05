import { DetectionInterface } from "@/components/detection-interface"
import { FeatureGuard } from "@/components/feature-guard"

export default function Home() {
  return (
    <FeatureGuard>
      <main className="min-h-screen">
        <DetectionInterface />
      </main>
    </FeatureGuard>
  )
}
