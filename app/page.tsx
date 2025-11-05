import { DetectionInterface } from "@/components/detection-interface"
import { LicenseGuard } from "@/components/license-guard"

export default function Home() {
  return (
    <LicenseGuard>
      <main className="min-h-screen">
        <DetectionInterface />
      </main>
    </LicenseGuard>
  )
}
