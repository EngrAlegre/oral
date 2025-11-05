"use client"

import { useEffect, useState } from "react"
import { ensureFeatureEnabled } from "@/lib/feature-check"

export function FeatureGuard({ children }: { children: React.ReactNode }) {
  const [isEnabled, setIsEnabled] = useState(true)

  useEffect(() => {
    const enabled = ensureFeatureEnabled()
    setIsEnabled(enabled)
  }, [])

  if (!isEnabled) {
    return null // Feature check already handled the UI
  }

  return <>{children}</>
}

