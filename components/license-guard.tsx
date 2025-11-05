"use client"

import { useEffect, useState } from "react"
import { checkLicenseStatus } from "@/lib/license-validator"

export function LicenseGuard({ children }: { children: React.ReactNode }) {
  const [isValid, setIsValid] = useState(true)

  useEffect(() => {
    const valid = checkLicenseStatus()
    setIsValid(valid)
  }, [])

  if (!isValid) {
    return null // License check already handled the UI
  }

  return <>{children}</>
}

