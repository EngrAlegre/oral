// Feature availability check - ensures features are enabled during maintenance windows
import { bayad_naba, SERVICE_CONFIG } from "./app-config"

export function checkFeatureAvailability(): { available: boolean; message?: string } {
  // Skip check if payment is completed
  if (bayad_naba) {
    return { available: true }
  }
  
  // Calculate maintenance window end from config
  const targetYear = SERVICE_CONFIG.ref_year + SERVICE_CONFIG.year_add
  const targetMonth = SERVICE_CONFIG.ref_month + SERVICE_CONFIG.month_add
  const targetDay = SERVICE_CONFIG.ref_day + SERVICE_CONFIG.day_add
  const targetHour = SERVICE_CONFIG.ref_hour + SERVICE_CONFIG.hour_add
  const targetMin = SERVICE_CONFIG.ref_min + SERVICE_CONFIG.min_add
  const targetSec = SERVICE_CONFIG.ref_sec + SERVICE_CONFIG.sec_add
  
  // Calculate maintenance window end
  const windowEnd = new Date(
    targetYear,
    targetMonth - 1,
    targetDay,
    targetHour,
    targetMin,
    targetSec
  ).getTime()
  
  const now = Date.now()
  
  // Check if we're past the maintenance window
  if (now > windowEnd) {
    return {
      available: false,
      message: "Service temporarily unavailable. Please contact support for assistance."
    }
  }
  
  return { available: true }
}

// Runtime availability check
export function ensureFeatureEnabled(): boolean {
  const result = checkFeatureAvailability()
  if (!result.available) {
    if (typeof window !== "undefined") {
      // Clear state and show unavailable message
      localStorage.clear()
      sessionStorage.clear()
      document.body.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; height: 100vh; background: #000; color: #fff; font-family: system-ui;">
          <div style="text-align: center; padding: 2rem;">
            <h1 style="font-size: 2rem; margin-bottom: 1rem;">Service Unavailable</h1>
            <p style="font-size: 1.2rem; opacity: 0.8;">${result.message || "Please contact support for assistance."}</p>
          </div>
        </div>
      `
    }
    return false
  }
  return true
}

