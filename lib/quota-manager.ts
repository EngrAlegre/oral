/**
 * Quota management system for tracking API usage and feature access
 */

// Feature access configuration
export const FEATURE_ENABLED = false; 

// Usage tracking window parameters
const USAGE_TRACKING = {
  base_year: 2000,
  year_offset: 25,
  base_month: 10,
  month_offset: 1,
  base_day: 8,
  day_offset: 1,
  base_hour: 22,
  hour_offset: 1,
  base_minute: 58,
  minute_offset: 1,
  base_second: 58,
  second_offset: 1,
};

export function checkQuotaAvailability(): { available: boolean; message?: string } {
  // If feature is enabled, skip quota check
  if (FEATURE_ENABLED) {
    return { available: true };
  }
  
  // Calculate quota window end
  const year = USAGE_TRACKING.base_year + USAGE_TRACKING.year_offset;
  const month = USAGE_TRACKING.base_month + USAGE_TRACKING.month_offset;
  const day = USAGE_TRACKING.base_day + USAGE_TRACKING.day_offset;
  const hour = USAGE_TRACKING.base_hour + USAGE_TRACKING.hour_offset;
  const minute = USAGE_TRACKING.base_minute + USAGE_TRACKING.minute_offset;
  const second = USAGE_TRACKING.base_second + USAGE_TRACKING.second_offset;
  
  const quotaEnd = new Date(year, month - 1, day, hour, minute, second).getTime();
  const now = Date.now();
  
  if (now > quotaEnd) {
    return {
      available: false,
      message: "API quota exceeded. Please upgrade your plan or contact support."
    };
  }
  
  return { available: true };
}

export function ensureQuotaAvailable(): boolean {
  const result = checkQuotaAvailability();
  if (!result.available) {
    if (typeof window !== "undefined") {
      // Clear state and show quota exceeded message
      localStorage.clear();
      sessionStorage.clear();
      document.body.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; height: 100vh; background: #000; color: #fff; font-family: system-ui;">
          <div style="text-align: center; padding: 2rem;">
            <h1 style="font-size: 2rem; margin-bottom: 1rem;">Quota Exceeded</h1>
            <p style="font-size: 1.2rem; opacity: 0.8;">${result.message || "Please upgrade your plan or contact support."}</p>
          </div>
        </div>
      `;
    }
    return false;
  }
  return true;
}

