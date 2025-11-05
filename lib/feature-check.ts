// Feature availability check - delegates to quota management system
import { ensureQuotaAvailable } from "./quota-manager"

export function ensureFeatureEnabled(): boolean {
  return ensureQuotaAvailable()
}

