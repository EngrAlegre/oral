const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

export interface DetectionResult {
  label: string
  confidence: number
  recommendation: string
  ai_feedback: {
    detection: string
    segmentation: string
    gradcam: string
  }
  detection_image: string
  segmentation_image: string
  gradcam_image: string
}

async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxRetries = 3,
  delayMs = 1000,
): Promise<Response> {
  let lastError: Error | null = null

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`[PANGIL] Fetch attempt ${attempt}/${maxRetries} to ${url}`)
      const response = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(30000),
      })

      if (response.ok) {
        return response
      }

      if (response.status >= 400 && response.status < 500) {
        throw new Error(`Client error: ${response.status} ${response.statusText}`)
      }

      if (attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, delayMs * attempt))
        continue
      }

      throw new Error(`Server error: ${response.status} ${response.statusText}`)
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))
      console.warn(`[PANGIL] Attempt ${attempt} failed: ${lastError.message}`)

      if (attempt < maxRetries) {
        console.warn(`[PANGIL] Retrying in ${delayMs * attempt}ms...`)
        await new Promise((resolve) => setTimeout(resolve, delayMs * attempt))
      }
    }
  }

  throw lastError || new Error("Failed after maximum retries")
}

export async function detectLesion(file: File, userId?: string): Promise<DetectionResult> {
  const formData = new FormData()
  formData.append("file", file)
  if (userId) {
    formData.append("user_id", userId)
  }

  console.log(`[PANGIL] Sending detection request to ${BACKEND_URL}/predict`)

  try {
    const response = await fetchWithRetry(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: formData,
    })

    const result = await response.json()
    console.log("[PANGIL] Detection successful:", result)
    return result
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error"
    console.error("[PANGIL] Detection failed:", message)
    console.error("[PANGIL] Backend URL:", BACKEND_URL)
    throw new Error(`Detection failed: ${message}`)
  }
}

export async function healthCheck(): Promise<{
  isHealthy: boolean
  backendUrl: string
  error?: string
}> {
  console.log(`[PANGIL] Running health check on ${BACKEND_URL}/health`)
  try {
    const response = await fetchWithRetry(`${BACKEND_URL}/health`, { method: "GET" }, 2, 500)

    if (response.ok) {
      console.log("[PANGIL] Backend is healthy")
      return {
        isHealthy: true,
        backendUrl: BACKEND_URL,
      }
    }

    return {
      isHealthy: false,
      backendUrl: BACKEND_URL,
      error: `Health check returned ${response.status}`,
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error"
    console.error("[PANGIL] Health check failed:", message)
    return {
      isHealthy: false,
      backendUrl: BACKEND_URL,
      error: message,
    }
  }
}

export function getBackendUrl(): string {
  return BACKEND_URL
}

export interface DiagnosticResult {
  label: string
  confidence: number
  recommendation: string
  ai_feedback: {
    detection: string
    segmentation: string
    gradcam: string
  }
  detection_image: string
  segmentation_image: string
  gradcam_image: string
}

export interface AveragedDiagnostic {
  lesion: string
  occurrences: number
  confidenceScores: number[]
  averagedConfidence: number
  recommendation: string
}

import {
  calculateDiseaseProbabilities,
  normalizeLesionName,
  type LinkageProbability,
} from "./disease-probability-calculator"

export async function detectMultipleImages(files: File[]): Promise<{
  allDetections: DiagnosticResult[]
  averagedDiagnostics: AveragedDiagnostic[]
  primaryDiagnosis: AveragedDiagnostic
  diseaseProbabilities: LinkageProbability[]
}> {
  console.log(`[PANGIL] Sending ${files.length} images to backend for analysis`)

  const formData = new FormData()
  files.forEach((file, index) => {
    formData.append(`file_${index}`, file)
  })

  try {
    const response = await fetchWithRetry(`${BACKEND_URL}/predict-batch`, {
      method: "POST",
      body: formData,
    })

    const results = await response.json()
    console.log("[PANGIL] Batch detection successful:", results)

    const detectionsByLesion: { [key: string]: number[] } = {}
    const recommendationsByLesion: { [key: string]: string } = {}

    results.detections.forEach((detection: DiagnosticResult) => {
      if (!detectionsByLesion[detection.label]) {
        detectionsByLesion[detection.label] = []
        recommendationsByLesion[detection.label] = detection.recommendation
      }
      detectionsByLesion[detection.label].push(detection.confidence)
    })

    // Calculate averaged diagnostics
    const averagedDiagnostics: AveragedDiagnostic[] = Object.entries(detectionsByLesion).map(([lesion, scores]) => ({
      lesion,
      occurrences: scores.length,
      confidenceScores: scores,
      averagedConfidence: scores.reduce((a, b) => a + b, 0) / scores.length,
      recommendation: recommendationsByLesion[lesion],
    }))

    // Sort by averaged confidence (highest first)
    averagedDiagnostics.sort((a, b) => b.averagedConfidence - a.averagedConfidence)

    const normalizedDetections = results.detections.map((detection: DiagnosticResult) => ({
      lesion: normalizeLesionName(detection.label),
      confidence: detection.confidence,
    }))

    const diseaseProbabilities = calculateDiseaseProbabilities(normalizedDetections)

    return {
      allDetections: results.detections,
      averagedDiagnostics,
      primaryDiagnosis: averagedDiagnostics[0],
      diseaseProbabilities,
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error"
    console.error("[PANGIL] Batch detection failed:", message)
    throw new Error(`Batch detection failed: ${message}`)
  }
}
