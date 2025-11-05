// Oral lesions that can be detected
export const ORAL_LESIONS = [
  "Gingivitis",
  "Dental Caries",
  "Aphthous Ulcer",
  "Oral Candidiasis",
  "Mucosal Tags",
  "Xerostomia",
] as const

// GI diseases that can be predicted
export const GI_DISEASES = ["Crohn's Disease", "Ulcerative Colitis", "Celiac Disease", "Chronic Liver Disease"] as const

// Mapping of oral lesions to GI diseases with adjustment factors
// Based on the computation shown in the provided image
export const LESION_TO_DISEASE_MAP: Record<string, Array<{ disease: string; adjustmentFactor: number }>> = {
  "Aphthous Ulcer": [
    { disease: "Crohn's Disease", adjustmentFactor: 0.62 },
    { disease: "Ulcerative Colitis", adjustmentFactor: 0.62 },
    { disease: "Celiac Disease", adjustmentFactor: 0.62 },
  ],
  Xerostomia: [
    { disease: "Chronic Liver Disease", adjustmentFactor: 0.7 },
    { disease: "Celiac Disease", adjustmentFactor: 0.7 },
  ],
  "Dental Caries": [{ disease: "Celiac Disease", adjustmentFactor: 0.82 }],
  "Mucosal Tags": [
    { disease: "Crohn's Disease", adjustmentFactor: 0.68 },
    { disease: "Celiac Disease", adjustmentFactor: 0.68 },
  ],
  Gingivitis: [
    { disease: "Crohn's Disease", adjustmentFactor: 0.72 },
    { disease: "Chronic Liver Disease", adjustmentFactor: 0.72 },
    { disease: "Ulcerative Colitis", adjustmentFactor: 0.72 },
  ],
  "Oral Candidiasis": [
    { disease: "Chronic Liver Disease", adjustmentFactor: 0.75 },
    { disease: "Celiac Disease", adjustmentFactor: 0.75 },
  ],
}

export interface DetectedLesion {
  type: string
  confidence: number
  bbox?: {
    x: number
    y: number
    width: number
    height: number
  }
}

export interface DiseaseComputation {
  disease: string
  contributions: Array<{
    lesion: string
    confidence: number
    adjustmentFactor: number
    adjustedValue: number
    sharedWith: number
    finalContribution: number
  }>
  totalProbability: number
}

/**
 * Computes disease probabilities based on detected oral lesions
 * Following the algorithm shown in the provided image:
 * 1. For each lesion, multiply confidence by adjustment factor
 * 2. If a lesion maps to multiple diseases, divide by number of diseases
 * 3. Sum contributions per disease to get final probability
 * 4. Cap maximum probability at 100%
 */
export function computeDiseaseProbabilities(detectedLesions: DetectedLesion[]): DiseaseComputation[] {
  const diseaseContributions: Record<
    string,
    Array<{
      lesion: string
      confidence: number
      adjustmentFactor: number
      adjustedValue: number
      sharedWith: number
      finalContribution: number
    }>
  > = {}

  // Initialize all diseases with empty arrays
  GI_DISEASES.forEach((disease) => {
    diseaseContributions[disease] = []
  })

  // Process each detected lesion
  detectedLesions.forEach((lesion) => {
    const mappings = LESION_TO_DISEASE_MAP[lesion.type]
    if (!mappings) return

    const numDiseases = mappings.length

    mappings.forEach(({ disease, adjustmentFactor }) => {
      // Step 1: Multiply confidence by adjustment factor
      const adjustedValue = lesion.confidence * adjustmentFactor

      // Step 2: Divide by number of diseases if shared
      const finalContribution = adjustedValue / numDiseases

      diseaseContributions[disease].push({
        lesion: lesion.type,
        confidence: lesion.confidence,
        adjustmentFactor,
        adjustedValue,
        sharedWith: numDiseases,
        finalContribution,
      })
    })
  })

  // Step 3: Sum contributions per disease
  const results: DiseaseComputation[] = GI_DISEASES.map((disease) => {
    const contributions = diseaseContributions[disease]
    const totalProbability = contributions.reduce((sum, c) => sum + c.finalContribution, 0)

    const cappedProbability = Math.min(totalProbability, 100)

    return {
      disease,
      contributions,
      totalProbability: Math.round(cappedProbability * 100) / 100, // Round to 2 decimal places
    }
  })

  // Sort by probability (highest first)
  return results.sort((a, b) => b.totalProbability - a.totalProbability)
}
