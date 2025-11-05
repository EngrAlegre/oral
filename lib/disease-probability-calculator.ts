// Based on Table 2: Linkage Probability Rates of Oral Lesions Across Gastrointestinal Diseases

export interface LinkageProbability {
  disease: string
  probability: number
  detectedLesions: string[]
  contributingLesionProbabilities: { lesion: string; probability: number }[]
}

// Table 2 data: Linkage probability rates for each lesion per disease
export const LESION_DISEASE_LINKAGE = {
  "Crohn's Disease": {
    "Mucosal Tags": 0.68,
    Xerostomia: 0.35,
    "Aphthous Ulcer": 0.1,
    "Dental Caries": 0.02,
    "Oral Candidiasis": 0.05,
    Gingivitis: 0.5,
  },
  "Celiac Disease": {
    "Mucosal Tags": 0.02,
    Xerostomia: 0.25,
    "Aphthous Ulcer": 0.25,
    "Dental Caries": 0.1,
    "Oral Candidiasis": 0.15,
    Gingivitis: 0.2,
  },
  "Ulcerative Colitis": {
    "Mucosal Tags": 0.05,
    Xerostomia: 0.6,
    "Aphthous Ulcer": 0.6,
    "Dental Caries": 0.2,
    "Oral Candidiasis": 0.2,
    Gingivitis: 0.5,
  },
  "Chronic Liver Disease": {
    "Mucosal Tags": 0.02,
    Xerostomia: 0.3,
    "Aphthous Ulcer": 0.07,
    "Dental Caries": 0.4,
    "Oral Candidiasis": 0.45,
    Gingivitis: 0.18,
  },
}

export const DISEASE_NAMES = Object.keys(LESION_DISEASE_LINKAGE) as (keyof typeof LESION_DISEASE_LINKAGE)[]

export function calculateDiseaseProbabilities(
  detectedLesions: Array<{ lesion: string }>, // Changed to binary detection (no confidence)
): LinkageProbability[] {
  console.log("[v0] Calculating disease probabilities for detected lesions:", detectedLesions)

  // Initialize disease probability accumulator
  const diseaseScores: { [key: string]: { score: number; lesions: Array<{ lesion: string; probability: number }> } } =
    {}

  DISEASE_NAMES.forEach((disease) => {
    diseaseScores[disease] = { score: 0, lesions: [] }
  })

  detectedLesions.forEach(({ lesion }) => {
    DISEASE_NAMES.forEach((disease) => {
      // Get linkage probability for this lesion-disease pair
      const linkageProbability = LESION_DISEASE_LINKAGE[disease][lesion as keyof typeof LESION_DISEASE_LINKAGE] || 0

      if (linkageProbability > 0) {
        // Use linkage probability directly (no confidence multiplication)
        diseaseScores[disease].score += linkageProbability * 100
        diseaseScores[disease].lesions.push({
          lesion,
          probability: linkageProbability * 100,
        })
      }
    })
  })

  // Normalize scores to get final probabilities
  const maxScore = Math.max(...Object.values(diseaseScores).map((d) => d.score))
  const probabilites: LinkageProbability[] = DISEASE_NAMES.map((disease) => {
    const normalizedProbability = maxScore > 0 ? (diseaseScores[disease].score / maxScore) * 100 : 0

    return {
      disease,
      probability: Math.round(normalizedProbability * 100) / 100,
      detectedLesions: Array.from(new Set(detectedLesions.map((d) => d.lesion))),
      contributingLesionProbabilities: diseaseScores[disease].lesions,
    }
  })

  // Sort by probability (highest first)
  probabilites.sort((a, b) => b.probability - a.probability)

  console.log("[v0] Calculated disease probabilities:", probabilites)
  return probabilites
}

export function normalizeLesionName(detectedLabel: string): string {
  const labelMap: { [key: string]: string } = {
    // Map YOLO model output names to Table 2 lesion names
    "mucosal tags": "Mucosal Tags",
    "mucosal-tags": "Mucosal Tags",
    xerostomia: "Xerostomia",
    "dry mouth": "Xerostomia",
    "aphthous ulcer": "Aphthous Ulcer",
    "aphthous-ulcer": "Aphthous Ulcer",
    ulcer: "Aphthous Ulcer",
    "dental caries": "Dental Caries",
    "dental-caries": "Dental Caries",
    caries: "Dental Caries",
    cavity: "Dental Caries",
    "oral candidiasis": "Oral Candidiasis",
    "oral-candidiasis": "Oral Candidiasis",
    candidiasis: "Oral Candidiasis",
    thrush: "Oral Candidiasis",
    gingivitis: "Gingivitis",
    "gum disease": "Gingivitis",
  }

  const normalized = labelMap[detectedLabel.toLowerCase()] || detectedLabel
  console.log(`[v0] Normalized lesion: "${detectedLabel}" -> "${normalized}"`)
  return normalized
}
