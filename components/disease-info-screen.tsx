"use client"

import { ArrowLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useState } from "react"

interface DiseaseInfoScreenProps {
  onBack: () => void
}

const diseases = [
  {
    name: "🦷 Crohn's Disease",
    description:
      "Crohn's disease is a type of inflammatory bowel disease (IBD) that causes chronic inflammation anywhere along the digestive tract, most commonly in the small intestine and colon. It is an autoimmune condition where the body's immune system attacks its own intestinal tissues.",
    symptoms: [
      "Abdominal pain and cramping",
      "Chronic diarrhea",
      "Fatigue and weight loss",
      "Loss of appetite",
      "Blood in the stool",
    ],
    oralManifestations: [
      "Aphthous Ulcers: Small, painful sores due to immune system overactivity and nutrient deficiencies",
      "Mucosal Tags: Small, skin-like growths on inner cheeks or lips from chronic inflammation",
      "Gingivitis: Gum inflammation due to reduced immunity and poor oral hygiene",
    ],
    treatment:
      "Anti-inflammatory drugs (corticosteroids), immunosuppressants or biologic therapy, nutritional supplements (vitamin B12, folate, iron), good oral hygiene with antiseptic mouth rinses, and topical corticosteroids for oral lesions.",
  },
  {
    name: "🦷 Ulcerative Colitis",
    description:
      "Ulcerative colitis is another form of inflammatory bowel disease (IBD) that specifically affects the colon and rectum, causing long-term inflammation and ulceration of the colon lining.",
    symptoms: [
      "Frequent diarrhea with blood or mucus",
      "Abdominal pain and urgency to defecate",
      "Fatigue and weight loss",
      "Fever during flare-ups",
    ],
    oralManifestations: [
      "Aphthous Ulcers: Caused by immune dysregulation, stress, and nutritional deficiencies",
      "Gingivitis: Gum inflammation from changes in immune response and bacterial plaque accumulation",
    ],
    treatment:
      "Anti-inflammatory drugs (mesalamine, corticosteroids), immunomodulators or biologics for severe cases, iron and vitamin supplementation, and proper oral care with mouth rinses and topical corticosteroids.",
  },
  {
    name: "🦷 Celiac Disease",
    description:
      "Celiac disease is an autoimmune disorder in which the ingestion of gluten (a protein found in wheat, rye, and barley) triggers an immune reaction that damages the small intestine lining, leading to malabsorption of nutrients.",
    symptoms: [
      "Chronic diarrhea or constipation",
      "Bloating, gas, and abdominal pain",
      "Weight loss",
      "Fatigue",
      "Nutrient deficiencies (iron, calcium, vitamin D, B12)",
    ],
    oralManifestations: [
      "Aphthous Ulcers: Result from immune activation and vitamin deficiencies",
      "Mucosal Tags: Appear due to chronic inflammation and epithelial changes",
      "Gingivitis: Caused by nutritional deficiencies that weaken gum tissue",
      "Dental Caries: Poor enamel development from malabsorption of calcium and vitamin D",
    ],
    treatment:
      "Strict lifelong gluten-free diet, vitamin and mineral supplementation, regular dental care with fluoride use, and topical treatments for mouth ulcers if needed.",
  },
  {
    name: "🦷 Chronic Liver Disease",
    description:
      "Chronic liver disease refers to long-term liver damage from various causes such as hepatitis, alcohol use, or fatty liver disease. It leads to scarring (cirrhosis) and impaired liver function.",
    symptoms: [
      "Jaundice (yellowing of skin and eyes)",
      "Fatigue and weakness",
      "Abdominal swelling (ascites)",
      "Easy bruising or bleeding",
      "Nausea and poor appetite",
    ],
    oralManifestations: [
      "Xerostomia (Dry Mouth): Decreased saliva flow from medications or altered liver metabolism",
      "Oral Candidiasis: Fungal infection from reduced saliva and weakened immune defenses",
      "Gingivitis: Inflammation and bleeding gums from reduced clotting factors",
    ],
    treatment:
      "Manage underlying liver condition (antiviral therapy, avoiding alcohol), maintain good oral hygiene and hydration, use saliva substitutes or sugar-free lozenges, antifungal medications for oral candidiasis, and regular dental checkups.",
  },
]

export function DiseaseInfoScreen({ onBack }: DiseaseInfoScreenProps) {
  const [selectedDisease, setSelectedDisease] = useState<(typeof diseases)[0] | null>(null)

  if (selectedDisease) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card shadow-sm">
          <div className="container mx-auto px-4 py-5">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="icon" onClick={() => setSelectedDisease(null)} className="h-10 w-10">
                <ArrowLeft className="h-5 w-5" />
              </Button>
              <div className="relative h-12 w-24">
                <img src="/images/pangil-logo.png" alt="PANGIL" className="h-full w-full object-contain" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">{selectedDisease.name}</h1>
                <p className="text-sm text-muted-foreground">Disease Information</p>
              </div>
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-8">
          <div className="mx-auto max-w-4xl space-y-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">Description</h3>
              <p className="text-muted-foreground leading-relaxed">{selectedDisease.description}</p>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">Common Symptoms</h3>
              <ul className="space-y-2">
                {selectedDisease.symptoms.map((symptom, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span className="text-muted-foreground">{symptom}</span>
                  </li>
                ))}
              </ul>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">Oral Manifestations</h3>
              <ul className="space-y-3">
                {selectedDisease.oralManifestations.map((manifestation, idx) => (
                  <li key={idx} className="text-sm text-muted-foreground leading-relaxed">
                    <span className="font-medium text-foreground">{manifestation.split(":")[0]}:</span>
                    {manifestation.split(":")[1]}
                  </li>
                ))}
              </ul>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">Treatment</h3>
              <p className="text-muted-foreground leading-relaxed">{selectedDisease.treatment}</p>
            </Card>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card shadow-sm">
        <div className="container mx-auto px-4 py-5">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={onBack} className="h-10 w-10">
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div className="relative h-12 w-24">
              <img src="/images/pangil-logo.png" alt="PANGIL" className="h-full w-full object-contain" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">GI Diseases</h1>
              <p className="text-sm text-muted-foreground">Learn about gastrointestinal diseases</p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="mx-auto max-w-4xl space-y-4">
          {diseases.map((disease) => (
            <Card
              key={disease.name}
              className="p-6 hover:shadow-lg transition-all cursor-pointer group"
              onClick={() => setSelectedDisease(disease)}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">
                    {disease.name}
                  </h3>
                  <p className="text-muted-foreground line-clamp-2">{disease.description}</p>
                </div>
                <ChevronRight className="h-6 w-6 text-muted-foreground group-hover:text-primary transition-colors" />
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
