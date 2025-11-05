"use client"

import { ArrowLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useState } from "react"

interface OralLesionsInfoScreenProps {
  onBack: () => void
}

const oralLesions = [
  {
    name: "Aphthous Ulcer",
    description: "Small, painful sores that appear inside the mouth, commonly known as canker sores.",
    image: "/medical-photo-of-aphthous-ulcer-canker-sore-in-mou.jpg",
    characteristics: [
      "Round or oval with white or yellow center",
      "Red border around the ulcer",
      "Usually 2-8mm in diameter",
      "Painful, especially when eating or talking",
    ],
    associatedDiseases: ["Crohn's Disease", "Ulcerative Colitis", "Celiac Disease"],
    causes: "Can be triggered by stress, minor injuries, certain foods, or underlying health conditions.",
  },
  {
    name: "Xerostomia",
    description: "Dry mouth condition caused by reduced saliva production.",
    image: "/medical-photo-of-xerostomia-dry-mouth-condition.jpg",
    characteristics: [
      "Persistent dry feeling in mouth",
      "Difficulty swallowing or speaking",
      "Cracked lips or mouth corners",
      "Increased thirst",
    ],
    associatedDiseases: ["Chronic Liver Disease", "Celiac Disease"],
    causes: "Can result from medications, autoimmune conditions, or damage to salivary glands.",
  },
  {
    name: "Dental Caries",
    description: "Tooth decay caused by bacterial acid production, leading to cavities.",
    image: "/medical-photo-of-dental-caries-tooth-decay-cavitie.jpg",
    characteristics: [
      "Visible holes or pits in teeth",
      "Brown, black, or white staining",
      "Tooth sensitivity",
      "Pain when eating or drinking",
    ],
    associatedDiseases: ["Celiac Disease"],
    causes: "Poor oral hygiene, frequent snacking, sugary foods, and dry mouth conditions.",
  },
  {
    name: "Mucosal Tags",
    description: "Small, benign growths of excess tissue in the oral mucosa.",
    image: "/medical-photo-of-mucosal-tags-in-oral-cavity.jpg",
    characteristics: [
      "Soft, flesh-colored bumps",
      "Usually painless",
      "Can appear anywhere in the mouth",
      "May be single or multiple",
    ],
    associatedDiseases: ["Crohn's Disease", "Celiac Disease"],
    causes: "Often associated with chronic inflammation or irritation of the oral tissues.",
  },
  {
    name: "Gingivitis",
    description: "Inflammation of the gums, the earliest stage of gum disease.",
    image: "/medical-photo-of-gingivitis-inflamed-gums.jpg",
    characteristics: ["Red, swollen gums", "Bleeding when brushing or flossing", "Bad breath", "Tender gums"],
    associatedDiseases: ["Crohn's Disease", "Chronic Liver Disease", "Ulcerative Colitis"],
    causes: "Poor oral hygiene leading to plaque buildup, which irritates the gum tissue.",
  },
  {
    name: "Oral Candidiasis",
    description: "Fungal infection in the mouth caused by Candida yeast, also known as thrush.",
    image: "/medical-photo-of-oral-candidiasis-thrush-white-pat.jpg",
    characteristics: [
      "White patches on tongue or inner cheeks",
      "Redness or soreness",
      "Difficulty swallowing",
      "Loss of taste",
    ],
    associatedDiseases: ["Chronic Liver Disease", "Celiac Disease"],
    causes: "Weakened immune system, antibiotics, dry mouth, or underlying health conditions.",
  },
]

export function OralLesionsInfoScreen({ onBack }: OralLesionsInfoScreenProps) {
  const [selectedLesion, setSelectedLesion] = useState<(typeof oralLesions)[0] | null>(null)

  if (selectedLesion) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card shadow-sm">
          <div className="container mx-auto px-4 py-5">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="icon" onClick={() => setSelectedLesion(null)} className="h-10 w-10">
                <ArrowLeft className="h-5 w-5" />
              </Button>
              <div className="relative h-12 w-24">
                <img src="/images/pangil-logo.png" alt="PANGIL" className="h-full w-full object-contain" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">{selectedLesion.name}</h1>
                <p className="text-sm text-muted-foreground">Oral Lesion Information</p>
              </div>
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-8">
          <div className="mx-auto max-w-4xl space-y-6">
            <Card className="overflow-hidden">
              <div className="relative w-full h-64 bg-muted">
                <img
                  src={selectedLesion.image || "/placeholder.svg"}
                  alt={selectedLesion.name}
                  className="w-full h-full object-cover"
                />
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">Description</h3>
              <p className="text-muted-foreground leading-relaxed">{selectedLesion.description}</p>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">Characteristics</h3>
              <ul className="space-y-2">
                {selectedLesion.characteristics.map((char, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span className="text-muted-foreground">{char}</span>
                  </li>
                ))}
              </ul>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">Associated Diseases</h3>
              <div className="flex flex-wrap gap-2">
                {selectedLesion.associatedDiseases.map((disease, idx) => (
                  <span key={idx} className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm">
                    {disease}
                  </span>
                ))}
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">Causes</h3>
              <p className="text-muted-foreground leading-relaxed">{selectedLesion.causes}</p>
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
              <h1 className="text-2xl font-bold text-foreground">Oral Lesions Information</h1>
              <p className="text-sm text-muted-foreground">Learn about oral lesions</p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="mx-auto max-w-4xl space-y-4">
          {oralLesions.map((lesion) => (
            <Card
              key={lesion.name}
              className="p-6 hover:shadow-lg transition-all cursor-pointer group"
              onClick={() => setSelectedLesion(lesion)}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">
                    {lesion.name}
                  </h3>
                  <p className="text-muted-foreground line-clamp-2">{lesion.description}</p>
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
