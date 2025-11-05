"use client"

import { useState, useEffect } from "react"
import { ArrowLeft, Calendar, FileText, Eye, Info, ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { BoundingBoxViewer } from "./bounding-box-viewer"

interface HistoryScreenProps {
  onBack: () => void
}

interface HistoryItem {
  id: string
  date: string
  time: string
  imageUrl: string
  detectionImage?: string // Image with detection overlay
  gradcamImage?: string // GradCAM heatmap
  detections: Array<{
    type: string
    bbox: { x: number; y: number; width: number; height: number }
  }>
  diseaseProbabilities: Array<{
    disease: string
    probability: number
    risk: string
  }>
}

export function HistoryScreen({ onBack }: HistoryScreenProps) {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null)
  const [currentView, setCurrentView] = useState(0)

  useEffect(() => {
    const stored = localStorage.getItem("pangil_history")
    if (stored) {
      setHistoryItems(JSON.parse(stored))
    }
  }, [])

  const getRecommendations = (disease: string, probability: number) => {
    if (probability === 0) return null

    const recommendations: Record<string, { urgent: string; followUp: string; lifestyle: string }> = {
      "Crohn's Disease": {
        urgent: "Consult a gastroenterologist for comprehensive evaluation and possible colonoscopy",
        followUp: "Monitor symptoms including abdominal pain, diarrhea, weight loss, and fatigue",
        lifestyle: "Consider dietary modifications and stress management techniques",
      },
      "Ulcerative Colitis": {
        urgent: "Schedule appointment with gastroenterologist for endoscopic examination",
        followUp: "Track bowel movement frequency, blood in stool, and abdominal cramping",
        lifestyle: "Maintain hydration and discuss anti-inflammatory diet options with healthcare provider",
      },
      "Celiac Disease": {
        urgent: "Request serological testing (tTG-IgA, EMA) and possible intestinal biopsy",
        followUp: "Document digestive symptoms, skin rashes, and nutritional deficiencies",
        lifestyle: "Prepare for potential gluten-free diet transition under medical supervision",
      },
      "Chronic Liver Disease": {
        urgent: "Immediate liver function tests (LFTs) and hepatology consultation recommended",
        followUp: "Monitor for jaundice, fatigue, abdominal swelling, and easy bruising",
        lifestyle: "Avoid alcohol and hepatotoxic medications; discuss vaccination status",
      },
    }

    return recommendations[disease]
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
              <h1 className="text-2xl font-bold text-foreground">Detection History</h1>
              <p className="text-sm text-muted-foreground">View your past results</p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="mx-auto max-w-4xl space-y-4">
          {historyItems.length === 0 ? (
            <Card className="p-12 text-center">
              <FileText className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
              <h3 className="text-xl font-semibold text-foreground mb-2">No History Yet</h3>
              <p className="text-muted-foreground">Your detection results will appear here</p>
            </Card>
          ) : (
            historyItems.map((item) => {
              const topDisease = item.diseaseProbabilities?.[0] || {
                disease: "Unknown",
                probability: 0,
                risk: "Low",
              }

              return (
                <Card
                  key={item.id}
                  className="p-6 hover:shadow-lg transition-shadow cursor-pointer"
                  onClick={() => setSelectedItem(item)}
                >
                  <div className="flex items-start gap-4">
                    <div className="relative w-24 h-24 rounded-lg overflow-hidden bg-muted flex-shrink-0">
                      <img
                        src={item.imageUrl || "/placeholder.svg"}
                        alt="Detection"
                        className="w-full h-full object-cover"
                      />
                    </div>

                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <Calendar className="h-5 w-5 text-primary" />
                        <span className="font-semibold text-foreground">{item.date}</span>
                        <span className="text-sm text-muted-foreground">{item.time}</span>
                      </div>

                      <div className="mb-3">
                        <p className="text-sm text-muted-foreground mb-2">Detected Lesions:</p>
                        <div className="flex flex-wrap gap-2">
                          {item.detections.map((detection, idx) => (
                            <Badge key={idx} variant="secondary">
                              {detection.type}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {item.diseaseProbabilities && item.diseaseProbabilities.length > 0 && (
                        <div className="flex items-center gap-2">
                          <p className="text-sm text-muted-foreground">Top Result:</p>
                          <span className="font-medium text-foreground">{topDisease.disease}</span>
                          <Badge
                            variant={
                              topDisease.probability >= 70
                                ? "destructive"
                                : topDisease.probability >= 40
                                  ? "default"
                                  : "secondary"
                            }
                          >
                            {topDisease.probability.toFixed(1)}%
                          </Badge>
                        </div>
                      )}
                    </div>

                    <Button
                      variant="outline"
                      size="icon"
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedItem(item)
                      }}
                    >
                      <Eye className="h-5 w-5" />
                    </Button>
                  </div>
                </Card>
              )
            })
          )}
        </div>
      </div>

      <Dialog open={!!selectedItem} onOpenChange={() => setSelectedItem(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Detection Details</DialogTitle>
          </DialogHeader>
          {selectedItem && (
            <div className="space-y-6">
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <Calendar className="h-4 w-4" />
                <span>
                  {selectedItem.date} at {selectedItem.time}
                </span>
              </div>

              {(() => {
                const views = [
                  { name: "Detection", image: selectedItem.detectionImage, description: "Original image with detection overlay" },
                  { name: "GradCAM", image: selectedItem.gradcamImage, description: "AI attention heatmap showing focus areas" }
                ].filter(view => view.image)

                const handlePrevious = () => {
                  setCurrentView((prev) => (prev === 0 ? views.length - 1 : prev - 1))
                }

                const handleNext = () => {
                  setCurrentView((prev) => (prev === views.length - 1 ? 0 : prev + 1))
                }

                if (views.length > 0) {
                  return (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h3 className="text-base font-semibold text-foreground">{views[currentView].name}</h3>
                        <Badge variant="secondary" className="text-xs">
                          {currentView + 1} / {views.length}
                        </Badge>
                      </div>
                      
                      <div className="relative w-full overflow-hidden rounded-lg border border-border bg-black">
                        <img 
                          src={views[currentView].image} 
                          alt={views[currentView].name} 
                          className="w-full h-auto"
                          style={{ maxHeight: "600px", objectFit: "contain" }}
                        />
                        
                        {views.length > 1 && (
                          <>
                            <Button
                              variant="secondary"
                              size="icon"
                              className="absolute left-4 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-black/70 hover:bg-black/90 text-white"
                              onClick={handlePrevious}
                            >
                              <ChevronLeft className="h-6 w-6" />
                            </Button>
                            <Button
                              variant="secondary"
                              size="icon"
                              className="absolute right-4 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-black/70 hover:bg-black/90 text-white"
                              onClick={handleNext}
                            >
                              <ChevronRight className="h-6 w-6" />
                            </Button>
                          </>
                        )}
                      </div>
                      
                      <p className="text-sm text-muted-foreground text-center">{views[currentView].description}</p>
                    </div>
                  )
                } else {
                  return <BoundingBoxViewer imageUrl={selectedItem.imageUrl} detections={selectedItem.detections} />
                }
              })()}

              <div className="space-y-3">
                <h3 className="text-base font-semibold text-foreground">Detected Oral Lesions</h3>
                <div className="space-y-2">
                  {selectedItem.detections.map((lesion, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between rounded-lg border border-border bg-card p-4 transition-colors hover:bg-muted/50"
                    >
                      <p className="font-medium text-foreground">{lesion.type}</p>
                    </div>
                  ))}
                </div>
              </div>

              {selectedItem.diseaseProbabilities && selectedItem.diseaseProbabilities.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-base font-semibold text-foreground">Disease Linkage Probability</h3>

                  {selectedItem.diseaseProbabilities.map((disease, index) => (
                    <div
                      key={index}
                      className="rounded-lg border border-border bg-muted/30 p-4 transition-all hover:shadow-md"
                    >
                      <div className="flex items-center justify-between">
                        <h4 className="font-semibold text-foreground">{disease.disease}</h4>
                        <Badge
                          variant={disease.probability > 50 ? "default" : "secondary"}
                          className="text-base px-3 py-1"
                        >
                          {disease.probability.toFixed(1)}%
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="space-y-4">
                <h3 className="text-base font-semibold text-foreground">Medical Recommendations</h3>

                {selectedItem.diseaseProbabilities.filter((d) => d.probability > 0).length > 0 ? (
                  <div className="space-y-3">
                    {selectedItem.diseaseProbabilities
                      .filter((d) => d.probability > 0)
                      .sort((a, b) => b.probability - a.probability)
                      .map((disease, index) => {
                        const recs = getRecommendations(disease.disease, disease.probability)
                        if (!recs) return null

                        const isHighRisk = disease.probability > 60
                        const isMediumRisk = disease.probability > 30

                        return (
                          <div
                            key={index}
                            className={`rounded-lg border p-4 transition-all hover:shadow-md ${
                              isHighRisk
                                ? "border-destructive/50 bg-destructive/5"
                                : isMediumRisk
                                  ? "border-warning/50 bg-warning/5"
                                  : "border-border bg-card"
                            }`}
                          >
                            <div className="mb-3 flex items-center justify-between">
                              <h4 className="font-semibold text-foreground">{disease.disease}</h4>
                              <Badge
                                variant={isHighRisk ? "destructive" : isMediumRisk ? "default" : "secondary"}
                                className="text-sm px-3 py-1"
                              >
                                {disease.probability.toFixed(1)}% Probability
                              </Badge>
                            </div>

                            <div className="space-y-3 text-sm">
                              <div className="flex gap-3">
                                <div className="mt-1 flex-shrink-0">
                                  <div className="h-2 w-2 rounded-full bg-destructive" />
                                </div>
                                <div>
                                  <p className="font-medium text-foreground">Immediate Action:</p>
                                  <p className="text-muted-foreground leading-relaxed">{recs.urgent}</p>
                                </div>
                              </div>

                              <div className="flex gap-3">
                                <div className="mt-1 flex-shrink-0">
                                  <div className="h-2 w-2 rounded-full bg-primary" />
                                </div>
                                <div>
                                  <p className="font-medium text-foreground">Monitor Symptoms:</p>
                                  <p className="text-muted-foreground leading-relaxed">{recs.followUp}</p>
                                </div>
                              </div>

                              <div className="flex gap-3">
                                <div className="mt-1 flex-shrink-0">
                                  <div className="h-2 w-2 rounded-full bg-accent" />
                                </div>
                                <div>
                                  <p className="font-medium text-foreground">Lifestyle Considerations:</p>
                                  <p className="text-muted-foreground leading-relaxed">{recs.lifestyle}</p>
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                  </div>
                ) : (
                  <Alert variant="default" className="bg-muted/30">
                    <Info className="h-4 w-4" />
                    <AlertDescription className="text-sm">
                      No significant disease probabilities detected. Continue regular dental check-ups and maintain good
                      oral hygiene.
                    </AlertDescription>
                  </Alert>
                )}
              </div>

              <Alert className="border-accent/50 bg-accent/5">
                <Info className="h-4 w-4 text-accent" />
                <AlertDescription className="text-sm leading-relaxed">
                  This is an AI-assisted diagnostic tool. Results should be verified by a qualified healthcare
                  professional. Always consult with a dentist or gastroenterologist for proper diagnosis and treatment.
                </AlertDescription>
              </Alert>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
