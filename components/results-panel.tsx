"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, Info, ChevronLeft, ChevronRight } from "lucide-react"
import { Spinner } from "@/components/ui/spinner"
import { Button } from "@/components/ui/button"
import { BoundingBoxViewer } from "@/components/bounding-box-viewer"

interface ResultsPanelProps {
  result: any
  isAnalyzing: boolean
}

export function ResultsPanel({ result, isAnalyzing }: ResultsPanelProps) {
  if (isAnalyzing) {
    return (
      <Card className="p-6 shadow-lg">
        <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-4">
          <Spinner className="h-12 w-12" />
          <p className="text-sm text-muted-foreground">Analyzing image...</p>
        </div>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card className="p-6 shadow-lg">
        <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-4">
          <Info className="h-12 w-12 text-muted-foreground" />
          <div className="text-center">
            <p className="text-base font-medium text-foreground">No analysis yet</p>
            <p className="text-sm text-muted-foreground">Capture or upload an image to begin detection</p>
          </div>
        </div>
      </Card>
    )
  }

  const detections = result.detections || []
  const imageUrl = result.imageUrl
  const detectionImage = result.detectionImage // Backend's image with label overlay
  const gradcamImage = result.gradcamImage // GradCAM rainbow heatmap
  const recommendations = result.recommendations || {}
  const disease = result.diseaseProbabilities?.[0]?.disease || "Unknown"
  const probability = result.diseaseProbabilities?.[0]?.probability || 0 // Renamed from confidence to probability

  // Carousel state for switching between visualizations
  const [currentView, setCurrentView] = useState(0)
  
  // Available visualization views
  const views = [
    { name: "Detection", image: detectionImage, description: "Original image with detection overlay" },
    { name: "GradCAM", image: gradcamImage, description: "AI attention heatmap showing focus areas" }
  ].filter(view => view.image) // Only show views that have images

  const handlePrevious = () => {
    setCurrentView((prev) => (prev === 0 ? views.length - 1 : prev - 1))
  }

  const handleNext = () => {
    setCurrentView((prev) => (prev === views.length - 1 ? 0 : prev + 1))
  }

  return (
    <Card className="p-6 shadow-lg">
      <h2 className="mb-4 text-lg font-semibold text-card-foreground">Detection Results</h2>

      <div className="space-y-6">
        {views.length > 0 && detections.length > 0 && (
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
              
              {/* Navigation Buttons */}
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
        )}

        {/* Detection Status */}
        {detections.length > 0 ? (
          <Alert variant="default" className="border-primary/50 bg-primary/5">
            <AlertCircle className="h-5 w-5 text-primary" />
            <AlertTitle className="text-base">Oral Lesion Detected</AlertTitle>
            <AlertDescription>
              {disease} detected with {probability.toFixed(1)}% linkage probability
            </AlertDescription>
          </Alert>
        ) : (
          <Alert variant="default">
            <Info className="h-5 w-5" />
            <AlertTitle className="text-base">No Lesions Detected</AlertTitle>
            <AlertDescription>No oral lesions were detected in the image</AlertDescription>
          </Alert>
        )}

        {detections.length > 0 && (
          <>
            <div className="space-y-3">
              <h3 className="text-base font-semibold text-foreground">Detected Lesions</h3>
              <div className="space-y-2">
                {detections.map((detection: any, idx: number) => (
                  <div key={idx} className="rounded-lg border border-border bg-card p-4">
                    <p className="font-medium text-foreground">{detection.type}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <h3 className="text-base font-semibold text-foreground">Disease Probability</h3>
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="flex items-center justify-between">
                  <p className="font-medium text-foreground">{disease}</p>
                  <Badge variant="secondary" className="text-base px-3 py-1">
                    {probability.toFixed(1)}%
                  </Badge>
                </div>
              </div>
            </div>

            {/* Medical Recommendations */}
            {recommendations && Object.keys(recommendations).length > 0 && (
              <div className="space-y-4">
                <h3 className="text-base font-semibold text-foreground">Medical Recommendations</h3>

                <div className="rounded-lg border border-primary/50 bg-primary/5 p-4">
                  <div className="space-y-3 text-sm">
                    {recommendations.urgent_actions && (
                      <div className="flex gap-3">
                        <div className="mt-1 flex-shrink-0">
                          <div className="h-2 w-2 rounded-full bg-destructive" />
                        </div>
                        <div>
                          <p className="font-medium text-foreground">Urgent Actions:</p>
                          <ul className="text-muted-foreground leading-relaxed list-disc list-inside">
                            {recommendations.urgent_actions.map((action: string, idx: number) => (
                              <li key={idx}>{action}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}

                    {recommendations.monitoring && (
                      <div className="flex gap-3">
                        <div className="mt-1 flex-shrink-0">
                          <div className="h-2 w-2 rounded-full bg-primary" />
                        </div>
                        <div>
                          <p className="font-medium text-foreground">Monitoring:</p>
                          <ul className="text-muted-foreground leading-relaxed list-disc list-inside">
                            {recommendations.monitoring.map((item: string, idx: number) => (
                              <li key={idx}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}

                    {recommendations.lifestyle && (
                      <div className="flex gap-3">
                        <div className="mt-1 flex-shrink-0">
                          <div className="h-2 w-2 rounded-full bg-accent" />
                        </div>
                        <div>
                          <p className="font-medium text-foreground">Lifestyle Considerations:</p>
                          <ul className="text-muted-foreground leading-relaxed list-disc list-inside">
                            {recommendations.lifestyle.map((item: string, idx: number) => (
                              <li key={idx}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Disclaimer */}
            <Alert className="border-accent/50 bg-accent/5">
              <Info className="h-4 w-4 text-accent" />
              <AlertDescription className="text-sm leading-relaxed">
                This is an AI-assisted diagnostic tool. Results should be verified by a qualified healthcare
                professional. Always consult with a dentist or healthcare provider for proper diagnosis and treatment.
              </AlertDescription>
            </Alert>
          </>
        )}
      </div>
    </Card>
  )
}
