"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "lucide-react"

interface ResultSlidesProps {
  result: any
}

export function ResultSlides({ result }: ResultSlidesProps) {
  const [currentSlide, setCurrentSlide] = useState(0)

  if (!result) return null

  const slides = [
    {
      title: "Detection Result",
      image: result.detection_image,
      label: result.label,
      confidence: result.confidence,
      feedback: result.ai_feedback?.detection,
    },
    {
      title: "Segmentation Result",
      image: result.segmentation_image,
      label: "Lesion Segmentation",
      confidence: result.confidence,
      feedback: result.ai_feedback?.segmentation,
    },
    {
      title: "Grad-CAM Result",
      image: result.gradcam_image,
      label: "Activation Heatmap",
      confidence: result.confidence,
      feedback: result.ai_feedback?.gradcam,
    },
  ]

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length)
  }

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length)
  }

  return (
    <div className="space-y-4">
      <div className="relative aspect-video overflow-hidden rounded-lg bg-muted">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentSlide}
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0"
          >
            {slides[currentSlide].image && (
              <img
                src={slides[currentSlide].image || "/placeholder.svg"}
                alt={slides[currentSlide].title}
                className="h-full w-full object-contain"
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-foreground">{slides[currentSlide].title}</h3>
            <Badge variant="secondary" className="text-base px-3 py-1">
              {(slides[currentSlide].confidence * 100).toFixed(1)}%
            </Badge>
          </div>

          <p className="text-sm text-muted-foreground leading-relaxed">{slides[currentSlide].feedback}</p>

          <div className="flex items-center justify-between pt-4">
            <Button onClick={prevSlide} variant="outline" size="icon">
              <ChevronLeft className="h-5 w-5" />
            </Button>

            <div className="flex gap-2">
              {slides.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => setCurrentSlide(idx)}
                  className={`h-2 rounded-full transition-all ${
                    idx === currentSlide ? "bg-primary w-8" : "bg-muted-foreground w-2"
                  }`}
                />
              ))}
            </div>

            <Button onClick={nextSlide} variant="outline" size="icon">
              <ChevronRight className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}
