"use client"

import type React from "react"
import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Upload, X, Scan } from "lucide-react"
import Image from "next/image"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { detectLesion } from "@/lib/api-client"
import { calculateDiseaseProbabilities, normalizeLesionName } from "@/lib/disease-probability-calculator"

interface ImageUploadProps {
  onDetection: (result: any) => void
  isAnalyzing: boolean
  setIsAnalyzing: (value: boolean) => void
  userId?: string
}

export function ImageUpload({ onDetection, isAnalyzing, setIsAnalyzing, userId }: ImageUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("image/")) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
      setError(null)
    }
  }

  const clearImage = () => {
    setSelectedImage(null)
    setImageFile(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const analyzeImage = async () => {
    if (!imageFile || !selectedImage) return

    setIsAnalyzing(true)
    setError(null)

    try {
      console.log("[v0] Starting detection with file:", imageFile.name)
      const result = await detectLesion(imageFile, userId)
      console.log("[v0] Detection result:", result)

      const normalizedLesion = normalizeLesionName(result.label)
      const detectedLesion = {
        lesion: normalizedLesion,
        confidence: result.confidence * 100,
      }
      const diseaseProbabilities = calculateDiseaseProbabilities([detectedLesion])

      // Convert FastAPI response to expected format
      const detections = [
        {
          type: result.label,
          confidence: Math.round(result.confidence * 100),
          bbox: {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
          },
        },
      ]

      onDetection({
        imageUrl: selectedImage,
        detections,
        diseaseProbabilities: diseaseProbabilities.map((dp) => ({
          disease: dp.disease,
          probability: dp.probability,
          risk: dp.probability > 60 ? "High" : dp.probability > 30 ? "Medium" : "Low",
        })),
        recommendations: result.recommendation,
        aiFeeback: result.ai_feedback,
        detectionImage: result.detection_image,
        segmentationImage: result.segmentation_image,
        gradcamImage: result.gradcam_image,
      })
    } catch (err) {
      console.error("[v0] Detection error:", err)
      const errorMessage = err instanceof Error ? err.message : "Detection failed. Please try again."
      setError(errorMessage)
      console.error("[v0] Full error details:", {
        error: err,
        backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL,
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-4">
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="relative aspect-video overflow-hidden rounded-lg border-2 border-dashed border-border bg-muted hover:border-primary/50 transition-colors">
        {selectedImage ? (
          <>
            <Image src={selectedImage || "/placeholder.svg"} alt="Selected image" fill className="object-contain" />
            <Button size="icon" variant="destructive" className="absolute right-3 top-3 h-10 w-10" onClick={clearImage}>
              <X className="h-5 w-5" />
            </Button>
          </>
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center p-4">
              <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
              <p className="mt-3 text-base font-medium text-foreground">Click to upload or drag and drop</p>
              <p className="text-sm text-muted-foreground">PNG, JPG up to 10MB</p>
            </div>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="absolute inset-0 cursor-pointer opacity-0"
        />
      </div>

      <Button onClick={analyzeImage} disabled={!selectedImage || isAnalyzing} className="w-full h-12 text-base">
        <Scan className="mr-2 h-5 w-5" />
        {isAnalyzing ? "Analyzing..." : "Analyze Image"}
      </Button>
    </div>
  )
}
