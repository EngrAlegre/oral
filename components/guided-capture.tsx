"use client"

import { useRef, useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Camera, CameraOff, Check } from "lucide-react"
import { detectMultipleImages } from "@/lib/api-client"

interface GuidedCaptureProps {
  onAnalysisComplete: (result: any) => void
  isAnalyzing: boolean
  setIsAnalyzing: (value: boolean) => void
}

const CAPTURE_ANGLES = [
  { id: 1, label: "Front Teeth", instruction: "Show your front teeth clearly" },
  { id: 2, label: "Top Teeth", instruction: "Tilt head back and show top teeth" },
  { id: 3, label: "Bottom Teeth", instruction: "Tilt head forward and show bottom teeth" },
  { id: 4, label: "Left Side", instruction: "Turn head to show left side of mouth" },
  { id: 5, label: "Right Side", instruction: "Turn head to show right side of mouth" },
  { id: 6, label: "Tongue Out", instruction: "Stick out your tongue" },
  { id: 7, label: "Under Tongue", instruction: "Lift your tongue to show underneath" },
  { id: 8, label: "Gums", instruction: "Show your gums clearly" },
]

export function GuidedCapture({ onAnalysisComplete, isAnalyzing, setIsAnalyzing }: GuidedCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [currentAngleIndex, setCurrentAngleIndex] = useState(0)
  const [capturedImages, setCapturedImages] = useState<File[]>([])
  const [error, setError] = useState<string | null>(null)

  const currentAngle = CAPTURE_ANGLES[currentAngleIndex]
  const progress = (capturedImages.length / CAPTURE_ANGLES.length) * 100

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsCameraActive(true)
        setError(null)
      }
    } catch (err) {
      setError("Unable to access camera. Please check permissions.")
      console.error("[v0] Camera error:", err)
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
      setIsCameraActive(false)
    }
  }

  const captureFrame = async (): Promise<File | null> => {
    if (!videoRef.current || !canvasRef.current) return null

    const canvas = canvasRef.current
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext("2d")

    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0)
      return new Promise((resolve) => {
        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(new File([blob], `angle-${currentAngleIndex + 1}.jpg`, { type: "image/jpeg" }))
            } else {
              resolve(null)
            }
          },
          "image/jpeg",
          0.9,
        )
      })
    }
    return null
  }

  const handleCapture = async () => {
    const frame = await captureFrame()
    if (frame) {
      setCapturedImages([...capturedImages, frame])

      if (currentAngleIndex < CAPTURE_ANGLES.length - 1) {
        setCurrentAngleIndex(currentAngleIndex + 1)
      }
    }
  }

  const handleRetake = () => {
    if (capturedImages.length > 0) {
      setCapturedImages(capturedImages.slice(0, -1))
      setCurrentAngleIndex(Math.max(0, currentAngleIndex - 1))
    }
  }

  const handleAnalyze = async () => {
    if (capturedImages.length !== CAPTURE_ANGLES.length) {
      setError("Please capture all 8 angles before analyzing")
      return
    }

    setIsAnalyzing(true)
    try {
      console.log("[v0] Sending 8 images to backend for analysis")
      const result = await detectMultipleImages(capturedImages)

      const firstCapturedImage = await new Promise<string>((resolve) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.readAsDataURL(capturedImages[0])
      })

      const detectionResult = {
        imageUrl: firstCapturedImage,
        detections: result.allDetections.map((d) => ({
          type: d.label,
          confidence: Math.round(d.confidence * 100) / 100,
          bbox: { x: 0, y: 0, width: 0, height: 0 },
        })),
        diseaseProbabilities: result.diseaseProbabilities.map((dp) => ({
          disease: dp.disease,
          probability: dp.probability,
          risk: dp.probability > 60 ? "High" : dp.probability > 30 ? "Medium" : "Low",
        })),
      }

      onAnalysisComplete(detectionResult)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed")
      console.error("[v0] Analysis error:", err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setCapturedImages([])
    setCurrentAngleIndex(0)
    setError(null)
  }

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  if (capturedImages.length === CAPTURE_ANGLES.length) {
    return (
      <Card className="p-6 space-y-4">
        <div className="text-center space-y-2">
          <Check className="h-12 w-12 text-green-600 mx-auto" />
          <h3 className="text-lg font-semibold">All Angles Captured!</h3>
          <p className="text-sm text-muted-foreground">Ready to analyze all 8 images</p>
        </div>

        <div className="space-y-2">
          <h4 className="font-medium text-sm">Captured Angles:</h4>
          <div className="grid grid-cols-2 gap-2">
            {CAPTURE_ANGLES.map((angle, idx) => (
              <div key={angle.id} className="flex items-center gap-2 text-sm">
                <Check className="h-4 w-4 text-green-600" />
                <span>{angle.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex gap-3">
          <Button onClick={handleReset} variant="outline" className="flex-1 bg-transparent">
            Retake All
          </Button>
          <Button onClick={handleAnalyze} disabled={isAnalyzing} className="flex-1">
            {isAnalyzing ? "Analyzing..." : "Analyze All Images"}
          </Button>
        </div>
      </Card>
    )
  }

  return (
    <Card className="p-6 space-y-4">
      {/* Progress */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <h3 className="font-semibold">Guided Multi-Angle Capture</h3>
          <span className="text-sm font-medium text-muted-foreground">
            {capturedImages.length}/{CAPTURE_ANGLES.length}
          </span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {/* Current Angle Instruction */}
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center space-y-2">
        <h4 className="font-semibold text-red-900">{currentAngle.label}</h4>
        <p className="text-sm text-red-800">{currentAngle.instruction}</p>
      </div>

      {/* Camera View */}
      <div className="relative aspect-video overflow-hidden rounded-lg bg-muted">
        {isCameraActive ? (
          <>
            <video ref={videoRef} autoPlay playsInline className="h-full w-full object-cover" />
            <canvas ref={canvasRef} className="hidden" />
          </>
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <Camera className="mx-auto h-12 w-12 text-muted-foreground" />
              <p className="mt-2 text-sm text-muted-foreground">Camera inactive</p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Controls */}
      <div className="flex gap-3">
        {!isCameraActive ? (
          <Button onClick={startCamera} className="flex-1 h-12">
            <Camera className="mr-2 h-5 w-5" />
            Start Camera
          </Button>
        ) : (
          <>
            <Button onClick={stopCamera} variant="outline" className="flex-1 h-12 bg-transparent">
              <CameraOff className="mr-2 h-5 w-5" />
              Stop
            </Button>
            <Button onClick={handleCapture} disabled={isAnalyzing} className="flex-1 h-12">
              Capture Angle {capturedImages.length + 1}
            </Button>
            {capturedImages.length > 0 && (
              <Button onClick={handleRetake} variant="secondary" className="flex-1 h-12">
                Retake
              </Button>
            )}
          </>
        )}
      </div>
    </Card>
  )
}
