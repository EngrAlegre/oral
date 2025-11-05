"use client"

import { useRef, useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Camera, CameraOff, Scan } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { detectLesion } from "@/lib/api-client"

interface CameraViewProps {
  onDetection: (result: any) => void
  isAnalyzing: boolean
  setIsAnalyzing: (value: boolean) => void
}

export function CameraView({ onDetection, isAnalyzing, setIsAnalyzing }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isRealTimeDetection, setIsRealTimeDetection] = useState(false)
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const startCamera = async () => {
    try {
      console.log("[Camera] Starting camera...")
      setError(null)
      
      // Prefer back camera on phones, no audio
      let stream: MediaStream | null = null
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: { ideal: "environment" } }, 
          audio: false 
        })
        console.log("[Camera] Got back camera stream")
      } catch {
        // Fallback to front
        try {
          stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "user" }, 
            audio: false 
          })
          console.log("[Camera] Got front camera stream")
        } catch (e2) {
          // Fallback to first available device
          const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput")
          console.log("[Camera] Available devices:", devices.length)
          if (devices[0]) {
            stream = await navigator.mediaDevices.getUserMedia({ 
              video: { deviceId: { exact: devices[0].deviceId } }, 
              audio: false 
            })
            console.log("[Camera] Got device stream:", devices[0].label)
          } else {
            throw e2
          }
        }
      }
      
      console.log("[Camera] Checking refs - videoRef:", !!videoRef.current, "stream:", !!stream)
      
      if (!videoRef.current) {
        throw new Error("Video element not found")
      }
      
      if (!stream) {
        throw new Error("No stream available")
      }
      
      videoRef.current.srcObject = stream
      console.log("[Camera] Stream attached to video element")
      
      // Wait for video to be ready
      await new Promise<void>((resolve, reject) => {
        if (!videoRef.current) return resolve()
        
        const timeout = setTimeout(() => {
          console.warn("[Camera] Timeout waiting for video ready")
          resolve()
        }, 3000)
        
        const onReady = () => {
          clearTimeout(timeout)
          console.log("[Camera] Video ready, dimensions:", videoRef.current?.videoWidth, "x", videoRef.current?.videoHeight)
          resolve()
        }
        
        videoRef.current.onloadedmetadata = onReady
        videoRef.current.oncanplay = onReady
      })
      
      // Start playback
      try { 
        await videoRef.current.play()
        console.log("[Camera] Video playing")
      } catch (playErr) {
        console.error("[Camera] Play error:", playErr)
      }
      
      setIsCameraActive(true)
      console.log("[Camera] Camera activated successfully")
    } catch (err: any) {
      console.error("[Camera] Failed to start:", err)
      let msg = "Unable to access camera. Please check permissions."
      if (err?.name === "NotReadableError") msg = "Camera is in use by another app. Close it and try again."
      if (err?.name === "NotAllowedError") msg = "Camera permission denied. Allow it in site settings."
      if (err?.name === "NotFoundError") msg = "No camera device found."
      setError(msg)
      setIsCameraActive(false)
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
      setIsCameraActive(false)
    }
    stopRealTimeDetection()
  }

  const captureFrame = async (): Promise<File | null> => {
    if (!videoRef.current || !canvasRef.current) return null

    const video = videoRef.current
    const canvas = canvasRef.current
    
    // Check if video is actually playing
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
      console.warn("[Camera] Video not ready")
      return null
    }

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext("2d")
    
    if (!ctx) return null

    ctx.drawImage(video, 0, 0)
    
    return new Promise((resolve) => {
      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(new File([blob], "frame.jpg", { type: "image/jpeg" }))
          } else {
            resolve(null)
          }
        },
        "image/jpeg",
        0.9,
      )
    })
  }

  const startRealTimeDetection = async () => {
    setIsRealTimeDetection(true)
    setIsAnalyzing(true)

    detectionIntervalRef.current = setInterval(async () => {
      const frame = await captureFrame()
      if (frame) {
        try {
          const result = await detectLesion(frame)
          onDetection({
            imageUrl: URL.createObjectURL(frame),
            ...result,
          })
        } catch (err) {
          console.error("[v0] Real-time detection error:", err)
        }
      }
    }, 2000)
  }

  const stopRealTimeDetection = () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }
    setIsRealTimeDetection(false)
    setIsAnalyzing(false)
  }

  const captureAndAnalyze = async () => {
    if (!videoRef.current) return

    setIsAnalyzing(true)
    const frame = await captureFrame()
    if (frame) {
      try {
        const result = await detectLesion(frame)
        const imageUrl = URL.createObjectURL(frame)
        onDetection({
          imageUrl,
          ...result,
        })
      } catch (err) {
        console.error("[v0] Detection error:", err)
        setError(err instanceof Error ? err.message : "Detection failed. Please try again.")
      } finally {
        setIsAnalyzing(false)
      }
    }
  }

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="space-y-4">
      <div className="relative aspect-video overflow-hidden rounded-lg bg-muted">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className={`h-full w-full object-cover ${!isCameraActive ? 'hidden' : ''}`}
        />
        <canvas ref={canvasRef} className="hidden" />
        
        {!isCameraActive && (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <Camera className="mx-auto h-12 w-12 text-muted-foreground" />
              <p className="mt-2 text-sm text-muted-foreground">Camera inactive</p>
            </div>
          </div>
        )}
        
        {isRealTimeDetection && (
          <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-medium">
            <div className="h-2 w-2 bg-white rounded-full animate-pulse" />
            Live Detection
          </div>
        )}
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex gap-3">
        {!isCameraActive ? (
          <Button onClick={startCamera} className="flex-1 h-12 text-base">
            <Camera className="mr-2 h-5 w-5" />
            Start Camera
          </Button>
        ) : (
          <>
            <Button onClick={stopCamera} variant="outline" className="flex-1 h-12 text-base bg-transparent">
              <CameraOff className="mr-2 h-5 w-5" />
              Stop Camera
            </Button>
            <Button
              onClick={captureAndAnalyze}
              disabled={isAnalyzing || isRealTimeDetection}
              className="flex-1 h-12 text-base"
            >
              <Scan className="mr-2 h-5 w-5" />
              {isAnalyzing ? "Analyzing..." : "Capture"}
            </Button>
            <Button
              onClick={isRealTimeDetection ? stopRealTimeDetection : startRealTimeDetection}
              variant={isRealTimeDetection ? "destructive" : "secondary"}
              className="flex-1 h-12 text-base"
            >
              {isRealTimeDetection ? "Stop Live" : "Start Live"}
            </Button>
          </>
        )}
      </div>
    </div>
  )
}
