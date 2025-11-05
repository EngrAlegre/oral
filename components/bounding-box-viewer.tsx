"use client"

import { useEffect, useRef, useState } from "react"
import type { DetectedLesion } from "@/lib/disease-computation"

interface BoundingBoxViewerProps {
  imageUrl: string
  detections: DetectedLesion[]
}

// Colors for different bounding boxes (matching Roboflow style)
const BBOX_COLORS = [
  "#00D4FF", // Cyan
  "#FF6B9D", // Pink
  "#FFB800", // Orange
  "#00FF88", // Green
  "#9D00FF", // Purple
]

export function BoundingBoxViewer({ imageUrl, detections }: BoundingBoxViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.src = imageUrl

    img.onload = () => {
      imageRef.current = img
      const canvas = canvasRef.current
      if (!canvas) return

      // Set canvas size to match container while maintaining aspect ratio
      const containerWidth = canvas.parentElement?.clientWidth || 800
      const aspectRatio = img.height / img.width
      const canvasWidth = Math.min(containerWidth, img.width)
      const canvasHeight = canvasWidth * aspectRatio

      setDimensions({ width: canvasWidth, height: canvasHeight })

      canvas.width = canvasWidth
      canvas.height = canvasHeight

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      // Draw the image
      ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight)

      // Draw bounding boxes
      const scaleX = canvasWidth / img.width
      const scaleY = canvasHeight / img.height

      detections.forEach((detection, index) => {
        if (!detection.bbox) return

        const color = BBOX_COLORS[index % BBOX_COLORS.length]
        const { x, y, width, height } = detection.bbox

        // Scale coordinates
        const scaledX = x * scaleX
        const scaledY = y * scaleY
        const scaledWidth = width * scaleX
        const scaledHeight = height * scaleY

        // Draw bounding box
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight)

        // Draw confidence badge
        const confidence = `${detection.confidence}%`
        ctx.font = "bold 16px system-ui"
        const textMetrics = ctx.measureText(confidence)
        const badgeWidth = textMetrics.width + 16
        const badgeHeight = 28

        // Badge background
        ctx.fillStyle = color
        ctx.fillRect(scaledX, scaledY - badgeHeight - 4, badgeWidth, badgeHeight)

        // Badge text
        ctx.fillStyle = "#000000"
        ctx.textBaseline = "middle"
        ctx.fillText(confidence, scaledX + 8, scaledY - badgeHeight / 2 - 4)
      })

      // Draw object count badge in bottom right
      const objectCount = `${detections.length} object${detections.length !== 1 ? "s" : ""} detected`
      ctx.font = "14px system-ui"
      const countMetrics = ctx.measureText(objectCount)
      const countBadgeWidth = countMetrics.width + 20
      const countBadgeHeight = 32
      const countBadgeX = canvasWidth - countBadgeWidth - 16
      const countBadgeY = canvasHeight - countBadgeHeight - 16

      // Count badge background
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
      ctx.fillRect(countBadgeX, countBadgeY, countBadgeWidth, countBadgeHeight)

      // Count badge text
      ctx.fillStyle = "#FFFFFF"
      ctx.textBaseline = "middle"
      ctx.fillText(objectCount, countBadgeX + 10, countBadgeY + countBadgeHeight / 2)
    }
  }, [imageUrl, detections])

  return (
    <div className="relative w-full overflow-hidden rounded-lg border border-border bg-black">
      <canvas
        ref={canvasRef}
        className="w-full h-auto"
        style={{ maxHeight: "600px", objectFit: "contain" }}
        width={dimensions.width}
        height={dimensions.height}
      />
    </div>
  )
}
