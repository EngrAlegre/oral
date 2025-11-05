"use client"

import { useEffect, useState } from "react"
import Image from "next/image"
import { Loader2 } from "lucide-react"

interface SplashScreenProps {
  onComplete: () => void
}

export function SplashScreen({ onComplete }: SplashScreenProps) {
  const [loadingProgress, setLoadingProgress] = useState(0)

  useEffect(() => {
    const progressInterval = setInterval(() => {
      setLoadingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(progressInterval)
          return 100
        }
        return prev + 10
      })
    }, 200)

    const timer = setTimeout(() => {
      onComplete()
    }, 2500)

    return () => {
      clearTimeout(timer)
      clearInterval(progressInterval)
    }
  }, [onComplete])

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-br from-primary via-primary/90 to-primary/80">
      <div className="animate-fade-in text-center">
        <div className="mb-8 flex justify-center">
          <div className="relative h-48 w-96 animate-bounce-slow">
            <Image src="/images/pangil-logo.png" alt="PANGIL Logo" fill className="object-contain" priority />
          </div>
        </div>

        <div className="mt-8 flex flex-col items-center gap-4 animate-slide-up">
          <Loader2 className="h-8 w-8 text-white animate-spin" />
          <p className="text-lg text-white/90">Initializing System...</p>
          <div className="w-64 h-2 bg-white/20 rounded-full overflow-hidden">
            <div
              className="h-full bg-white transition-all duration-300 ease-out"
              style={{ width: `${loadingProgress}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
