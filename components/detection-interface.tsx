"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { CameraView } from "@/components/camera-view"
import { ImageUpload } from "@/components/image-upload"
import { ResultsPanel } from "@/components/results-panel"
import { SplashScreen } from "@/components/splash-screen"
import { SelectionScreen } from "@/components/selection-screen"
import { HistoryScreen } from "@/components/history-screen"
import { DiseaseInfoScreen } from "@/components/disease-info-screen"
import { OralLesionsInfoScreen } from "@/components/oral-lesions-info-screen"
import { AccountScreen } from "@/components/account-screen"
import { SignInScreen } from "@/components/sign-in-screen"
import { SignUpScreen } from "@/components/sign-up-screen"
import { ArrowLeft } from "lucide-react"

type Screen =
  | "splash"
  | "auth"
  | "selection"
  | "camera"
  | "upload"
  | "results"
  | "history"
  | "diseaseInfo"
  | "oralLesionsInfo"
  | "account"
  | "signIn"
  | "signUp"

export function DetectionInterface() {
  const [currentScreen, setCurrentScreen] = useState<Screen>("splash")
  const [detectionResult, setDetectionResult] = useState<any>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [user, setUser] = useState<any>(null)

  useEffect(() => {
    const storedUser = localStorage.getItem("pangil_user")
    if (storedUser) {
      setUser(JSON.parse(storedUser))
    }
  }, [])

  const handleDetection = (result: any) => {
    const historyItem = {
      id: Date.now().toString(),
      date: new Date().toLocaleDateString(),
      time: new Date().toLocaleTimeString(),
      imageUrl: result.imageUrl,
      detectionImage: result.detectionImage, // Detection overlay
      gradcamImage: result.gradcamImage, // GradCAM heatmap
      detections: result.detections,
      diseaseProbabilities: result.diseaseProbabilities || [],
    }

    const existingHistory = localStorage.getItem("pangil_history")
    const history = existingHistory ? JSON.parse(existingHistory) : []
    history.unshift(historyItem)
    localStorage.setItem("pangil_history", JSON.stringify(history.slice(0, 50)))

    setDetectionResult(result)
    setCurrentScreen("results")
  }

  const handleBack = () => {
    setDetectionResult(null)
    setCurrentScreen("selection")
  }

  const handleSignInSuccess = (userData: any) => {
    setUser(userData)
    setCurrentScreen("selection")
  }

  const handleLogout = () => {
    localStorage.removeItem("pangil_user")
    setUser(null)
    setCurrentScreen("signIn")
  }

  if (currentScreen === "splash") {
    return (
      <SplashScreen
        onComplete={() => {
          // If user is already logged in, go to selection, otherwise go to sign in
          setCurrentScreen(user ? "selection" : "signIn")
        }}
      />
    )
  }

  if (currentScreen === "signIn") {
    return (
      <SignInScreen
        onBack={() => {}} // No back button on sign in screen
        onSignInSuccess={handleSignInSuccess}
        onSwitchToSignUp={() => setCurrentScreen("signUp")}
      />
    )
  }

  if (currentScreen === "signUp") {
    return (
      <SignUpScreen onBack={() => setCurrentScreen("signIn")} onSwitchToSignIn={() => setCurrentScreen("signIn")} />
    )
  }

  if (!user) {
    setCurrentScreen("signIn")
    return null
  }

  if (currentScreen === "selection") {
    return (
      <SelectionScreen
        onSelectCamera={() => setCurrentScreen("camera")}
        onSelectUpload={() => setCurrentScreen("upload")}
        onSelectHistory={() => setCurrentScreen("history")}
        onSelectDiseaseInfo={() => setCurrentScreen("diseaseInfo")}
        onSelectOralLesionsInfo={() => setCurrentScreen("oralLesionsInfo")}
        onSelectAccount={() => setCurrentScreen("account")}
        user={user}
        onSignIn={() => setCurrentScreen("signIn")}
      />
    )
  }

  if (currentScreen === "history") {
    return <HistoryScreen onBack={handleBack} />
  }

  if (currentScreen === "diseaseInfo") {
    return <DiseaseInfoScreen onBack={handleBack} />
  }

  if (currentScreen === "oralLesionsInfo") {
    return <OralLesionsInfoScreen onBack={handleBack} />
  }

  if (currentScreen === "account") {
    return <AccountScreen onBack={handleBack} user={user} onLogout={handleLogout} />
  }

  if (currentScreen === "results") {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card shadow-sm">
          <div className="container mx-auto px-4 py-5">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="icon" onClick={handleBack} className="h-10 w-10">
                <ArrowLeft className="h-5 w-5" />
              </Button>
              <div className="relative h-12 w-24">
                <img src="/images/pangil-logo.png" alt="PANGIL" className="h-full w-full object-contain" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">PANGIL</h1>
                <p className="text-sm text-muted-foreground">Detection Results</p>
              </div>
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-8">
          <ResultsPanel result={detectionResult} isAnalyzing={false} />
        </div>
      </div>
    )
  }

  // Camera or Upload Screen
  return (
    <div className="min-h-screen bg-background">
      {/* Header with Back Button */}
      <header className="border-b border-border bg-card shadow-sm">
        <div className="container mx-auto px-4 py-5">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={handleBack} className="h-10 w-10">
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div className="relative h-12 w-24">
              <img src="/images/pangil-logo.png" alt="PANGIL" className="h-full w-full object-contain" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">PANGIL</h1>
              <p className="text-sm text-muted-foreground">
                {currentScreen === "camera" ? "Real-time Camera" : "Upload Image"}
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Detection Input */}
      <div className="container mx-auto px-4 py-8">
        <Card className="p-6 shadow-lg">
          {currentScreen === "camera" ? (
            <CameraView onDetection={handleDetection} isAnalyzing={isAnalyzing} setIsAnalyzing={setIsAnalyzing} />
          ) : (
            <ImageUpload onDetection={handleDetection} isAnalyzing={isAnalyzing} setIsAnalyzing={setIsAnalyzing} />
          )}
        </Card>
      </div>
    </div>
  )
}
