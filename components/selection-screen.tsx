"use client"
import { Camera, Upload, History, Menu, User, Info, Stethoscope, LogIn } from "lucide-react"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Badge } from "@/components/ui/badge"
import { useState } from "react"

interface SelectionScreenProps {
  onSelectCamera: () => void
  onSelectUpload: () => void
  onSelectHistory: () => void
  onSelectDiseaseInfo: () => void
  onSelectOralLesionsInfo: () => void
  onSelectAccount: () => void
  user: any
  onSignIn: () => void
}

export function SelectionScreen({
  onSelectCamera,
  onSelectUpload,
  onSelectHistory,
  onSelectDiseaseInfo,
  onSelectOralLesionsInfo,
  onSelectAccount,
  user,
  onSignIn,
}: SelectionScreenProps) {
  const [menuOpen, setMenuOpen] = useState(false)

  const handleMenuItemClick = (action: () => void) => {
    setMenuOpen(false)
    action()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 flex flex-col">
      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="w-80">
          <SheetHeader>
            <SheetTitle>Information</SheetTitle>
          </SheetHeader>
          <div className="mt-8 space-y-2">
            <button
              onClick={() => handleMenuItemClick(onSelectDiseaseInfo)}
              className="w-full flex items-center gap-4 rounded-lg p-4 text-left transition-colors hover:bg-muted"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Stethoscope className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Disease Information</h3>
                <p className="text-sm text-muted-foreground">Learn about GI diseases</p>
              </div>
            </button>
            <button
              onClick={() => handleMenuItemClick(onSelectOralLesionsInfo)}
              className="w-full flex items-center gap-4 rounded-lg p-4 text-left transition-colors hover:bg-muted"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Info className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Oral Lesions Info</h3>
                <p className="text-sm text-muted-foreground">Learn about oral lesions</p>
              </div>
            </button>
          </div>
        </SheetContent>
      </Sheet>

      <header className="border-b border-border bg-card/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setMenuOpen(true)}
              className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-primary transition-all hover:bg-primary/20 hover:scale-105 active:scale-95"
              aria-label="Menu"
            >
              <Menu className="h-6 w-6" />
            </button>

            {/* Center: Logo */}
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
              <img src="/images/pangil-logo.png" alt="PANGIL" className="h-12 w-24 object-contain" />
            </div>

            {/* Right: History and Account */}
            <div className="flex items-center gap-2">
              <button
                onClick={onSelectHistory}
                className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-primary transition-all hover:bg-primary/20 hover:scale-105 active:scale-95"
                aria-label="History"
              >
                <History className="h-6 w-6" />
              </button>
              {user ? (
                <button
                  onClick={onSelectAccount}
                  className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-primary transition-all hover:bg-primary/20 hover:scale-105 active:scale-95 relative"
                  aria-label="Account"
                >
                  <User className="h-6 w-6" />
                  <Badge className="absolute -top-1 -right-1 h-3 w-3 p-0 bg-green-500" />
                </button>
              ) : (
                <button
                  onClick={onSignIn}
                  className="flex h-12 px-4 items-center justify-center gap-2 rounded-lg bg-primary text-primary-foreground transition-all hover:bg-primary/90 hover:scale-105 active:scale-95"
                  aria-label="Sign In"
                >
                  <LogIn className="h-5 w-5" />
                  <span className="text-sm font-medium">Sign In</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 flex items-center justify-center px-4 py-8">
        <div className="w-full max-w-2xl">
          <div className="mb-12 text-center">
            {user ? (
              <>
                <h2 className="text-4xl font-bold text-foreground mb-4">
                  Welcome, {user.full_name?.split(" ")[0] || "Doctor"}!
                </h2>
                <p className="text-lg text-muted-foreground">Choose how you want to detect oral lesions</p>
              </>
            ) : (
              <>
                <h2 className="text-4xl font-bold text-foreground mb-4">Start Detection</h2>
                <p className="text-lg text-muted-foreground">Choose how you want to detect oral lesions</p>
              </>
            )}
          </div>

          <div className="grid gap-8 md:grid-cols-2">
            <button
              onClick={onSelectCamera}
              className="group relative overflow-hidden rounded-3xl border-2 border-border bg-card p-12 shadow-xl transition-all hover:scale-105 hover:border-primary hover:shadow-2xl active:scale-95"
            >
              <div className="flex flex-col items-center gap-6">
                <div className="flex h-28 w-28 items-center justify-center rounded-full bg-primary/10 transition-colors group-hover:bg-primary/20">
                  <Camera className="h-14 w-14 text-primary" />
                </div>
                <div className="text-center">
                  <h3 className="text-2xl font-semibold text-foreground mb-2">Camera</h3>
                  <p className="text-base text-muted-foreground">Real-time detection</p>
                </div>
              </div>
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
            </button>

            <button
              onClick={onSelectUpload}
              className="group relative overflow-hidden rounded-3xl border-2 border-border bg-card p-12 shadow-xl transition-all hover:scale-105 hover:border-primary hover:shadow-2xl active:scale-95"
            >
              <div className="flex flex-col items-center gap-6">
                <div className="flex h-28 w-28 items-center justify-center rounded-full bg-primary/10 transition-colors group-hover:bg-primary/20">
                  <Upload className="h-14 w-14 text-primary" />
                </div>
                <div className="text-center">
                  <h3 className="text-2xl font-semibold text-foreground mb-2">Upload</h3>
                  <p className="text-base text-muted-foreground">Select an image</p>
                </div>
              </div>
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
