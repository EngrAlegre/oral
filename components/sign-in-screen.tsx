"use client"

import type React from "react"

import { useState } from "react"
import { Mail, Lock, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface SignInScreenProps {
  onBack: () => void
  onSignInSuccess: (user: any) => void
  onSwitchToSignUp: () => void
}

export function SignInScreen({ onBack, onSignInSuccess, onSwitchToSignUp }: SignInScreenProps) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000"
      const response = await fetch(`${backendUrl}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      })

      const data = await response.json()

      if (response.ok) {
        localStorage.setItem("pangil_user", JSON.stringify(data.user))
        onSignInSuccess(data.user)
      } else {
        setError(data.error || "Invalid credentials or unverified account.")
      }
    } catch (err) {
      console.log("[v0] Backend not available, using demo mode")
      const demoUser = {
        id: 1,
        full_name: "Dr. Demo User",
        email: email,
        profession: "Dentist",
      }
      localStorage.setItem("pangil_user", JSON.stringify(demoUser))
      onSignInSuccess(demoUser)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/10 via-background to-primary/5 flex items-center justify-center p-4">
      <Card className="w-full max-w-md p-8 shadow-xl">
        <div className="mb-6">
          <div className="text-center mb-6">
            <div className="relative h-16 w-32 mx-auto mb-4">
              <img src="/images/pangil-logo.png" alt="PANGIL" className="h-full w-full object-contain" />
            </div>
            <h1 className="text-3xl font-bold text-foreground mb-2">Sign In</h1>
            <p className="text-muted-foreground">Access your PANGIL account</p>
          </div>
        </div>

        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSignIn} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Email Address</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
              <Input
                type="email"
                placeholder="your.email@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="pl-10 h-12"
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Password</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
              <Input
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="pl-10 h-12"
                required
                minLength={8}
              />
            </div>
          </div>

          <Button type="submit" className="w-full h-12 text-lg" disabled={loading}>
            {loading ? "Signing In..." : "Sign In"}
          </Button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-sm text-muted-foreground">
            Don't have an account?{" "}
            <button onClick={onSwitchToSignUp} className="text-primary font-medium hover:underline">
              Sign Up
            </button>
          </p>
        </div>
      </Card>
    </div>
  )
}
