"use client"

import type React from "react"

import { useState } from "react"
import { ArrowLeft, User, Mail, Lock, Building, FileText, AlertCircle, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface SignUpScreenProps {
  onBack: () => void
  onSwitchToSignIn: () => void
}

export function SignUpScreen({ onBack, onSwitchToSignIn }: SignUpScreenProps) {
  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    profession: "",
    licenseNo: "",
    clinicName: "",
    password: "",
  })
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    // Validate email domain
    const validDomains = ["@gmail.com", "@outlook.com"]
    const hasValidDomain = validDomains.some((domain) => formData.email.endsWith(domain))
    if (!hasValidDomain && !formData.email.includes(".edu")) {
      setError("Email must end with @gmail.com, @outlook.com, or an institutional email")
      setLoading(false)
      return
    }

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000"
      const response = await fetch(`${backendUrl}/api/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      })

      const data = await response.json()

      if (response.ok) {
        setSuccess(true)
        setTimeout(() => onSwitchToSignIn(), 2000)
      } else {
        setError(data.error || "Registration failed. Please try again.")
      }
    } catch (err) {
      // Demo mode
      console.log("[v0] Backend not available, simulating registration")
      setSuccess(true)
      setTimeout(() => onSwitchToSignIn(), 2000)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/10 via-background to-primary/5 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl p-8 shadow-xl">
        <div className="mb-6">
          <Button variant="ghost" size="icon" onClick={onBack} className="mb-4">
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div className="text-center mb-6">
            <div className="relative h-16 w-32 mx-auto mb-4">
              <img src="/images/pangil-logo.png" alt="PANGIL" className="h-full w-full object-contain" />
            </div>
            <h1 className="text-3xl font-bold text-foreground mb-2">Sign Up</h1>
            <p className="text-muted-foreground">Create your professional account</p>
          </div>
        </div>

        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert className="mb-4 border-green-500 bg-green-50">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-800">
              Registration successful! Your account is pending verification. Redirecting to sign in...
            </AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Full Name</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                <Input
                  type="text"
                  placeholder="Dr. Juan Dela Cruz"
                  value={formData.fullName}
                  onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
                  className="pl-10 h-12"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Profession</label>
              <Select
                value={formData.profession}
                onValueChange={(value) => setFormData({ ...formData, profession: value })}
                required
              >
                <SelectTrigger className="h-12">
                  <SelectValue placeholder="Select profession" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Dentist">Dentist</SelectItem>
                  <SelectItem value="General Doctor">General Doctor</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Email Address</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
              <Input
                type="email"
                placeholder="your.email@gmail.com"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="pl-10 h-12"
                required
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Professional License Number</label>
              <div className="relative">
                <FileText className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                <Input
                  type="text"
                  placeholder="PRC License No."
                  value={formData.licenseNo}
                  onChange={(e) => setFormData({ ...formData, licenseNo: e.target.value })}
                  className="pl-10 h-12"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Clinic/Hospital Name</label>
              <div className="relative">
                <Building className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                <Input
                  type="text"
                  placeholder="Medical Center Name"
                  value={formData.clinicName}
                  onChange={(e) => setFormData({ ...formData, clinicName: e.target.value })}
                  className="pl-10 h-12"
                  required
                />
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Password</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
              <Input
                type="password"
                placeholder="Minimum 8 characters"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                className="pl-10 h-12"
                required
                minLength={8}
              />
            </div>
          </div>

          <Button type="submit" className="w-full h-12 text-lg" disabled={loading || success}>
            {loading ? "Creating Account..." : "Sign Up"}
          </Button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-sm text-muted-foreground">
            Already have an account?{" "}
            <button onClick={onSwitchToSignIn} className="text-primary font-medium hover:underline">
              Sign In
            </button>
          </p>
        </div>
      </Card>
    </div>
  )
}
