"use client"

import { ArrowLeft, User, Mail, Building, FileText, Settings, LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

interface AccountScreenProps {
  onBack: () => void
  user: any
  onLogout: () => void
}

export function AccountScreen({ onBack, user, onLogout }: AccountScreenProps) {
  const userData = user || {
    full_name: "Guest User",
    email: "guest@example.com",
    profession: "Guest",
    clinic_name: "N/A",
    license_no: "N/A",
  }

  const history = localStorage.getItem("pangil_history")
  const totalScans = history ? JSON.parse(history).length : 0
  const thisWeek = history
    ? JSON.parse(history).filter((item: any) => {
        const itemDate = new Date(item.date)
        const weekAgo = new Date()
        weekAgo.setDate(weekAgo.getDate() - 7)
        return itemDate >= weekAgo
      }).length
    : 0

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card shadow-sm">
        <div className="container mx-auto px-4 py-5">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={onBack} className="h-10 w-10">
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div className="relative h-12 w-24">
              <img src="/images/pangil-logo.png" alt="PANGIL" className="h-full w-full object-contain" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">Account</h1>
              <p className="text-sm text-muted-foreground">Profile & Settings</p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="mx-auto max-w-4xl space-y-6">
          {/* Profile Card */}
          <Card className="p-8">
            <div className="flex items-center gap-6 mb-6">
              <div className="flex h-24 w-24 items-center justify-center rounded-full bg-primary/10">
                <User className="h-12 w-12 text-primary" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-foreground mb-1">{userData.full_name}</h2>
                <p className="text-muted-foreground">{userData.profession}</p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                <Mail className="h-5 w-5 text-primary" />
                <div>
                  <p className="text-sm text-muted-foreground">Email</p>
                  <p className="font-medium text-foreground">{userData.email}</p>
                </div>
              </div>

              {userData.license_no && userData.license_no !== "N/A" && (
                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <FileText className="h-5 w-5 text-primary" />
                  <div>
                    <p className="text-sm text-muted-foreground">License Number</p>
                    <p className="font-medium text-foreground">{userData.license_no}</p>
                  </div>
                </div>
              )}

              {userData.clinic_name && userData.clinic_name !== "N/A" && (
                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <Building className="h-5 w-5 text-primary" />
                  <div>
                    <p className="text-sm text-muted-foreground">Clinic/Hospital</p>
                    <p className="font-medium text-foreground">{userData.clinic_name}</p>
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Statistics Card */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Statistics</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-primary/5 text-center">
                <p className="text-3xl font-bold text-primary mb-1">{totalScans}</p>
                <p className="text-sm text-muted-foreground">Total Scans</p>
              </div>
              <div className="p-4 rounded-lg bg-primary/5 text-center">
                <p className="text-3xl font-bold text-primary mb-1">{thisWeek}</p>
                <p className="text-sm text-muted-foreground">This Week</p>
              </div>
            </div>
          </Card>

          {/* Settings Card */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Settings
            </h3>
            <div className="space-y-3">
              <Button variant="outline" className="w-full justify-start bg-transparent">
                Edit Profile
              </Button>
              <Button variant="outline" className="w-full justify-start bg-transparent">
                Change Password
              </Button>
              <Button variant="outline" className="w-full justify-start bg-transparent">
                Notification Preferences
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start text-destructive hover:text-destructive hover:bg-destructive/10 bg-transparent"
                onClick={onLogout}
              >
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
