import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.text()
    console.log("[v0] Received image data, length:", body.length)

    const roboflowUrl = "https://detect.roboflow.com/kerl-ir77j-gg4mf/14?api_key=8EY0LXKbk70w1nxht2HN"

    console.log("[v0] Forwarding request to Roboflow API")

    const response = await fetch(roboflowUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: body,
    })

    console.log("[v0] Roboflow response status:", response.status)

    const responseText = await response.text()
    console.log("[v0] Roboflow response text (first 200 chars):", responseText.substring(0, 200))

    if (!response.ok) {
      console.error("[v0] Roboflow API error:", responseText)
      return NextResponse.json({ error: "Detection failed", details: responseText }, { status: response.status })
    }

    try {
      const result = JSON.parse(responseText)
      console.log("[v0] Roboflow API response parsed successfully")
      return NextResponse.json(result)
    } catch (parseError) {
      console.error("[v0] Failed to parse Roboflow response as JSON:", parseError)
      console.error("[v0] Response text:", responseText)
      return NextResponse.json(
        {
          error: "Invalid response from detection API",
          details: "Response was not valid JSON",
          responseText: responseText.substring(0, 500),
        },
        { status: 500 },
      )
    }
  } catch (error) {
    console.error("[v0] API route error:", error)
    return NextResponse.json({ error: "Internal server error", details: String(error) }, { status: 500 })
  }
}
