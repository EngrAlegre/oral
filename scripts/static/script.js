// Global state
let stream = null
let currentImage = null
let isAnalyzing = false

// DOM Elements
const tabs = document.querySelectorAll(".tab-button")
const tabContents = document.querySelectorAll(".tab-content")
const video = document.getElementById("video")
const canvas = document.getElementById("canvas")
const startCameraBtn = document.getElementById("start-camera")
const captureBtn = document.getElementById("capture")
const dropZone = document.getElementById("drop-zone")
const fileInput = document.getElementById("file-input")
const previewContainer = document.getElementById("preview-container")
const previewImage = document.getElementById("preview-image")
const analyzeUploadBtn = document.getElementById("analyze-upload")
const resultsPanel = document.getElementById("results-panel")

// Color palette for bounding boxes
const COLORS = [
  "#3b82f6", // blue
  "#10b981", // green
  "#f59e0b", // orange
  "#ef4444", // red
  "#8b5cf6", // purple
  "#ec4899", // pink
]

// Recommendations database
const RECOMMENDATIONS = {
  "Crohn's Disease": {
    high: "Immediate gastroenterology consultation recommended. Monitor for abdominal pain, diarrhea, and weight loss.",
    medium: "Schedule gastroenterology appointment within 2 weeks. Track digestive symptoms.",
    low: "Consider gastroenterology screening if symptoms develop. Maintain oral hygiene.",
  },
  "Ulcerative Colitis": {
    high: "Urgent gastroenterology evaluation needed. Watch for bloody stools and severe abdominal cramping.",
    medium: "Gastroenterology consultation advised within 2-3 weeks. Monitor bowel habits.",
    low: "Routine screening recommended if family history present. Continue regular dental care.",
  },
  "Celiac Disease": {
    high: "Immediate evaluation for celiac disease recommended. Consider gluten-free diet trial and serological testing.",
    medium: "Schedule celiac screening within 2 weeks. Document dietary symptoms.",
    low: "Consider celiac screening if symptoms persist. Monitor for digestive issues.",
  },
  "Chronic Liver Disease": {
    high: "Urgent hepatology consultation required. Monitor for jaundice, fatigue, and abdominal swelling.",
    medium: "Liver function tests and hepatology consultation recommended within 2 weeks.",
    low: "Routine liver screening advised. Maintain healthy lifestyle and avoid alcohol.",
  },
}

// Tab switching
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const tabName = tab.dataset.tab

    // Update active tab
    tabs.forEach((t) => t.classList.remove("active"))
    tab.classList.add("active")

    // Update active content
    tabContents.forEach((content) => {
      content.classList.remove("active")
      if (content.id === `${tabName}-tab`) {
        content.classList.add("active")
      }
    })

    // Stop camera if switching away from camera tab
    if (tabName !== "camera" && stream) {
      stopCamera()
    }
  })
})

// Camera functions
startCameraBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
    })
    video.srcObject = stream
    startCameraBtn.disabled = true
    captureBtn.disabled = false
  } catch (error) {
    alert("Error accessing camera: " + error.message)
  }
})

captureBtn.addEventListener("click", () => {
  // Capture frame from video
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const ctx = canvas.getContext("2d")
  ctx.drawImage(video, 0, 0)

  // Convert to blob and analyze
  canvas.toBlob((blob) => {
    analyzeImage(blob, canvas.toDataURL())
  }, "image/jpeg")
})

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop())
    stream = null
    video.srcObject = null
    startCameraBtn.disabled = false
    captureBtn.disabled = true
  }
}

// Upload functions
dropZone.addEventListener("click", () => {
  fileInput.click()
})

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault()
  dropZone.style.borderColor = "var(--primary)"
})

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "var(--border)"
})

dropZone.addEventListener("drop", (e) => {
  e.preventDefault()
  dropZone.style.borderColor = "var(--border)"

  const file = e.dataTransfer.files[0]
  if (file && file.type.startsWith("image/")) {
    handleImageFile(file)
  }
})

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0]
  if (file) {
    handleImageFile(file)
  }
})

function handleImageFile(file) {
  const reader = new FileReader()
  reader.onload = (e) => {
    previewImage.src = e.target.result
    currentImage = file
    dropZone.style.display = "none"
    previewContainer.style.display = "block"
  }
  reader.readAsDataURL(file)
}

analyzeUploadBtn.addEventListener("click", () => {
  if (currentImage) {
    analyzeImage(currentImage, previewImage.src)
  }
})

// Analysis function
async function analyzeImage(imageBlob, imageDataUrl) {
  if (isAnalyzing) return

  isAnalyzing = true
  resultsPanel.style.display = "none"

  // Show loading state
  const loadingMsg = document.createElement("div")
  loadingMsg.className = "banner banner-info"
  loadingMsg.innerHTML = "<span>Analyzing image...</span>"
  document.querySelector(".main-content").appendChild(loadingMsg)

  try {
    const formData = new FormData()
    formData.append("image", imageBlob)

    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      throw new Error("Analysis failed")
    }

    const data = await response.json()
    displayResults(data, imageDataUrl)
  } catch (error) {
    console.error("Analysis error:", error)
    alert("Error analyzing image: " + error.message)
  } finally {
    isAnalyzing = false
    loadingMsg.remove()
  }
}

// Display results
function displayResults(data, imageUrl) {
  const { detections, disease_probabilities } = data

  // Show results panel
  resultsPanel.style.display = "block"
  resultsPanel.scrollIntoView({ behavior: "smooth" })

  // Display image with bounding boxes
  displayDetectionImage(imageUrl, detections)

  // Display detected lesions
  displayLesions(detections)

  // Display disease probabilities
  displayDiseaseProbabilities(disease_probabilities)

  // Display recommendations
  displayRecommendations(disease_probabilities)
}

function displayDetectionImage(imageUrl, detections) {
  const detectionImage = document.getElementById("detection-image")
  const boundingBoxes = document.getElementById("bounding-boxes")
  const objectCount = document.getElementById("object-count")

  detectionImage.src = imageUrl
  boundingBoxes.innerHTML = ""
  objectCount.textContent = `${detections.length} object${detections.length !== 1 ? "s" : ""} detected`

  // Wait for image to load to get correct dimensions
  detectionImage.onload = () => {
    const imgRect = detectionImage.getBoundingClientRect()
    const scaleX = imgRect.width / detectionImage.naturalWidth
    const scaleY = imgRect.height / detectionImage.naturalHeight

    detections.forEach((detection, index) => {
      const color = COLORS[index % COLORS.length]
      const bbox = detection.bbox

      const box = document.createElement("div")
      box.className = "bbox"
      box.style.left = `${bbox.x * scaleX}px`
      box.style.top = `${bbox.y * scaleY}px`
      box.style.width = `${bbox.width * scaleX}px`
      box.style.height = `${bbox.height * scaleY}px`
      box.style.borderColor = color

      const label = document.createElement("div")
      label.className = "bbox-label"
      label.style.backgroundColor = color
      label.textContent = `${Math.round(detection.confidence)}%`

      box.appendChild(label)
      boundingBoxes.appendChild(box)
    })
  }
}

function displayLesions(detections) {
  const lesionsList = document.getElementById("lesions-list")
  lesionsList.innerHTML = ""

  detections.forEach((detection, index) => {
    const color = COLORS[index % COLORS.length]
    const item = document.createElement("div")
    item.className = "lesion-item"
    item.style.borderLeftColor = color
    item.innerHTML = `
            <span class="lesion-name">${detection.type}</span>
            <span class="lesion-confidence">${Math.round(detection.confidence)}%</span>
        `
    lesionsList.appendChild(item)
  })
}

function displayDiseaseProbabilities(probabilities) {
  const diseaseList = document.getElementById("disease-list")
  diseaseList.innerHTML = ""

  Object.entries(probabilities).forEach(([disease, probability]) => {
    const item = document.createElement("div")
    item.className = "disease-item"
    item.innerHTML = `
            <div class="disease-header">
                <span class="disease-name">${disease}</span>
                <span class="disease-probability">${probability.toFixed(1)}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${probability}%"></div>
            </div>
        `
    diseaseList.appendChild(item)
  })
}

function displayRecommendations(probabilities) {
  const recommendationsList = document.getElementById("recommendations-list")
  recommendationsList.innerHTML = ""

  Object.entries(probabilities).forEach(([disease, probability]) => {
    if (probability > 0) {
      const severity = probability >= 60 ? "high" : probability >= 30 ? "medium" : "low"
      const recommendation = RECOMMENDATIONS[disease][severity]

      const item = document.createElement("div")
      item.className = `recommendation-item ${severity}`
      item.innerHTML = `
                <div class="recommendation-header">
                    <span class="recommendation-disease">${disease}</span>
                    <span class="recommendation-badge ${severity}">${severity} risk</span>
                </div>
                <p class="recommendation-text">${recommendation}</p>
            `
      recommendationsList.appendChild(item)
    }
  })
}
