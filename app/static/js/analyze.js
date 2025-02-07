// State management
let currentFile = null;
let isUploading = false;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('preview-image');
const loadingOverlay = document.getElementById('loadingOverlay');

/**
 * Initialize all event listeners and UI components
 */
function initializeUI() {
    setupUploadZone();
}

/**
 * Set up the upload zone with drag and drop functionality
 */
function setupUploadZone() {
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });
}

/**
 * Handle file selection and preview
 */
function handleFile(file) {
    if (file && file.type.startsWith('image/')) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            hideAnalysisCards();
        };
        reader.readAsDataURL(file);
    }
}

/**
 * Hide all analysis result cards
 */
function hideAnalysisCards() {
    const cards = document.querySelectorAll('.analysis-card');
    cards.forEach(card => {
        card.style.display = 'none';
        card.classList.remove('show');
    });
}

/**
 * Show loading overlay with optional message
 */
function showLoading(message = 'Processing...') {
    loadingOverlay.style.display = 'flex';
    document.getElementById('loadingText').textContent = message;
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

/**
 * Show a toast notification
 */
function showToast(message, type = 'info') {
    Toastify({
        text: message,
        duration: 3000,
        gravity: "bottom",
        position: "right",
        style: {
            background: type === 'error' ? "#dc3545" : 
                       type === 'success' ? "#198754" : 
                       type === 'warning' ? "#ffc107" : "#0d6efd"
        }
    }).showToast();
}

/**
 * Analyze the current image using the API
 */
async function analyzeImage() {
    if (!currentFile || isUploading) return;

    isUploading = true;
    showLoading('Analyzing image...');

    try {
        // Upload image
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('type', 'pets');
        
        const uploadResponse = await fetch('/api/v1/dataset/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Failed to upload image');
        }

        const uploadResult = await uploadResponse.json();
        
        if (!uploadResult.filename) {
            throw new Error('No filename received from server');
        }

        // Wait for image processing
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Analyze image
        const analysisResponse = await fetch(`/api/v1/analysis/analyze/pet/${uploadResult.filename}`, {
            method: 'POST'
        });

        if (!analysisResponse.ok) {
            const errorData = await analysisResponse.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        const result = await analysisResponse.json();
        
        if (!result || !result.result) {
            throw new Error('Invalid analysis result received');
        }
        
        // Check confidence level
        if (result.result.confidence < 0.5) {
            showToast('Analysis completed, but confidence is low due to image quality', 'warning');
        } else {
            showToast('Analysis completed successfully', 'success');
        }
        
        displayResults(result.result);

    } catch (error) {
        console.error('Error:', error);
        showToast(error.message || 'Error analyzing image. Please try again.', 'error');
    } finally {
        isUploading = false;
        hideLoading();
    }
}

/**
 * Display analysis results in the UI
 */
function displayResults(result) {
    // Update Animal Info
    document.getElementById('animalType').textContent = capitalizeFirst(result.labels.animal_type);
    document.getElementById('animalSize').textContent = capitalizeFirst(result.labels.size);
    setConfidenceBar('animalConfidence', result.confidence);
    setConfidenceBar('sizeConfidence', result.size_confidence);

    // Update Health Assessment
    document.getElementById('bodyCondition').textContent = capitalizeFirst(result.labels.body_condition);
    document.getElementById('healthIssues').textContent = capitalizeFirst(result.labels.visible_health_issues);
    document.getElementById('pregnancyIndicators').textContent = capitalizeFirst(result.labels.pregnancy_indicators);
    setConfidenceBar('healthConfidence', result.health_confidence);
    setConfidenceBar('bodyConditionConfidence', result.body_condition_confidence);
    setConfidenceBar('pregnancyConfidence', result.pregnancy_confidence);

    // Update Environment
    document.getElementById('context').textContent = capitalizeFirst(result.labels.context);
    document.getElementById('imageQuality').textContent = capitalizeFirst(result.labels.image_quality);
    setConfidenceBar('contextConfidence', result.context_confidence);
    setConfidenceBar('qualityConfidence', result.quality_confidence);

    // Update Analysis Quality
    setConfidenceBar('overallConfidence', result.confidence);

    // Update Recommendations
    const recommendationsContainer = document.getElementById('recommendations');
    const recommendations = [
        `Based on ${result.labels.visible_health_issues} health status: ${getHealthRecommendation(result.labels.visible_health_issues)}`,
        `Based on ${result.labels.body_condition} body condition: ${getBodyConditionRecommendation(result.labels.body_condition)}`,
        `Based on ${result.labels.pregnancy_indicators} pregnancy status: ${getPregnancyRecommendation(result.labels.pregnancy_indicators)}`
    ].filter(rec => rec);
    recommendationsContainer.innerHTML = recommendations
        .map(rec => `<div class="recommendation">${rec}</div>`)
        .join('');

    // Update Limitations
    const limitationsContainer = document.getElementById('limitations');
    const limitations = getLimitationsFromResults(result);
    limitationsContainer.innerHTML = limitations
        .map(lim => `<div class="limitation">${lim}</div>`)
        .join('');

    // Show cards with animation
    const cards = document.querySelectorAll('.analysis-card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.style.display = 'block';
            setTimeout(() => card.classList.add('show'), 50);
        }, index * 100);
    });
}

/**
 * Set confidence bar value and color
 */
function setConfidenceBar(elementId, value) {
    const bar = document.getElementById(elementId);
    if (bar) {
        bar.style.width = `${value * 100}%`;
        bar.style.backgroundColor = getConfidenceColor(value);
    }
}

/**
 * Get color based on confidence value
 */
function getConfidenceColor(value) {
    if (value >= 0.7) return '#198754';  // Green
    if (value >= 0.4) return '#ffc107';  // Yellow
    return '#dc3545';  // Red
}

/**
 * Helper function to capitalize first letter
 */
function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Get health recommendation based on status
 */
function getHealthRecommendation(healthIssues) {
    switch(healthIssues) {
        case 'none':
            return 'No immediate health concerns detected';
        case 'wounds':
            return 'Veterinary attention recommended for wound treatment';
        case 'skin_issues':
            return 'Dermatological evaluation recommended';
        default:
            return 'General veterinary check-up recommended';
    }
}

/**
 * Get body condition recommendation
 */
function getBodyConditionRecommendation(condition) {
    switch(condition) {
        case 'underweight':
            return 'Nutritional assessment and feeding plan recommended';
        case 'overweight':
            return 'Diet and exercise plan recommended';
        case 'normal':
            return 'Maintain current diet and exercise routine';
        default:
            return '';
    }
}

/**
 * Get pregnancy recommendation
 */
function getPregnancyRecommendation(status) {
    switch(status) {
        case 'visible':
            return 'Immediate veterinary care recommended for pregnancy monitoring';
        case 'possible':
            return 'Veterinary evaluation recommended to confirm pregnancy';
        case 'none':
            return 'No pregnancy indicators detected';
        default:
            return '';
    }
}

/**
 * Get analysis limitations based on confidence values
 */
function getLimitationsFromResults(result) {
    const limitations = [];
    
    if (result.confidence < 0.7) {
        limitations.push('Overall confidence is below optimal threshold');
    }
    
    if (result.quality_confidence < 0.6) {
        limitations.push('Image quality may affect analysis accuracy');
    }
    
    if (result.health_confidence < 0.7) {
        limitations.push('Health assessment confidence is limited');
    }
    
    return limitations.length ? limitations : ['No significant limitations in analysis'];
}

// Initialize UI when document is ready
document.addEventListener('DOMContentLoaded', initializeUI); 