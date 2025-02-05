/**
 * Custom logger implementation for consistent logging across the application
 * @type {Object}
 */
const logger = {
    info: (msg) => console.log('[INFO]:', msg),
    error: (msg) => console.error('[ERROR]:', msg),
    warn: (msg) => console.warn('[WARN]:', msg)
};

/**
 * Global state management variables
 * @type {Object}
 */
let currentImageIndex = 0;
let images = [];
let labels = {};
let zoomModal;
let currentPage = 0;
let imagesPerPage = 20;
let isLoading = false;
let currentImage;
let isCarouselInitialized = false;

/**
 * Carousel state management
 * @type {Object}
 */
let thumbnailsTrack;
let thumbnailsContainer;
let carouselPrevBtn;
let carouselNextBtn;
let thumbnailWidth = 110; // 100px + 10px gap
let visibleThumbnails = 0;
let currentTranslate = 0;

/**
 * Image preloading system configuration
 * @type {Object}
 */
const imageCache = new Map();
const preloadQueue = [];
let isPreloading = false;
const PRELOAD_BATCH_SIZE = 5;

// Upload Modal Management
let uploadModal;
let selectedFiles = [];
let isUploading = false;

// Variable global para controlar la cancelación
let isCanceling = false;

// Initialize carousel
/**
 * Initializes the carousel component with event listeners and state
 * @returns {boolean} Success status of initialization
 */
function initializeCarousel() {
    if (isCarouselInitialized) return;
    
    thumbnailsTrack = document.querySelector('.thumbnails-track');
    thumbnailsContainer = document.querySelector('.thumbnails-container');
    carouselPrevBtn = document.querySelector('.carousel-nav.prev');
    carouselNextBtn = document.querySelector('.carousel-nav.next');
    
    if (!thumbnailsTrack || !thumbnailsContainer || !carouselPrevBtn || !carouselNextBtn) {
        logger.warn('Carousel elements not found in DOM');
        return false;
    }
    
    carouselPrevBtn.addEventListener('click', () => moveCarousel('prev'));
    carouselNextBtn.addEventListener('click', () => moveCarousel('next'));
    
    updateVisibleThumbnails();
    window.addEventListener('resize', updateVisibleThumbnails);
    
    isCarouselInitialized = true;
    return true;
}

// Initialization
document.addEventListener('DOMContentLoaded', async function() {
    try {
        // Initialize basic UI elements
        zoomModal = new bootstrap.Modal(document.getElementById('zoomModal'));
        
        // Load saved data
        loadFromLocalStorage();
        
        // Initialize panels
        setupPanels();
        
        // Initialize carousel (retry if needed)
        let retryCount = 0;
        const initCarousel = async () => {
            if (!initializeCarousel() && retryCount < 3) {
                retryCount++;
                logger.warn(`Retrying carousel initialization (attempt ${retryCount})`);
                await new Promise(resolve => setTimeout(resolve, 500));
                return initCarousel();
            }
        };
        await initCarousel();
        
        // Load images and setup
        await loadImages();
        if (images.length > 0) {
            await loadImage(0);
        }
        
        setupValidation();
        setupShortcutsPanel();
        setupOfflineSupport();
        loadStats();
        
    } catch (error) {
        logger.error('Error during initialization:', error);
        showToast('Error initializing application', 'error');
    }
});

// Create and update thumbnails
function updateThumbnails() {
    if (!isCarouselInitialized) {
        if (!initializeCarousel()) {
            logger.warn('Cannot update thumbnails: carousel not initialized');
            return;
        }
    }
    
    try {
        thumbnailsTrack.innerHTML = '';
        
        if (!Array.isArray(images) || images.length === 0) {
            logger.warn('No images available for thumbnails');
            return;
        }
        
        images.forEach((imageId, index) => {
            const thumbnail = document.createElement('div');
            thumbnail.className = `thumbnail ${index === currentImageIndex ? 'active' : ''}`;
            
            const img = document.createElement('img');
            img.alt = `Thumbnail ${index + 1}`;
            img.loading = 'lazy';
            
            // Add error handling for thumbnail images
            img.onerror = () => {
                img.src = '/static/img/placeholder.png'; // Fallback image
                logger.warn(`Failed to load thumbnail for image ${imageId}`);
            };
            
            img.src = `/api/v1/dataset/image/${imageId}?thumbnail=true`;
            
            thumbnail.appendChild(img);
            thumbnail.addEventListener('click', () => loadImage(index));
            thumbnailsTrack.appendChild(thumbnail);
        });
        
        updateCarouselButtons();
        
    } catch (error) {
        logger.error('Error updating thumbnails:', error);
        showToast('Error updating thumbnails', 'error');
    }
}

// Setup panels
function setupPanels() {
    // Restore detection panel state
    const detectionPanel = document.getElementById('detection-panel');
    const detectionContent = document.getElementById('detection-content');
    const detectionLoading = document.getElementById('detection-loading');
    
    if (detectionPanel && localStorage.getItem('detection_panel_collapsed') === 'true') {
        detectionPanel.classList.add('collapsed');
        if (detectionContent) detectionContent.style.display = 'none';
        if (detectionLoading) detectionLoading.style.display = 'none';
    }
    
    // Restore shortcuts panel state
    const shortcutsPanel = document.getElementById('shortcuts-panel');
    const shortcutsContent = document.getElementById('shortcuts-content');
    
    if (shortcutsPanel && localStorage.getItem('shortcuts_panel_collapsed') === 'true') {
        shortcutsPanel.classList.add('collapsed');
        if (shortcutsContent) shortcutsContent.style.display = 'none';
    }
}

// Load current image
/**
 * Loads and displays an image at the specified index
 * Handles caching, preloading, and UI updates
 * @param {number} index - The index of the image to load
 * @returns {Promise<void>}
 */
async function loadImage(index) {
    try {
        if (index < 0 || index >= images.length) {
            logger.warn(`Invalid image index: ${index}`);
            return;
        }

        currentImageIndex = index;
        currentImage = images[index];
        
        logger.info(`Loading image at index ${index}: ${currentImage}`);
        
        // Update carousel state and UI
        if (isCarouselInitialized) {
            document.querySelectorAll('.thumbnail').forEach((thumb, i) => {
                thumb.classList.toggle('active', i === index);
            });
            
            // Calculate and update carousel position
            const thumbnailPosition = index * thumbnailWidth;
            if (thumbnailPosition < -currentTranslate || 
                thumbnailPosition > -currentTranslate + (visibleThumbnails * thumbnailWidth)) {
                currentTranslate = Math.max(
                    -(Math.max(0, images.length - visibleThumbnails) * thumbnailWidth),
                    Math.min(0, -(Math.floor(index / visibleThumbnails) * visibleThumbnails * thumbnailWidth))
                );
                thumbnailsTrack.style.transform = `translateX(${currentTranslate}px)`;
                updateCarouselButtons();
            }
        }

        // Load and display main image
        const img = document.getElementById('current-image');
        if (!img) {
            throw new Error('Main image element not found');
        }
        
        img.classList.remove('loaded');
        
        try {
            if (imageCache.has(currentImage)) {
                img.src = imageCache.get(currentImage);
            } else {
                const response = await fetch(`/api/v1/dataset/image/${currentImage}`);
                if (!response.ok) throw new Error(`Failed to load image: ${response.status}`);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                imageCache.set(currentImage, url);
                img.src = url;
            }
            
            img.onload = () => img.classList.add('loaded');
        } catch (imgError) {
            logger.error('Error loading main image:', imgError);
            img.src = '/static/img/placeholder.png';
            showToast('Error loading image', 'error');
        }
        
        // Queue next images for preloading
        queueImagesForPreload(index);
        
        // Load and apply labels
        try {
            const labelsResponse = await fetch(`/api/v1/dataset/labels/${currentImage}`);
            if (labelsResponse.ok) {
                const savedLabels = await labelsResponse.json();
                if (savedLabels) {
                    Object.entries(savedLabels).forEach(([key, value]) => {
                        const select = document.getElementById(key);
                        if (select) {
                            select.value = value;
                            validateField(select);
                        }
                    });
                }
            } else if (labelsResponse.status === 404) {
                // Clear form if no labels exist
                document.getElementById('label-form').reset();
            }
        } catch (labelError) {
            logger.warn('Error loading labels:', labelError);
        }

        // Update detection panel
        const detectionLoading = document.getElementById('detection-loading');
        const detectionContent = document.getElementById('detection-content');
        
        if (detectionLoading && detectionContent) {
            detectionLoading.style.display = 'block';
            detectionContent.style.display = 'none';
            
            try {
                const detectionResponse = await fetch(`/api/v1/dataset/prelabel/${currentImage}`);
                if (detectionResponse.ok) {
                    const predictions = await detectionResponse.json();
                    updateDetectionPanel(predictions);
                }
            } catch (detectionError) {
                logger.warn('Error loading detection results:', detectionError);
            } finally {
                detectionLoading.style.display = 'none';
                detectionContent.style.display = 'block';
            }
        }
        
        // Update progress
        updateProgress();
        
    } catch (error) {
        logger.error('Error in loadImage:', error);
        showToast('Error loading image: ' + error.message, 'error');
    }
}

// Load from localStorage
function loadFromLocalStorage() {
    try {
        const savedLabels = localStorage.getItem('ept_vision_labels');
        if (savedLabels) {
            labels = JSON.parse(savedLabels);
        }
        
        const savedProgress = localStorage.getItem('ept_vision_progress');
        if (savedProgress) {
            currentImageIndex = parseInt(savedProgress);
        }
        
        logger.info("Loaded data from localStorage");
    } catch (error) {
        console.error('Error loading from localStorage:', error);
    }
}

// Save to localStorage
function saveToLocalStorage() {
    try {
        localStorage.setItem('ept_vision_labels', JSON.stringify(labels));
        localStorage.setItem('ept_vision_progress', currentImageIndex.toString());
    } catch (error) {
        console.error('Error saving to localStorage:', error);
    }
}

// Load images with pagination
async function loadImages() {
    if (isLoading) return;
    
    try {
        isLoading = true;
        showLoadingIndicator();
        
        // Obtener el valor actual del selector
        const perPageSelect = document.getElementById('imagesPerPage');
        const selectedPerPage = parseInt(perPageSelect.value);
        
        const response = await fetch(`/api/v1/dataset/images?skip=${currentPage * selectedPerPage}&limit=${selectedPerPage}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        if (data.images && Array.isArray(data.images)) {
            // Replace current images with new page
            images = data.images;
            imagesPerPage = selectedPerPage; // Actualizar la variable global
            
            // Update pagination UI
            const totalImages = data.total || 0;
            const totalPages = Math.ceil(totalImages / imagesPerPage);
            document.getElementById('currentPage').textContent = currentPage + 1;
            document.getElementById('totalPages').textContent = totalPages;
            document.getElementById('prevPage').disabled = currentPage === 0;
            document.getElementById('nextPage').disabled = currentPage >= totalPages - 1;
            
            // Update thumbnails
            if (thumbnailsTrack && thumbnailsContainer) {
                updateThumbnails();
            }
            
            // Load first image of the page if available
            if (images.length > 0) {
                await loadImage(0);
            }
            
            // Start preloading images
            queueImagesForPreload(0);
        }
    } catch (error) {
        logger.error('Error loading images:', error);
        showToast('Error loading images: ' + error.message, 'error');
    } finally {
        isLoading = false;
        hideLoadingIndicator();
    }
}

// Loading indicators
function showLoadingIndicator() {
    document.querySelector('.image-container').classList.add('loading');
}

function hideLoadingIndicator() {
    document.querySelector('.image-container').classList.remove('loading');
}

// Offline support
function setupOfflineSupport() {
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    async function handleOnline() {
        showToast('Connection restored - Synchronizing...', 'info');
        await syncLabels();
    }
    
    function handleOffline() {
        showToast('Offline mode - Changes will be saved locally', 'warning');
    }
}

// Sync labels
/**
 * Handles offline data persistence and synchronization
 * @returns {Promise<void>}
 */
async function syncLabels() {
    const unsyncedLabels = JSON.parse(localStorage.getItem('ept_vision_unsynced_labels') || '{}');
    
    for (const [imageId, labelData] of Object.entries(unsyncedLabels)) {
        try {
            const response = await fetch(`/api/v1/dataset/labels/${imageId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(labelData)
            });
            
            if (response.ok) {
                delete unsyncedLabels[imageId];
            }
        } catch (error) {
            logger.error(`Error syncing labels for ${imageId}:`, error);
        }
    }
    
    localStorage.setItem('ept_vision_unsynced_labels', JSON.stringify(unsyncedLabels));
}

// Save labels with offline support
async function saveLabels() {
    const formData = {};
    document.querySelectorAll('#label-form select').forEach(select => {
        formData[select.id] = select.value;
    });

    const imageId = images[currentImageIndex];
    
    // Save locally first
    labels[imageId] = formData;
    saveToLocalStorage();
    
    try {
        if (navigator.onLine) {
            const response = await fetch(`/api/v1/dataset/labels/${imageId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (response.ok) {
                showToast('Labels saved successfully', 'success');
            } else {
                throw new Error('Error saving labels');
            }
        } else {
            // Save for later sync
            const unsyncedLabels = JSON.parse(
                localStorage.getItem('ept_vision_unsynced_labels') || '{}'
            );
            unsyncedLabels[imageId] = formData;
            localStorage.setItem(
                'ept_vision_unsynced_labels',
                JSON.stringify(unsyncedLabels)
            );
            
            showToast('Labels saved locally', 'info');
        }
    } catch (error) {
        console.error('Error saving labels:', error);
        showToast('Error saving labels', 'error');
    }
}

// Real-time validation
function setupValidation() {
    const form = document.getElementById('label-form');
    const selects = form.querySelectorAll('select');
    
    selects.forEach(select => {
        select.addEventListener('change', () => validateField(select));
    });
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (validateForm()) {
            await saveLabels();
        }
    });
}

function validateField(field) {
    const feedback = field.nextElementSibling;
    if (field.value) {
        field.classList.remove('is-invalid');
        field.classList.add('is-valid');
        feedback.textContent = '✓';
        return true;
    } else {
        field.classList.remove('is-valid');
        field.classList.add('is-invalid');
        feedback.textContent = 'This field is required';
        return false;
    }
}

function validateForm() {
    const selects = document.querySelectorAll('#label-form select');
    let isValid = true;
    
    selects.forEach(select => {
        if (!validateField(select)) {
            isValid = false;
        }
    });
    
    return isValid;
}

// Update progress bar
async function updateProgress() {
    try {
        const response = await fetch('/api/v1/dataset/progress');
        const data = await response.json();
        
        const progress = document.querySelector('.progress-bar');
        progress.style.width = `${data.progress_percentage}%`;
        progress.textContent = `${data.labeled_images}/${data.total_images} images`;
    } catch (error) {
        console.error('Error updating progress:', error);
    }
}

// Zoom functions
function openZoomModal() {
    const currentImg = document.getElementById('current-image');
    const zoomedImg = document.getElementById('zoomed-image');
    zoomedImg.src = currentImg.src;
    zoomModal.show();
}

// Shortcuts panel
function setupShortcutsPanel() {
    // No longer needed - panel should only respond to toggle button
}

// Toggle shortcuts panel
function toggleShortcutsPanel() {
    const panel = document.getElementById('shortcuts-panel');
    const content = document.getElementById('shortcuts-content');
    
    panel.classList.toggle('collapsed');
    
    // Save state
    if (panel.classList.contains('collapsed')) {
        content.style.display = 'none';
        localStorage.setItem('shortcuts_panel_collapsed', 'true');
    } else {
        content.style.display = 'block';
        localStorage.setItem('shortcuts_panel_collapsed', 'false');
    }
}

// Show notifications
function showToast(message, type = 'info') {
    const style = {
        background: type === "error" ? "linear-gradient(to right, #ff5f6d, #ffc371)" :
                    type === "warning" ? "linear-gradient(to right, #f6d365, #fda085)" :
                    "linear-gradient(to right, #00b09b, #96c93d)",
        color: "#fff"
    };
    
    Toastify({
        text: message,
        duration: 5000,
        close: true,
        style: style
    }).showToast();
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }
    
    const key = e.key.toLowerCase();
    
    // Navigation
    if (e.key === 'ArrowLeft') {
        loadImage(currentImageIndex - 1);
    } else if (e.key === 'ArrowRight') {
        loadImage(currentImageIndex + 1);
    } else if (e.key === 'Enter' && !e.shiftKey) {
        document.querySelector('button[type="submit"]').click();
    }
    
    // Animal type (1-3)
    else if (key === '1') selectOption('animal_type', 0);
    else if (key === '2') selectOption('animal_type', 1);
    else if (key === '3') selectOption('animal_type', 2);
    
    // Size (4-6)
    else if (key === '4') selectOption('size', 0);
    else if (key === '5') selectOption('size', 1);
    else if (key === '6') selectOption('size', 2);
    
    // Body condition (7-9)
    else if (key === '7') selectOption('body_condition', 0);
    else if (key === '8') selectOption('body_condition', 1);
    else if (key === '9') selectOption('body_condition', 2);
    
    // Health issues (Q-T)
    else if (key === 'q') selectOption('visible_health_issues', 0);
    else if (key === 'w') selectOption('visible_health_issues', 1);
    else if (key === 'e') selectOption('visible_health_issues', 2);
    else if (key === 't') selectOption('visible_health_issues', 3);
    
    // Pregnancy (Y-I)
    else if (key === 'y') selectOption('pregnancy_indicators', 0);
    else if (key === 'u') selectOption('pregnancy_indicators', 1);
    else if (key === 'i') selectOption('pregnancy_indicators', 2);
    
    // Image quality (O-P)
    else if (key === 'o') selectOption('image_quality', 0);
    else if (key === 'p') selectOption('image_quality', 1);
    
    // Context (Z-M)
    else if (key === 'z') selectOption('context', 0);
    else if (key === 'x') selectOption('context', 1);
    else if (key === 'c') selectOption('context', 2);
    else if (key === 'm') selectOption('context', 3);
    
    // Discard image
    else if (key === 'd') {
        discardImage(event);
    }
});

function selectOption(selectId, index) {
    const select = document.getElementById(selectId);
    const options = select.options;
    if (index < options.length) {
        select.selectedIndex = index;
        validateField(select);
    }
}

// Load and display statistics
async function loadStats() {
    try {
        const response = await fetch('/api/v1/dataset/stats');
        if (!response.ok) throw new Error('Failed to load statistics');
        
        const stats = await response.json();
        
        // Update progress bar
        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.width = `${stats.progress_percentage}%`;
        progressBar.textContent = `${Math.round(stats.progress_percentage)}%`;
        
        // Update summary badges
        document.getElementById('total-images').textContent = stats.total_images;
        document.getElementById('labeled-images').textContent = stats.labeled_images;
        document.getElementById('unlabeled-images').textContent = stats.unlabeled_images;
        document.getElementById('discarded-images').textContent = stats.discarded_images;
        
        // Update distribution table
        const row = document.getElementById('distribution-row');
        row.innerHTML = '';
        
        // Helper function to create distribution cell
        const createDistributionCell = (distribution) => {
            const cell = document.createElement('td');
            const items = Object.entries(distribution || {})
                .map(([key, value]) => `${key}: ${value}`)
                .join('<br>');
            cell.innerHTML = items || 'None: 0';
            return cell;
        };
        
        // Add cells for each category
        row.appendChild(createDistributionCell(stats.distributions.animal_type));
        row.appendChild(createDistributionCell(stats.distributions.size));
        row.appendChild(createDistributionCell(stats.distributions.body_condition));
        row.appendChild(createDistributionCell(stats.distributions.visible_health_issues));
        row.appendChild(createDistributionCell(stats.distributions.pregnancy_indicators));
        row.appendChild(createDistributionCell(stats.distributions.image_quality));
        row.appendChild(createDistributionCell(stats.distributions.context));
        
    } catch (error) {
        logger.error('Error loading statistics:', error);
        showError('Error loading statistics');
    }
}

async function exportDataset(format) {
    try {
        showToast('Starting export...', 'info');
        
        const response = await fetch(`/api/v1/dataset/export/${format}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `dataset_export.${format}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            
            showToast('Dataset exported successfully', 'success');
        } else {
            throw new Error('Export error');
        }
    } catch (error) {
        console.error('Error exporting dataset:', error);
        showToast('Error exporting dataset', 'error');
    }
}

// Update statistics periodically
setInterval(loadStats, 30000); // Every 30 seconds

async function discardImage(event) {
    if (event) {
        event.preventDefault();
    }
    
    try {
        if (!currentImage) {
            logger.error("No image selected to discard");
            showError("No image selected to discard");
            return;
        }

        logger.info("Attempting to discard image:", currentImage);
        showLoading('Discarding image...');

        const response = await fetch(`/api/v1/dataset/discard/${currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            const responseText = await response.text();
            let errorMessage = `Server error (${response.status})`;
            try {
                const responseData = JSON.parse(responseText);
                errorMessage = responseData.detail || errorMessage;
            } catch (e) {
                if (responseText) {
                    errorMessage = responseText;
                }
            }
            throw new Error(errorMessage);
        }

        // Remove the discarded image from the images array
        const index = images.indexOf(currentImage);
        if (index > -1) {
            images.splice(index, 1);
            
            // Clear the image from cache if it exists
            if (imageCache.has(currentImage)) {
                URL.revokeObjectURL(imageCache.get(currentImage));
                imageCache.delete(currentImage);
            }
            
            // Update thumbnails
            updateThumbnails();
            
            // Adjust currentImageIndex if necessary
            if (currentImageIndex >= images.length) {
                currentImageIndex = Math.max(0, images.length - 1);
            }
            
            // Load the next available image
            if (images.length > 0) {
                await loadImage(currentImageIndex);
            } else {
                // If no images left, clear the current image
                const img = document.getElementById('current-image');
                if (img) {
                    img.src = '';
                    img.classList.remove('loaded');
                }
                document.getElementById('label-form').reset();
            }
            
            // Update carousel position
            if (isCarouselInitialized) {
                const maxTranslate = -(Math.max(0, images.length - visibleThumbnails) * thumbnailWidth);
                currentTranslate = Math.max(maxTranslate, currentTranslate);
                thumbnailsTrack.style.transform = `translateX(${currentTranslate}px)`;
                updateCarouselButtons();
            }
        }

        // Update statistics and UI
        await loadStats();
        showToast('Image discarded successfully', 'success');

    } catch (error) {
        logger.error("Error discarding image:", error);
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
}

// Loading indicator functions
function showLoading(message = 'Processing...') {
    document.getElementById('loadingText').textContent = message;
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

/**
 * Updates the detection panel with AI prediction results
 * @param {Object} result - The prediction results from the AI model
 * @param {Object} result.labels - The predicted labels
 * @param {number} result.confidence - The overall confidence score
 */
function updateDetectionPanel(result) {
    const updateLog = (id, value, confidence) => {
        const confSpan = document.getElementById(id + 'Conf');
        const confBar = document.getElementById(id + 'Bar');
        if (confSpan && confBar) {
            const confValue = parseFloat(confidence) || 0;
            confSpan.textContent = `${value} (${(confValue * 100).toFixed(1)}%)`;
            confBar.style.width = `${confValue * 100}%`;
            
            // Color coding based on confidence levels
            if (confValue > 0.7) {
                confBar.style.backgroundColor = '#28a745';
            } else if (confValue > 0.4) {
                confBar.style.backgroundColor = '#ffc107';
            } else {
                confBar.style.backgroundColor = '#dc3545';
            }
        }
    };

    if (result.labels) {
        updateLog('animalType', result.labels.animal_type, result.confidence);
        updateLog('size', result.labels.size, result.size_confidence);
        updateLog('bodyCondition', result.labels.body_condition, result.body_condition_confidence);
        updateLog('healthIssues', result.labels.visible_health_issues, result.health_confidence);
        updateLog('pregnancy', result.labels.pregnancy_indicators, result.pregnancy_confidence);
        updateLog('quality', result.labels.image_quality, result.quality_confidence);
        updateLog('context', result.labels.context, result.context_confidence);
    }
}

async function prelabelImage() {
    if (!currentImage) {
        showError("No image selected");
        return;
    }

    try {
        showLoading('Pre-labeling image...');
        
        // Clean up any existing buttons
        cleanupLoadingOverlayButtons();
        
        // Get or create log div
        let logDiv = document.getElementById('autolabel-logs');
        if (!logDiv) {
            logDiv = createLogDiv();
        }
        
        function addLog(message) {
            const line = document.createElement('div');
            line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(line);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        // Add cancel button to loading overlay
        const loadingOverlay = document.getElementById('loadingOverlay');
        const closeButton = document.createElement('button');
        closeButton.className = 'btn btn-primary mt-3';
        closeButton.textContent = 'Close';
        closeButton.onclick = () => hideLoading();
        loadingOverlay.appendChild(closeButton);
        
        addLog(`Starting pre-labeling for image: ${currentImage}`);
        
        const response = await fetch(`/api/v1/dataset/prelabel/${currentImage}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log("Pre-labeling result:", result);
        
        // Update detection logs
        updateDetectionPanel(result);
        
        if (result.should_discard) {
            const message = `❌ Image ${currentImage} - discarded (${result.reason}). Confidence: ${(result.confidence * 100).toFixed(1)}%`;
            addLog(message);
            showToast(message, "warning");
            
            if (confirm(message + "\nDo you want to discard this image?")) {
                await discardImage();
            }
            return;
        }
        
        // Fill form with predicted labels
        const labels = result.labels;
        if (labels) {
            addLog(`✅ Image ${currentImage} - labeled (${(result.confidence * 100).toFixed(1)}% confidence)`);
            for (const [key, value] of Object.entries(labels)) {
                const select = document.getElementById(key);
                if (select) {
                    const option = Array.from(select.options).find(
                        opt => opt.value.toLowerCase() === String(value).toLowerCase()
                    );
                    if (option) {
                        select.value = option.value;
                        validateField(select);
                    } else {
                        console.warn(`Option ${value} not found for ${key}`);
                    }
                }
            }
            
            // Save the labels automatically but don't navigate
            await saveLabels();
            showToast('Image auto-labeled successfully', 'success');
        } else {
            addLog(`❌ Image ${currentImage} - error (No labels generated)`);
        }
        
        addLog('\nClick "Close" to continue...');
        
    } catch (error) {
        console.error("Error pre-labeling image:", error);
        addLog(`❌ Error pre-labeling image: ${error.message}`);
        showError(`Error pre-labeling image: ${error.message}`);
    }
}

// Add floating log button to HTML
document.addEventListener('DOMContentLoaded', function() {
    const floatingLogBtn = document.createElement('button');
    floatingLogBtn.className = 'btn btn-info position-fixed';
    floatingLogBtn.style.cssText = 'bottom: 20px; right: 20px; z-index: 1000; opacity: 0.8;';
    floatingLogBtn.innerHTML = '<i class="fas fa-terminal"></i> Show Logs';
    document.body.appendChild(floatingLogBtn);

    floatingLogBtn.onclick = () => {
        showLoading('Process Logs');
        let logDiv = document.getElementById('autolabel-logs');
        if (!logDiv) {
            logDiv = createLogDiv();
        }
        const closeButton = document.createElement('button');
        closeButton.className = 'btn btn-primary mt-3';
        closeButton.textContent = 'Close';
        closeButton.onclick = () => hideLoading();
        
        // Clean up any existing buttons
        cleanupLoadingOverlayButtons();
        document.getElementById('loadingOverlay').appendChild(closeButton);
    };
});

// Function to clean up loading overlay buttons
function cleanupLoadingOverlayButtons() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const buttons = loadingOverlay.querySelectorAll('button');
    buttons.forEach(button => button.remove());
}

// Function to create log div
function createLogDiv() {
    const logDiv = document.createElement('div');
    logDiv.id = 'autolabel-logs';
    logDiv.style.maxHeight = '200px';
    logDiv.style.overflow = 'auto';
    logDiv.style.backgroundColor = '#f8f9fa';
    logDiv.style.padding = '10px';
    logDiv.style.marginTop = '10px';
    logDiv.style.fontFamily = 'monospace';
    document.querySelector('.loading-text').after(logDiv);
    return logDiv;
}

/**
 * Processes batch AI labeling for multiple images
 * Handles real-time updates and progress tracking
 * @returns {Promise<void>}
 */
async function prelabelAll() {
    try {
        if (!images || images.length === 0) {
            showError("No images available to process");
            return;
        }

        if (!confirm(`This will auto-label ${images.length} images from the current page. Continue?`)) {
            return;
        }

        isCanceling = false;
        showLoading(`Auto-labeling ${images.length} images from current page...`);
        
        try {
            // Clean up any existing buttons
            cleanupLoadingOverlayButtons();
            
            // Get or create log div
            let logDiv = document.getElementById('autolabel-logs');
            if (!logDiv) {
                logDiv = createLogDiv();
            }
            
            function addLog(message) {
                const line = document.createElement('div');
                line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                logDiv.appendChild(line);
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            // Add cancel button to loading overlay
            const loadingOverlay = document.getElementById('loadingOverlay');
            const cancelButton = document.createElement('button');
            cancelButton.className = 'btn btn-danger mt-3 me-2';
            cancelButton.textContent = 'Cancel';
            let eventSource;
            
            cancelButton.onclick = () => {
                isCanceling = true;
                addLog('Canceling... Please wait for current operation to complete...');
                cancelButton.disabled = true;
                cancelButton.textContent = 'Canceling...';
                
                // Close the EventSource connection
                if (eventSource) {
                    eventSource.close();
                    finishProcessing('Operation was canceled by user.');
                }
            };
            loadingOverlay.appendChild(cancelButton);
            
            // Iniciar el procesamiento por lotes
            addLog(`Starting batch pre-labeling for ${images.length} images...`);
            
            // Enviar la lista exacta de imágenes a procesar
            const imageList = images.map(img => ({ id: img }));
            
            // Request batch pre-labeling for current page using EventSource for real-time updates
            const queryParams = new URLSearchParams({
                images: JSON.stringify(imageList)
            });
            eventSource = new EventSource(`/api/v1/dataset/prelabel/batch/stream?${queryParams}`);
            
            let totalLabeled = 0;
            let totalDiscarded = 0;
            let totalErrors = 0;
            let totalProcessed = 0;
            
            function finishProcessing(message = null) {
                if (eventSource) {
                    eventSource.close();
                }
                
                addLog(`\nSummary:`);
                addLog(`- Labeled: ${totalLabeled} images`);
                addLog(`- Discarded: ${totalDiscarded} images`);
                addLog(`- Errors: ${totalErrors} images`);
                addLog(`- Total processed: ${totalProcessed} images`);
                
                if (message) {
                    addLog(`\n${message}`);
                }
                
                addLog('\nClick "Close" to continue...');
                
                // Clean up any existing buttons
                cleanupLoadingOverlayButtons();
                
                const closeButton = document.createElement('button');
                closeButton.className = 'btn btn-primary mt-3';
                closeButton.textContent = 'Close';
                closeButton.onclick = () => {
                    hideLoading();
                    // Refresh stats and current page
                    loadStats();
                    loadImages();
                };
                loadingOverlay.appendChild(closeButton);
                
                showToast(`Auto-labeled ${totalLabeled} images, discarded ${totalDiscarded} images`, 'success');
            }
            
            eventSource.onmessage = (event) => {
                const detail = JSON.parse(event.data);
                
                if (detail.type === 'progress') {
                    const statusEmoji = detail.status === 'labeled' ? '✅' :
                                      detail.status === 'discarded' ? '🗑️' :
                                      detail.status === 'error' ? '❌' : '⏭️';
                    
                    let message = `${statusEmoji} Image ${detail.index + 1}: ${detail.image_id} - ${detail.status}`;
                    
                    if (detail.confidence) {
                        message += ` (${(detail.confidence * 100).toFixed(1)}% confidence)`;
                    }
                    
                    if (detail.reason) {
                        message += ` - ${detail.reason}`;
                    }
                    
                    addLog(message);
                    
                    // Actualizar contadores
                    if (detail.status === 'labeled') totalLabeled++;
                    else if (detail.status === 'discarded') totalDiscarded++;
                    else if (detail.status === 'error') totalErrors++;
                    totalProcessed++;
                    
                } else if (detail.type === 'complete') {
                    const message = detail.status === 'canceled' ? 
                        'Operation was canceled.' : 
                        'Operation completed successfully.';
                    finishProcessing(message);
                }
            };
            
            eventSource.onerror = (error) => {
                logger.error("Error in EventSource:", error);
                addLog(`Error: Connection lost or server error`);
                showError("Error in batch pre-labeling: Connection lost");
                finishProcessing('Operation failed due to connection error.');
            };

        } catch (error) {
            logger.error("Error in batch:", error);
            addLog(`Error: ${error.message}`);
            showError("Error in batch pre-labeling: " + error.message);
            hideLoading();
        }

    } catch (error) {
        logger.error("Error in batch pre-labeling:", error);
        showError("Error in batch pre-labeling: " + error.message);
        hideLoading();
    }
}

// Navigation functions
async function nextImage() {
    if (currentImageIndex < images.length - 1) {
        await loadImage(currentImageIndex + 1);
    } else {
        showToast('No more images to show', 'info');
    }
}

async function previousImage() {
    if (currentImageIndex > 0) {
        await loadImage(currentImageIndex - 1);
    } else {
        showToast('This is the first image', 'info');
    }
}

// Update navigation buttons to use preventDefault
document.addEventListener('DOMContentLoaded', function() {
    // Use more specific selectors based on button content
    const prevButton = document.querySelector('button.btn-secondary i.bi-arrow-left').parentElement;
    const nextButton = document.querySelector('button.btn-secondary i.bi-arrow-right').parentElement;
    
    if (prevButton) {
        prevButton.onclick = async (e) => {
            e.preventDefault();
            await previousImage();
        };
    }
    
    if (nextButton) {
        nextButton.onclick = async (e) => {
            e.preventDefault();
            await nextImage();
        };
    }
});

// Toggle detection panel
function toggleDetectionPanel() {
    const panel = document.getElementById('detection-panel');
    const content = document.getElementById('detection-content');
    const loading = document.getElementById('detection-loading');
    
    panel.classList.toggle('collapsed');
    
    // Save state
    if (panel.classList.contains('collapsed')) {
        content.style.display = 'none';
        loading.style.display = 'none';
        localStorage.setItem('detection_panel_collapsed', 'true');
    } else {
        content.style.display = 'block';
        // Only show loading if it was previously visible
        if (loading.getAttribute('data-was-visible') === 'true') {
            loading.style.display = 'block';
        }
        localStorage.setItem('detection_panel_collapsed', 'false');
    }
}

// Update carousel state
function updateVisibleThumbnails() {
    visibleThumbnails = Math.floor(thumbnailsContainer.offsetWidth / thumbnailWidth);
    updateCarouselButtons();
}

// Move carousel
function moveCarousel(direction) {
    const maxTranslate = -(Math.max(0, images.length - visibleThumbnails) * thumbnailWidth);
    
    if (direction === 'prev') {
        currentTranslate = Math.min(0, currentTranslate + thumbnailWidth * visibleThumbnails);
    } else {
        currentTranslate = Math.max(maxTranslate, currentTranslate - thumbnailWidth * visibleThumbnails);
    }
    
    thumbnailsTrack.style.transform = `translateX(${currentTranslate}px)`;
    updateCarouselButtons();
}

// Update carousel navigation buttons
function updateCarouselButtons() {
    const maxTranslate = -(Math.max(0, images.length - visibleThumbnails) * thumbnailWidth);
    carouselPrevBtn.disabled = currentTranslate >= 0;
    carouselNextBtn.disabled = currentTranslate <= maxTranslate;
}

// Preload images
async function preloadImages() {
    if (isPreloading || preloadQueue.length === 0) return;
    
    isPreloading = true;
    const batch = preloadQueue.splice(0, PRELOAD_BATCH_SIZE);
    
    await Promise.all(batch.map(async (imageId) => {
        if (!imageCache.has(imageId)) {
            try {
                const response = await fetch(`/api/v1/dataset/image/${imageId}`);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                imageCache.set(imageId, url);
            } catch (error) {
                console.error(`Error preloading image ${imageId}:`, error);
            }
        }
    }));
    
    isPreloading = false;
    preloadImages(); // Continue with next batch if any
}

/**
 * Manages the image preloading queue for smoother navigation
 * @param {number} currentIndex - The index of the currently displayed image
 */
function queueImagesForPreload(currentIndex) {
    preloadQueue.length = 0; // Clear existing queue
    
    // Queue next images
    for (let i = currentIndex + 1; i < Math.min(currentIndex + 10, images.length); i++) {
        if (!imageCache.has(images[i])) {
            preloadQueue.push(images[i]);
        }
    }
    
    // Queue previous images
    for (let i = currentIndex - 1; i >= Math.max(0, currentIndex - 5); i--) {
        if (!imageCache.has(images[i])) {
            preloadQueue.push(images[i]);
        }
    }
    
    preloadImages();
}

function openUploadModal() {
    if (!uploadModal) {
        uploadModal = new bootstrap.Modal(document.getElementById('uploadModal'));
    }
    resetUploadUI();
    uploadModal.show();
}

function resetUploadUI() {
    selectedFiles = [];
    document.querySelector('.upload-progress').style.display = 'none';
    document.querySelector('.upload-zone').style.display = 'block';
    document.querySelector('.upload-items').innerHTML = '';
    updateUploadStats();
}

function updateUploadStats() {
    document.getElementById('totalFiles').textContent = selectedFiles.length;
    const uploaded = selectedFiles.filter(f => f.status === 'success').length;
    const failed = selectedFiles.filter(f => f.status === 'error').length;
    document.getElementById('uploadedFiles').textContent = uploaded;
    document.getElementById('failedFiles').textContent = failed;
}

// File Selection and Drag & Drop
document.getElementById('fileInput').addEventListener('change', handleFileSelect);
const dropZone = document.getElementById('uploadZone');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
});

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    handleFiles(files);
}

function handleFiles(files) {
    // Filter only image files
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    // Add files to the queue
    imageFiles.forEach(file => {
        selectedFiles.push({
            file: file,
            status: 'pending',
            error: null
        });
    });

    // Show upload UI
    document.querySelector('.upload-progress').style.display = 'block';
    updateUploadStats();
    updateUploadList();
}

function updateUploadList() {
    const container = document.querySelector('.upload-items');
    container.innerHTML = '';

    selectedFiles.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'upload-item';
        div.innerHTML = `
            <span class="filename">${item.file.name}</span>
            <span class="status ${item.status}">
                ${getStatusIcon(item.status)}
                ${item.error || getStatusText(item.status)}
            </span>
        `;
        container.appendChild(div);
    });
}

function getStatusIcon(status) {
    switch(status) {
        case 'success':
            return '<i class="bi bi-check-circle"></i>';
        case 'error':
            return '<i class="bi bi-x-circle"></i>';
        case 'uploading':
            return '<i class="bi bi-arrow-repeat"></i>';
        default:
            return '<i class="bi bi-hourglass"></i>';
    }
}

function getStatusText(status) {
    switch(status) {
        case 'success':
            return 'Uploaded';
        case 'error':
            return 'Failed';
        case 'uploading':
            return 'Uploading...';
        default:
            return 'Pending';
    }
}

/**
 * Handles file upload process with batch processing and progress tracking
 * @returns {Promise<void>}
 */
async function startUpload() {
    if (isUploading || selectedFiles.length === 0) return;
    
    isUploading = true;
    const startUploadBtn = document.getElementById('startUploadBtn');
    startUploadBtn.disabled = true;
    
    const progressBar = document.querySelector('.upload-progress .progress-bar');
    let uploaded = 0;
    let uploadedFiles = [];

    try {
        // Process files in chunks of 5 for better performance
        for (let i = 0; i < selectedFiles.length; i += 5) {
            const chunk = selectedFiles.slice(i, i + 5);
            await Promise.all(chunk.map(async (item) => {
                try {
                    item.status = 'uploading';
                    updateUploadList();

                    const formData = new FormData();
                    formData.append('file', item.file);

                    const response = await fetch('/api/v1/dataset/upload', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    uploadedFiles.push(result.filename);  // Store uploaded filename
                    item.status = 'success';
                } catch (error) {
                    item.status = 'error';
                    item.error = error.message;
                    console.error(`Error uploading ${item.file.name}:`, error);
                }

                uploaded++;
                const progress = (uploaded / selectedFiles.length) * 100;
                progressBar.style.width = `${progress}%`;
                updateUploadList();
                updateUploadStats();
            }));
        }

        // Show completion message
        showToast('Upload completed!', 'success');
        
        // Get total number of images to calculate last page
        const response = await fetch('/api/v1/dataset/images?skip=0&limit=1');
        const data = await response.json();
        const totalImages = data.total;
        const totalPages = Math.ceil(totalImages / imagesPerPage);
        
        // Go to last page
        currentPage = totalPages - 1;
        await loadImages();
        
        // Try to find and select the first uploaded image in the current page
        if (uploadedFiles.length > 0 && images.length > 0) {
            const firstUploadedIndex = images.findIndex(img => uploadedFiles.includes(img));
            if (firstUploadedIndex !== -1) {
                await loadImage(firstUploadedIndex);
            }
        }
        
    } catch (error) {
        logger.error('Upload error:', error);
        showToast('Error during upload', 'error');
    } finally {
        isUploading = false;
        startUploadBtn.disabled = false;
        uploadModal.hide();
    }
}

// Add event listeners for pagination
document.addEventListener('DOMContentLoaded', function() {
    const imagesPerPageSelect = document.getElementById('imagesPerPage');
    const prevPageBtn = document.getElementById('prevPage');
    const nextPageBtn = document.getElementById('nextPage');
    const goToPageBtn = document.getElementById('goToPage');
    const pageInput = document.getElementById('pageInput');

    if (imagesPerPageSelect) {
        imagesPerPageSelect.addEventListener('change', async function() {
            imagesPerPage = parseInt(this.value);
            currentPage = 0;
            await loadImages();
        });
    }

    if (prevPageBtn) {
        prevPageBtn.addEventListener('click', async function() {
            if (currentPage > 0) {
                currentPage--;
                await loadImages();
            }
        });
    }

    if (nextPageBtn) {
        nextPageBtn.addEventListener('click', async function() {
            currentPage++;
            await loadImages();
        });
    }

    if (goToPageBtn && pageInput) {
        goToPageBtn.addEventListener('click', async function() {
            const page = parseInt(pageInput.value) - 1;
            if (page >= 0) {
                currentPage = page;
                await loadImages();
            }
        });
    }
});
