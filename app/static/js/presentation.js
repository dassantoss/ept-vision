// Initialize AOS (Animate On Scroll)
AOS.init({
    duration: 800,
    once: true,
    offset: 100
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Active navigation link based on scroll position
window.addEventListener('scroll', () => {
    let current = '';
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (scrollY >= sectionTop - 100) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').slice(1) === current) {
            link.classList.add('active');
        }
    });
});

// Navbar background opacity on scroll
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.backgroundColor = 'rgba(33, 37, 41, 0.95)';
    } else {
        navbar.style.backgroundColor = 'rgba(33, 37, 41, 0.8)';
    }
});

// Initialize tooltips
const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
});

// Section Navigation
const sections = [
    'introduction',
    'demo',
    'architecture',
    'training',
    'dataset',
    'development',
    'ethics',
    'next-steps',
    'conclusions'
];

let currentSectionIndex = 0;

// Function to scroll to a specific section
function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({ behavior: 'smooth' });
    updateActiveDot(sectionId);
}

// Function to update the active dot
function updateActiveDot(sectionId) {
    document.querySelectorAll('.nav-dot').forEach(dot => {
        dot.classList.remove('active');
    });
    
    // Handle special case for next-steps section
    let tooltipText = sectionId
        .split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    
    const dot = document.querySelector(`.nav-dot[data-tooltip="${tooltipText}"]`);
    if (dot) {
        dot.classList.add('active');
    }
}

// Function to handle keyboard navigation
function handleKeyPress(e) {
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        e.preventDefault();
        if (currentSectionIndex < sections.length - 1) {
            currentSectionIndex++;
            scrollToSection(sections[currentSectionIndex]);
        }
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        e.preventDefault();
        if (currentSectionIndex > 0) {
            currentSectionIndex--;
            scrollToSection(sections[currentSectionIndex]);
        }
    }
}

// Function to handle scroll and update active section
function handleScroll() {
    const scrollPosition = window.scrollY;
    
    sections.forEach((sectionId, index) => {
        const section = document.getElementById(sectionId);
        const sectionTop = section.offsetTop - 100;
        const sectionBottom = sectionTop + section.offsetHeight;
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
            currentSectionIndex = index;
            updateActiveDot(sectionId);
        }
    });
}

// Add event listeners
document.addEventListener('keydown', handleKeyPress);
window.addEventListener('scroll', handleScroll);

// Set initial active dot
document.addEventListener('DOMContentLoaded', () => {
    updateActiveDot(sections[0]);
}); 