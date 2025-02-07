// Función para añadir la barra de navegación personalizada
function addCustomNavigation() {
    // Crear la barra de navegación
    const nav = document.createElement('div');
    nav.className = 'custom-nav';
    nav.innerHTML = `
        <a href="/analyze" style="margin-right: 15px;">
            <i class="bi bi-camera"></i> Demo
        </a>
    `;

    // Insertar la barra de navegación después del header de Swagger
    const header = document.querySelector('.swagger-ui .topbar');
    if (header && !document.querySelector('.custom-nav')) {
        header.parentNode.insertBefore(nav, header.nextSibling);
    }
}

// Esperar a que el DOM esté completamente cargado
window.addEventListener('load', function() {
    // Añadir la barra de navegación
    addCustomNavigation();

    // Añadir los estilos de Bootstrap Icons
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css';
    document.head.appendChild(link);
}); 