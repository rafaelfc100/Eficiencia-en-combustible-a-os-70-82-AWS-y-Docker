document.getElementById("predictForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const features = [
        parseInt(document.getElementById("cylinders").value),
        parseFloat(document.getElementById("displacement").value),
        parseFloat(document.getElementById("horsepower").value),
        parseFloat(document.getElementById("weight").value),
        parseFloat(document.getElementById("acceleration").value),
        parseInt(document.getElementById("model_year").value),
        parseInt(document.getElementById("origin").value)
    ];

    // URL dinámica - usa el hostname actual para desarrollo y producción
    const backendUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? 'http://localhost:5000' 
        : 'http://18.116.200.132:5000';

    try {
        const response = await fetch(`${backendUrl}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: features })
        });

        if (!response.ok) {
            throw new Error(`Error HTTP: ${response.status}`);
        }

        const data = await response.json();
        document.getElementById("result").innerText = data.prediction;

        const img = document.getElementById("result-img");
        if (data.prediction.includes("eficiente")) {
            img.src = "img/eficiente.jpg";
            img.style.display = "block";
        } else {
            img.src = "img/menos_eficiente.jpg";
            img.style.display = "block";
        }

    } catch (error) {
        document.getElementById("result").innerText = "Error al conectar con el backend: " + error.message;
        console.error("Error completo:", error);
    }
});