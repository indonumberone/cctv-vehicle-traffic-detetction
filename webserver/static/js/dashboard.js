// Theme Toggle Functionality
const themeToggle = document.getElementById("theme-toggle");
const lightIcon = document.getElementById("light-icon");
const darkIcon = document.getElementById("dark-icon");
const htmlElement = document.documentElement;

// Load theme from localStorage
const currentTheme = localStorage.getItem("theme") || "dark";
if (currentTheme === "light") {
  htmlElement.classList.add("light");
  lightIcon.classList.remove("hidden");
  darkIcon.classList.add("hidden");
} else {
  htmlElement.classList.remove("light");
  lightIcon.classList.add("hidden");
  darkIcon.classList.remove("hidden");
}

// Toggle theme
themeToggle.addEventListener("click", () => {
  if (htmlElement.classList.contains("light")) {
    htmlElement.classList.remove("light");
    lightIcon.classList.add("hidden");
    darkIcon.classList.remove("hidden");
    localStorage.setItem("theme", "dark");
  } else {
    htmlElement.classList.add("light");
    lightIcon.classList.remove("hidden");
    darkIcon.classList.add("hidden");
    localStorage.setItem("theme", "light");
  }
});

// Vehicle Counts and Chart
let vehicleChart = null;

// Initialize Chart
function initializeChart() {
  const ctx = document.getElementById("vehicleChart").getContext("2d");

  // Get theme
  const isDarkMode = !document.documentElement.classList.contains("light");
  const gridColor = isDarkMode
    ? "rgba(255, 255, 255, 0.1)"
    : "rgba(0, 0, 0, 0.1)";
  const textColor = isDarkMode
    ? "rgba(255, 255, 255, 0.7)"
    : "rgba(0, 0, 0, 0.7)";

  vehicleChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Car",
          data: [],
          borderColor: "rgb(59, 130, 246)",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          tension: 0.4,
          fill: true,
          pointRadius: 3,
          pointHoverRadius: 5,
        },
        {
          label: "Motorcycle",
          data: [],
          borderColor: "rgb(34, 197, 94)",
          backgroundColor: "rgba(34, 197, 94, 0.1)",
          tension: 0.4,
          fill: true,
          pointRadius: 3,
          pointHoverRadius: 5,
        },
        {
          label: "Truck",
          data: [],
          borderColor: "rgb(249, 115, 22)",
          backgroundColor: "rgba(249, 115, 22, 0.1)",
          tension: 0.4,
          fill: true,
          pointRadius: 3,
          pointHoverRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      animation: {
        duration: 750,
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: {
            color: textColor,
            usePointStyle: true,
            padding: 15,
            font: {
              size: 12,
            },
          },
        },
        tooltip: {
          backgroundColor: isDarkMode
            ? "rgba(0, 0, 0, 0.8)"
            : "rgba(255, 255, 255, 0.9)",
          titleColor: isDarkMode ? "#fff" : "#000",
          bodyColor: isDarkMode ? "#fff" : "#000",
          borderColor: isDarkMode
            ? "rgba(255, 255, 255, 0.1)"
            : "rgba(0, 0, 0, 0.1)",
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          grid: {
            color: gridColor,
          },
          ticks: {
            color: textColor,
            maxRotation: 45,
            minRotation: 45,
            autoSkip: true,
            maxTicksLimit: 12,
          },
        },
        y: {
          beginAtZero: true,
          grid: {
            color: gridColor,
          },
          ticks: {
            color: textColor,
            precision: 0,
          },
        },
      },
    },
  });
}

// Fetch vehicle counts
async function updateVehicleCounts() {
  try {
    const response = await fetch("/api/vehicle-counts");
    if (response.ok) {
      const data = await response.json();

      // Update count displays with animation
      updateCountWithAnimation("car-count", data.car || 0);
      updateCountWithAnimation("motorcycle-count", data.motorcycle || 0);
      updateCountWithAnimation("truck-count", data.truck || 0);
    }
  } catch (error) {
    console.error("Error fetching vehicle counts:", error);
  }
}

// Animate count update
function updateCountWithAnimation(elementId, newValue) {
  const element = document.getElementById(elementId);
  const currentValue = parseInt(element.textContent) || 0;

  if (currentValue !== newValue) {
    element.textContent = newValue;
  }
}

// Fetch chart data
async function updateVehicleChart() {
  try {
    const hours = document.getElementById("chart-time-range").value;

    // Determine interval based on time range
    let interval;
    if (parseFloat(hours) <= 0.5) {
      interval = "1m"; // 1 minute for 10-30 min range
    } else if (parseFloat(hours) <= 3) {
      interval = "5m"; // 5 minutes for 1-3 hour range
    } else if (parseFloat(hours) <= 12) {
      interval = "10m"; // 10 minutes for 6-12 hour range
    } else {
      interval = "30m"; // 30 minutes for 24 hour range
    }

    const response = await fetch(
      `/api/vehicle-chart?hours=${hours}&interval=${interval}`
    );

    if (response.ok) {
      const data = await response.json();

      // Combine all timestamps
      const allTimestamps = new Set();
      ["car", "motorcycle", "truck"].forEach((vehicleType) => {
        data[vehicleType].forEach((item) => {
          allTimestamps.add(item.time);
        });
      });

      // Sort timestamps
      const sortedTimestamps = Array.from(allTimestamps).sort();

      // Create data maps
      const carMap = new Map(data.car.map((item) => [item.time, item.count]));
      const motorcycleMap = new Map(
        data.motorcycle.map((item) => [item.time, item.count])
      );
      const truckMap = new Map(
        data.truck.map((item) => [item.time, item.count])
      );

      // Format labels and data
      const labels = sortedTimestamps.map((timestamp) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString("id-ID", {
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        });
      });

      const carData = sortedTimestamps.map((t) => carMap.get(t) || 0);
      const motorcycleData = sortedTimestamps.map(
        (t) => motorcycleMap.get(t) || 0
      );
      const truckData = sortedTimestamps.map((t) => truckMap.get(t) || 0);

      // Update chart
      if (vehicleChart) {
        vehicleChart.data.labels = labels;
        vehicleChart.data.datasets[0].data = carData;
        vehicleChart.data.datasets[1].data = motorcycleData;
        vehicleChart.data.datasets[2].data = truckData;
        vehicleChart.update("none"); // Update without animation for better performance
      }
    }
  } catch (error) {
    console.error("Error fetching vehicle chart data:", error);
  }
}

// Initialize chart on page load
initializeChart();

// Update vehicle data every 10 seconds (reduced frequency)
setInterval(updateVehicleCounts, 10000);
updateVehicleCounts();

// Update chart every 60 seconds (reduced frequency)
setInterval(updateVehicleChart, 60000);
updateVehicleChart();

// Update chart when time range changes
document
  .getElementById("chart-time-range")
  .addEventListener("change", updateVehicleChart);

// Update chart theme when theme changes
themeToggle.addEventListener("click", () => {
  setTimeout(() => {
    if (vehicleChart) {
      vehicleChart.destroy();
      initializeChart();
      updateVehicleChart();
    }
  }, 100);
});

// Hamburger Menu Toggle
const hamburgerBtn = document.getElementById("hamburger-btn");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");

function toggleSidebar() {
  sidebar.classList.toggle("translate-x-0");
  sidebar.classList.toggle("-translate-x-full");
  sidebarOverlay.classList.toggle("hidden");

  // Animate hamburger button
  const spans = hamburgerBtn.querySelectorAll("span");
  if (sidebar.classList.contains("translate-x-0")) {
    spans[0].style.transform = "rotate(45deg) translate(8px, 8px)";
    spans[1].style.opacity = "0";
    spans[2].style.transform = "rotate(-45deg) translate(7px, -7px)";
  } else {
    spans[0].style.transform = "";
    spans[1].style.opacity = "";
    spans[2].style.transform = "";
  }
}

hamburgerBtn.addEventListener("click", toggleSidebar);
sidebarOverlay.addEventListener("click", toggleSidebar);

// Close sidebar on nav link click (mobile)
const navLinks = document.querySelectorAll(".nav-link");
navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    if (window.innerWidth <= 1024) {
      toggleSidebar();
    }
  });
});

// Handle window resize
window.addEventListener("resize", () => {
  if (window.innerWidth > 1024) {
    sidebar.classList.remove("-translate-x-full");
    sidebar.classList.add("translate-x-0");
    sidebarOverlay.classList.add("hidden");
    const spans = hamburgerBtn.querySelectorAll("span");
    spans[0].style.transform = "";
    spans[1].style.opacity = "";
    spans[2].style.transform = "";
  } else {
    sidebar.classList.add("-translate-x-full");
    sidebar.classList.remove("translate-x-0");
  }
});
