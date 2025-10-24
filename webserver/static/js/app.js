// HLS Video Player Setup
const video = document.getElementById("video-player");
const videoSource = "/output/playlist.m3u8";
const errorContainer = document.getElementById("error-container");
const errorMessage = document.getElementById("error-message");
const streamStatus = document.getElementById("stream-status");

function showError(msg) {
  errorContainer.classList.remove("d-none");
  errorMessage.textContent = msg;
  streamStatus.innerHTML = `
    <span class="status-badge offline">
      <span class="status-dot"></span>
      Disconnected
    </span>
  `;
}

function hideError() {
  errorContainer.classList.add("d-none");
  streamStatus.innerHTML = `
    <span class="status-badge online">
      <span class="status-dot"></span>
      Connected
    </span>
  `;
}

// Check if HLS is natively supported (Safari, iOS)
if (video.canPlayType("application/vnd.apple.mpegurl")) {
  video.src = videoSource;
  video.addEventListener("loadedmetadata", function () {
    console.log("HLS stream loaded (native)");
    hideError();
  });
  video.addEventListener("error", function () {
    showError("Failed to load video stream (native player)");
  });
}
// Use hls.js for other browsers
else if (Hls.isSupported()) {
  const hls = new Hls({
    enableWorker: true,
    lowLatencyMode: true,
    backBufferLength: 90,
  });

  hls.loadSource(videoSource);
  hls.attachMedia(video);

  hls.on(Hls.Events.MANIFEST_PARSED, function () {
    console.log("HLS manifest parsed");
    hideError();
    video.play().catch((e) => {
      console.log("Autoplay prevented:", e);
    });
  });

  hls.on(Hls.Events.ERROR, function (event, data) {
    console.error("HLS Error:", data);
    if (data.fatal) {
      switch (data.type) {
        case Hls.ErrorTypes.NETWORK_ERROR:
          showError("Network error - attempting to recover...");
          hls.startLoad();
          break;
        case Hls.ErrorTypes.MEDIA_ERROR:
          showError("Media error - attempting to recover...");
          hls.recoverMediaError();
          break;
        default:
          showError("Fatal error - cannot recover stream");
          hls.destroy();
          break;
      }
    }
  });
} else {
  showError("Browser tidak mendukung HLS streaming");
}

// Uptime counter
let uptimeSeconds = 0;
setInterval(() => {
  uptimeSeconds++;
  const hours = Math.floor(uptimeSeconds / 3600);
  const minutes = Math.floor((uptimeSeconds % 3600) / 60);
  const seconds = uptimeSeconds % 60;
  document.getElementById("uptime").textContent = `${String(hours).padStart(
    2,
    "0"
  )}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}, 1000);

// Fetch FPS data from API
async function updateStats() {
  try {
    const response = await fetch("/api/stats");
    if (response.ok) {
      const data = await response.json();

      // Update FPS display
      if (data.fps !== undefined && data.fps > 0) {
        document.getElementById("fps").textContent = `${data.fps.toFixed(
          1
        )} fps`;
      } else {
        document.getElementById("fps").textContent = "-- fps";
      }

      // Update target FPS
      if (data.target_fps !== undefined) {
        document.getElementById("target-fps").textContent = data.target_fps;
      }
    }
  } catch (error) {
    console.error("Error fetching stats:", error);
    document.getElementById("fps").textContent = "-- fps";
  }
}

// Update stats every 1 second
setInterval(updateStats, 1000);

// Initial fetch
updateStats();

// Hamburger Menu Toggle
const hamburgerBtn = document.getElementById("hamburger-btn");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");

function toggleSidebar() {
  hamburgerBtn.classList.toggle("active");
  sidebar.classList.toggle("active");
  sidebarOverlay.classList.toggle("active");
}

hamburgerBtn.addEventListener("click", toggleSidebar);
sidebarOverlay.addEventListener("click", toggleSidebar);

// Close sidebar on nav link click (mobile)
const navLinks = document.querySelectorAll(".nav-link");
navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    if (window.innerWidth <= 768) {
      toggleSidebar();
    }
  });
});

// Handle window resize
window.addEventListener("resize", () => {
  if (window.innerWidth > 768) {
    sidebar.classList.remove("active");
    sidebarOverlay.classList.remove("active");
    hamburgerBtn.classList.remove("active");
  }
});
